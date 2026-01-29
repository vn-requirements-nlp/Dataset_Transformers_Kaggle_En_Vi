#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict multi-label requirements from a .txt file (one requirement per line) and export CSV UTF-8 with BOM
for correct Vietnamese display in Excel.

Notes:
- Supports --use_vitokenizer / --no_vitokenizer to match training preprocessing.
- If not provided, will auto-read from <model_dir>/train_config.json (preprocess.use_vitokenizer).
- Hides noisy CUDA/XLA/absl logs by capturing STDERR early; on crash, captured STDERR is printed.
"""

# -----------------------------
# Quiet mode: hide noisy library logs/warnings (TF/absl/cuda, FutureWarning, etc.)
# - By default we suppress STDERR and only print it if the script crashes.
# - Set environment variable QUIET_STDERR=0 to disable.
# -----------------------------
import os
import sys
import tempfile
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BAR", "1")

_QUIET_STDERR = os.environ.get("QUIET_STDERR", "1") not in ("0", "false", "False", "NO", "no")

if _QUIET_STDERR:
    warnings.filterwarnings("ignore")

class _StderrCaptureOnError:
    """Redirect OS-level STDERR (fd=2) to a temp file; dump it only if an exception happens."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._orig_fd = None
        self._tmp = None
        self._tmp_path = None

    def __enter__(self):
        if not self.enabled:
            return self
        self._orig_fd = os.dup(2)
        self._tmp = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        self._tmp_path = self._tmp.name
        os.dup2(self._tmp.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False

        # restore stderr
        try:
            os.dup2(self._orig_fd, 2)
        finally:
            try:
                os.close(self._orig_fd)
            except Exception:
                pass

        # flush/close temp
        try:
            try:
                self._tmp.flush()
            except Exception:
                pass
            try:
                self._tmp.close()
            except Exception:
                pass
        finally:
            self._tmp = None

        # on error: dump captured stderr
        if exc_type is not None and self._tmp_path and os.path.exists(self._tmp_path):
            try:
                data = open(self._tmp_path, "rb").read()
                if data:
                    sys.stderr.write("\n===== Captured STDERR (suppressed logs/warnings) =====\n")
                    try:
                        sys.stderr.buffer.write(data)
                    except Exception:
                        sys.stderr.write(data.decode("utf-8", errors="replace"))
                    sys.stderr.write("\n===== End Captured STDERR =====\n")
            finally:
                try:
                    os.remove(self._tmp_path)
                except Exception:
                    pass
            return False

        # success: discard
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.remove(self._tmp_path)
            except Exception:
                pass
        return False

# Start capturing STDERR as early as possible (hide XLA/CUDA/absl logs during imports)
if _QUIET_STDERR:
    _stderr_cap = _StderrCaptureOnError(enabled=True)
    _stderr_cap.__enter__()
    _stderr_closed = False

    import atexit
    _orig_excepthook = sys.excepthook

    def _close_stderr_capture(exc_type=None, exc=None, tb=None):
        global _stderr_closed
        if _stderr_closed:
            return
        _stderr_closed = True
        _stderr_cap.__exit__(exc_type, exc, tb)

    def _excepthook(exc_type, exc, tb):
        _close_stderr_capture(exc_type, exc, tb)  # dump captured stderr on crash
        _orig_excepthook(exc_type, exc, tb)

    sys.excepthook = _excepthook
    atexit.register(lambda: _close_stderr_capture(None, None, None))  # discard on success

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from transformer_feature_utils import (
    build_seed_features,
    load_seed_words_from_model_dir,
    seed_words_enabled,
)
from transformer_seeded_model import SeededSequenceClassifier


def load_train_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "train_config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def maybe_tokenize_vi(text: str, use_vitokenizer: bool) -> str:
    if not use_vitokenizer:
        return text
    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    except Exception:
        return text


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_thresholds(thresholds_json: Optional[str]) -> Optional[Dict[str, float]]:
    """Load per-label thresholds.

    Expected formats:
      {
        "Label A": {"thr": 0.39, ...},
        "Label B": 0.5,
        ...
      }

    Returns: {label_name: thr_float} or None.
    """
    if not thresholds_json:
        return None

    with open(thresholds_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    thr_map: Dict[str, float] = {}
    for label, v in raw.items():
        if isinstance(v, dict) and "thr" in v:
            thr_map[label] = float(v["thr"])
        elif isinstance(v, (float, int)):
            thr_map[label] = float(v)
    return thr_map


def load_model(model_dir: str) -> Tuple[AutoTokenizer, torch.nn.Module, str, Dict[int, str]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    config = AutoConfig.from_pretrained(model_dir)
    if int(getattr(config, "seed_words_dim", 0) or 0) > 0:
        model = SeededSequenceClassifier.from_pretrained(model_dir, config=config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # id2label in config sometimes has str keys
    id2label_cfg = getattr(model.config, "id2label", None)
    if id2label_cfg and isinstance(next(iter(id2label_cfg.keys())), str):
        id2label = {int(k): v for k, v in id2label_cfg.items()}
    else:
        id2label = id2label_cfg or {}

    if not id2label:
        # fallback
        num_labels = getattr(model.config, "num_labels", None)
        if num_labels is None:
            num_labels = model.classifier.out_proj.out_features if hasattr(model, "classifier") else 0
        id2label = {i: f"LABEL_{i}" for i in range(int(num_labels))}

    return tokenizer, model, device, id2label


def predict_proba_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: str,
    max_length: int,
    seed_feats: Optional[np.ndarray] = None,
) -> np.ndarray:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if seed_feats is not None:
        inputs["seed_feats"] = torch.tensor(seed_feats, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(**inputs).logits.detach().cpu().numpy()

    return sigmoid(logits)  # shape: [N, L]


def read_txt_lines(path: str) -> List[str]:
    # Accept UTF-8 (with/without BOM). Also handles Windows newlines.
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def write_csv_utf8_sig(path: str, fieldnames: List[str], rows: List[Dict[str, object]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Dự đoán nhãn multi-label cho nhiều requirement từ file .txt và xuất CSV UTF-8 (có BOM) để mở bằng Excel không lỗi tiếng Việt."
        )
    )

    ap.add_argument("--model_dir", type=str, default="models/phobert-multilabel")

    # Input source (required)
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--text", type=str, help="Một câu requirement duy nhất")
    inp.add_argument("--input_txt", type=str, help="Đường dẫn file .txt (mỗi dòng là 1 requirementText)")

    # ViTokenizer option (optional; default None => auto from train_config.json)
    vt = ap.add_mutually_exclusive_group()
    vt.add_argument("--use_vitokenizer", dest="use_vitokenizer", action="store_true",
                    help="Tokenize Vietnamese text with pyvi (word segmentation).")
    vt.add_argument("--no_vitokenizer", dest="use_vitokenizer", action="store_false",
                    help="Disable Vietnamese word segmentation.")
    ap.set_defaults(use_vitokenizer=None)

    ap.add_argument("--output_csv", type=str, default="predictions.csv",
                    help="File CSV đầu ra (UTF-8 with BOM).")

    # threshold chung (fallback)
    ap.add_argument("--threshold", type=float, default=0.5)

    # threshold per-label
    ap.add_argument(
        "--thresholds_json",
        type=str,
        default=None,
        help=(
            "Đường dẫn thresholds.json (per-label). Nếu có, sẽ ưu tiên dùng theo từng nhãn; nhãn nào thiếu sẽ fallback về --threshold."
        ),
    )

    # Let max_length default to train_config.json if possible
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)

    # extras
    ap.add_argument("--include_probs", action="store_true",
                    help="Xuất thêm cột xác suất cho mỗi nhãn (suffix: __prob).")
    ap.add_argument("--include_active_labels", action="store_true",
                    help="Xuất thêm cột ActiveLabels (danh sách nhãn=1).")

    args = ap.parse_args()

    # Auto defaults from train_config.json
    cfg = load_train_config(Path(args.model_dir))
    if args.use_vitokenizer is None:
        args.use_vitokenizer = bool(cfg.get("preprocess", {}).get("use_vitokenizer", False))
    if args.max_length is None:
        args.max_length = int(cfg.get("tokenization", {}).get("max_length", 256))

    # Reduce HF console noise (still compatible with our stderr capture)
    try:
        from transformers.utils import logging as _hf_logging
        _hf_logging.set_verbosity_error()
    except Exception:
        pass

    tokenizer, model, device, id2label = load_model(args.model_dir)
    labels_in_order = [id2label[i] for i in sorted(id2label.keys())]

    seed_words_map = load_seed_words_from_model_dir(Path(args.model_dir))
    seed_dim = int(getattr(model.config, "seed_words_dim", 0) or 0)
    use_seed_words = seed_dim > 0
    if not use_seed_words and seed_words_enabled(seed_words_map):
        print("[WARN] seed_words.json found but model was trained without seed features. Ignoring.")

    thr_map = load_thresholds(args.thresholds_json)
    thr_vec = np.array([thr_map.get(lab, args.threshold) if thr_map else args.threshold for lab in labels_in_order])

    if args.text is not None:
        texts = [args.text.strip()]
    else:
        texts = read_txt_lines(args.input_txt)
        if not texts:
            raise SystemExit("input_txt không có dòng nào (hoặc toàn dòng trống).")

    # Apply ViTokenizer preprocessing (must be BEFORE tokenizer)
    if args.use_vitokenizer:
        texts = [maybe_tokenize_vi(t, True) for t in texts]

    rows: List[Dict[str, object]] = []

    # batching
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        seed_feats = None
        if use_seed_words:
            seed_feats = build_seed_features(batch, seed_words_map, labels_in_order)
        probs = predict_proba_batch(
            batch,
            tokenizer,
            model,
            device,
            args.max_length,
            seed_feats=seed_feats,
        )  # [B, L]

        for t, p in zip(batch, probs):
            pred01 = (p >= thr_vec).astype(int)

            row: Dict[str, object] = {"RequirementText": t}
            for lab, v01 in zip(labels_in_order, pred01.tolist()):
                row[lab] = int(v01)

            if args.include_probs:
                for lab, pv in zip(labels_in_order, p.tolist()):
                    row[f"{lab}__prob"] = float(pv)

            if args.include_active_labels:
                active = [lab for lab, v01 in zip(labels_in_order, pred01.tolist()) if v01 == 1]
                row["ActiveLabels"] = "; ".join(active)

            rows.append(row)

    # fieldnames
    fieldnames = ["RequirementText"] + labels_in_order
    if args.include_probs:
        fieldnames += [f"{lab}__prob" for lab in labels_in_order]
    if args.include_active_labels:
        fieldnames += ["ActiveLabels"]

    write_csv_utf8_sig(args.output_csv, fieldnames, rows)

    print(f"✅ Wrote: {args.output_csv}")
    print(f"Rows: {len(rows)} | Labels: {len(labels_in_order)}")
    print(f"use_vitokenizer: {args.use_vitokenizer} | max_length: {args.max_length} | batch_size: {args.batch_size}")
    if args.thresholds_json:
        print(f"Per-label thresholds: {args.thresholds_json} (fallback={args.threshold})")
    else:
        print(f"Threshold: {args.threshold}")


if __name__ == "__main__":
    main()
