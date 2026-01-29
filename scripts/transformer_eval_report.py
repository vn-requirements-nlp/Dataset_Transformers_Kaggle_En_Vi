#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-grade evaluation script:
- Evaluate on a FIXED split (from split_seedXX.json)
- Apply per-label thresholds (thresholds.json) tuned on VAL
- Print per-label classification_report and summary metrics
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
    try:
        from transformers.utils import logging as _hf_logging
        _hf_logging.set_verbosity_error()
    except Exception:
        pass

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

        try:
            os.dup2(self._orig_fd, 2)
        finally:
            try:
                os.close(self._orig_fd)
            except Exception:
                pass

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
        else:
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
        nonlocal_vars = None  # no-op, keeps snippet copy/paste friendly
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
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labelmap(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    label_names = obj.get("label_names") or obj.get("label_columns")
    if not label_names:
        raise ValueError("labelmap must contain 'label_names' or 'label_columns'")
    label2id = {str(k): int(v) for k, v in obj["label2id"].items()}
    id2label = obj.get("id2label")
    if isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}
    else:
        id2label = {i: label_names[i] for i in range(len(label_names))}
    return label_names, label2id, id2label


def to_multihot(row, label2id, num_labels: int):
    y = np.zeros(num_labels, dtype=np.int32)
    if "label_ids" in row and isinstance(row["label_ids"], list):
        for lid in row["label_ids"]:
            y[int(lid)] = 1
        return y
    for lab in row.get("labels", []):
        if str(lab) in label2id:
            y[label2id[str(lab)]] = 1
    return y


def load_thresholds(thresholds_json: str | None):
    if not thresholds_json:
        return None
    obj = json.loads(Path(thresholds_json).read_text(encoding="utf-8"))
    # Accept 2 formats:
    # 1) {"Label": {"thr": 0.42, ...}, ...}
    # 2) {"Label": 0.42, ...}
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict) and "thr" in v:
            out[k] = float(v["thr"])
        else:
            out[k] = float(v)
    return out


def apply_thresholds(probs: np.ndarray, label_names, thresholds: dict | None, fallback_thr: float):
    y_pred = np.zeros_like(probs, dtype=np.int32)
    used = []
    for j, name in enumerate(label_names):
        thr = thresholds.get(name, fallback_thr) if thresholds else fallback_thr
        used.append(thr)
        y_pred[:, j] = (probs[:, j] >= thr).astype(np.int32)
    return y_pred, used


def main():
    ap = argparse.ArgumentParser(description="Evaluate multi-label model on fixed split with per-label thresholds")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--labelmap_path", required=True)
    ap.add_argument("--split_path", default=None, help="split_seedXX.json ... (optional if --eval_all)")
    ap.add_argument("--eval_all", action="store_true", help="Evaluate on ALL rows of data_path (ignore split)")
    ap.add_argument("--split_name", default="test", choices=["train", "val", "test"])
    ap.add_argument("--thresholds_json", default=None, help="per-label thresholds (tuned on val). Optional.")
    ap.add_argument("--threshold", type=float, default=0.5, help="fallback threshold if label missing in thresholds_json")
    ap.add_argument("--max_length", type=int, default=None, help="Max length (default: read from train_config.json if available)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--run_name", default=None, help="Title to show on console/report header (e.g., 'VAL-SILVER seed42')")

    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--use_vitokenizer",
        dest="use_vitokenizer",
        action="store_true",
        help="Tokenize Vietnamese text with pyvi (word segmentation).",
    )
    g.add_argument(
        "--no_vitokenizer",
        dest="use_vitokenizer",
        action="store_false",
        help="Disable Vietnamese word segmentation.",
    )
    ap.set_defaults(use_vitokenizer=None)
    ap.add_argument("--out_report", default=None, help="Save classification_report text to file")
    ap.add_argument("--out_metrics", default=None, help="Save summary metrics json to file")
    args = ap.parse_args()

    cfg = load_train_config(Path(args.model_dir))
    if args.use_vitokenizer is None:
        args.use_vitokenizer = bool(cfg.get("preprocess", {}).get("use_vitokenizer", False))
    if args.max_length is None:
        args.max_length = int(cfg.get("tokenization", {}).get("max_length", 256))

    label_names, label2id, _ = load_labelmap(Path(args.labelmap_path))
    num_labels = len(label_names)

    rows = read_jsonl(Path(args.data_path))
    if args.eval_all:
        subset = rows
    else:
        if not args.split_path:
            raise ValueError("Need --split_path unless using --eval_all")
        split_obj = json.loads(Path(args.split_path).read_text(encoding="utf-8"))
        key = {"train": "train_idx", "val": "val_idx", "test": "test_idx"}[args.split_name]
        idxs = split_obj[key]
        subset = [rows[i] for i in idxs]

    texts = [maybe_tokenize_vi(r["text"], True) for r in subset] if args.use_vitokenizer else [r["text"] for r in subset]
    y_true = np.stack([to_multihot(r, label2id, num_labels) for r in subset], axis=0)

    seed_words_map = load_seed_words_from_model_dir(Path(args.model_dir))
    model_cfg = AutoConfig.from_pretrained(args.model_dir)
    use_seed_words = int(getattr(model_cfg, "seed_words_dim", 0) or 0) > 0
    if not use_seed_words and seed_words_enabled(seed_words_map):
        print("[WARN] seed_words.json found but model was trained without seed features. Ignoring.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    if use_seed_words:
        model = SeededSequenceClassifier.from_pretrained(args.model_dir, config=model_cfg)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, config=model_cfg)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    probs_all = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i+args.batch_size]
        seed_feats = None
        if use_seed_words:
            seed_feats = build_seed_features(batch, seed_words_map, label_names)
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=args.max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        if seed_feats is not None:
            enc["seed_feats"] = torch.tensor(seed_feats, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(**enc).logits.detach().cpu().numpy()
        probs_all.append(sigmoid(logits))

    probs = np.concatenate(probs_all, axis=0)

    thr_map = load_thresholds(args.thresholds_json)
    y_pred, used_thrs = apply_thresholds(probs, label_names, thr_map, args.threshold)

    # Summary metrics
    metrics = {
        "split": args.split_name,
        "n": int(y_true.shape[0]),
        "threshold_fallback": float(args.threshold),
        "thresholds_json": args.thresholds_json,
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
    }

    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    data_tag = Path(args.data_path).name
    mode = "ALL" if args.eval_all else args.split_name.upper()
    title = args.run_name or f"{mode} | data={data_tag}"

    print("\n" + "="*100)
    print(f"📊 Classification report (per label) | {title}")
    print("="*100)
    print(report)
    print("\n📌 Summary metrics:")
    for k in ["f1_micro","f1_macro","precision_micro","recall_micro","exact_match"]:
        print(f"- {k}: {metrics[k]:.6f}")

    if args.out_report:
        Path(args.out_report).write_text(report, encoding="utf-8")
        print("Saved report:", args.out_report)

    if args.out_metrics:
        Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved metrics:", args.out_metrics)


if __name__ == "__main__":
    main()
