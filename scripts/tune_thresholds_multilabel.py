#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tune per-label thresholds on a FIXED validation split (paper-grade).

Output: <model_dir>/thresholds.json

thresholds.json format:
{
  "Quality (Q)": {"thr": 0.42, "f1": 0.71, "p": 0.75, "r": 0.68},
  ...
}
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
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer



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
    return label_names, label2id, len(label_names)


def to_multihot(row, label2id, num_labels: int):
    y = np.zeros(num_labels, dtype=np.int32)
    if "label_ids" in row and isinstance(row["label_ids"], list):
        for lid in row["label_ids"]:
            y[int(lid)] = 1
        return y
    for lab in row.get("labels", []):
        k = str(lab)
        if k in label2id:
            y[label2id[k]] = 1
    return y


@torch.no_grad()
def predict_probs(texts, model_dir: str, max_length: int = 256, batch_size: int = 32):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    probs_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu().numpy()
        probs_all.append(sigmoid(logits))

    return np.concatenate(probs_all, axis=0)


def best_threshold_for_label(y_true_col, prob_col, grid):
    best = {"thr": 0.5, "f1": -1.0, "p": 0.0, "r": 0.0}
    for thr in grid:
        y_pred_col = (prob_col >= thr).astype(np.int32)
        f1 = f1_score(y_true_col, y_pred_col, zero_division=0)
        if f1 > best["f1"]:
            p = precision_score(y_true_col, y_pred_col, zero_division=0)
            r = recall_score(y_true_col, y_pred_col, zero_division=0)
            best = {"thr": float(thr), "f1": float(f1), "p": float(p), "r": float(r)}
    return best


def main():
    ap = argparse.ArgumentParser(description="Tune per-label thresholds on VAL split (fixed split json).")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--labelmap_path", required=True)

    # Paper-grade: use fixed split
    ap.add_argument("--split_path", required=True, help="split_seedXX.json from scripts/make_splits_stratified.py")
    ap.add_argument("--split_name", default="val", choices=["train", "val", "test"])


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

    ap.add_argument("--max_length", type=int, default=None, help="Max length (default: read from train_config.json if available)")
    ap.add_argument("--batch_size", type=int, default=32)

    # threshold search grid
    ap.add_argument("--thr_min", type=float, default=0.05)
    ap.add_argument("--thr_max", type=float, default=0.95)
    ap.add_argument("--thr_step", type=float, default=0.01)

    args = ap.parse_args()

    cfg = load_train_config(Path(args.model_dir))
    if args.use_vitokenizer is None:
        args.use_vitokenizer = bool(cfg.get("preprocess", {}).get("use_vitokenizer", False))
    if args.max_length is None:
        args.max_length = int(cfg.get("tokenization", {}).get("max_length", 256))

    rows = read_jsonl(Path(args.data_path))
    label_names, label2id, num_labels = load_labelmap(Path(args.labelmap_path))

    split_obj = json.loads(Path(args.split_path).read_text(encoding="utf-8"))
    key = {"train": "train_idx", "val": "val_idx", "test": "test_idx"}[args.split_name]
    idxs = split_obj[key]
    subset = [rows[i] for i in idxs]

    texts = [maybe_tokenize_vi(r["text"], True) for r in subset] if args.use_vitokenizer else [r["text"] for r in subset]
    y_true = np.stack([to_multihot(r, label2id, num_labels) for r in subset], axis=0)

    probs = predict_probs(texts, args.model_dir, max_length=args.max_length, batch_size=args.batch_size)

    grid = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)

    out = {}
    f1s = []
    for j, name in enumerate(label_names):
        best = best_threshold_for_label(y_true[:, j], probs[:, j], grid)
        out[name] = best
        f1s.append(best["f1"])

    out_path = Path(args.model_dir) / "thresholds.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_f1 = float(np.mean(f1s))
    print("\n✅ Threshold tuning DONE")
    print(f"📌 Split: {args.split_name} | Samples: {len(subset)} | Labels: {num_labels}")
    print(f"📊 Avg per-label F1 ({args.split_name}): {avg_f1:.4f}")
    print(f"💾 Saved thresholds to: {out_path}")


if __name__ == "__main__":
    main()
