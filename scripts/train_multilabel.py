#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune transformer for SENTENCE-LEVEL MULTI-LABEL classification.

This version is adjusted to MATCH your dataset template:

Each JSONL line:
{
  "text": "...",
  "labels": ["Functional (F)", "Quality (Q)", ...],   # list[str]
  "label_ids": [0, 1, 8]                              # list[int], same order as "labels"
}

Why we also load labelmap?
- To keep label ordering stable across runs (so reports show the same label names in the same order),
- And to optionally include labels that might have 0 support in your dataset (e.g. "Other (OT)").

Recommended labelmap format (like labelmap_multilabel.json):
{
  "label_columns": [...],               # or "label_names"
  "label2id": {"Functional (F)": 0, ...},
  "id2label": {"0": "Functional (F)", ...}
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

# Make the script self-contained on Kaggle (these env vars must be set BEFORE importing TF/JAX/Transformers).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BAR", "1")

_QUIET_STDERR = os.environ.get("QUIET_STDERR", "1") not in ("0", "false", "False", "NO", "no")

if _QUIET_STDERR:
    # Hide Python warnings (FutureWarning, UserWarning, etc.)
    warnings.filterwarnings("ignore")

    # Hide HuggingFace logger warnings like "Some weights ... newly initialized"
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
            return False  # don't swallow
        # Restore STDERR first
        try:
            os.dup2(self._orig_fd, 2)
        finally:
            try:
                os.close(self._orig_fd)
            except Exception:
                pass
            self._orig_fd = None

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

        # If error: print captured stderr
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
            return False  # re-raise
        else:
            # Success: discard captured stderr
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
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def maybe_tokenize_vi(text: str, use_vitokenizer: bool) -> str:
    """Optional Vietnamese word segmentation using pyvi (same as baseline).

    If pyvi is not installed, this function falls back to raw text.
    """
    if not use_vitokenizer:
        return text
    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    except Exception:
        return text



def _as_int_key_dict(d):
    """Convert dict keys to int if they are strings of digits."""
    out = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except Exception:
            out[k] = v
    return out


def load_labelmap(labelmap_path: Path):
    """
    Load labelmap with stable order.

    Accepts:
      - {"label_names": [...], "label2id": {...}, "id2label": {...}}
      - {"label_columns": [...], "label2id": {...}, "id2label": {...}}
    """
    obj = json.loads(labelmap_path.read_text(encoding="utf-8"))
    label_names = obj.get("label_names") or obj.get("label_columns")
    if not label_names:
        raise ValueError("labelmap must contain 'label_names' or 'label_columns' list")

    label2id = obj.get("label2id")
    id2label = obj.get("id2label")

    if not isinstance(label2id, dict) or not isinstance(id2label, dict):
        raise ValueError("labelmap must contain dicts: 'label2id' and 'id2label'")

    # normalize types
    label2id = {str(k): int(v) for k, v in label2id.items()}
    id2label = _as_int_key_dict(id2label)

    # sanity: label_names must align with label2id
    for i, name in enumerate(label_names):
        if str(name) not in label2id:
            raise ValueError(f"labelmap missing label2id for: {name!r}")
        if label2id[str(name)] != i:
            # We rely on label_names order == 0..N-1
            raise ValueError(
                f"labelmap order mismatch: label_names[{i}]={name!r} but label2id gives {label2id[str(name)]}"
            )

    return label_names, label2id, {int(k): v for k, v in id2label.items()}


def load_jsonl_multilabel(path: Path):
    """
    Read dataset JSONL. Must match:
      {"text": "...", "labels": [...], "label_ids": [...]}

    Backward-compat:
      - if file has {"text": "...", "label": [...]}, we map it to "labels" and will compute ids using labelmap.
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "text" not in obj:
                raise ValueError(f"Line {ln}: missing key 'text'")

            # normalize labels field name
            if "labels" not in obj and "label" in obj:
                obj["labels"] = obj["label"]

            if "labels" not in obj:
                raise ValueError(f"Line {ln}: missing key 'labels' (or legacy 'label')")

            if not isinstance(obj["labels"], list):
                raise ValueError(f"Line {ln}: 'labels' must be list[str]")

            # label_ids optional if you want to compute from labelmap
            if "label_ids" in obj and not isinstance(obj["label_ids"], list):
                raise ValueError(f"Line {ln}: 'label_ids' must be list[int]")

            rows.append(obj)

    return rows


def train_val_test_split(examples, test_ratio=0.1, val_ratio=0.1, seed=42):
    n = len(examples)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_idx = idxs[:n_test]
    val_idx = idxs[n_test : n_test + n_val]
    train_idx = idxs[n_test + n_val :]

    def subset(idxs_):
        return [examples[i] for i in idxs_]

    return subset(train_idx), subset(val_idx), subset(test_idx)


def to_hf_dataset(examples, label2id, num_labels: int):
    """
    Convert examples to HF Dataset with multi-hot vectors in key "labels" (float32).
    Priority:
      1) use label_ids if present
      2) else use label names -> label2id
    """
    texts = []
    ys = []

    for ex in examples:
        texts.append(ex["text"])
        y = np.zeros(num_labels, dtype=np.float32)

        if "label_ids" in ex and ex["label_ids"] is not None:
            for lid in ex["label_ids"]:
                if not isinstance(lid, int):
                    raise ValueError(f"label_ids must be ints, got {type(lid)}")
                if 0 <= lid < num_labels:
                    y[lid] = 1.0
        else:
            for lab in ex["labels"]:
                lid = label2id.get(str(lab))
                if lid is not None:
                    y[lid] = 1.0

        ys.append(y.tolist())

    return Dataset.from_dict({"text": texts, "labels": ys})


def compute_metrics_builder(threshold: float):
    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_true = np.array(y_true, dtype=np.int32)

        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        y_pred = (probs >= threshold).astype(np.int32)

        exact_match = float(np.mean(np.all(y_true == y_pred, axis=1)))

        return {
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_micro": precision_score(
                y_true, y_pred, average="micro", zero_division=0
            ),
            "recall_micro": recall_score(
                y_true, y_pred, average="micro", zero_division=0
            ),
            "exact_match": exact_match,
        }

    return compute_metrics


# -----------------------------
# Custom Trainer: BCEWithLogits + pos_weight for imbalance
# -----------------------------
class WeightedBCETrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight  # torch.Tensor shape [num_labels] or None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self._pos_weight is not None:
            pos_w = self._pos_weight.to(logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        else:
            loss_fct = nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune transformer for multi-label sentence classification (MATCH dataset with labels + label_ids)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="JSONL path (text + labels + label_ids)",
    )
    parser.add_argument(
        "--labelmap_path",
        type=str,
        default="data/labelmap_multilabel.json",
        help="Labelmap JSON path (stable label order).",
    )
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base")
    parser.add_argument(
        "--use_vitokenizer",
        action="store_true",
        help="Tokenize Vietnamese text with pyvi (word segmentation) before transformer tokenization.",
    )
    parser.add_argument("--output_dir", type=str, default="models/phobert-multilabel")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="Path to fixed split JSON (train/val/test indices) created by scripts/make_splits_stratified.py",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Sigmoid threshold"
    )
    parser.add_argument(
        "--use_pos_weight",
        action="store_true",
        help="Use automatic per-label pos_weight (helps rare NFR labels in imbalanced data).",
    )
    parser.add_argument(
        "--pos_weight_max",
        type=float,
        default=20.0,
        help="Clip pos_weight to this max to avoid extreme weights for very rare labels.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist preprocessing/tokenization settings so tune/eval scripts can reuse them.
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "preprocess": {"use_vitokenizer": bool(args.use_vitokenizer)},
                "tokenization": {"max_length": int(args.max_length)},
                "training": {
                    "seed": int(args.seed),
                    "use_pos_weight": bool(args.use_pos_weight),
                    "pos_weight_max": float(args.pos_weight_max),
                },
                "model": {"name": str(args.model_name)},
                "data": {
                    "data_path": str(args.data_path),
                    "labelmap_path": str(args.labelmap_path),
                    "split_path": str(args.split_path) if args.split_path else None,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    labelmap_path = Path(args.labelmap_path)
    if not labelmap_path.exists():
        raise FileNotFoundError(f"Labelmap file not found: {labelmap_path}")

    print(f"📂 Read dataset: {data_path}")
    examples = load_jsonl_multilabel(data_path)
    print(f"🧾 Total samples: {len(examples)}")

    print(f"🏷️ Load labelmap: {labelmap_path}")
    label_names, label2id, id2label = load_labelmap(labelmap_path)
    num_labels = len(label_names)
    print(f"🏷️ Num labels: {num_labels}")
    print("Label order:", label_names)

    # Quick sanity check between dataset's (labels,label_ids) and labelmap (if label_ids exist)
    # (Fails fast if mismatch)
    for ex in random.sample(examples, k=min(50, len(examples))):
        if "label_ids" not in ex:
            continue
        for lab, lid in zip(ex["labels"], ex["label_ids"]):
            if label2id.get(str(lab)) != int(lid):
                raise ValueError(
                    "Dataset label_ids do NOT match labelmap!\n"
                    f"Example text: {ex['text']}\n"
                    f"Pair: ({lab!r}, {lid}) but labelmap says {label2id.get(str(lab))}"
                )

    print("🔀 Split train/val/test ...")
    if args.split_path:
        split_obj = json.loads(Path(args.split_path).read_text(encoding="utf-8"))
        train_idx = split_obj["train_idx"]
        val_idx = split_obj["val_idx"]
        test_idx = split_obj["test_idx"]
        train_ex = [examples[i] for i in train_idx]
        val_ex = [examples[i] for i in val_idx]
        test_ex = [examples[i] for i in test_idx]
    else:
        train_ex, val_ex, test_ex = train_val_test_split(
            examples,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    print(f"  Train: {len(train_ex)}, Val: {len(val_ex)}, Test: {len(test_ex)}")

    if args.use_vitokenizer:
        # Apply the same word segmentation as baseline to make comparisons fair.
        for ex in train_ex:
            ex["text"] = maybe_tokenize_vi(ex["text"], True)
        for ex in val_ex:
            ex["text"] = maybe_tokenize_vi(ex["text"], True)
        for ex in test_ex:
            ex["text"] = maybe_tokenize_vi(ex["text"], True)

    ds_train = to_hf_dataset(train_ex, label2id, num_labels)
    ds_val = to_hf_dataset(val_ex, label2id, num_labels)
    ds_test = to_hf_dataset(test_ex, label2id, num_labels)
    
    dataset = DatasetDict({"train": ds_train, "validation": ds_val, "test": ds_test})

    # ---- pos_weight (auto) to help rare labels (imbalanced multi-label) ----
    pos_weight_t = None
    if args.use_pos_weight:
        y = np.array(dataset["train"]["labels"], dtype=np.float32)  # [N, L]
        pos = y.sum(axis=0)
        neg = y.shape[0] - pos
        pos_weight = neg / (pos + 1e-8)
        pos_weight = np.clip(pos_weight, 1.0, args.pos_weight_max)
        pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32)
        print("✅ Using pos_weight (clipped):", [round(float(x), 4) for x in pos_weight.tolist()])

    print(f"🧩 Load tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    dataset = dataset.map(tok, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    print(f"🤖 Load model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=str(out_dir / "logs"),
        logging_steps=200,
        logging_first_step=True,
        disable_tqdm=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedBCETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(threshold=args.threshold),
        pos_weight=pos_weight_t,
    )

    print("🚀 Training ...")
    trainer.train()

    print(f"💾 Save model to: {out_dir}")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Save the exact labelmap used (so predict keeps correct id2label)
    out_labelmap = out_dir / "labelmap.json"
    out_labelmap.write_text(
        json.dumps(
            {
                "label_names": label_names,
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("🎉 Done!")


if __name__ == "__main__":
    main()
