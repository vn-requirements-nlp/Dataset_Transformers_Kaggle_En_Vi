# scripts/transformer_make_splits_kfold.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create multi-label STRATIFIED k-fold splits and save indices to JSON.

This produces one split file per fold:
  <prefix>_seed{seed}_fold{fold}.json

Each file contains train_idx/val_idx/test_idx for that fold.

Requires:
  pip install iterative-stratification
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "text" not in obj:
                raise ValueError(f"Line {ln}: missing key 'text'")

            if "labels" not in obj and "label" in obj:
                obj["labels"] = obj["label"]

            if "labels" not in obj and "label_ids" not in obj:
                raise ValueError(f"Line {ln}: missing 'labels'/'label_ids' (multi-label)")

            rows.append(obj)
    return rows


def load_labelmap(path: Path) -> Tuple[List[str], Dict[str, int]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    label_names = obj.get("label_names") or obj.get("label_columns")
    if not isinstance(label_names, list):
        raise ValueError("labelmap must contain 'label_names' or 'label_columns'")
    label2id = obj.get("label2id")
    if not isinstance(label2id, dict):
        raise ValueError("labelmap missing 'label2id'")
    label2id = {str(k): int(v) for k, v in label2id.items()}
    return label_names, label2id


def to_multihot(row: Dict[str, Any], label2id: Dict[str, int], num_labels: int) -> np.ndarray:
    y = np.zeros((num_labels,), dtype=np.int64)
    if "label_ids" in row and row["label_ids"] is not None:
        for lid in row["label_ids"]:
            y[int(lid)] = 1
        return y

    labels = row.get("labels", [])
    if labels is None:
        labels = []
    if not isinstance(labels, list):
        raise ValueError("Expected 'labels' to be a list[str]")

    for lab in labels:
        k = str(lab)
        if k not in label2id:
            raise ValueError(f"Unknown label name in data: {lab}")
        y[label2id[k]] = 1
    return y


def build_multihot_matrix(
    rows: List[Dict[str, Any]],
    label2id: Dict[str, int],
    num_labels: int,
) -> np.ndarray:
    return np.stack([to_multihot(r, label2id, num_labels) for r in rows], axis=0)


def _warn_rare_labels(y: np.ndarray, label_names: List[str], n_splits: int) -> None:
    counts = y.sum(axis=0)
    rare = [(label_names[i], int(c)) for i, c in enumerate(counts) if c < n_splits]
    if not rare:
        return
    print("[WARN] Some labels appear fewer times than n_splits; stratification may be imperfect.")
    for name, c in rare:
        print(f" - {name}: {c} samples")


def _write_split_file(
    out_path: Path,
    meta: Dict[str, Any],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
) -> None:
    obj = dict(meta)
    obj["train_idx"] = train_idx
    obj["val_idx"] = val_idx
    obj["test_idx"] = test_idx
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def make_kfold_splits(
    data_path: Path,
    labelmap_path: Path,
    out_dir: Path,
    seed: int,
    n_splits: int,
    val_ratio: float,
    prefix: str,
) -> None:
    rows = read_jsonl(data_path)
    label_names, label2id = load_labelmap(labelmap_path)
    num_labels = len(label_names)
    y = build_multihot_matrix(rows, label2id, num_labels)

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in (0, 1)")

    test_ratio = 1.0 / n_splits
    train_ratio = 1.0 - test_ratio - val_ratio
    if train_ratio <= 0:
        raise ValueError("Invalid ratios: train_ratio must be > 0")

    _warn_rare_labels(y, label_names, n_splits)

    out_dir.mkdir(parents=True, exist_ok=True)

    X_dummy = np.zeros((len(rows), 1), dtype=np.float32)
    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )

    val_rel = val_ratio / (1.0 - test_ratio)

    for fold, (train_val_idx, test_idx) in enumerate(mskf.split(X_dummy, y)):
        train_val_idx = np.array(train_val_idx, dtype=np.int64)
        test_idx = np.array(test_idx, dtype=np.int64)

        y_train_val = y[train_val_idx]
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=val_rel, random_state=seed + fold
        )
        train_rel, val_rel_idx = next(msss.split(X_dummy[train_val_idx], y_train_val))

        train_idx = train_val_idx[train_rel].tolist()
        val_idx = train_val_idx[val_rel_idx].tolist()
        test_idx_list = test_idx.tolist()

        meta = {
            "seed": seed,
            "fold": fold,
            "n_splits": n_splits,
            "data_path": str(data_path),
            "labelmap_path": str(labelmap_path),
            "n": len(rows),
            "test_ratio": test_ratio,
            "val_ratio": val_ratio,
            "train_ratio": train_ratio,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx_list),
            "label_names": label_names,
        }

        out_path = out_dir / f"{prefix}_seed{seed}_fold{fold}.json"
        _write_split_file(out_path, meta, train_idx, val_idx, test_idx_list)
        print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Make k-fold stratified splits (multi-label)")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--labelmap_path", required=True)
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_splits", type=int, default=10)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--prefix", default="split_kfold")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    make_kfold_splits(
        data_path=Path(args.data_path),
        labelmap_path=Path(args.labelmap_path),
        out_dir=Path(args.out_dir),
        seed=args.seed,
        n_splits=args.n_splits,
        val_ratio=args.val_ratio,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
