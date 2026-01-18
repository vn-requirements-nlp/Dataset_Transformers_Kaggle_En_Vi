#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create multi-label STRATIFIED train/val/test splits and save indices to JSON.

Why this is "paper-grade":
- fixed split file => reproducible experiments
- multi-label stratification => label distribution is similar across splits (important for rare NFR labels)

Requires:
  pip install iterative-stratification
"""

import argparse
import json
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


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
    label2id = obj["label2id"]
    label2id = {str(k): int(v) for k, v in label2id.items()}
    num_labels = len(label_names)
    return label_names, label2id, num_labels


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


def main():
    ap = argparse.ArgumentParser(description="Make multi-label stratified splits and save indices")
    ap.add_argument("--data_path", required=True, help="JSONL dataset path")
    ap.add_argument("--labelmap_path", required=True, help="labelmap json path")
    ap.add_argument("--out_split", default="data/splits/split_seed42.json", help="output split json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--export_dir", default=None, help="If set, export train/val/test.jsonl here")
    args = ap.parse_args()

    data_path = Path(args.data_path)
    labelmap_path = Path(args.labelmap_path)
    out_split = Path(args.out_split)
    out_split.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(data_path)
    label_names, label2id, num_labels = load_labelmap(labelmap_path)

    Y = np.stack([to_multihot(r, label2id, num_labels) for r in rows], axis=0)
    idx = np.arange(len(rows))

    temp_ratio = args.val_ratio + args.test_ratio
    if not (0 < temp_ratio < 1):
        raise ValueError("val_ratio + test_ratio must be in (0, 1)")

    # Split: train vs (val+test)
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=temp_ratio, random_state=args.seed
    )
    train_idx, temp_idx = next(msss1.split(idx, Y))

    # Split: temp -> val vs test (relative)
    test_rel = args.test_ratio / temp_ratio
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_rel, random_state=args.seed
    )
    val_pos, test_pos = next(msss2.split(np.arange(len(temp_idx)), Y[temp_idx]))
    val_idx = temp_idx[val_pos]
    test_idx = temp_idx[test_pos]

    train_idx = sorted(map(int, train_idx))
    val_idx = sorted(map(int, val_idx))
    test_idx = sorted(map(int, test_idx))

    # Sanity
    if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
        raise RuntimeError("Split overlap detected")
    if len(train_idx) + len(val_idx) + len(test_idx) != len(rows):
        raise RuntimeError("Split size mismatch")

    split_obj = {
        "seed": args.seed,
        "data_path": str(data_path),
        "labelmap_path": str(labelmap_path),
        "n": len(rows),
        "test_ratio": args.test_ratio,
        "val_ratio": args.val_ratio,
        "train_ratio": 1.0 - temp_ratio,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "label_names": label_names,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    out_split.write_text(json.dumps(split_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Saved split:", out_split)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    if args.export_dir:
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        def dump(name, indices):
            p = export_dir / f"{name}.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for i in indices:
                    f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
            print("✅ Exported:", p)

        dump("train", train_idx)
        dump("val", val_idx)
        dump("test", test_idx)


if __name__ == "__main__":
    main()
