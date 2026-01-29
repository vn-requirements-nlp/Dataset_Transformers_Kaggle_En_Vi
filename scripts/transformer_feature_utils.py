# scripts/transformer_feature_utils.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def _normalize_key(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _label_aliases(label: str) -> List[str]:
    base = label.split("(", 1)[0].strip()
    return list({
        _normalize_key(label),
        _normalize_key(base),
    })


def load_seed_words(path: Path, label_names: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("seed_words_json must be a JSON object: {label: [word, ...]}")

    alias_to_label: Dict[str, str] = {}
    for lab in label_names:
        for alias in _label_aliases(lab):
            alias_to_label[alias] = lab

    out: Dict[str, List[str]] = {lab: [] for lab in label_names}
    unknown_keys: List[str] = []

    for key, words in raw.items():
        if not isinstance(words, list):
            raise ValueError(f"Seed words for '{key}' must be a list")
        norm_key = _normalize_key(str(key))
        label = alias_to_label.get(norm_key)
        if label is None:
            unknown_keys.append(str(key))
            continue
        for w in words:
            if not isinstance(w, str):
                continue
            w = w.strip()
            if not w:
                continue
            out[label].append(w)

    # de-duplicate (case-insensitive) while preserving order
    for lab, words in out.items():
        seen = set()
        dedup: List[str] = []
        for w in words:
            key = w.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(w)
        out[lab] = dedup

    return out, unknown_keys


def normalize_seed_words(
    seed_words_map: Dict[str, List[str]],
    tokenizer: Optional[Callable[[str], str]] = None,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for lab, words in seed_words_map.items():
        new_list: List[str] = []
        for w in words:
            t = tokenizer(w) if tokenizer else w
            if not isinstance(t, str):
                t = str(t)
            t = t.strip()
            if not t:
                continue
            new_list.append(t)
        out[lab] = new_list
    return out


def seed_words_enabled(seed_words_map: Dict[str, List[str]]) -> bool:
    return any(bool(v) for v in seed_words_map.values())


def build_seed_features(
    texts: List[str],
    seed_words_map: Dict[str, List[str]],
    label_names: List[str],
) -> np.ndarray:
    n = len(texts)
    num_labels = len(label_names)
    feats = np.zeros((n, num_labels), dtype=np.float32)
    if n == 0:
        return feats

    lower_texts = [t.lower() for t in texts]

    for j, lab in enumerate(label_names):
        words = seed_words_map.get(lab, [])
        if not words:
            continue
        words_lc = [w.lower() for w in words if w]
        if not words_lc:
            continue
        for i, t in enumerate(lower_texts):
            for w in words_lc:
                if w in t:
                    feats[i, j] = 1.0
                    break

    return feats


def load_seed_words_from_model_dir(model_dir: Path) -> Dict[str, List[str]]:
    cfg_path = model_dir / "train_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            seed_cfg = cfg.get("seed_words") or {}
            if isinstance(seed_cfg, dict):
                mapping = seed_cfg.get("label_to_words")
                if isinstance(mapping, dict):
                    out: Dict[str, List[str]] = {}
                    for k, v in mapping.items():
                        if isinstance(v, list):
                            out[str(k)] = [str(x) for x in v if isinstance(x, str)]
                        else:
                            out[str(k)] = []
                    return out
        except Exception:
            pass

    seed_path = model_dir / "seed_words.json"
    if seed_path.exists():
        try:
            raw = json.loads(seed_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                out = {}
                for k, v in raw.items():
                    if isinstance(v, list):
                        out[str(k)] = [str(x) for x in v if isinstance(x, str)]
                    else:
                        out[str(k)] = []
                return out
        except Exception:
            pass

    return {}
