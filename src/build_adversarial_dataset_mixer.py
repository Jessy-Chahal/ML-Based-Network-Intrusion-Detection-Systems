"""
Task J1: Build adversarial mixed training splits.

For each dataset (cicids2017, nslkdd, unswnb15):
- Load clean training split from data/splits/<dataset>.npz
- Keep ONLY benign samples from the clean split
- Load adversarial samples from data/adversarial/adv_train_<short>.npz
  and keep only source_id > 0 (true adversarial samples)
- Build a 70% benign / 30% adversarial mixed training set
- Preserve adversarial class balance via stratified sampling
- Save to data/splits/adversarial_mixed_<dataset>.npz
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


SPLITS_DIR = Path("data/splits")
ADVERSARIAL_DIR = Path("data/adversarial")
OUT_DIR = SPLITS_DIR

DATASETS = ["cicids2017", "nslkdd", "unswnb15"]
DATASET_NAME_MAP = {
    "cicids2017": "cicids",
    "nslkdd": "nslkdd",
    "unswnb15": "unswnb15",
}


def load_label_map(dataset: str) -> Dict[int, str]:
    path = SPLITS_DIR / f"{dataset}_label_map.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing label map: {path}")
    raw = np.load(path, allow_pickle=True).item()
    return {int(k): str(v) for k, v in raw.items()}


def find_benign_label_id(label_map: Dict[int, str]) -> int:
    for idx, name in label_map.items():
        if name in {"BENIGN", "Normal"}:
            return idx
    raise ValueError(f"Could not find benign label in {label_map}")


def stratified_indices(
    y: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Return indices of length n_samples sampled from y while preserving class proportions.
    """
    if n_samples <= 0:
        return np.empty((0,), dtype=np.int64)
    if n_samples > len(y):
        raise ValueError(f"Requested {n_samples} samples from only {len(y)} rows")

    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    ideal = proportions * n_samples

    base = np.floor(ideal).astype(int)
    remainder = n_samples - int(base.sum())
    frac = ideal - base

    if remainder > 0:
        order = np.argsort(-frac)
        for i in order[:remainder]:
            base[i] += 1

    class_to_take = {cls: int(take) for cls, take in zip(classes, base)}
    sampled_parts = []

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        take = min(class_to_take[cls], len(cls_idx))
        if take > 0:
            sampled_parts.append(cls_idx[:take])

    sampled = np.concatenate(sampled_parts) if sampled_parts else np.empty((0,), dtype=np.int64)

    # If min() clipped any class due to edge cases, fill deficit randomly from remaining pool.
    deficit = n_samples - len(sampled)
    if deficit > 0:
        selected = np.zeros(len(y), dtype=bool)
        selected[sampled] = True
        remaining = np.where(~selected)[0]
        rng.shuffle(remaining)
        sampled = np.concatenate([sampled, remaining[:deficit]])

    rng.shuffle(sampled)
    return sampled.astype(np.int64)


def resolve_mix_counts(
    n_benign_available: int, n_adv_available: int, clean_ratio: float = 0.70
) -> Tuple[int, int]:
    """
    Compute feasible sample counts satisfying target clean/adv ratio.
    """
    adv_ratio = 1.0 - clean_ratio
    max_total_from_benign = int(np.floor(n_benign_available / clean_ratio))
    max_total_from_adv = int(np.floor(n_adv_available / adv_ratio))
    total = min(max_total_from_benign, max_total_from_adv)

    n_clean = int(np.floor(total * clean_ratio))
    n_adv = total - n_clean

    # Safety corrections
    n_clean = min(n_clean, n_benign_available)
    n_adv = min(n_adv, n_adv_available)
    return n_clean, n_adv


def class_count_map(y: np.ndarray, label_map: Dict[int, str]) -> Dict[str, int]:
    classes, counts = np.unique(y, return_counts=True)
    out = {}
    for cls, cnt in zip(classes, counts):
        out[label_map.get(int(cls), str(int(cls)))] = int(cnt)
    return out


def build_one(dataset: str, rng: np.random.Generator) -> dict:
    split_path = SPLITS_DIR / f"{dataset}.npz"
    adv_short = DATASET_NAME_MAP[dataset]
    adv_path = ADVERSARIAL_DIR / f"adv_train_{adv_short}.npz"
    out_path = OUT_DIR / f"adversarial_mixed_{dataset}.npz"

    if not split_path.exists():
        raise FileNotFoundError(f"Missing clean split file: {split_path}")
    if not adv_path.exists():
        raise FileNotFoundError(f"Missing adversarial file: {adv_path}")

    label_map = load_label_map(dataset)
    benign_id = find_benign_label_id(label_map)

    clean_npz = np.load(split_path, allow_pickle=True)
    X_clean_all = clean_npz["X_train"].astype(np.float32)
    y_clean_all = clean_npz["y_train"].astype(np.int64)

    # Clean means benign only (per task clarification).
    benign_mask = y_clean_all == benign_id
    X_benign = X_clean_all[benign_mask]
    y_benign = y_clean_all[benign_mask]

    adv_npz = np.load(adv_path, allow_pickle=True)
    X_adv_all = adv_npz["X_train"].astype(np.float32)
    y_adv_all = adv_npz["y_train"].astype(np.int64)
    if "source_id" not in adv_npz.files:
        raise KeyError(f"{adv_path} missing source_id; cannot isolate adversarial rows.")
    source_id = adv_npz["source_id"].astype(np.int64)

    # Keep only true adversarial rows (1/2/3), never clean (0).
    adv_mask = source_id > 0
    X_adv = X_adv_all[adv_mask]
    y_adv = y_adv_all[adv_mask]

    # Enforce adversarial-only labels in adv part.
    attack_label_mask = y_adv != benign_id
    X_adv = X_adv[attack_label_mask]
    y_adv = y_adv[attack_label_mask]

    n_clean, n_adv = resolve_mix_counts(len(X_benign), len(X_adv), clean_ratio=0.70)
    if n_clean == 0 or n_adv == 0:
        raise ValueError(
            f"Insufficient data for {dataset}: benign={len(X_benign)}, adversarial={len(X_adv)}"
        )

    clean_idx = rng.choice(len(X_benign), size=n_clean, replace=False)
    adv_idx = stratified_indices(y_adv, n_adv, rng)

    X_mix_clean = X_benign[clean_idx]
    y_mix_clean = y_benign[clean_idx]
    X_mix_adv = X_adv[adv_idx]
    y_mix_adv = y_adv[adv_idx]

    X_mix = np.concatenate([X_mix_clean, X_mix_adv], axis=0).astype(np.float32)
    y_mix = np.concatenate([y_mix_clean, y_mix_adv], axis=0).astype(np.int64)
    source_mix = np.concatenate(
        [np.zeros(len(X_mix_clean), dtype=np.int64), np.ones(len(X_mix_adv), dtype=np.int64)],
        axis=0,
    )

    perm = rng.permutation(len(X_mix))
    X_mix = X_mix[perm]
    y_mix = y_mix[perm]
    source_mix = source_mix[perm]

    np.savez_compressed(
        out_path,
        X_train=X_mix,
        y_train=y_mix,
        source_id=source_mix,  # 0=benign clean, 1=adversarial
        clean_ratio_target=np.array([0.70], dtype=np.float32),
        adversarial_ratio_target=np.array([0.30], dtype=np.float32),
    )

    mix_benign = int((y_mix == benign_id).sum())
    mix_adv = int(len(y_mix) - mix_benign)

    summary = {
        "dataset": dataset,
        "output_path": str(out_path),
        "benign_label_id": benign_id,
        "n_benign_available": int(len(X_benign)),
        "n_adversarial_available": int(len(X_adv)),
        "n_clean_used": int(n_clean),
        "n_adversarial_used": int(n_adv),
        "n_final": int(len(X_mix)),
        "final_clean_ratio": float(mix_benign / len(y_mix)),
        "final_adversarial_ratio": float(mix_adv / len(y_mix)),
        "adversarial_class_counts": class_count_map(y_mix_adv, label_map),
    }
    return summary


def main():
    rng = np.random.default_rng(42)
    full_summary = {}

    for dataset in DATASETS:
        s = build_one(dataset, rng)
        full_summary[dataset] = s
        print(
            f"{dataset}: clean={s['n_clean_used']}, adv={s['n_adversarial_used']}, "
            f"final={s['n_final']}, ratios={s['final_clean_ratio']:.3f}/{s['final_adversarial_ratio']:.3f}"
        )

    summary_path = OUT_DIR / "adversarial_mixed_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
