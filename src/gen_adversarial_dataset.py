"""
Build adversarial training datasets for Sprint 3.

For each dataset:
- Keep 70% stratified clean training data.
- Generate adversarial samples from the remaining attack-only partition using
  three attack families (A/B/C).
- Merge and save:
  data/adversarial/adv_train_{cicids,nslkdd,unswnb15}.npz
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
import sys

import numpy as np

# Allow both:
# - python -m src.gen_adversarial_dataset
# - python src/gen_adversarial_dataset.py
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.attacks.protocol_exploitation import (
    add_tcp_options,
    fragment_payload,
    shift_ack_timing,
)
from src.constraints import CICIDSFeatures as F
from src.constraints import TCPConstraintValidator
from src.mutations import blend_with_benign, split_packets

SPLITS_DIR = Path("data/splits")
OUT_DIR = Path("data/adversarial")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME_MAP = {
    "cicids2017": "cicids",
    "nslkdd": "nslkdd",
    "unswnb15": "unswnb15",
}


def load_split(dataset: str):
    npz_path = SPLITS_DIR / f"{dataset}.npz"
    label_path = SPLITS_DIR / f"{dataset}_label_map.npy"
    with np.load(npz_path, allow_pickle=True) as data:
        X_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.int64)
    label_map = np.load(label_path, allow_pickle=True).item()
    return X_train, y_train, label_map


def find_benign_label_id(label_map: dict[int, str]) -> int:
    for idx, name in label_map.items():
        if name in {"BENIGN", "Normal"}:
            return int(idx)
    raise ValueError(f"Could not find benign label in label_map={label_map}")


def stratified_clean_split(y: np.ndarray, clean_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    clean_idx_parts = []
    remain_idx_parts = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_clean = int(np.floor(clean_ratio * len(cls_idx)))
        n_clean = max(1, min(n_clean, len(cls_idx)))
        clean_idx_parts.append(cls_idx[:n_clean])
        remain_idx_parts.append(cls_idx[n_clean:])
    clean_idx = np.concatenate(clean_idx_parts)
    remain_idx = np.concatenate(remain_idx_parts)
    rng.shuffle(clean_idx)
    rng.shuffle(remain_idx)
    return clean_idx, remain_idx


def _safe_attack_call(fn: Callable[[np.ndarray], np.ndarray], sample: np.ndarray):
    try:
        out = fn(sample)
        if out is None:
            return None
        return np.asarray(out, dtype=np.float32)
    except Exception:
        return None


def build_cicids_adversarial(
    X_attack: np.ndarray, benign_pool: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Attack A/B/C outputs for CICIDS using mutation + protocol modules.
    """
    validator = TCPConstraintValidator()

    # Use benign median IAT as protocol timing target (ms).
    if len(benign_pool) > 0:
        target_iat_ms = float(np.median(np.clip(benign_pool[:, F.FLOW_IAT_MEAN], 0, None)) / 1000.0)
        target_iat_ms = float(np.clip(target_iat_ms, 1.0, 2000.0))
    else:
        target_iat_ms = 50.0

    n = len(X_attack)
    order = rng.permutation(n)
    thirds = np.array_split(order, 3)

    X_adv = []
    src = []

    # Attack A: feature obfuscation style (split packets)
    for idx in thirds[0]:
        sample = X_attack[idx]
        pert = _safe_attack_call(lambda s: split_packets(s, number_of_fragments=2), sample)
        if pert is not None and validator.validate(sample, pert):
            X_adv.append(pert)
            src.append(1)

    # Attack B: behavioral mimicry style (blend with benign)
    for idx in thirds[1]:
        sample = X_attack[idx]
        pert = _safe_attack_call(
            lambda s: blend_with_benign(s, benign_pool=benign_pool, k_samples=3, random_number_generator=rng),
            sample,
        )
        if pert is not None and validator.validate(sample, pert):
            X_adv.append(pert)
            src.append(2)

    # Attack C: protocol exploitation (cycle through three protocol mutations)
    protocol_fns = [
        lambda s: fragment_payload(s, n_fragments=4),
        add_tcp_options,
        lambda s: shift_ack_timing(s, target_iat_ms=target_iat_ms),
    ]
    for k, idx in enumerate(thirds[2]):
        sample = X_attack[idx]
        fn = protocol_fns[k % len(protocol_fns)]
        pert = _safe_attack_call(fn, sample)
        if pert is not None and validator.validate(sample, pert):
            X_adv.append(pert)
            src.append(3)

    if len(X_adv) == 0:
        return np.empty((0, X_attack.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X_adv).astype(np.float32), np.asarray(src, dtype=np.int64)


def generic_attack_a(sample: np.ndarray, feature_std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = sample.astype(np.float32, copy=True)
    n_features = x.shape[0]
    k = max(1, int(0.10 * n_features))
    idx = rng.choice(n_features, size=k, replace=False)
    noise = rng.normal(0.0, 0.10 * (feature_std[idx] + 1e-6), size=k)
    x[idx] = x[idx] + noise.astype(np.float32)
    return x


def generic_attack_b(sample: np.ndarray, benign_pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = sample.astype(np.float32, copy=True)
    if len(benign_pool) == 0:
        return x
    ref = benign_pool[rng.integers(0, len(benign_pool))]
    alpha = 0.70
    return (alpha * x + (1.0 - alpha) * ref).astype(np.float32)


def generic_attack_c(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = sample.astype(np.float32, copy=True)
    n_features = x.shape[0]
    k = max(1, int(0.08 * n_features))
    idx = rng.choice(n_features, size=k, replace=False)
    x[idx] = x[idx] * rng.uniform(0.75, 0.98, size=k).astype(np.float32)
    return x


def build_generic_adversarial(
    X_attack: np.ndarray, benign_pool: np.ndarray, feature_std: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X_attack)
    order = rng.permutation(n)
    thirds = np.array_split(order, 3)

    X_adv = []
    src = []

    for idx in thirds[0]:
        X_adv.append(generic_attack_a(X_attack[idx], feature_std, rng))
        src.append(1)
    for idx in thirds[1]:
        X_adv.append(generic_attack_b(X_attack[idx], benign_pool, rng))
        src.append(2)
    for idx in thirds[2]:
        X_adv.append(generic_attack_c(X_attack[idx], rng))
        src.append(3)

    if len(X_adv) == 0:
        return np.empty((0, X_attack.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X_adv).astype(np.float32), np.asarray(src, dtype=np.int64)


def build_for_dataset(dataset: str, rng: np.random.Generator):
    X_train, y_train, label_map = load_split(dataset)
    benign_id = find_benign_label_id(label_map)

    clean_idx, remain_idx = stratified_clean_split(y_train, clean_ratio=0.70, rng=rng)
    attack_idx = remain_idx[y_train[remain_idx] != benign_id]

    X_clean = X_train[clean_idx]
    y_clean = y_train[clean_idx]

    X_attack = X_train[attack_idx]
    y_attack = y_train[attack_idx]
    benign_pool = X_train[y_train == benign_id]

    if dataset == "cicids2017":
        X_adv, source_id = build_cicids_adversarial(X_attack, benign_pool, rng)
    else:
        feature_std = X_train.std(axis=0).astype(np.float32)
        X_adv, source_id = build_generic_adversarial(X_attack, benign_pool, feature_std, rng)

    # Keep original attack labels for adversarial samples.
    y_adv = y_attack[: len(X_adv)].astype(np.int64)

    X_out = np.concatenate([X_clean, X_adv], axis=0).astype(np.float32)
    y_out = np.concatenate([y_clean, y_adv], axis=0).astype(np.int64)
    source_out = np.concatenate(
        [np.zeros(len(X_clean), dtype=np.int64), source_id],
        axis=0,
    )

    # Shuffle final training set.
    perm = rng.permutation(len(X_out))
    X_out = X_out[perm]
    y_out = y_out[perm]
    source_out = source_out[perm]

    short_name = DATASET_NAME_MAP[dataset]
    out_path = OUT_DIR / f"adv_train_{short_name}.npz"
    np.savez_compressed(
        out_path,
        X_train=X_out,
        y_train=y_out,
        source_id=source_out,   # 0=clean,1=attack_a,2=attack_b,3=attack_c
    )

    return {
        "dataset": dataset,
        "output_path": str(out_path),
        "n_clean": int(len(X_clean)),
        "n_attack_candidates": int(len(X_attack)),
        "n_adv_generated": int(len(X_adv)),
        "n_final": int(len(X_out)),
        "feature_dim": int(X_out.shape[1]),
    }


def main():
    rng = np.random.default_rng(42)
    summary = {}

    for dataset in ["cicids2017", "nslkdd", "unswnb15"]:
        stats = build_for_dataset(dataset, rng)
        summary[dataset] = stats
        print(
            f"{dataset}: clean={stats['n_clean']}, adv={stats['n_adv_generated']}, "
            f"final={stats['n_final']} -> {stats['output_path']}"
        )

    summary_path = OUT_DIR / "adv_generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
