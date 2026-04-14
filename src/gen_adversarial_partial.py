"""
Build "partially trained" adversarial datasets — training data perturbed with
exactly ONE attack family (A, B, or C).

Usage:
    python src/gen_adversarial_partial.py --attack a   # Attack A only
    python src/gen_adversarial_partial.py --attack b   # Attack B only
    python src/gen_adversarial_partial.py --attack c   # Attack C only
    python src/gen_adversarial_partial.py --attack a --datasets cicids2017   # single dataset

For each dataset the script:
  - Keeps 70% stratified clean training data (identical split logic to gen_adversarial_dataset.py).
  - Applies ONLY the chosen attack family to all remaining attack samples.
  - Saves to data/adversarial/adv_train_{dataset}_attack_{a|b|c}_only.npz

Output files (for --attack a):
    data/adversarial/adv_train_cicids_attack_a_only.npz
    data/adversarial/adv_train_nslkdd_attack_a_only.npz
    data/adversarial/adv_train_unswnb15_attack_a_only.npz

A summary JSON is written alongside each file and to:
    data/adversarial/adv_partial_generation_summary_{a|b|c}.json

These files are consumed by train_adversarial_partial.py to produce
partially-trained adversarial models (adv_{a|b|c}_only_{rf,xgb,mlp,lstm}_{dataset}).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional
import sys

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.attacks.behavioral_mimicry import mimic_packet_size, mimic_timing
from src.attacks.feature_obfuscation import dilute_scan_pattern, inject_decoy_flows
from src.attacks.protocol_exploitation import (
    add_tcp_options,
    fragment_payload,
    shift_ack_timing,
)
from src.constraints import CICIDSFeatures as F
from src.mutations import blend_with_benign

# ── constants ────────────────────────────────────────────────────────────────

SECONDS_TO_MICROSECONDS = 1_000_000.0
INJECT_DECOY_K = 5
DILUTE_COVER_TRAFFIC_RATE = 1.0

SPLITS_DIR = Path("data/splits")
OUT_DIR = Path("data/adversarial")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME_MAP = {
    "cicids2017": "cicids",
    "nslkdd": "nslkdd",
    "unswnb15": "unswnb15",
}

ALL_DATASETS = ["cicids2017", "nslkdd", "unswnb15"]


# ── data helpers (identical to gen_adversarial_dataset.py) ────────────────────

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
        if name in {"BENIGN", "Normal", "normal"}:
            return int(idx)
    raise ValueError(f"Could not find benign label in label_map={label_map}")


def stratified_clean_split(
    y: np.ndarray, clean_ratio: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    clean_idx_parts = []
    remain_idx_parts = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_clean = max(1, min(int(np.floor(clean_ratio * len(cls_idx))), len(cls_idx)))
        clean_idx_parts.append(cls_idx[:n_clean])
        remain_idx_parts.append(cls_idx[n_clean:])
    clean_idx = np.concatenate(clean_idx_parts)
    remain_idx = np.concatenate(remain_idx_parts)
    rng.shuffle(clean_idx)
    rng.shuffle(remain_idx)
    return clean_idx, remain_idx


def _safe_attack_call(
    fn: Callable[[np.ndarray], Optional[np.ndarray]], sample: np.ndarray
) -> Optional[np.ndarray]:
    try:
        out = fn(sample)
        if out is None:
            return None
        return np.asarray(out, dtype=np.float32)
    except Exception:
        return None


def _label_name(label_map: dict, label_id: int) -> str:
    for key in (label_id, int(label_id), str(label_id), str(int(label_id))):
        if key in label_map:
            return str(label_map[key])
    return ""


def compute_benign_profile_from_data(
    X_train: np.ndarray, y_train: np.ndarray, benign_id: int
) -> dict:
    benign = X_train[y_train == benign_id]
    if len(benign) == 0:
        return {
            "flow_iat_mean": {"mean_us": 50_000.0},
            "flow_iat_std": {"mean_us": 10_000.0},
            "pkt_len_mean": {"mean_us": 200.0},
        }
    iat_mean_us = np.clip(benign[:, F.FLOW_IAT_MEAN], 0.0, None)
    iat_std_us = np.clip(benign[:, F.FLOW_IAT_STD], 0.0, None)
    total_pkts = benign[:, F.TOT_FWD_PKTS] + benign[:, F.TOT_BWD_PKTS]
    total_bytes = benign[:, F.TOT_LEN_FWD_PKTS] + benign[:, F.TOT_LEN_BWD_PKTS]
    pkt_sizes = np.full(total_bytes.shape, np.nan, dtype=np.float64)
    np.divide(total_bytes, total_pkts, out=pkt_sizes, where=total_pkts > 0)
    pkt_sizes = pkt_sizes[~np.isnan(pkt_sizes)]
    pkt_mean_bytes = float(np.clip(
        float(np.median(pkt_sizes)) if len(pkt_sizes) > 0 else 200.0,
        20.0, 1500.0,
    ))
    return {
        "flow_iat_mean": {"mean_us": float(np.median(iat_mean_us))},
        "flow_iat_std": {"mean_us": float(np.median(iat_std_us))},
        "pkt_len_mean": {"mean_us": pkt_mean_bytes},
    }


def _recompute_rates_after_packet_size(flow: np.ndarray) -> np.ndarray:
    duration_sec = flow[F.FLOW_DURATION] / SECONDS_TO_MICROSECONDS
    if duration_sec <= 0:
        return flow
    total_bytes = flow[F.TOT_LEN_FWD_PKTS] + flow[F.TOT_LEN_BWD_PKTS]
    total_pkts = flow[F.TOT_FWD_PKTS] + flow[F.TOT_BWD_PKTS]
    flow[F.FLOW_BYTS_S] = total_bytes / duration_sec
    flow[F.FLOW_PKTS_S] = total_pkts / duration_sec
    flow[F.FWD_PKTS_S] = flow[F.TOT_FWD_PKTS] / duration_sec
    flow[F.BWD_PKTS_S] = flow[F.TOT_BWD_PKTS] / duration_sec
    return flow


# ── per-attack-family perturb functions ──────────────────────────────────────

def perturb_attack_a_cicids(
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    benign_pool: np.ndarray,
    label_map: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Attack A (feature obfuscation) to ALL samples — CICIDS variant."""
    benign_fp64 = benign_pool.astype(np.float64, copy=False)
    X_adv: list[np.ndarray] = []
    y_adv: list[int] = []

    for idx in range(len(X_attack)):
        sample = X_attack[idx].astype(np.float64, copy=False)
        lid = int(y_attack[idx])
        name = _label_name(label_map, lid)
        pert: Optional[np.ndarray] = None
        try:
            if name == "PortScan":
                out, _meta = dilute_scan_pattern(
                    sample, cover_traffic_rate=DILUTE_COVER_TRAFFIC_RATE, rng=rng
                )
                pert = out
            else:
                out, _meta = inject_decoy_flows(
                    sample, benign_fp64, k=INJECT_DECOY_K, attack_type="dos", rng=rng
                )
                pert = out
        except Exception:
            pert = None
        if pert is not None:
            X_adv.append(np.asarray(pert, dtype=np.float32))
            y_adv.append(lid)

    if not X_adv:
        return np.empty((0, X_attack.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X_adv).astype(np.float32), np.asarray(y_adv, dtype=np.int64)


def perturb_attack_a_generic(
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    benign_pool: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Attack A (blend_with_benign) to ALL samples — generic variant."""
    X_adv: list[np.ndarray] = []
    y_adv: list[int] = []

    for idx in range(len(X_attack)):
        sample = X_attack[idx]
        lid = int(y_attack[idx])
        pert = _safe_attack_call(
            lambda s, bp=benign_pool: blend_with_benign(
                s, benign_pool=bp, k_samples=3, random_number_generator=rng
            ),
            sample,
        )
        if pert is not None:
            X_adv.append(pert)
            y_adv.append(lid)

    if not X_adv:
        return np.empty((0, X_attack.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X_adv).astype(np.float32), np.asarray(y_adv, dtype=np.int64)


def perturb_attack_b(
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    benign_profile: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Attack B (behavioral mimicry) to ALL samples."""
    X_adv: list[np.ndarray] = []
    y_adv: list[int] = []

    for j in range(len(X_attack)):
        sample = X_attack[j].astype(np.float64, copy=False)
        lid = int(y_attack[j])
        try:
            if j % 2 == 0:
                pert = mimic_timing(sample, benign_profile)
            else:
                pert = mimic_packet_size(sample, benign_profile)
                pert = _recompute_rates_after_packet_size(pert)
        except Exception:
            pert = None
        if pert is not None:
            X_adv.append(np.asarray(pert, dtype=np.float32))
            y_adv.append(lid)

    if not X_adv:
        return np.empty((0, X_attack.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X_adv).astype(np.float32), np.asarray(y_adv, dtype=np.int64)


def perturb_attack_c_cicids(
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    benign_pool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Attack C (protocol exploitation) to ALL samples — CICIDS variant."""
    target_iat_ms = float(np.clip(
        float(np.median(np.clip(benign_pool[:, F.FLOW_IAT_MEAN], 0, None))) / 1000.0,
        1.0, 2000.0,
    ))
    protocol_fns = [
        lambda s: fragment_payload(s, n_fragments=4),
        add_tcp_options,
        lambda s: shift_ack_timing(s, target_iat_ms=target_iat_ms),
    ]
    X_adv: list[np.ndarray] = []
    y_adv: list[int] = []

    for k in range(len(X_attack)):
        sample = X_attack[k]
        lid = int(y_attack[k])
        fn = protocol_fns[k % len(protocol_fns)]
        pert = _safe_attack_call(fn, sample)
        if pert is not None:
            X_adv.append(pert)
            y_adv.append(lid)

    if not X_adv:
        return np.empty((0, X_attack.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X_adv).astype(np.float32), np.asarray(y_adv, dtype=np.int64)


def _generic_attack_c_sample(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = sample.astype(np.float32, copy=True)
    n_features = x.shape[0]
    k = max(1, int(0.08 * n_features))
    idx = rng.choice(n_features, size=k, replace=False)
    x[idx] = x[idx] * rng.uniform(0.75, 0.98, size=k).astype(np.float32)
    return x


def perturb_attack_c_generic(
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Attack C (generic feature scaling) to ALL samples — generic variant."""
    X_adv = [_generic_attack_c_sample(X_attack[i], rng) for i in range(len(X_attack))]
    y_adv = y_attack.tolist()
    return np.stack(X_adv).astype(np.float32), np.asarray(y_adv, dtype=np.int64)


# ── dataset builder ───────────────────────────────────────────────────────────

def build_for_dataset(dataset: str, attack: str, rng: np.random.Generator) -> dict:
    """
    Generate a partial adversarial training set using only one attack family.

    Args:
        dataset: one of "cicids2017", "nslkdd", "unswnb15"
        attack:  one of "a", "b", "c"
        rng:     seeded RNG for reproducibility

    Returns:
        summary dict with paths and sample counts
    """
    X_train, y_train, label_map = load_split(dataset)
    benign_id = find_benign_label_id(label_map)

    clean_idx, remain_idx = stratified_clean_split(y_train, clean_ratio=0.70, rng=rng)
    attack_idx = remain_idx[y_train[remain_idx] != benign_id]

    X_clean = X_train[clean_idx]
    y_clean = y_train[clean_idx]
    X_attack = X_train[attack_idx]
    y_attack = y_train[attack_idx]
    benign_pool = X_train[y_train == benign_id]

    n_train = len(y_train)
    clean_ratio_actual = len(X_clean) / n_train if n_train else 0.0

    print(f"  clean={len(X_clean)}, attack candidates={len(X_attack)}", flush=True)

    # ── apply the chosen attack family to ALL attack samples ─────────────────
    if attack == "a":
        if dataset == "cicids2017":
            X_adv, y_adv = perturb_attack_a_cicids(
                X_attack, y_attack, benign_pool, label_map, rng
            )
        else:
            X_adv, y_adv = perturb_attack_a_generic(
                X_attack, y_attack, benign_pool, rng
            )

    elif attack == "b":
        if dataset == "cicids2017":
            benign_profile = compute_benign_profile_from_data(X_train, y_train, benign_id)
        else:
            benign_profile = compute_benign_profile_from_data(X_train, y_train, benign_id)
        X_adv, y_adv = perturb_attack_b(X_attack, y_attack, benign_profile)

    else:  # attack == "c"
        if dataset == "cicids2017":
            X_adv, y_adv = perturb_attack_c_cicids(X_attack, y_attack, benign_pool)
        else:
            X_adv, y_adv = perturb_attack_c_generic(X_attack, y_attack, rng)

    # ── merge clean + adversarial and shuffle ────────────────────────────────
    source_id = np.concatenate([
        np.zeros(len(X_clean), dtype=np.int64),
        np.full(len(X_adv), {"a": 1, "b": 2, "c": 3}[attack], dtype=np.int64),
    ])
    X_out = np.concatenate([X_clean, X_adv], axis=0).astype(np.float32)
    y_out = np.concatenate([y_clean, y_adv], axis=0).astype(np.int64)

    perm = rng.permutation(len(X_out))
    X_out = X_out[perm]
    y_out = y_out[perm]
    source_id = source_id[perm]

    short_name = DATASET_NAME_MAP[dataset]
    out_path = OUT_DIR / f"adv_train_{short_name}_attack_{attack}_only.npz"
    np.savez_compressed(out_path, X_train=X_out, y_train=y_out, source_id=source_id)

    return {
        "dataset": dataset,
        "attack": attack,
        "output_path": str(out_path),
        "n_clean": int(len(X_clean)),
        "n_attack_candidates": int(len(X_attack)),
        "n_adv_generated": int(len(X_adv)),
        "n_final": int(len(X_out)),
        "feature_dim": int(X_out.shape[1]),
        "n_train_total": int(n_train),
        "clean_ratio_actual": round(float(clean_ratio_actual), 6),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate partial adversarial training data using a single attack family."
    )
    parser.add_argument(
        "--attack",
        required=True,
        choices=["a", "b", "c"],
        help="Which attack family to apply to all attack samples (a=feature obfuscation, "
             "b=behavioral mimicry, c=protocol exploitation).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=ALL_DATASETS,
        help="Datasets to generate for (default: all three).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    summary = {}

    for dataset in args.datasets:
        print(f"{dataset} (attack {args.attack.upper()}): generating...", flush=True)
        stats = build_for_dataset(dataset, args.attack, rng)
        summary[dataset] = stats
        print(
            f"{dataset}: clean={stats['n_clean']}/{stats['n_train_total']} "
            f"(ratio~{stats['clean_ratio_actual']:.4f}), "
            f"adv={stats['n_adv_generated']}, final={stats['n_final']} "
            f"-> {stats['output_path']}",
            flush=True,
        )

    summary_path = OUT_DIR / f"adv_partial_generation_summary_{args.attack}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
