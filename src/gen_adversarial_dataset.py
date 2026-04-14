"""
Build adversarial training datasets for Sprint 3.

For each dataset:
- Keep 70% stratified clean training data.
- Generate adversarial samples from the remaining attack-only partition using
  three attack families (A/B/C).
- Merge and save:
  data/adversarial/adv_train_{cicids,nslkdd,unswnb15}.npz

Performance: inject_decoy_flows, dilute_scan_pattern, and protocol_exploitation
already run their own constraint validation. Do not re-run TCPConstraintValidator
on every sample — that duplicated work and made full CICIDS runs orders of
magnitude slower than the old proxy pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional
import sys

import numpy as np

# Allow both:
# - python -m src.gen_adversarial_dataset
# - python src/gen_adversarial_dataset.py
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.dotenv_utils import get_env_float, get_env_int

from src.attacks.behavioral_mimicry import mimic_packet_size, mimic_timing
from src.attacks.feature_obfuscation import dilute_scan_pattern, inject_decoy_flows
from src.attacks.protocol_exploitation import (
    add_tcp_options,
    fragment_payload,
    shift_ack_timing,
)
from src.constraints import CICIDSFeatures as F
from src.mutations import blend_with_benign

SECONDS_TO_MICROSECONDS = 1_000_000.0
INJECT_DECOY_K = get_env_int("ADV_INJECT_DECOY_K")
DILUTE_COVER_TRAFFIC_RATE = get_env_float("ADV_DILUTE_COVER_TRAFFIC_RATE")
ADV_BLEND_WITH_BENIGN_K_SAMPLES = get_env_int("ADV_BLEND_WITH_BENIGN_K_SAMPLES")
ADV_GENERIC_ATTACK_C_FEATURE_FRAC = get_env_float("ADV_GENERIC_ATTACK_C_FEATURE_FRAC")
ADV_GENERIC_ATTACK_C_SCALE_MIN = get_env_float("ADV_GENERIC_ATTACK_C_SCALE_MIN")
ADV_GENERIC_ATTACK_C_SCALE_MAX = get_env_float("ADV_GENERIC_ATTACK_C_SCALE_MAX")
ADV_CLEAN_RATIO = get_env_float("ADV_CLEAN_RATIO")
ADV_GENERATION_SEED = get_env_int("ADV_GENERATION_SEED")
PROTOCOL_N_FRAGMENTS = get_env_int("PROTOCOL_N_FRAGMENTS")

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
        if name in {"BENIGN", "Normal", "normal"}:
            return int(idx)
    raise ValueError(f"Could not find benign label in label_map={label_map}")


def compute_benign_profile_from_data(
    X_train: np.ndarray, y_train: np.ndarray, benign_id: int
) -> dict:
    """
    Build a benign profile dict from training data for use with mimic_timing and mimic_packet_size.
    Shape matches what load_benign_profile() returns from benign_profiles.json.
    """
    benign = X_train[y_train == benign_id]
    if len(benign) == 0:
        # Fallback defaults if no benign samples
        return {
            "flow_iat_mean": {"mean_us": 50_000.0},
            "flow_iat_std": {"mean_us": 10_000.0},
            "pkt_len_mean": {"mean_us": 200.0},
        }

    iat_mean_us = np.clip(benign[:, F.FLOW_IAT_MEAN], 0.0, None)
    iat_std_us = np.clip(benign[:, F.FLOW_IAT_STD], 0.0, None)
    total_pkts = benign[:, F.TOT_FWD_PKTS] + benign[:, F.TOT_BWD_PKTS]
    total_bytes = benign[:, F.TOT_LEN_FWD_PKTS] + benign[:, F.TOT_LEN_BWD_PKTS]
    # Per-flow average packet size (bytes); avoid div-by-zero
    pkt_sizes = np.full(total_bytes.shape, np.nan, dtype=np.float64)
    np.divide(total_bytes, total_pkts, out=pkt_sizes, where=total_pkts > 0)
    pkt_sizes = pkt_sizes[~np.isnan(pkt_sizes)]
    pkt_mean_bytes = float(np.median(pkt_sizes)) if len(pkt_sizes) > 0 else 200.0
    pkt_mean_bytes = float(np.clip(pkt_mean_bytes, 20.0, 1500.0))

    return {
        "flow_iat_mean": {"mean_us": float(np.median(iat_mean_us))},
        "flow_iat_std": {"mean_us": float(np.median(iat_std_us))},
        "pkt_len_mean": {"mean_us": pkt_mean_bytes},
    }


def _recompute_rates_after_packet_size(flow: np.ndarray) -> np.ndarray:
    """
    Recompute FLOW_BYTS_S, FLOW_PKTS_S, FWD_PKTS_S, BWD_PKTS_S from totals and duration.
    mimic_packet_size does not update these (evaluate_attack_b does this before constraint checks).
    """
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


def _safe_attack_call(fn: Callable[[np.ndarray], Optional[np.ndarray]], sample: np.ndarray):
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


def build_cicids_adversarial(
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    benign_pool: np.ndarray,
    benign_profile: dict,
    label_map: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Attack A: dilute_scan_pattern (PortScan), inject_decoy_flows (all other labels in this third).
    Attack B: mimic_timing / mimic_packet_size alternating (profile from training benign).
    Attack C: fragment_payload, add_tcp_options, shift_ack_timing (unchanged).

    Validity: trust feature_obfuscation and protocol_exploitation validators only — no extra TCP pass.
    """
    if len(benign_pool) == 0:
        raise RuntimeError("CICIDS adversarial generation requires a non-empty benign_pool.")

    benign_fp64 = benign_pool.astype(np.float64, copy=False)

    target_iat_ms = float(np.median(np.clip(benign_pool[:, F.FLOW_IAT_MEAN], 0, None)) / 1000.0)
    target_iat_ms = float(np.clip(target_iat_ms, 1.0, 2000.0))

    n = len(X_attack)
    order = rng.permutation(n)
    thirds = np.array_split(order, 3)

    X_adv: list[np.ndarray] = []
    src: list[int] = []
    y_adv: list[int] = []

    # Attack A
    for idx in thirds[0]:
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
                    sample,
                    benign_fp64,
                    k=INJECT_DECOY_K,
                    attack_type="dos",
                    rng=rng,
                )
                pert = out
        except Exception:
            pert = None
        if pert is not None:
            X_adv.append(np.asarray(pert, dtype=np.float32))
            src.append(1)
            y_adv.append(lid)

    # Attack B
    for j, idx in enumerate(thirds[1]):
        sample = X_attack[idx].astype(np.float64, copy=False)
        lid = int(y_attack[idx])
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
            src.append(2)
            y_adv.append(lid)

    # Attack C (protocol helpers already TCP-validate internally)
    protocol_fns = [
        lambda s: fragment_payload(s, n_fragments=PROTOCOL_N_FRAGMENTS),
        add_tcp_options,
        lambda s: shift_ack_timing(s, target_iat_ms=target_iat_ms),
    ]
    for k, idx in enumerate(thirds[2]):
        sample = X_attack[idx]
        lid = int(y_attack[idx])
        fn = protocol_fns[k % len(protocol_fns)]
        pert = _safe_attack_call(fn, sample)
        if pert is not None:
            X_adv.append(pert)
            src.append(3)
            y_adv.append(lid)

    if len(X_adv) == 0:
        empty = np.empty((0, X_attack.shape[1]), dtype=np.float32)
        return empty, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return (
        np.stack(X_adv).astype(np.float32),
        np.asarray(src, dtype=np.int64),
        np.asarray(y_adv, dtype=np.int64),
    )


def generic_attack_c(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = sample.astype(np.float32, copy=True)
    n_features = x.shape[0]
    k = max(1, int(ADV_GENERIC_ATTACK_C_FEATURE_FRAC * n_features))
    idx = rng.choice(n_features, size=k, replace=False)
    x[idx] = x[idx] * rng.uniform(
        ADV_GENERIC_ATTACK_C_SCALE_MIN,
        ADV_GENERIC_ATTACK_C_SCALE_MAX,
        size=k,
    ).astype(np.float32)
    return x


def build_generic_adversarial(
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    benign_id: int,
    benign_pool: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Attack A: blend_with_benign (inject_decoy/dilute are CICIDS-scoped).
    Attack B: mimic_timing / mimic_packet_size alternating.
    Attack C: generic feature scaling (no CICIDS protocol mutations).
    """
    benign_profile = compute_benign_profile_from_data(X_train, y_train, benign_id)

    n = len(X_attack)
    order = rng.permutation(n)
    thirds = np.array_split(order, 3)

    X_adv: list[np.ndarray] = []
    src: list[int] = []
    y_adv: list[int] = []

    for idx in thirds[0]:
        sample = X_attack[idx]
        lid = int(y_attack[idx])
        pert = _safe_attack_call(
            lambda s, bp=benign_pool: blend_with_benign(
                s,
                benign_pool=bp,
                k_samples=ADV_BLEND_WITH_BENIGN_K_SAMPLES,
                random_number_generator=rng,
            ),
            sample,
        )
        if pert is not None:
            X_adv.append(pert)
            src.append(1)
            y_adv.append(lid)

    for j, idx in enumerate(thirds[1]):
        sample = X_attack[idx].astype(np.float64, copy=False)
        lid = int(y_attack[idx])
        try:
            if j % 2 == 0:
                pert = mimic_timing(sample, benign_profile)
            else:
                pert = mimic_packet_size(sample, benign_profile)
        except Exception:
            pert = None
        if pert is not None:
            X_adv.append(np.asarray(pert, dtype=np.float32))
            src.append(2)
            y_adv.append(lid)

    for idx in thirds[2]:
        X_adv.append(generic_attack_c(X_attack[idx], rng))
        src.append(3)
        y_adv.append(int(y_attack[idx]))

    if len(X_adv) == 0:
        empty = np.empty((0, X_attack.shape[1]), dtype=np.float32)
        return empty, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return (
        np.stack(X_adv).astype(np.float32),
        np.asarray(src, dtype=np.int64),
        np.asarray(y_adv, dtype=np.int64),
    )


def _validate_adversarial_families(source_id: np.ndarray, dataset: str) -> dict[str, int]:
    adv_src = source_id[source_id > 0]
    counts = {f"family_{k}": int((adv_src == k).sum()) for k in (1, 2, 3)}
    zero = [k for k in (1, 2, 3) if counts[f"family_{k}"] == 0]
    if zero:
        raise RuntimeError(
            f"{dataset}: adversarial generation produced zero samples for attack family(ies) {zero}. "
            f"Counts per family (source_id 1=A, 2=B, 3=C): {counts}."
        )
    return counts


def build_for_dataset(dataset: str, rng: np.random.Generator):
    X_train, y_train, label_map = load_split(dataset)
    benign_id = find_benign_label_id(label_map)

    clean_idx, remain_idx = stratified_clean_split(
        y_train, clean_ratio=ADV_CLEAN_RATIO, rng=rng
    )
    attack_idx = remain_idx[y_train[remain_idx] != benign_id]

    X_clean = X_train[clean_idx]
    y_clean = y_train[clean_idx]

    X_attack = X_train[attack_idx]
    y_attack = y_train[attack_idx]
    benign_pool = X_train[y_train == benign_id]

    n_train = len(y_train)
    clean_ratio_actual = len(X_clean) / n_train if n_train else 0.0

    if dataset == "cicids2017":
        benign_profile = compute_benign_profile_from_data(X_train, y_train, benign_id)
        X_adv, source_id, y_adv = build_cicids_adversarial(
            X_attack, y_attack, benign_pool, benign_profile, label_map, rng
        )
    else:
        X_adv, source_id, y_adv = build_generic_adversarial(
            X_attack, y_attack, X_train, y_train, benign_id, benign_pool, rng
        )

    family_counts = _validate_adversarial_families(
        np.concatenate([np.zeros(len(X_clean), dtype=np.int64), source_id]),
        dataset,
    )

    X_out = np.concatenate([X_clean, X_adv], axis=0).astype(np.float32)
    y_out = np.concatenate([y_clean, y_adv], axis=0).astype(np.int64)
    source_out = np.concatenate(
        [np.zeros(len(X_clean), dtype=np.int64), source_id],
        axis=0,
    )

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
        source_id=source_out,
    )

    return {
        "dataset": dataset,
        "output_path": str(out_path),
        "n_clean": int(len(X_clean)),
        "n_attack_candidates": int(len(X_attack)),
        "n_adv_generated": int(len(X_adv)),
        "n_final": int(len(X_out)),
        "feature_dim": int(X_out.shape[1]),
        "n_train_total": int(n_train),
        "clean_ratio_actual": round(float(clean_ratio_actual), 6),
        "adv_family_counts": family_counts,
    }


def main():
    rng = np.random.default_rng(ADV_GENERATION_SEED)
    summary = {}

    for dataset in ["cicids2017", "nslkdd", "unswnb15"]:
        print(f"{dataset}: generating...", flush=True)
        stats = build_for_dataset(dataset, rng)
        summary[dataset] = stats
        fc = stats["adv_family_counts"]
        cr = stats["clean_ratio_actual"]
        n_tot = stats["n_train_total"]
        print(
            f"{dataset}: clean={stats['n_clean']}/{n_tot} (clean_ratio~{cr:.4f}, target 0.70), "
            f"adv={stats['n_adv_generated']}, final={stats['n_final']} -> {stats['output_path']}"
        )
        print(
            f"  Adversarial samples per family: A={fc['family_1']}, B={fc['family_2']}, C={fc['family_3']}",
            flush=True,
        )

    summary_path = OUT_DIR / "adv_generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
