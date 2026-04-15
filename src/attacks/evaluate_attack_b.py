"""
Evaluate Attack B (behavioral mimicry) evasion metrics across datasets.

Computes:
- ESR (Evasion Success Rate): among originally detected attack samples,
  the fraction classified as benign/normal after mutation.
- Constraint satisfaction rate on CICIDS2017 only (TCPConstraintValidator).

For NSL-KDD and UNSW-NB15, constraint validation is intentionally skipped
because validator checks are scoped to the CICIDS2017 feature schema.

Writes output to results/attack_b_metrics_all_datasets.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import sys

import joblib
import numpy as np
import tensorflow as tf

# Allow both:
# - python -m src.evaluate_attack_b
# - python src/evaluate_attack_b.py
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.attacks.behavioral_mimicry import (
    load_benign_profile,
    mimic_timing,
    mimic_packet_size,
)
from src.constraints import CICIDSFeatures as F
from src.constraints import TCPConstraintValidator
from src.dotenv_utils import get_env_float

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "attack_b_metrics_all_datasets.json"

SECONDS_TO_MICROSECONDS = get_env_float("SECONDS_TO_MICROSECONDS")
DATASETS = ["cicids2017", "nslkdd", "unswnb15"]

# Only CICIDS2017 has validators scoped to its feature schema
CONSTRAINT_VALIDATION_DATASETS = {"cicids2017"}

# Benign label names across datasets
BENIGN_LABEL_CANDIDATES = {"BENIGN", "Normal", "normal"}


###
# Data loading
###
def load_dataset_data(dataset: str):
    npz_path = SPLITS_DIR / f"{dataset}.npz"
    label_path = SPLITS_DIR / f"{dataset}_label_map.npy"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing split file: {npz_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label map: {label_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
    label_map = np.load(label_path, allow_pickle=True).item()
    return X_train, y_train, X_test, y_test, label_map


def benign_label_id(label_map: dict[int, str], dataset: str) -> int:
    for idx, name in label_map.items():
        if name in BENIGN_LABEL_CANDIDATES:
            return int(idx)
    raise ValueError(f"Could not find benign/normal label id in {dataset}_label_map.npy")


###
# Profile helpers
###
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


def get_benign_profile(
    profile_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    benign_id: int,
) -> dict:
    """Load profile from JSON if name is not 'computed', else compute from training data."""
    if profile_name == "computed":
        return compute_benign_profile_from_data(X_train, y_train, benign_id)
    return load_benign_profile(profile_name)


def _recompute_rates_after_packet_size(flow: np.ndarray) -> np.ndarray:
    """
    Recompute FLOW_BYTS_S, FLOW_PKTS_S, FWD_PKTS_S, BWD_PKTS_S from totals and duration.
    mimic_packet_size does not update these
    Call this so TCPConstraintValidator passes.
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


###
# Model loading and prediction
###
def load_models(dataset: str):
    rf = joblib.load(MODELS_DIR / f"rf_{dataset}.pkl")
    xgb = joblib.load(MODELS_DIR / f"xgb_{dataset}.pkl")
    scaler = joblib.load(MODELS_DIR / f"scaler_{dataset}.pkl")
    mlp = tf.keras.models.load_model(MODELS_DIR / f"mlp_{dataset}.h5", compile=False)
    return rf, xgb, scaler, mlp


def predict_by_model(model_name: str, model_obj, X: np.ndarray, scaler=None) -> np.ndarray:
    if model_name == "mlp":
        if scaler is None:
            raise ValueError("MLP prediction requires scaler.")
        X_scaled = scaler.transform(X)
        probs = model_obj.predict(X_scaled, verbose=0)
        return np.argmax(probs, axis=1).astype(np.int64)
    return model_obj.predict(X).astype(np.int64)


###
# Metrics helpers
###
def apply_mutation_batch(
    X: np.ndarray,
    mutation: Callable[[np.ndarray], np.ndarray],
    validate_constraints: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the given mutation to every sample in X.

    Constraint validation is only meaningful for CICIDS2017. 
    For other datasets validate_constraints=False and all mutations are accepted
    if the call succeeds without raising an exception.
    """
    validator = TCPConstraintValidator() if validate_constraints else None
    mutated = np.zeros_like(X, dtype=np.float32)
    valid = np.zeros(len(X), dtype=bool)

    for i, sample in enumerate(X):
        try:
            perturbed = mutation(sample)
            if validator is not None and not validator.validate(sample, perturbed):
                valid[i] = False
                continue
            mutated[i] = perturbed.astype(np.float32, copy=False)
            valid[i] = True
        except Exception:
            valid[i] = False
    return mutated, valid


def evaluate_mutation(
    model_name: str,
    orig_pred: np.ndarray,
    pert_pred: np.ndarray,
    valid_mask: np.ndarray,
    benign_id: int,
) -> dict:
    originally_detected = orig_pred != benign_id
    n_originally_detected = int(originally_detected.sum())

    valid_and_detected = originally_detected & valid_mask
    n_valid_and_detected = int(valid_and_detected.sum())

    evaded = valid_and_detected & (pert_pred == benign_id)
    n_evaded = int(evaded.sum())

    esr = (n_evaded / n_originally_detected) if n_originally_detected > 0 else 0.0
    esr_on_valid = (n_evaded / n_valid_and_detected) if n_valid_and_detected > 0 else 0.0

    return {
        "model": model_name,
        "n_originally_detected": n_originally_detected,
        "n_originally_detected_and_valid": n_valid_and_detected,
        "n_evaded": n_evaded,
        "esr": esr,
        "esr_on_valid": esr_on_valid,
    }


###
# Per-dataset evaluation
###
def evaluate_dataset(dataset: str, args) -> dict:
    X_train, y_train, X_test, y_test, label_map = load_dataset_data(dataset)
    benign_id = benign_label_id(label_map, dataset)
    validate_constraints = dataset in CONSTRAINT_VALIDATION_DATASETS

    profile = get_benign_profile(args.profile, X_train, y_train, benign_id)

    # Attack samples drawn from test split only (don't touch training data)
    attack_mask = y_test != benign_id
    attack_indices = np.where(attack_mask)[0]

    if args.max_attack_samples is not None and len(attack_indices) > args.max_attack_samples:
        rng = np.random.default_rng(args.seed)
        attack_indices = rng.choice(
            attack_indices, size=args.max_attack_samples, replace=False
        )
        attack_indices = np.sort(attack_indices)

    X_attack = X_test[attack_indices]

    ### Models ###
    rf, xgb, scaler, mlp = load_models(dataset)
    models = {
        "random_forest": (rf, None),
        "xgboost": (xgb, None),
        "mlp": (mlp, scaler),
    }

    baseline_predictions = {}
    for model_name, (model_obj, model_scaler) in models.items():
        baseline_predictions[model_name] = predict_by_model(
            model_name, model_obj, X_attack, scaler=model_scaler
        )

    ### Mutation setup ###
    # Closures capture profile and args to match apply_mutation_batch's expected signature
    def _mimic_timing_fn(row: np.ndarray) -> np.ndarray:
        return mimic_timing(
            row,
            profile,
            max_delay_ms=args.max_delay_ms,
            maximum_duration_ratio=args.max_duration_ratio,
        )

    def _mimic_packet_size_fn(row: np.ndarray) -> np.ndarray:
        out = mimic_packet_size(row, profile)
        return _recompute_rates_after_packet_size(out)

    mutation_specs = {
        "mimic_timing": _mimic_timing_fn,
        "mimic_packet_size": _mimic_packet_size_fn,
    }

    ### Constraint validation note ###
    if validate_constraints:
        constraint_note = "TCPConstraintValidator applied."
    else:
        constraint_note = (
            "Constraint validation skipped: validators are scoped to CICIDS2017 feature schema."
        )

    dataset_output = {
        "dataset": dataset,
        "parameters": {
            "profile": args.profile,
            "max_delay_ms": args.max_delay_ms,
            "max_duration_ratio": args.max_duration_ratio,
            "max_attack_samples": args.max_attack_samples,
            "seed": args.seed,
        },
        "profile_summary": {
            "flow_iat_mean_us": profile.get("flow_iat_mean", {}).get("mean_us"),
            "flow_iat_std_us": profile.get("flow_iat_std", {}).get("mean_us"),
            "pkt_len_mean_bytes": profile.get("pkt_len_mean", {}).get("mean_us"),
        },
        "constraint_validation": {
            "enabled": validate_constraints,
            "validator": "TCPConstraintValidator" if validate_constraints else None,
            "note": constraint_note,
        },
        "n_attack_samples_evaluated": int(len(X_attack)),
        "n_attack_samples_total_in_test": int(attack_mask.sum()),
        "metrics": {},
    }

    ### Mutations ###
    for mutation_name, mutation_fn in mutation_specs.items():
        mutated_X, valid_mask = apply_mutation_batch(
            X_attack, mutation_fn, validate_constraints=validate_constraints
        )
        eval_X = np.where(valid_mask[:, None], mutated_X, X_attack)

        mutation_metrics = {
            "constraint_satisfaction_rate": (
                float(valid_mask.mean()) if validate_constraints and len(valid_mask) > 0 else None
            ),
            "n_constraint_pass": int(valid_mask.sum()) if validate_constraints else None,
            "n_samples": int(len(valid_mask)),
            "n_mutation_success": int(valid_mask.sum()),
            "models": {},
        }

        for model_name, (model_obj, model_scaler) in models.items():
            pert_pred = predict_by_model(
                model_name, model_obj, eval_X, scaler=model_scaler
            )
            model_metrics = evaluate_mutation(
                model_name=model_name,
                orig_pred=baseline_predictions[model_name],
                pert_pred=pert_pred,
                valid_mask=valid_mask,
                benign_id=benign_id,
            )
            mutation_metrics["models"][model_name] = model_metrics

        dataset_output["metrics"][mutation_name] = mutation_metrics

    return dataset_output


###
# Main
###
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Attack B (behavioral mimicry) evasion metrics."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to evaluate. Default: all.",
    )
    parser.add_argument(
        "--max-attack-samples",
        type=int,
        default=None,
        help="Optional cap on number of attack samples from each dataset test split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile",
        type=str,
        default="computed",
        help="Benign profile: 'computed' or name from benign_profiles.json (e.g. https, dns).",
    )
    parser.add_argument(
        "--max-delay-ms",
        type=float,
        default=500.0,
        help="Max IAT (ms) for mimic_timing; delays above this are capped.",
    )
    parser.add_argument(
        "--max-duration-ratio",
        type=float,
        default=2.0,
        help="mimic_timing: flow duration cannot exceed this ratio of original.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/attack_b_metrics_all_datasets.json).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_PATH

    dataset_results = [evaluate_dataset(dataset, args) for dataset in args.datasets]

    output = {
        "attack": "Attack B - Behavioral Mimicry",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets_evaluated": args.datasets,
        "results": dataset_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {out_path}")
    for entry in dataset_results:
        print(
            f"[{entry['dataset']}] attack_samples={entry['n_attack_samples_evaluated']}, "
            f"constraint_validation={entry['constraint_validation']['enabled']}"
        )


if __name__ == "__main__":
    main()
