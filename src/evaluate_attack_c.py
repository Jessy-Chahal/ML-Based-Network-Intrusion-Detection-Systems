"""
Evaluate Attack C (protocol exploitation) evasion metrics across datasets.

Computes:
- ESR (Evasion Success Rate): among originally detected attack samples,
  the fraction classified as benign/normal after mutation.
- Constraint satisfaction rate on CICIDS2017 only (TCPConstraintValidator).

For NSL-KDD and UNSW-NB15, constraint validation is intentionally skipped
because validator checks are scoped to the CICIDS2017 feature schema.

Writes output to results/attack_c_metrics_all_datasets.json.
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
# - python -m src.evaluate_attack_c
# - python src/evaluate_attack_c.py
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

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "attack_c_metrics_all_datasets.json"

DATASETS = ["cicids2017", "nslkdd", "unswnb15"]
CONSTRAINT_VALIDATION_DATASETS = {"cicids2017"}
BENIGN_LABEL_CANDIDATES = {"BENIGN", "Normal", "normal"}


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


def compute_target_iat_ms(X_train: np.ndarray, y_train: np.ndarray, benign_id: int) -> float:
    benign = X_train[y_train == benign_id]
    if len(benign) == 0:
        return 50.0
    benign_iat_us = np.clip(benign[:, F.FLOW_IAT_MEAN], a_min=0.0, a_max=None)
    target_iat_us = float(np.median(benign_iat_us))
    target_iat_ms = target_iat_us / 1000.0
    return float(np.clip(target_iat_ms, 1.0, 2000.0))


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


def apply_mutation_batch(
    X: np.ndarray,
    mutation: Callable[[np.ndarray], np.ndarray],
    validate_constraints: bool,
) -> tuple[np.ndarray, np.ndarray]:
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


def evaluate_dataset(dataset: str, args) -> dict:
    X_train, y_train, X_test, y_test, label_map = load_dataset_data(dataset)
    benign_id = benign_label_id(label_map, dataset)
    validate_constraints = dataset in CONSTRAINT_VALIDATION_DATASETS

    attack_mask = y_test != benign_id
    attack_indices = np.where(attack_mask)[0]

    if args.max_attack_samples is not None and len(attack_indices) > args.max_attack_samples:
        rng = np.random.default_rng(args.seed)
        attack_indices = rng.choice(
            attack_indices, size=args.max_attack_samples, replace=False
        )
        attack_indices = np.sort(attack_indices)

    X_attack = X_test[attack_indices]

    iat_target_ms = (
        float(args.target_iat_ms)
        if args.target_iat_ms is not None
        else compute_target_iat_ms(X_train, y_train, benign_id)
    )

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

    mutation_specs = {
        "fragment_payload": lambda row: fragment_payload(row, args.n_fragments),
        "add_tcp_options": add_tcp_options,
        "shift_ack_timing": lambda row: shift_ack_timing(row, iat_target_ms),
    }

    if validate_constraints:
        constraint_note = "TCPConstraintValidator applied."
    else:
        constraint_note = (
            "Constraint validation skipped: validators are scoped to CICIDS2017 feature schema."
        )

    dataset_output = {
        "dataset": dataset,
        "parameters": {
            "n_fragments": args.n_fragments,
            "target_iat_ms": iat_target_ms,
            "max_attack_samples": args.max_attack_samples,
            "seed": args.seed,
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Attack C evasion metrics.")
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
    parser.add_argument("--n-fragments", type=int, default=4)
    parser.add_argument(
        "--target-iat-ms",
        type=float,
        default=None,
        help="Override target IAT (ms) for shift_ack_timing. If omitted, uses benign median IAT.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/attack_c_metrics_all_datasets.json).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_PATH

    dataset_results = [evaluate_dataset(dataset, args) for dataset in args.datasets]

    output = {
        "attack": "Attack C - Protocol Exploitation",
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
