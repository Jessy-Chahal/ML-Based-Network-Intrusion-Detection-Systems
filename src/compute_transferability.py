"""
Compute transferability metrics for Attack A/B/C mutations.

Transferability definition used here:
For each mutation, source dataset, and target model trained on a different
dataset, compute ESR:

    ESR = n_evaded / n_originally_detected

Where "detected/evaded" are defined with respect to the target model's benign
label class.

Output:
results/transferability_metrics.json
with records containing:
  mutation, source_dataset, target_dataset, target_model, esr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np

# Allow both:
# - python -m src.compute_transferability
# - python src/compute_transferability.py
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src import evaluate_attack_a as attack_a_eval
from src import evaluate_attack_b as attack_b_eval
from src import evaluate_attack_c as attack_c_eval
from src.attacks.behavioral_mimicry import mimic_timing, mimic_packet_size
from src.attacks.protocol_exploitation import (
    add_tcp_options,
    fragment_payload,
    shift_ack_timing,
)


DATASETS = ["cicids2017", "nslkdd", "unswnb15"]
RESULTS_DIR = Path("results")
DEFAULT_OUTPUT = RESULTS_DIR / "transferability_metrics.json"


def _subsample_indices(idx: np.ndarray, max_n: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_n is None or len(idx) <= max_n:
        return idx
    chosen = rng.choice(idx, size=max_n, replace=False)
    return np.sort(chosen)


def _compute_esr(
    orig_pred: np.ndarray,
    pert_pred: np.ndarray,
    valid_mask: np.ndarray,
    benign_id_target: int,
) -> float:
    originally_detected = orig_pred != benign_id_target
    n_originally_detected = int(originally_detected.sum())
    if n_originally_detected == 0:
        return 0.0

    valid_and_detected = originally_detected & valid_mask
    evaded = valid_and_detected & (pert_pred == benign_id_target)
    n_evaded = int(evaded.sum())
    return float(n_evaded / n_originally_detected)


def _build_target_context(target_datasets: List[str]) -> Dict[str, dict]:
    contexts: Dict[str, dict] = {}
    for dataset in target_datasets:
        # Load model bundle
        rf, xgb, scaler, mlp = attack_c_eval.load_models(dataset)
        _, _, _, _, label_map = attack_c_eval.load_dataset_data(dataset)
        benign_id = attack_c_eval.benign_label_id(label_map, dataset)

        # Expected model input dimensions for compatibility checks.
        rf_n = int(getattr(rf, "n_features_in_", -1))
        xgb_n = int(getattr(xgb, "n_features_in_", -1))
        mlp_n = int(mlp.input_shape[-1])

        contexts[dataset] = {
            "benign_id": benign_id,
            "models": {
                "random_forest": (rf, None),
                "xgboost": (xgb, None),
                "mlp": (mlp, scaler),
            },
            "n_features": {
                "random_forest": rf_n,
                "xgboost": xgb_n,
                "mlp": mlp_n,
            },
        }
    return contexts


def _build_attack_a_batches(source_dataset: str, args, rng: np.random.Generator):
    """
    Return list of tuples:
      (mutation_name, X_source_original, X_source_eval, valid_mask)
    """
    X_train, y_train, X_test, y_test, label_map = attack_a_eval.load_dataset(source_dataset)
    benign_id = attack_a_eval.benign_label_id(label_map, source_dataset)
    validate_constraints = source_dataset in attack_a_eval.CONSTRAINT_VALIDATION_DATASETS

    benign_pool = X_train[y_train == benign_id].astype(np.float64)
    if len(benign_pool) == 0:
        return []

    if source_dataset == "cicids2017":
        dos_ids = attack_a_eval.dos_label_ids(label_map)
        scan_id = attack_a_eval.portscan_label_id(label_map)

        dos_idx = np.where(np.isin(y_test, dos_ids))[0]
        dos_idx = _subsample_indices(dos_idx, args.max_dos_samples, rng)
        X_inject = X_test[dos_idx].astype(np.float64)

        if scan_id is not None:
            scan_idx = np.where(y_test == scan_id)[0]
            scan_idx = _subsample_indices(scan_idx, args.max_scan_samples, rng)
            X_dilute = X_test[scan_idx].astype(np.float64)
        else:
            X_dilute = np.empty((0, X_test.shape[1]), dtype=np.float64)
    else:
        attack_idx = np.where(y_test != benign_id)[0]
        attack_idx = _subsample_indices(attack_idx, args.max_attack_samples, rng)
        X_inject = X_test[attack_idx].astype(np.float64)
        X_dilute = X_inject.copy()

    batches = []

    if len(X_inject) > 0:
        mutated, valid_mask, _ = attack_a_eval.apply_inject_decoy_batch(
            X_inject,
            benign_pool=benign_pool,
            k=args.k,
            rng=rng,
            validate_constraints=validate_constraints,
        )
        eval_X = np.where(valid_mask[:, None], mutated, X_inject.astype(np.float32))
        batches.append(("inject_decoy_flows", X_inject.astype(np.float32), eval_X, valid_mask))

    if len(X_dilute) > 0:
        mutated, valid_mask, _ = attack_a_eval.apply_dilute_scan_batch(
            X_dilute,
            cover_traffic_rate=args.cover_traffic_rate,
            rng=rng,
            validate_constraints=validate_constraints,
        )
        eval_X = np.where(valid_mask[:, None], mutated, X_dilute.astype(np.float32))
        batches.append(("dilute_scan_pattern", X_dilute.astype(np.float32), eval_X, valid_mask))

    return batches


def _build_attack_b_batches(source_dataset: str, args, rng: np.random.Generator):
    X_train, y_train, X_test, y_test, label_map = attack_b_eval.load_dataset_data(source_dataset)
    benign_id = attack_b_eval.benign_label_id(label_map, source_dataset)
    validate_constraints = source_dataset in attack_b_eval.CONSTRAINT_VALIDATION_DATASETS

    profile = attack_b_eval.get_benign_profile("computed", X_train, y_train, benign_id)

    attack_idx = np.where(y_test != benign_id)[0]
    attack_idx = _subsample_indices(attack_idx, args.max_attack_samples, rng)
    X_attack = X_test[attack_idx].astype(np.float64)
    if len(X_attack) == 0:
        return []

    def _mimic_timing_fn(row: np.ndarray) -> np.ndarray:
        return mimic_timing(
            row,
            profile,
            max_delay_ms=args.max_delay_ms,
            maximum_duration_ratio=args.max_duration_ratio,
        )

    def _mimic_packet_size_fn(row: np.ndarray) -> np.ndarray:
        out = mimic_packet_size(row, profile)
        return attack_b_eval._recompute_rates_after_packet_size(out)

    batches = []
    for mutation_name, mutation_fn in {
        "mimic_timing": _mimic_timing_fn,
        "mimic_packet_size": _mimic_packet_size_fn,
    }.items():
        mutated, valid_mask = attack_b_eval.apply_mutation_batch(
            X_attack,
            mutation_fn,
            validate_constraints=validate_constraints,
        )
        eval_X = np.where(valid_mask[:, None], mutated, X_attack.astype(np.float32))
        batches.append((mutation_name, X_attack.astype(np.float32), eval_X, valid_mask))

    return batches


def _build_attack_c_batches(source_dataset: str, args, rng: np.random.Generator):
    X_train, y_train, X_test, y_test, label_map = attack_c_eval.load_dataset_data(source_dataset)
    benign_id = attack_c_eval.benign_label_id(label_map, source_dataset)
    validate_constraints = source_dataset in attack_c_eval.CONSTRAINT_VALIDATION_DATASETS

    attack_idx = np.where(y_test != benign_id)[0]
    attack_idx = _subsample_indices(attack_idx, args.max_attack_samples, rng)
    X_attack = X_test[attack_idx].astype(np.float64)
    if len(X_attack) == 0:
        return []

    iat_target_ms = (
        float(args.target_iat_ms)
        if args.target_iat_ms is not None
        else attack_c_eval.compute_target_iat_ms(X_train, y_train, benign_id)
    )

    batches = []
    for mutation_name, mutation_fn in {
        "fragment_payload": lambda row: fragment_payload(row, args.n_fragments),
        "add_tcp_options": add_tcp_options,
        "shift_ack_timing": lambda row: shift_ack_timing(row, iat_target_ms),
    }.items():
        mutated, valid_mask = attack_c_eval.apply_mutation_batch(
            X_attack,
            mutation_fn,
            validate_constraints=validate_constraints,
        )
        eval_X = np.where(valid_mask[:, None], mutated, X_attack.astype(np.float32))
        batches.append((mutation_name, X_attack.astype(np.float32), eval_X, valid_mask))

    return batches


def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-dataset transferability ESR for Attack A/B/C mutations."
    )
    parser.add_argument(
        "--source-datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to draw attack samples from (source).",
    )
    parser.add_argument(
        "--target-datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets whose models are used as transfer targets.",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Attack A knobs
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cover-traffic-rate", type=float, default=1.0)
    parser.add_argument("--max-dos-samples", type=int, default=None)
    parser.add_argument("--max-scan-samples", type=int, default=None)

    # Shared cap (Attack B/C and non-CICIDS Attack A routing)
    parser.add_argument("--max-attack-samples", type=int, default=None)

    # Attack B knobs
    parser.add_argument("--max-delay-ms", type=float, default=500.0)
    parser.add_argument("--max-duration-ratio", type=float, default=2.0)

    # Attack C knobs
    parser.add_argument("--n-fragments", type=int, default=4)
    parser.add_argument("--target-iat-ms", type=float, default=None)

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path for transferability metrics.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    target_context = _build_target_context(args.target_datasets)
    records = []
    skipped_feature_mismatch = 0

    for source_dataset in args.source_datasets:
        source_batches = []
        source_batches.extend(_build_attack_a_batches(source_dataset, args, rng))
        source_batches.extend(_build_attack_b_batches(source_dataset, args, rng))
        source_batches.extend(_build_attack_c_batches(source_dataset, args, rng))

        for mutation_name, X_orig, X_eval, valid_mask in source_batches:
            for target_dataset in args.target_datasets:
                if target_dataset == source_dataset:
                    continue

                context = target_context[target_dataset]
                benign_id_target = context["benign_id"]
                for target_model, (model_obj, model_scaler) in context["models"].items():
                    expected_n = context["n_features"][target_model]
                    if X_orig.shape[1] != expected_n:
                        skipped_feature_mismatch += 1
                        continue

                    orig_pred = attack_c_eval.predict_by_model(
                        target_model, model_obj, X_orig, scaler=model_scaler
                    )
                    pert_pred = attack_c_eval.predict_by_model(
                        target_model, model_obj, X_eval, scaler=model_scaler
                    )
                    esr = _compute_esr(orig_pred, pert_pred, valid_mask, benign_id_target)
                    records.append(
                        {
                            "mutation": mutation_name,
                            "source_dataset": source_dataset,
                            "target_dataset": target_dataset,
                            "target_model": target_model,
                            "esr": round(float(esr), 4),
                        }
                    )

    # Stable ordering for downstream reporting/diffs.
    records.sort(
        key=lambda r: (
            r["mutation"],
            r["source_dataset"],
            r["target_dataset"],
            r["target_model"],
        )
    )

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {out_path}")
    print(f"Records: {len(records)}")
    if skipped_feature_mismatch > 0:
        print(
            "Skipped feature-incompatible evaluations: "
            f"{skipped_feature_mismatch} "
            "(source and target model use different feature dimensions)."
        )


if __name__ == "__main__":
    main()
