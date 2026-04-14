"""
Evaluate Attack A (feature obfuscation) evasion metrics across datasets.

Computes per-mutation:
  - ESR (Evasion Success Rate): among originally-detected attack samples,
    the fraction classified as BENIGN/Normal after mutation.
  - Constraint satisfaction rate: fraction of perturbed samples passing
    FunctionalConstraintValidator. CICIDS2017 only - validators are scoped
    to the CICIDS2017 feature schema and are skipped for other datasets.
  - FP score summary: mean pkt_rate_ratio and fraction passing FP threshold.
    CICIDS2017 only for the same reason.

Attack-to-label routing:
  - inject_decoy_flows  -> DoS* labels on CICIDS2017 / all attack labels on others
  - dilute_scan_pattern -> PortScan label on CICIDS2017 / all attack labels on others

For NSL-KDD and UNSW-NB15, both mutations are applied to all attack samples
(no label-specific routing) since those datasets do not have the same
DoS*/PortScan label structure as CICIDS2017.

Writes output to results/attack_a_metrics_all_datasets.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import sys

import joblib
import numpy as np
import tensorflow as tf

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.attacks.feature_obfuscation import (
    inject_decoy_flows,
    dilute_scan_pattern,
    compute_fp_score,
)
from src.constraints import (
    CICIDSFeatures as F,
    FunctionalConstraintValidator,
    PlausibilityConstraintValidator,
    CompositeConstraintValidator,
)

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "attack_a_metrics_all_datasets.json"

DATASETS = ["cicids2017", "nslkdd", "unswnb15"]

# Only CICIDS2017 has validators scoped to its feature schema
CONSTRAINT_VALIDATION_DATASETS = {"cicids2017"}

# Benign label names across datasets
BENIGN_LABEL_CANDIDATES = {"BENIGN", "Normal", "normal"}


###
# Data loading
###
def load_dataset(dataset: str):
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


def benign_label_id(label_map: dict, dataset: str) -> int:
    for idx, name in label_map.items():
        if name in BENIGN_LABEL_CANDIDATES:
            return int(idx)
    raise ValueError(f"Could not find benign label in {dataset}_label_map.npy")


def dos_label_ids(label_map: dict) -> list[int]:
    """Return all label IDs whose name starts with 'DoS'. CICIDS2017 only."""
    return [int(idx) for idx, name in label_map.items() if name.startswith("DoS")]


def portscan_label_id(label_map: dict) -> Optional[int]:
    """Return the PortScan label ID, or None if not present. CICIDS2017 only."""
    for idx, name in label_map.items():
        if name == "PortScan":
            return int(idx)
    return None


###
# Model loading and prediction
###
def load_models(dataset: str):
    rf = joblib.load(MODELS_DIR / f"rf_{dataset}.pkl")
    xgb = joblib.load(MODELS_DIR / f"xgb_{dataset}.pkl")
    scaler = joblib.load(MODELS_DIR / f"scaler_{dataset}.pkl")
    mlp = tf.keras.models.load_model(
        MODELS_DIR / f"mlp_{dataset}.h5", compile=False
    )
    return rf, xgb, scaler, mlp


def predict(model_name: str, model_obj, X: np.ndarray, scaler=None) -> np.ndarray:
    if model_name == "mlp":
        if scaler is None:
            raise ValueError("MLP prediction requires a scaler.")
        X_scaled = scaler.transform(X)
        probs = model_obj.predict(X_scaled, verbose=0)
        return np.argmax(probs, axis=1).astype(np.int64)
    return model_obj.predict(X).astype(np.int64)


###
# Metrics helpers
###
def evasion_metrics(
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

    esr = n_evaded / n_originally_detected if n_originally_detected > 0 else 0.0
    esr_on_valid = n_evaded / n_valid_and_detected  if n_valid_and_detected  > 0 else 0.0

    return {
        "model":                           model_name,
        "n_originally_detected":           n_originally_detected,
        "n_originally_detected_and_valid": n_valid_and_detected,
        "n_evaded":                        n_evaded,
        "esr":                             round(esr, 4),
        "esr_on_valid":                    round(esr_on_valid, 4),
    }


def fp_score_summary(fp_scores: list) -> Optional[dict]:
    """
    Aggregate FP score dicts into summary statistics.
    Returns None if no valid scores (e.g. non-CICIDS2017 datasets).
    """
    valid_scores = [s for s in fp_scores if s is not None]
    if not valid_scores:
        return None

    ratios = [s["pkt_rate_ratio"] for s in valid_scores]
    passing = [s["passes_threshold"] for s in valid_scores]

    return {
        "n_scored":               len(valid_scores),
        "mean_pkt_rate_ratio":    round(float(np.mean(ratios)), 4),
        "median_pkt_rate_ratio":  round(float(np.median(ratios)), 4),
        "frac_passing_threshold": round(float(np.mean(passing)), 4),
    }


###
# Mutation batch runners
###
def apply_inject_decoy_batch(
    X: np.ndarray,
    benign_pool: np.ndarray,
    k: int,
    rng: np.random.Generator,
    validate_constraints: bool,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Apply inject_decoy_flows to every sample in X.

    Constraint validation is only meaningful for CICIDS2017. For other
    datasets validate_constraints=False and all mutations are accepted
    if the call succeeds without raising an exception.
    """
    mutated = np.array(X, dtype=np.float64)
    valid = np.zeros(len(X), dtype=bool)
    fp_scores = []

    for i, sample in enumerate(X):
        try:
            perturbed, meta = inject_decoy_flows(
                sample, benign_pool, k=k, attack_type="dos", rng=rng
            )
            # FP scores reference CICIDS2017-specific feature indices - only
            # collect them when constraint validation is active.
            fp_scores.append(meta["fp_score"] if validate_constraints else None)

            if perturbed is not None:
                mutated[i] = perturbed
                valid[i] = True
        except Exception:
            fp_scores.append(None)

    return mutated.astype(np.float32), valid, fp_scores


def apply_dilute_scan_batch(
    X: np.ndarray,
    cover_traffic_rate: float,
    rng: np.random.Generator,
    validate_constraints: bool,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Apply dilute_scan_pattern to every sample in X.

    Same constraint validation logic as apply_inject_decoy_batch.
    """
    mutated = np.array(X, dtype=np.float64)
    valid = np.zeros(len(X), dtype=bool)
    fp_scores = []

    for i, sample in enumerate(X):
        try:
            perturbed, meta = dilute_scan_pattern(
                sample, cover_traffic_rate=cover_traffic_rate, rng=rng
            )
            fp_scores.append(meta["fp_score"] if validate_constraints else None)

            if perturbed is not None:
                mutated[i] = perturbed
                valid[i] = True
        except Exception:
            fp_scores.append(None)

    return mutated.astype(np.float32), valid, fp_scores


###
# Per-dataset evaluation
###
def evaluate_dataset(dataset: str, args) -> dict:
    X_train, y_train, X_test, y_test, label_map = load_dataset(dataset)
    benign_id = benign_label_id(label_map, dataset)
    validate_constraints = dataset in CONSTRAINT_VALIDATION_DATASETS
    rng = np.random.default_rng(args.seed)

    # Benign pool drawn from training set only - never touch test split
    benign_pool = X_train[y_train == benign_id].astype(np.float64)
    if len(benign_pool) == 0:
        raise RuntimeError(f"No benign samples found in {dataset} X_train.")

    ### Label routing ###
    # CICIDS2017: route each mutation to its designated label subset.
    # Other datasets: apply both mutations to all attack samples since they
    # do not have the same DoS*/PortScan label structure.
    if dataset == "cicids2017":
        dos_ids = dos_label_ids(label_map)
        scan_id = portscan_label_id(label_map)

        dos_idx = np.where(np.isin(y_test, dos_ids))[0]
        if args.max_dos_samples is not None and len(dos_idx) > args.max_dos_samples:
            dos_idx = np.sort(rng.choice(dos_idx, size=args.max_dos_samples, replace=False))
        X_inject = X_test[dos_idx].astype(np.float64)

        if scan_id is not None:
            scan_idx = np.where(y_test == scan_id)[0]
            if args.max_scan_samples is not None and len(scan_idx) > args.max_scan_samples:
                scan_idx = np.sort(rng.choice(scan_idx, size=args.max_scan_samples, replace=False))
            X_dilute = X_test[scan_idx].astype(np.float64)
        else:
            X_dilute = np.empty((0, X_test.shape[1]), dtype=np.float64)

        inject_label_note = "DoS* labels only"
        dilute_label_note = "PortScan label only"

    else:
        attack_idx = np.where(y_test != benign_id)[0]
        if args.max_dos_samples is not None and len(attack_idx) > args.max_dos_samples:
            attack_idx = np.sort(rng.choice(attack_idx, size=args.max_dos_samples, replace=False))
        X_inject = X_test[attack_idx].astype(np.float64)
        X_dilute = X_inject.copy()

        inject_label_note = "All attack labels (no DoS-specific routing for this dataset)"
        dilute_label_note = "All attack labels (no PortScan-specific routing for this dataset)"

    ### Constraint validation note ###
    if validate_constraints:
        constraint_note = "FunctionalConstraintValidator applied."
    else:
        constraint_note = (
            "Constraint validation skipped: validators are scoped to "
            "the CICIDS2017 feature schema."
        )

    ### Models ###
    rf, xgb, scaler, mlp = load_models(dataset)
    models = {
        "random_forest": (rf,  None),
        "xgboost":       (xgb, None),
        "mlp":           (mlp, scaler),
    }

    inject_baseline = {
        name: predict(name, obj, X_inject, scaler=sc)
        for name, (obj, sc) in models.items()
    } if len(X_inject) > 0 else {}

    dilute_baseline = {
        name: predict(name, obj, X_dilute, scaler=sc)
        for name, (obj, sc) in models.items()
    } if len(X_dilute) > 0 else {}

    ### inject_decoy_flows ###
    inject_metrics: dict = {"skipped": True, "reason": "No samples available."}
    if len(X_inject) > 0:
        mutated, valid_mask, fp_scores = apply_inject_decoy_batch(
            X_inject, benign_pool, k=args.k, rng=rng,
            validate_constraints=validate_constraints,
        )
        eval_X = np.where(valid_mask[:, None], mutated, X_inject.astype(np.float32))

        inject_metrics = {
            "target_labels":              inject_label_note,
            "constraint_satisfaction_rate": (
                round(float(valid_mask.mean()), 4) if validate_constraints else None
            ),
            "n_constraint_pass":  int(valid_mask.sum()) if validate_constraints else None,
            "n_samples":          int(len(valid_mask)),
            "n_mutation_success": int(valid_mask.sum()),
            "fp_score_summary":   fp_score_summary(fp_scores),
            "constraint_validation": {
                "enabled": validate_constraints,
                "note":    constraint_note,
            },
            "models": {},
        }
        for model_name, (model_obj, model_scaler) in models.items():
            pert_pred = predict(model_name, model_obj, eval_X, scaler=model_scaler)
            inject_metrics["models"][model_name] = evasion_metrics(
                model_name, inject_baseline[model_name], pert_pred, valid_mask, benign_id
            )

    ### dilute_scan_pattern ###
    dilute_metrics: dict = {"skipped": True, "reason": "No samples available."}
    if len(X_dilute) > 0:
        mutated, valid_mask, fp_scores = apply_dilute_scan_batch(
            X_dilute, cover_traffic_rate=args.cover_traffic_rate, rng=rng,
            validate_constraints=validate_constraints,
        )
        eval_X = np.where(valid_mask[:, None], mutated, X_dilute.astype(np.float32))

        dilute_metrics = {
            "target_labels":              dilute_label_note,
            "constraint_satisfaction_rate": (
                round(float(valid_mask.mean()), 4) if validate_constraints else None
            ),
            "n_constraint_pass":  int(valid_mask.sum()) if validate_constraints else None,
            "n_samples":          int(len(valid_mask)),
            "n_mutation_success": int(valid_mask.sum()),
            "fp_score_summary":   fp_score_summary(fp_scores),
            "constraint_validation": {
                "enabled": validate_constraints,
                "note":    constraint_note,
            },
            "models": {},
        }
        for model_name, (model_obj, model_scaler) in models.items():
            pert_pred = predict(model_name, model_obj, eval_X, scaler=model_scaler)
            dilute_metrics["models"][model_name] = evasion_metrics(
                model_name, dilute_baseline[model_name], pert_pred, valid_mask, benign_id
            )
    
    # NSL-KDD is structurally incompatible with Attack A mutations -
    # both functions reference CICIDS2017-specific feature indices that
    # require at least 67 features. NSL-KDD has 40. Add a note so the
    # 0% results are not misread as meaningful evasion results.
    if dataset == "nslkdd":
        incompatible_note = (
            "Attack A mutations reference CICIDS2017-specific feature indices "
            "(minimum 67 features required). NSL-KDD has 40 features - "
            "mutations are structurally incompatible with this dataset."
        )
        inject_metrics["compatibility_note"] = incompatible_note
        dilute_metrics["compatibility_note"] = incompatible_note

    return {
        "dataset":                    dataset,
        "n_benign_pool":              int(len(benign_pool)),
        "n_inject_samples_evaluated": int(len(X_inject)),
        "n_dilute_samples_evaluated": int(len(X_dilute)),
        "parameters": {
            "k":                  args.k,
            "cover_traffic_rate": args.cover_traffic_rate,
            "max_dos_samples":    args.max_dos_samples,
            "max_scan_samples":   args.max_scan_samples,
            "seed":               args.seed,
        },
        "metrics": {
            "inject_decoy_flows":  inject_metrics,
            "dilute_scan_pattern": dilute_metrics,
        },
    }


###
# Main
###
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Attack A evasion metrics across datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to evaluate. Default: all three.",
    )
    parser.add_argument(
        "--max-dos-samples",
        type=int,
        default=None,
        help="Cap on DoS/attack samples for inject_decoy_flows.",
    )
    parser.add_argument(
        "--max-scan-samples",
        type=int,
        default=None,
        help="Cap on PortScan samples for dilute_scan_pattern (CICIDS2017 only).",
    )
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--k",                  type=int,   default=5,
                        help="Number of decoy flows for inject_decoy_flows.")
    parser.add_argument("--cover-traffic-rate", type=float, default=1.0,
                        help="Dilution intensity for dilute_scan_pattern (0.0–2.0).")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/attack_a_metrics_all_datasets.json).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_PATH

    dataset_results = [evaluate_dataset(dataset, args) for dataset in args.datasets]

    output = {
        "attack":             "Attack A - Feature Obfuscation",
        "generated_at_utc":   datetime.now(timezone.utc).isoformat(),
        "datasets_evaluated": args.datasets,
        "results":            dataset_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {out_path}")
    for entry in dataset_results:
        cv = entry["dataset"] in CONSTRAINT_VALIDATION_DATASETS
        print(
            f"[{entry['dataset']}]"
            f"  inject_samples={entry['n_inject_samples_evaluated']}"
            f"  dilute_samples={entry['n_dilute_samples_evaluated']}"
            f"  constraint_validation={'yes' if cv else 'no'}"
        )


if __name__ == "__main__":
    main()