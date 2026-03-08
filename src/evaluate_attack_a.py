"""
Evaluate Attack A (feature obfuscation) evasion metrics on CICIDS2017.

Computes per-mutation:
  - ESR (Evasion Success Rate): among originally-detected attack samples, the fraction classified as BENIGN after mutation.
  - Constraint satisfaction rate: fraction of perturbed samples that pass FunctionalConstraintValidator + PlausibilityConstraintValidator.
  - FP score summary: mean pkt_rate_ratio and fraction passing FP threshold.

Attack-to-label routing:
  - inject_decoy_flows  -> DoS* labels (DoS Hulk, DoS GoldenEye, etc.)
  - dilute_scan_pattern -> PortScan label only

Running either attack on the wrong label class would produce meaningless evasion numbers, 
so each mutation is evaluated on its designated subset.

Writes output to results/attack_a_metrics.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
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

SPLITS_DIR  = Path("data/splits")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "attack_a_metrics.json"


###
# Data loading
###
def load_cicids_data():
    npz_path   = SPLITS_DIR / "cicids2017.npz"
    label_path = SPLITS_DIR / "cicids2017_label_map.npy"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing split file: {npz_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label map: {label_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test  = data["X_test"]
        y_test  = data["y_test"]
    label_map = np.load(label_path, allow_pickle=True).item()
    return X_train, y_train, X_test, y_test, label_map


def benign_label_id(label_map: dict) -> int:
    for idx, name in label_map.items():
        if name == "BENIGN":
            return int(idx)
    raise ValueError("Could not find BENIGN label in cicids2017_label_map.npy")


def dos_label_ids(label_map: dict) -> list[int]:
    """Return all label IDs whose name starts with 'DoS'."""
    return [int(idx) for idx, name in label_map.items() if name.startswith("DoS")]


def portscan_label_id(label_map: dict) -> Optional[int]:
    for idx, name in label_map.items():
        if name == "PortScan":
            return int(idx)
    return None


###
# Model loading and prediction
###
def load_models():
    rf     = joblib.load(MODELS_DIR / "rf_cicids2017.pkl")
    xgb    = joblib.load(MODELS_DIR / "xgb_cicids2017.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler_cicids2017.pkl")
    mlp    = tf.keras.models.load_model(MODELS_DIR / "mlp_cicids2017.h5", compile=False)
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
# Mutation batch application
###
def apply_inject_decoy_batch(
    X: np.ndarray,
    benign_pool: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Apply inject_decoy_flows to every sample in X.

    Returns:
        mutated    — array of perturbed samples (invalid rows hold original)
        valid      — boolean mask, True where mutation passed validation
        fp_scores  — list of FP score dicts, one per sample
    """
    validator = CompositeConstraintValidator([
        FunctionalConstraintValidator(attack_class="dos"),
        PlausibilityConstraintValidator(),
    ])

    mutated   = np.array(X, dtype=np.float64)
    valid     = np.zeros(len(X), dtype=bool)
    fp_scores = []

    for i, sample in enumerate(X):
        try:
            perturbed, meta = inject_decoy_flows(
                sample, benign_pool, k=k, attack_type="dos", rng=rng
            )
            fp_scores.append(meta["fp_score"])
            if perturbed is not None and validator.validate(sample, perturbed):
                mutated[i] = perturbed
                valid[i]   = True
        except Exception:
            fp_scores.append(None)

    return mutated.astype(np.float32), valid, fp_scores


def apply_dilute_scan_batch(
    X: np.ndarray,
    cover_traffic_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Apply dilute_scan_pattern to every sample in X.

    Returns:
        mutated    — array of perturbed samples (invalid rows hold original)
        valid      — boolean mask, True where mutation passed validation
        fp_scores  — list of FP score dicts, one per sample
    """
    validator = CompositeConstraintValidator([
        FunctionalConstraintValidator(attack_class="portscan"),
        PlausibilityConstraintValidator(),
    ])

    mutated   = np.array(X, dtype=np.float64)
    valid     = np.zeros(len(X), dtype=bool)
    fp_scores = []

    for i, sample in enumerate(X):
        try:
            perturbed, meta = dilute_scan_pattern(
                sample, cover_traffic_rate=cover_traffic_rate, rng=rng
            )
            fp_scores.append(meta["fp_score"])
            if perturbed is not None and validator.validate(sample, perturbed):
                mutated[i] = perturbed
                valid[i]   = True
        except Exception:
            fp_scores.append(None)

    return mutated.astype(np.float32), valid, fp_scores


###
# Metrics computation
###
def evasion_metrics(
    model_name: str,
    orig_pred: np.ndarray,
    pert_pred: np.ndarray,
    valid_mask: np.ndarray,
    benign_id: int,
) -> dict:
    """
    Compute ESR for one model on one mutation.

    ESR denominator is all originally-detected samples (valid or not), consistent with the Attack C evaluator. 
    sr_on_valid uses only samples that also passed constraint validation, 
    which is the more meaningful number when constraint satisfaction rate is low.
    """
    originally_detected   = orig_pred != benign_id
    n_originally_detected = int(originally_detected.sum())

    valid_and_detected   = originally_detected & valid_mask
    n_valid_and_detected = int(valid_and_detected.sum())

    evaded   = valid_and_detected & (pert_pred == benign_id)
    n_evaded = int(evaded.sum())

    esr          = n_evaded / n_originally_detected if n_originally_detected > 0 else 0.0
    esr_on_valid = n_evaded / n_valid_and_detected  if n_valid_and_detected  > 0 else 0.0

    return {
        "model":                              model_name,
        "n_originally_detected":              n_originally_detected,
        "n_originally_detected_and_valid":    n_valid_and_detected,
        "n_evaded":                           n_evaded,
        "esr":                                round(esr, 4),
        "esr_on_valid":                       round(esr_on_valid, 4),
    }


def fp_score_summary(fp_scores: list[dict]) -> dict:
    """Aggregate FP score dicts into summary statistics."""
    valid_scores = [s for s in fp_scores if s is not None]
    if not valid_scores:
        return {"n_scored": 0}

    ratios   = [s["pkt_rate_ratio"] for s in valid_scores]
    passing  = [s["passes_threshold"] for s in valid_scores]

    return {
        "n_scored":              len(valid_scores),
        "mean_pkt_rate_ratio":   round(float(np.mean(ratios)), 4),
        "median_pkt_rate_ratio": round(float(np.median(ratios)), 4),
        "frac_passing_threshold": round(float(np.mean(passing)), 4),
    }


###
# Main
###
def main():
    parser = argparse.ArgumentParser(description="Evaluate Attack A evasion metrics.")
    parser.add_argument(
        "--max-dos-samples",
        type=int,
        default=None,
        help="Cap on DoS samples evaluated by inject_decoy_flows.",
    )
    parser.add_argument(
        "--max-scan-samples",
        type=int,
        default=None,
        help="Cap on PortScan samples evaluated by dilute_scan_pattern.",
    )
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--k",                  type=int,   default=5,
                        help="Number of decoy flows for inject_decoy_flows.")
    parser.add_argument("--cover-traffic-rate", type=float, default=1.0,
                        help="Dilution intensity for dilute_scan_pattern (0.0–2.0).")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    X_train, y_train, X_test, y_test, label_map = load_cicids_data()

    benign_id   = benign_label_id(label_map)
    dos_ids     = dos_label_ids(label_map)
    scan_id     = portscan_label_id(label_map)

    # Benign pool for inject_decoy_flows — drawn from training set only
    benign_pool = X_train[y_train == benign_id].astype(np.float64)
    if len(benign_pool) == 0:
        raise RuntimeError("No BENIGN samples found in X_train.")

    ### Build per-attack evaluation subsets from X_test ###
    dos_mask   = np.isin(y_test, dos_ids)
    dos_idx    = np.where(dos_mask)[0]

    if args.max_dos_samples is not None and len(dos_idx) > args.max_dos_samples:
        dos_idx = rng.choice(dos_idx, size=args.max_dos_samples, replace=False)
        dos_idx = np.sort(dos_idx)

    X_dos = X_test[dos_idx].astype(np.float64)

    scan_idx = np.array([], dtype=int)
    X_scan   = np.empty((0, X_test.shape[1]), dtype=np.float64)
    if scan_id is not None:
        scan_mask = y_test == scan_id
        scan_idx  = np.where(scan_mask)[0]
        if args.max_scan_samples is not None and len(scan_idx) > args.max_scan_samples:
            scan_idx = rng.choice(scan_idx, size=args.max_scan_samples, replace=False)
            scan_idx = np.sort(scan_idx)
        X_scan = X_test[scan_idx].astype(np.float64)

    rf, xgb, scaler, mlp = load_models()
    models = {
        "random_forest": (rf,  None),
        "xgboost":       (xgb, None),
        "mlp":           (mlp, scaler),
    }

    # Baseline predictions on original (unperturbed) subsets
    dos_baseline  = {
        name: predict(name, obj, X_dos,  scaler=sc)
        for name, (obj, sc) in models.items()
    }
    scan_baseline = {
        name: predict(name, obj, X_scan, scaler=sc)
        for name, (obj, sc) in models.items()
    } if len(X_scan) > 0 else {}

    output = {
        "attack":           "Attack A - Feature Obfuscation",
        "dataset":          "cicids2017",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "k":                  args.k,
            "cover_traffic_rate": args.cover_traffic_rate,
            "max_dos_samples":    args.max_dos_samples,
            "max_scan_samples":   args.max_scan_samples,
            "seed":               args.seed,
        },
        "n_benign_pool":                  int(len(benign_pool)),
        "n_dos_samples_evaluated":        int(len(X_dos)),
        "n_scan_samples_evaluated":       int(len(X_scan)),
        "n_dos_samples_total_in_test":    int(dos_mask.sum()),
        "n_scan_samples_total_in_test":   int((y_test == scan_id).sum()) if scan_id is not None else 0,
        "metrics": {},
    }

    ### inject_decoy_flows (DoS subset) ###
    mutated_dos, valid_dos, fp_dos = apply_inject_decoy_batch(
        X_dos, benign_pool, k=args.k, rng=rng
    )

    # For invalid mutations, fall back to the original so model predictions
    # do not spuriously inflate ESR; the valid_mask excludes them from ESR.
    eval_dos = np.where(valid_dos[:, None], mutated_dos, X_dos.astype(np.float32))

    inj_metrics = {
        "target_labels":              "DoS*",
        "constraint_satisfaction_rate": round(float(valid_dos.mean()), 4) if len(valid_dos) > 0 else 0.0,
        "n_constraint_pass":          int(valid_dos.sum()),
        "n_samples":                  int(len(valid_dos)),
        "fp_score_summary":           fp_score_summary(fp_dos),
        "models":                     {},
    }
    for model_name, (model_obj, model_scaler) in models.items():
        pert_pred = predict(model_name, model_obj, eval_dos, scaler=model_scaler)
        inj_metrics["models"][model_name] = evasion_metrics(
            model_name, dos_baseline[model_name], pert_pred, valid_dos, benign_id
        )

    output["metrics"]["inject_decoy_flows"] = inj_metrics

    ### dilute_scan_pattern (PortScan subset) ###
    if len(X_scan) > 0:
        mutated_scan, valid_scan, fp_scan = apply_dilute_scan_batch(
            X_scan, cover_traffic_rate=args.cover_traffic_rate, rng=rng
        )
        eval_scan = np.where(valid_scan[:, None], mutated_scan, X_scan.astype(np.float32))

        scan_metrics = {
            "target_labels":              "PortScan",
            "constraint_satisfaction_rate": round(float(valid_scan.mean()), 4) if len(valid_scan) > 0 else 0.0,
            "n_constraint_pass":          int(valid_scan.sum()),
            "n_samples":                  int(len(valid_scan)),
            "fp_score_summary":           fp_score_summary(fp_scan),
            "models":                     {},
        }
        for model_name, (model_obj, model_scaler) in models.items():
            pert_pred = predict(model_name, model_obj, eval_scan, scaler=model_scaler)
            scan_metrics["models"][model_name] = evasion_metrics(
                model_name, scan_baseline[model_name], pert_pred, valid_scan, benign_id
            )

        output["metrics"]["dilute_scan_pattern"] = scan_metrics
    else:
        output["metrics"]["dilute_scan_pattern"] = {
            "skipped": True,
            "reason": "PortScan label not found in test split or no samples available.",
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {RESULTS_PATH}")
    print(f"DoS samples evaluated:     {len(X_dos)}")
    print(f"PortScan samples evaluated: {len(X_scan)}")


if __name__ == "__main__":
    main()