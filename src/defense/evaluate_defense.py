"""
Defense evaluation script 

Evaluates the adversarially retrained ensemble across all three datasets (CICIDS2017, NSL-KDD, UNSW-NB15).  
For each dataset the correct baseline and adversarial model pair is loaded, 
compatible mutations are applied to the attack subset of the test split, and recovery delta is reported.

Mutation compatibility per dataset
-----------------------------------
CICIDS2017 (70 features):
    All 7 mutations are run.

NSL-KDD (40 features):
    Only mimic_timing is run.  
    All other mutations are structurally incompatible:
      - mimic_packet_size accesses indices 48, 52, 54 which exceed the 40-feature NSL-KDD dimensionality.
      - fragment_payload, add_tcp_options, and shift_ack_timing call TCPConstraintValidator, which requires >=67 features.
      - inject_decoy_flows and dilute_scan_pattern depend on the CICIDS2017-schema benign pool and constraint validators.

UNSW-NB15 (67 features):
    mimic_timing, mimic_packet_size, fragment_payload, add_tcp_options, and shift_ack_timing are run.  
    inject_decoy_flows and dilute_scan_pattern are skipped because they require the CICIDS2017-schema benign pool.

    Note: 
    For UNSW-NB15 and NSL-KDD, mutation functions operate on CICIDS2017-equivalent positional feature indices.  
    The mutations perturb the feature vector at those positions regardless of what those ositions represent in 
    the other dataset's schema.  
    This tests whether adversarial retraining provides general robustness to feature perturbations, 
    which is the cross-dataset research question.

Writes output to results/defense_metrics.json.

Run from repo root:
    python src/defense/evaluate_defense.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.defense.ensemble import Ensemble
from src.attacks.feature_obfuscation import inject_decoy_flows, dilute_scan_pattern
from src.attacks.behavioral_mimicry import mimic_timing, mimic_packet_size
from src.attacks.protocol_exploitation import (
    fragment_payload,
    add_tcp_options,
    shift_ack_timing,
)
from src.constraints import CICIDSFeatures as F

SPLITS_DIR = Path("data/splits")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "defense_metrics.json"

# Normal/benign label name per dataset
BENIGN_LABEL = {
    "cicids2017": "BENIGN",
    "nslkdd": "Normal",
    "unswnb15": "Normal",
}

# All mutation names in run order (used by CICIDS2017)
ALL_MUTATIONS = [
    "inject_decoy_flows",
    "dilute_scan_pattern",
    "mimic_timing",
    "mimic_packet_size",
    "fragment_payload",
    "add_tcp_options",
    "shift_ack_timing",
]

# Mutations to skip per dataset, with the reason stored as the dict value
SKIPPED_MUTATIONS: dict[str, dict[str, str]] = {
    "nslkdd": {
        "inject_decoy_flows": (
            "Requires CICIDS2017-schema benign pool and constraint validators "
            "tied to CICIDS2017 feature indices."
        ),
        "dilute_scan_pattern": (
            "Requires CICIDS2017 feature semantics for cover-traffic insertion "
            "and constraint validators."
        ),
        "mimic_packet_size": (
            "Accesses feature indices 48 (PKT_SIZE_AVG), 52 (SUBFLOW_FWD_BYTS), "
            "54 (SUBFLOW_BWD_BYTS) which exceed NSL-KDD dimensionality (40 features)."
        ),
        "fragment_payload": (
            "TCPConstraintValidator._check_inputs requires >=67 features; "
            "NSL-KDD has 40."
        ),
        "add_tcp_options": (
            "TCPConstraintValidator._check_inputs requires >=67 features; "
            "NSL-KDD has 40."
        ),
        "shift_ack_timing": (
            "TCPConstraintValidator._check_inputs requires >=67 features; "
            "NSL-KDD has 40."
        ),
    },
    "unswnb15": {
        "inject_decoy_flows": (
            "Requires a CICIDS2017-schema benign pool; UNSW-NB15 features at "
            "the same positional indices carry different semantics."
        ),
        "dilute_scan_pattern": (
            "Requires CICIDS2017 feature semantics for cover-traffic insertion."
        ),
    },
    "cicids2017": {},
}


### Data loading ###

def load_dataset(dataset: str):
    with np.load(SPLITS_DIR / f"{dataset}.npz", allow_pickle=True) as d:
        X_train = d["X_train"]
        y_train = d["y_train"]
        X_test = d["X_test"]
        y_test = d["y_test"]
    label_map = np.load(
        SPLITS_DIR / f"{dataset}_label_map.npy", allow_pickle=True
    ).item()
    return X_train, y_train, X_test, y_test, label_map


def get_benign_id(label_map: dict, dataset: str) -> int:
    target = BENIGN_LABEL[dataset]
    for idx, name in label_map.items():
        if name == target:
            return int(idx)
    raise ValueError(f"Benign label '{target}' not found in label map for {dataset}")


### Benign profile helpers ###

def build_benign_profile(X_benign: np.ndarray) -> dict:
    iat_mean = np.clip(X_benign[:, F.FLOW_IAT_MEAN], 0, None)
    iat_std = np.clip(X_benign[:, F.FLOW_IAT_STD],  0, None)

    total_pkts = X_benign[:, F.TOT_FWD_PKTS] + X_benign[:, F.TOT_BWD_PKTS]
    total_bytes = X_benign[:, F.TOT_LEN_FWD_PKTS] + X_benign[:, F.TOT_LEN_BWD_PKTS]
    pkt_sizes = np.where(total_pkts > 0, total_bytes / total_pkts, np.nan)

    return {
        "flow_iat_mean": {"mean_us": float(np.median(iat_mean))},
        "flow_iat_std": {"mean_us": float(np.median(iat_std))},
        "pkt_len_mean": {"mean_us": float(np.clip(np.nanmedian(pkt_sizes), 20.0, 1500.0))},
    }


def build_target_iat_ms(X_benign: np.ndarray) -> float:
    iat_us = np.clip(X_benign[:, F.FLOW_IAT_MEAN], 0, None)
    return float(np.clip(np.median(iat_us) / 1000.0, 1.0, 2000.0))


### Mutation application ###

def apply_mutation_batch(X: np.ndarray, mutate_fn) -> tuple[np.ndarray, np.ndarray]:
    mutated = np.array(X, dtype=np.float32)
    valid = np.zeros(len(X), dtype=bool)

    for i, sample in enumerate(X):
        try:
            result = mutate_fn(sample)
            if isinstance(result, tuple):
                perturbed, _ = result
            else:
                perturbed = result

            if perturbed is not None:
                mutated[i] = perturbed.astype(np.float32)
                valid[i] = True
        except Exception:
            pass

    return mutated, valid


### Metrics ###

def detection_rate_per_model(
    ensemble: Ensemble,
    X: np.ndarray,
    y_true: np.ndarray,
    benign_id: int,
) -> dict:
    """
    Returns the binary attack detection rate (TPR) per individual model inside the ensemble.
    """
    votes = ensemble._get_votes(np.array(X, dtype=np.float64))
    voter_names = ["random_forest", "xgboost", "mlp"][:len(votes)]

    is_attack = y_true != benign_id
    n_attack = int(is_attack.sum())

    result = {}
    for name, preds in zip(voter_names, votes):
        if n_attack == 0:
            result[name] = None
        else:
            result[name] = round(
                float((preds[is_attack] != benign_id).sum()) / n_attack, 4
            )
    return result


def detection_rate(
    ensemble: Ensemble,
    X: np.ndarray,
    y_true: np.ndarray,
    benign_id: int,
) -> float:
    preds = ensemble.predict(X)
    detected = preds != benign_id
    is_attack = y_true != benign_id
    n_attack = int(is_attack.sum())
    if n_attack == 0:
        return 0.0
    return float(detected[is_attack].sum()) / n_attack


def compute_clean_metrics(
    ensemble: Ensemble,
    X_test: np.ndarray,
    y_test: np.ndarray,
    benign_id: int,
) -> dict:
    """
    Computes overarching Accuracy, Precision, Recall, F1, and Detection Rate 
    on the clean, unperturbed test dataset (binary: Attack vs Benign).
    """
    preds = ensemble.predict(X_test.astype(np.float32))
    
    y_true_binary = (y_test != benign_id).astype(int)
    preds_binary = (preds != benign_id).astype(int)
    
    acc = accuracy_score(y_true_binary, preds_binary)
    prec = precision_score(y_true_binary, preds_binary, zero_division=0)
    rec = recall_score(y_true_binary, preds_binary, zero_division=0)
    f1 = f1_score(y_true_binary, preds_binary, zero_division=0)
    
    return {
        "accuracy":       round(float(acc), 4),
        "precision":      round(float(prec), 4),
        "recall":         round(float(rec), 4),
        "f1":             round(float(f1), 4),
        "detection_rate": round(float(rec), 4), # DR is identical to Recall in this binary setup
    }


def evaluate_mutation(
    mutation_name: str,
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    X_benign_test: np.ndarray,
    y_benign_test: np.ndarray,
    mutate_fn,
    baseline: Ensemble,
    adversarial: Ensemble,
    benign_id: int,
) -> dict:
    mutated, valid = apply_mutation_batch(X_attack, mutate_fn)

    # Reconstruct the test set attacks, keeping failed mutations as originals
    eval_X = np.where(valid[:, None], mutated, X_attack.astype(np.float32))
    
    X_valid = eval_X[valid]
    y_valid = y_attack[valid]
    n_valid = int(valid.sum())
    n_total = int(len(X_attack))

    if n_valid == 0:
        return {
            "mutation": mutation_name,
            "n_samples": n_total,
            "n_mutation_success": 0,
            "recovery_delta_pp": None,
            "meets_25pp_target": False,
            "baseline_metrics": None,
            "adversarial_metrics": None,
            "baseline_per_model": None,
            "adversarial_per_model": None,
            "note": "No valid mutations produced - cannot compute metrics.",
        }

    # 1. Detection Rate (Calculated specifically on successfully mutated samples)
    baseline_dr = detection_rate(baseline, X_valid, y_valid, benign_id)
    adversarial_dr = detection_rate(adversarial, X_valid, y_valid, benign_id)
    delta_pp = (adversarial_dr - baseline_dr) * 100.0

    # 2. Reconstruct the FULL test set (Benign + Mutated Attacks) for overarching binary metrics
    X_full_test = np.vstack([X_benign_test, eval_X]).astype(np.float32)
    y_full_test = np.concatenate([y_benign_test, y_attack])
    
    # Binarize labels for overarching metric calculations (Attack=1, Benign=0)
    y_true_binary = (y_full_test != benign_id).astype(int)

    def compute_full_metrics(ensemble, dr):
        preds = ensemble.predict(X_full_test)
        preds_binary = (preds != benign_id).astype(int)
        
        return {
            "accuracy": round(float(accuracy_score(y_true_binary, preds_binary)), 4),
            "precision": round(float(precision_score(y_true_binary, preds_binary, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_binary, preds_binary, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true_binary, preds_binary, zero_division=0)), 4),
            "detection_rate": round(dr, 4),
        }

    baseline_metrics = compute_full_metrics(baseline, baseline_dr)
    adversarial_metrics = compute_full_metrics(adversarial, adversarial_dr)

    # 3. Model level performance
    baseline_per_model = detection_rate_per_model(baseline, X_valid, y_valid, benign_id)
    adversarial_per_model = detection_rate_per_model(adversarial, X_valid, y_valid, benign_id)

    return {
        "mutation":                   mutation_name,
        "n_samples":                  n_total,
        "n_mutation_success":         n_valid,
        "recovery_delta_pp":          round(delta_pp, 2),
        "meets_25pp_target":          delta_pp >= 25.0,
        "baseline_metrics":           baseline_metrics,
        "adversarial_metrics":        adversarial_metrics,
        "baseline_per_model":         baseline_per_model,
        "adversarial_per_model":      adversarial_per_model,
    }


### Per-dataset evaluation ###

def evaluate_dataset(dataset: str, rng: np.random.Generator) -> dict:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")

    ### Data ###
    print("  Loading data...")
    X_train, y_train, X_test, y_test, label_map = load_dataset(dataset)
    benign_id = get_benign_id(label_map, dataset)

    X_benign_train = X_train[y_train == benign_id].astype(np.float64)

    # Separate benign and attack portions of the test set
    benign_mask_test = y_test == benign_id
    X_benign_test = X_test[benign_mask_test].astype(np.float64)
    y_benign_test = y_test[benign_mask_test]

    attack_mask = y_test != benign_id
    X_attack = X_test[attack_mask].astype(np.float64)
    y_attack = y_test[attack_mask]

    ### Benign profile ###
    benign_profile = build_benign_profile(X_benign_train)
    target_iat_ms = build_target_iat_ms(X_benign_train)

    ### Ensembles ###
    print("  Loading ensembles...")
    baseline_ens = Ensemble.baseline_for(dataset)
    adversarial_ens = Ensemble.adversarial_for(dataset)
    print(f"    Baseline   : {baseline_ens}")
    print(f"    Adversarial: {adversarial_ens}")

    ### Clean overarching metrics ###
    print("  Computing clean overall metrics...")
    baseline_clean = compute_clean_metrics(
        baseline_ens, X_test, y_test, benign_id
    )
    adversarial_clean = compute_clean_metrics(
        adversarial_ens, X_test, y_test, benign_id
    )

    ### Build mutation callables ###
    all_mutation_fns = {
        "inject_decoy_flows": lambda s: inject_decoy_flows(
            s, X_benign_train, k=5, attack_type="dos", rng=rng
        ),
        "dilute_scan_pattern": lambda s: dilute_scan_pattern(
            s, cover_traffic_rate=1.0, rng=rng
        ),
        "mimic_timing": lambda s: mimic_timing(
            s, benign_profile, max_delay_ms=500.0, maximum_duration_ratio=2.0
        ),
        "mimic_packet_size": lambda s: mimic_packet_size(s, benign_profile),
        "fragment_payload": lambda s: fragment_payload(s, n_fragments=4),
        "add_tcp_options": add_tcp_options,
        "shift_ack_timing": lambda s: shift_ack_timing(s, target_iat_ms=target_iat_ms),
    }

    skip_map = SKIPPED_MUTATIONS.get(dataset, {})

    ### Evaluate ###
    mutation_results = []

    for name in ALL_MUTATIONS:
        if name in skip_map:
            print(f"  Skipping  {name} (schema incompatibility)")
            mutation_results.append({
                "mutation": name,
                "skipped": True,
                "reason": skip_map[name],
            })
            continue

        print(f"  Evaluating {name}...")
        result = evaluate_mutation(
            mutation_name=name,
            X_attack=X_attack,
            y_attack=y_attack,
            X_benign_test=X_benign_test,
            y_benign_test=y_benign_test,
            mutate_fn=all_mutation_fns[name],
            baseline=baseline_ens,
            adversarial=adversarial_ens,
            benign_id=benign_id,
        )
        mutation_results.append(result)

        if result.get("recovery_delta_pp") is not None:
            print(
                f"    baseline={result['baseline_metrics']['detection_rate']:.3f}  "
                f"adversarial={result['adversarial_metrics']['detection_rate']:.3f}  "
                f"delta={result['recovery_delta_pp']:+.1f}pp  "
                f"Meets 25pp target: {'Yes' if result['meets_25pp_target'] else 'No'}"
            )

    ### Summary ###
    evaluated = [r for r in mutation_results if not r.get("skipped")]
    n_meeting = sum(1 for r in evaluated if r.get("meets_25pp_target"))

    schema_note = None
    if dataset != "cicids2017":
        schema_note = (
            "Mutations that ran on this dataset operate on CICIDS2017-equivalent positional feature indices. "
            "The perturbed index positions may carry different semantic meaning in this dataset's schema. "
            "Results test whether adversarial retraining provides general robustness to feature perturbations, not protocol-specific evasion."
        )

    return {
        "dataset": dataset,
        "n_attack_samples": int(attack_mask.sum()),
        "target_recovery_pp": 25.0,
        "clean_detection": {
            "note": (
                "Metrics on the unperturbed full test set (Binary: Attack vs. Benign). "
            ),
            "baseline":    baseline_clean,
            "adversarial": adversarial_clean,
        },
        "n_mutations_evaluated": len(evaluated),
        "n_mutations_skipped": len(mutation_results) - len(evaluated),
        "n_mutations_meeting_target": n_meeting,
        **({"schema_note": schema_note} if schema_note else {}),
        "mutations": mutation_results,
    }


### Main ###

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    datasets = ["cicids2017", "nslkdd", "unswnb15"]
    all_results = []

    for dataset in datasets:
        result = evaluate_dataset(dataset, rng)
        all_results.append(result)

        n_meet = result["n_mutations_meeting_target"]
        print(
            f"\n  [{dataset}] Mutations meeting 25pp target: "
            f"{n_meet} / {result['n_mutations_evaluated']}  "
            f"(skipped: {result['n_mutations_skipped']})"
        )
        print(
            f"  [{dataset}] Clean detection - "
            f"baseline: acc={result['clean_detection']['baseline']['accuracy']:.3f}  "
            f"dr={result['clean_detection']['baseline']['detection_rate']:.3f}  "
            f"adversarial: acc={result['clean_detection']['adversarial']['accuracy']:.3f}  "
            f"dr={result['clean_detection']['adversarial']['detection_rate']:.3f}"
        )

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()