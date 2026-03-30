"""
Defense evaluation script — Task A3.

Measures whether the adversarially retrained ensemble recovers detection
on samples that evade the baseline ensemble.

For each mutation:
  1. Re-generate perturbed test samples using the same mutations as the
     attack evaluators (Attack A, B, C).
  2. Run both the baseline and adversarial ensembles on the perturbed samples.
  3. Report detection accuracy on clean data, detection accuracy on adversarial
     data, and the recovery delta (adversarial minus baseline).

Target: at least 25-35 percentage point recovery on adversarial samples
relative to the undefended baseline (per proposal, Tafreshian & Zhang 2025,
Awad et al. 2025).

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

SPLITS_DIR  = Path("data/splits")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "defense_metrics.json"

# Benign label name in CICIDS2017
BENIGN_LABEL = "BENIGN"


### Data loading ###
def load_cicids():
    with np.load(SPLITS_DIR / "cicids2017.npz", allow_pickle=True) as d:
        X_train = d["X_train"]
        y_train = d["y_train"]
        X_test  = d["X_test"]
        y_test  = d["y_test"]
    label_map = np.load(SPLITS_DIR / "cicids2017_label_map.npy", allow_pickle=True).item()
    return X_train, y_train, X_test, y_test, label_map


def get_benign_id(label_map: dict) -> int:
    for idx, name in label_map.items():
        if name == BENIGN_LABEL:
            return int(idx)
    raise ValueError("BENIGN label not found in label map")


### Mutation helpers ###
def apply_mutation_batch(X: np.ndarray, mutate_fn) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply mutate_fn to each sample in X.

    Returns:
        mutated   — array of perturbed samples (invalid rows hold original)
        valid     — boolean mask, True where mutation succeeded
    """
    mutated = np.array(X, dtype=np.float32)
    valid   = np.zeros(len(X), dtype=bool)

    for i, sample in enumerate(X):
        try:
            result = mutate_fn(sample)
            # Attack A/B/C functions return (perturbed, meta) tuples;
            # raw mutation functions return just the array.
            if isinstance(result, tuple):
                perturbed, _ = result
            else:
                perturbed = result

            if perturbed is not None:
                mutated[i] = perturbed.astype(np.float32)
                valid[i]   = True
        except Exception:
            pass

    return mutated, valid


### Metrics ###
def detection_rate(ensemble: Ensemble, X: np.ndarray, y_true: np.ndarray, benign_id: int) -> float:
    """
    Fraction of originally-detected attack samples that are still detected
    (i.e. NOT classified as benign) after mutation.

    Higher = better defense. Lower = more evasion.
    """
    preds    = ensemble.predict(X)
    detected = (preds != benign_id)
    # Only count samples that were attacks in the first place
    is_attack = (y_true != benign_id)
    n_attack  = int(is_attack.sum())
    if n_attack == 0:
        return 0.0
    return float(detected[is_attack].sum()) / n_attack


def clean_detection_rate(ensemble: Ensemble, X_clean: np.ndarray, y_clean: np.ndarray, benign_id: int) -> float:
    """
    Fraction of attack samples in the clean (unperturbed) test set that are
    correctly detected. Used to confirm the ensemble baseline before comparing
    adversarial performance.
    """
    return detection_rate(ensemble, X_clean, y_clean, benign_id)


def evaluate_mutation(
    mutation_name: str,
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    mutate_fn,
    baseline: Ensemble,
    adversarial: Ensemble,
    benign_id: int,
) -> dict:
    """
    Run one mutation against both ensembles and return the metrics dict.
    """
    mutated, valid = apply_mutation_batch(X_attack, mutate_fn)

    # Use original sample for invalid mutations so they don't spuriously
    # inflate evasion — valid_mask excludes them from rate calculation.
    eval_X = np.where(valid[:, None], mutated, X_attack.astype(np.float32))

    # Detection rates on valid perturbed samples only
    X_valid  = eval_X[valid]
    y_valid  = y_attack[valid]
    n_valid  = int(valid.sum())
    n_total  = int(len(X_attack))

    if n_valid == 0:
        return {
            "mutation":                    mutation_name,
            "n_samples":                   n_total,
            "n_mutation_success":          0,
            "baseline_detection_rate":     None,
            "adversarial_detection_rate":  None,
            "recovery_delta_pp":           None,
            "meets_25pp_target":           False,
            "note": "No valid mutations produced — cannot compute detection rates.",
        }

    baseline_dr    = detection_rate(baseline,    X_valid, y_valid, benign_id)
    adversarial_dr = detection_rate(adversarial, X_valid, y_valid, benign_id)
    delta_pp       = (adversarial_dr - baseline_dr) * 100.0

    return {
        "mutation":                   mutation_name,
        "n_samples":                  n_total,
        "n_mutation_success":         n_valid,
        "baseline_detection_rate":    round(baseline_dr, 4),
        "adversarial_detection_rate": round(adversarial_dr, 4),
        "recovery_delta_pp":          round(delta_pp, 2),
        # Target from proposal: 25-35pp recovery
        "meets_25pp_target":          delta_pp >= 25.0,
    }


### Main ###
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    print("Loading data...")
    X_train, y_train, X_test, y_test, label_map = load_cicids()
    benign_id = get_benign_id(label_map)

    # Benign pool for Attack A mutations — training set only
    benign_pool = X_train[y_train == benign_id].astype(np.float64)

    # All attack samples from test split
    attack_mask = y_test != benign_id
    X_attack    = X_test[attack_mask].astype(np.float64)
    y_attack    = y_test[attack_mask]

    # Benign IAT target for shift_ack_timing (median benign IAT in ms)
    benign_iat_us = np.clip(X_train[y_train == benign_id][:, F.FLOW_IAT_MEAN], 0, None)
    target_iat_ms = float(np.clip(np.median(benign_iat_us) / 1000.0, 1.0, 2000.0))

    # Benign profile for Attack B mimic mutations
    benign      = X_train[y_train == benign_id]
    total_pkts  = benign[:, F.TOT_FWD_PKTS] + benign[:, F.TOT_BWD_PKTS]
    total_bytes = benign[:, F.TOT_LEN_FWD_PKTS] + benign[:, F.TOT_LEN_BWD_PKTS]
    pkt_sizes   = np.where(total_pkts > 0, total_bytes / total_pkts, np.nan)
    benign_profile = {
        "flow_iat_mean": {"mean_us": float(np.median(np.clip(benign[:, F.FLOW_IAT_MEAN], 0, None)))},
        "flow_iat_std":  {"mean_us": float(np.median(np.clip(benign[:, F.FLOW_IAT_STD],  0, None)))},
        "pkt_len_mean":  {"mean_us": float(np.clip(np.nanmedian(pkt_sizes), 20.0, 1500.0))},
    }

    print("Loading ensembles...")
    baseline_ensemble    = Ensemble.baseline()
    adversarial_ensemble = Ensemble.adversarial()
    print(f"  Baseline   : {baseline_ensemble}")
    print(f"  Adversarial: {adversarial_ensemble}")

    ### Clean detection rate ###
    # Run both ensembles on the unperturbed attack subset first.
    # This is the ceiling — the best either ensemble can do before any mutation.
    print("\nComputing clean detection rates...")
    baseline_clean_dr    = clean_detection_rate(baseline_ensemble,    X_attack.astype(np.float32), y_attack, benign_id)
    adversarial_clean_dr = clean_detection_rate(adversarial_ensemble, X_attack.astype(np.float32), y_attack, benign_id)

    ### Mutations to evaluate ###
    # These match the mutations used in the attack evaluators.
    mutations = {
        # Attack A
        "inject_decoy_flows": lambda s: inject_decoy_flows(
            s, benign_pool, k=5, attack_type="dos", rng=rng
        ),
        "dilute_scan_pattern": lambda s: dilute_scan_pattern(
            s, cover_traffic_rate=1.0, rng=rng
        ),
        # Attack B
        "mimic_timing": lambda s: mimic_timing(
            s, benign_profile, max_delay_ms=500.0, maximum_duration_ratio=2.0
        ),
        "mimic_packet_size": lambda s: mimic_packet_size(s, benign_profile),
        # Attack C
        "fragment_payload":  lambda s: fragment_payload(s, n_fragments=4),
        "add_tcp_options":   add_tcp_options,
        "shift_ack_timing":  lambda s: shift_ack_timing(s, target_iat_ms=target_iat_ms),
    }

    ### Evaluate each mutation ###
    mutation_results = []
    for name, fn in mutations.items():
        print(f"  Evaluating {name}...")
        result = evaluate_mutation(
            mutation_name=name,
            X_attack=X_attack,
            y_attack=y_attack,
            mutate_fn=fn,
            baseline=baseline_ensemble,
            adversarial=adversarial_ensemble,
            benign_id=benign_id,
        )
        mutation_results.append(result)
        if result["recovery_delta_pp"] is not None:
            print(
                f"    baseline={result['baseline_detection_rate']:.3f}  "
                f"adversarial={result['adversarial_detection_rate']:.3f}  "
                f"delta={result['recovery_delta_pp']:+.1f}pp  "
                f"Meets 25pp target: {'Yes' if result['meets_25pp_target'] else 'No'}"
            )

    ### Summary ###
    n_meeting_target = sum(
        1 for r in mutation_results
        if r["meets_25pp_target"]
    )

    output = {
        "generated_at_utc":  datetime.now(timezone.utc).isoformat(),
        "dataset":           "cicids2017",
        "target_recovery_pp": 25.0,
        "n_attack_samples":  int(attack_mask.sum()),
        "clean_detection": {
            "baseline_detection_rate":    round(baseline_clean_dr, 4),
            "adversarial_detection_rate": round(adversarial_clean_dr, 4),
            "note": (
                "Detection rate on unperturbed attack samples. "
                "Both ensembles should be near 1.0 here."
            ),
        },
        "n_mutations_evaluated":      len(mutation_results),
        "n_mutations_meeting_target": n_meeting_target,
        "mutations": mutation_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {RESULTS_PATH}")
    print(f"Mutations meeting 25pp target: {n_meeting_target} / {len(mutation_results)}")
    print(f"Clean detection — baseline: {baseline_clean_dr:.3f}  adversarial: {adversarial_clean_dr:.3f}")


if __name__ == "__main__":
    main()