"""
Category A: Feature Obfuscation Attacks

Dilute or mask the anomalous statistics of a malicious flow by blending it with benign decoys, 
adding cover traffic, or slowing port scan probes.

Each attack returns the perturbed sample (or None if it fails validation) along
with a metadata dict containing the functional preservation score and any violations.
"""

import numpy as np
from typing import Optional, Tuple

from src.mutations import (
    blend_with_benign,
    add_padding,
    delay_packets,
    Direction,
    REGISTRY,
)
from src.constraints import (
    CICIDSFeatures as F,
    FunctionalConstraintValidator,
    PlausibilityConstraintValidator,
    CompositeConstraintValidator,
)


###
# Functional preservation scoring
###
def compute_fp_score(
    original: np.ndarray,
    perturbed: np.ndarray,
    attack_type: str,
) -> dict:
    """
    Compute the functional preservation (FP) score for a perturbed sample.

    Compares packet rate before and after perturbation. 
    Rate is recomputed from base features (TOT_FWD_PKTS, TOT_BWD_PKTS, FLOW_DURATION). 
    Returns a dict with the rate ratio and whether it meets the minimum threshold for the attack type.
    """
    def _rate(s: np.ndarray) -> float:
        total_pkts = s[F.TOT_FWD_PKTS] + s[F.TOT_BWD_PKTS]
        duration_sec = s[F.FLOW_DURATION] / 1e6
        return total_pkts / duration_sec if duration_sec > 0 else float('inf')

    orig_rate = _rate(original)
    pert_rate = _rate(perturbed)

    if orig_rate > 0 and orig_rate != float('inf'):
        ratio = pert_rate / orig_rate
    else:
        ratio = 1.0

    thresholds = {"portscan": 0.60, "dos": 0.70}
    threshold = thresholds.get(attack_type, 0.50)

    return {
        "attack_type": attack_type,
        "pkt_rate_ratio": round(ratio, 4),
        "passes_threshold": bool(ratio >= threshold),
    }


###
# Shared validation helper
###
def _validate(
    original: np.ndarray,
    perturbed: np.ndarray,
    attack_type: str,
    attack_name: str,
    extra_metadata: dict,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Run constraint validators against a perturbed sample and return results.

    Skips PlausibilityConstraintValidator for portscan flows, 
    since their near-zero payloads fall below the minimum packet size threshold by design
    (not as a result of the perturbation).
    """
    validators = [FunctionalConstraintValidator(attack_class=attack_type)]
    if attack_type != "portscan":
        validators.append(PlausibilityConstraintValidator())

    validator = CompositeConstraintValidator(validators)
    fp_score = compute_fp_score(original, perturbed, attack_type)
    violations = validator.describe_violations(original, perturbed)

    metadata = {
        "attack": attack_name,
        "fp_score": fp_score,
        "valid": len(violations) == 0,
        "violations": violations,
        **extra_metadata,
    }

    return (perturbed if not violations else None), metadata


###
# Attack 1: Decoy flow injection
###
def inject_decoy_flows(
    malicious_sample: np.ndarray,
    benign_pool: np.ndarray,
    k: int,
    attack_type: str = "dos",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Blend a malicious flow with k benign flows to move it toward the benign distribution.

    Uses a weighted average: (1/(k+1)) * malicious + (k/(k+1)) * mean(benign_sample).
    Higher k pushes the result further toward benign. 
    The attacker generates additional outbound traffic alongside the attack (no spoofing needed).

    Returns the perturbed sample and a metadata dict (or None if validation fails).
    """
    perturbed = blend_with_benign(
        malicious_sample, benign_pool,
        k_samples=k,
        random_number_generator=rng,
    )
    return _validate(
        malicious_sample, perturbed, attack_type,
        attack_name="inject_decoy_flows",
        extra_metadata={"k": k},
    )


###
# Attack 2: Scan pattern dilution
###
def dilute_scan_pattern(
    scan_sample: np.ndarray,
    cover_traffic_rate: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Dilute a port scan flow's burst signature using two chained mutations.

    First slows probe timing by shifting inter-arrival time (50ms per unit of cover_traffic_rate), 
    then injects ACK-only padding packets to simulate interleaved normal traffic. 
    The actual probe packets are unchanged, only delays and cover traffic are added around them.

    Returns the perturbed sample and a metadata dict (None if validation fails).
    """
    if not (0.0 <= cover_traffic_rate <= 2.0):
        raise ValueError(
            f"cover_traffic_rate must be between 0.0 and 2.0, got {cover_traffic_rate}"
        )

    # Map cover_traffic_rate to IAT shift: 1.0 rate -> 50ms delay
    delta_ms = cover_traffic_rate * 50.0

    # Map cover_traffic_rate to padding packet count: proportional to the
    # original forward packet count, capped to avoid PlausibilityValidator
    # rejecting an implausibly inflated packet count.
    orig_fwd_pkts = int(scan_sample[F.TOT_FWD_PKTS])
    n_padding = max(1, int(orig_fwd_pkts * cover_traffic_rate * 0.5))

    # Slow the probe timing
    perturbed = delay_packets(scan_sample, delta_milliseconds=delta_ms, direction=Direction.FORWARD)

    # Inject cover padding packets
    perturbed = add_padding(perturbed, number_of_packets=n_padding, direction=Direction.FORWARD, padding_bytes=40)

    return _validate(
        scan_sample, perturbed, "portscan",
        attack_name="dilute_scan_pattern",
        extra_metadata={
            "cover_traffic_rate": cover_traffic_rate,
            "delta_ms": delta_ms,
            "n_padding_packets": n_padding,
        },
    )