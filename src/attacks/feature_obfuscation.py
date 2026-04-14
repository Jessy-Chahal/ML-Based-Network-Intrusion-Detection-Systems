"""
Category A: Feature Obfuscation Attacks.

Strategy: dilute or mask the anomalous statistics of a malicious flow by
blending it with benign-looking decoy flows, adding cover traffic padding,
or slowing the probe rate of a port scan. All perturbations are applied
through the shared mutations API in src/mutations.py rather than directly
modifying feature vectors.

Each public function in this module:
  1. Calls one or more mutations from src/mutations.py
  2. Validates the result using the appropriate constraint validators
  3. Returns the perturbed sample (or None if it fails validation) along
     with a metadata dict containing the FP score and any violations

Functional preservation thresholds are defined in
docs/functional_preservation.md. Samples that fail the FP threshold are
rejected - the perturbation has degraded the attack below operational utility.

Feasibility note: inject_decoy_flows requires the attacker to generate
additional outbound traffic from their own host - no source address spoofing
is needed. dilute_scan_pattern adds timing delays, which the attacker
controls directly via socket-level rate limiting.
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

    Packet rate is recomputed from mutable base features (TOT_FWD_PKTS,
    TOT_BWD_PKTS, FLOW_DURATION) rather than read from the stored FLOW_PKTS_S
    field, which is an immutable derived feature that will not reflect
    mutations to the base packet count or duration fields.

    Args:
        original:    Unmodified malicious feature vector.
        perturbed:   Perturbed feature vector.
        attack_type: One of 'portscan' or 'dos'. Selects the FP threshold
                     defined in docs/functional_preservation.md.

    Returns:
        Dict with keys:
            attack_type      - the attack class used for threshold selection
            pkt_rate_ratio   - perturbed_rate / original_rate (0.0–∞)
            passes_threshold - True if ratio meets the minimum threshold
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
    Run constraint validators against a perturbed sample.

    PlausibilityConstraintValidator is intentionally skipped for 'portscan'
    attack type. PortScan flows in CICIDS2017 are SYN-probe flows with
    near-zero payloads, which are structurally below the 20-byte average
    packet size minimum - this is a property of the flow type, not a
    violation introduced by the perturbation.
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
    Blend a malicious flow with k benign flows sampled from the BENIGN pool.

    Delegates to mutations.inject_decoy_flows, which computes:
        (1 / (k+1)) * malicious + (k / (k+1)) * mean(benign_sample)

    Higher k moves the blended vector further toward the benign distribution.
    The attacker generates additional outbound traffic (HTTP GETs, DNS queries)
    from their own host alongside the attack - no spoofing required.

    Args:
        malicious_sample: Feature vector of the attack flow (1D array).
        benign_pool:      2D array of BENIGN class samples from X_train,
                          shape (n_benign, n_features).
        k:                Number of decoy flows to inject. Must be >= 1.
        attack_type:      'dos' or 'portscan' - selects FP threshold.
        rng:              Optional random generator for reproducibility.

    Returns:
        Tuple of (perturbed_sample, metadata).
        perturbed_sample is None if validation fails.
        metadata contains fp_score, valid flag, and any violations.
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
    Dilute a PortScan flow's burst signature using two chained mutations:

        1. shift_inter_arrival_time - slows probe timing to break the burst
           pattern. cover_traffic_rate is mapped to a millisecond delay:
           each unit of cover_traffic_rate adds 50ms of IAT shift.

        2. add_padding_packets - injects cover packets (ACK-only, 40 bytes)
           to simulate interleaved HTTP/DNS traffic alongside the probes.
           The number of padding packets is proportional to cover_traffic_rate
           and the original forward packet count.

    The scan still sends the same probe packets - the mutations only add
    delays and padding traffic around them. Port coverage is preserved
    as long as the FP threshold (pkt_rate_ratio >= 0.60) is met.

    Args:
        scan_sample:        Feature vector of a PortScan flow (1D array).
        cover_traffic_rate: Dilution intensity in the range [0.0, 2.0].
                            0.0 = no dilution. 2.0 = maximum dilution.
        rng:                Optional random generator for reproducibility.

    Returns:
        Tuple of (perturbed_sample, metadata).
        perturbed_sample is None if validation fails.
        metadata contains fp_score, valid flag, violations, and the
        computed delta_ms and n_padding_packets used.
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

    # Step 1: Slow the probe timing
    perturbed = delay_packets(scan_sample, delta_milliseconds=delta_ms, direction=Direction.FORWARD)

    # Step 2: Inject cover padding packets
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