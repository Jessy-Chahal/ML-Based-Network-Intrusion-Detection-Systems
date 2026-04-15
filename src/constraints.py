"""
Problem-space constraint validators for adversarial example generation.
All perturbations must pass their relevant validator before being added to the adversarial dataset. 

The implementation details for this code are documented in docs/constraint_spec.md.

These validators operate on aggregated CICFlowMeter flow metrics, not raw packets. 
They enforce structural consistency rules (counts, ratios, and derived features must be mutually consistent).

Usage:
    validator = TCPConstraintValidator()
    if validator.validate(original_sample, perturbed_sample):
        adversarial_dataset.append(perturbed_sample)
    else:
        violations = validator.describe_violations(original_sample, perturbed_sample)
        print(f"Discarding sample: {violations}")
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from src.dotenv_utils import get_env_int

###
# Feature index constants for CICIDS2017 (71 features after preprocessing)
# These must match the column order produced by src/preprocess.py exactly
# NSL-KDD and UNSW-NB15 use different feature schemas and their classes will be created later for evaluation phases
###

class CICIDSFeatures:
    """
    These indices correspond to the CICIDS2017 feature vector columns (post-preprocessing).
    The ordering is defined by the src/preprocess.py script.
    """
    DEST_PORT = 0
    FLOW_DURATION = 1
    TOT_FWD_PKTS = 2
    TOT_BWD_PKTS = 3
    TOT_LEN_FWD_PKTS = 4
    TOT_LEN_BWD_PKTS = 5
    FWD_PKT_LEN_MAX = 6
    FWD_PKT_LEN_MIN = 7
    FWD_PKT_LEN_MEAN = 8
    FWD_PKT_LEN_STD = 9
    BWD_PKT_LEN_MAX = 10
    BWD_PKT_LEN_MIN = 11
    BWD_PKT_LEN_MEAN = 12
    BWD_PKT_LEN_STD = 13
    FLOW_BYTS_S = 14
    FLOW_PKTS_S = 15
    FLOW_IAT_MEAN = 16
    FLOW_IAT_STD = 17
    FLOW_IAT_MAX = 18
    FLOW_IAT_MIN = 19
    FWD_IAT_TOT = 20
    FWD_IAT_MEAN = 21
    FWD_IAT_STD = 22
    FWD_IAT_MAX = 23
    FWD_IAT_MIN = 24
    BWD_IAT_TOT = 25
    BWD_IAT_MEAN = 26
    BWD_IAT_STD = 27
    BWD_IAT_MAX = 28
    BWD_IAT_MIN = 29
    FWD_PSH_FLAGS = 30
    FWD_URG_FLAGS = 31
    FWD_HEADER_LEN = 32
    BWD_HEADER_LEN = 33
    FWD_PKTS_S = 34
    BWD_PKTS_S = 35
    PKT_LEN_MIN = 36
    PKT_LEN_MAX = 37
    PKT_LEN_MEAN = 38
    PKT_LEN_STD = 39
    PKT_LEN_VAR = 40
    FIN_FLAG_CNT = 41
    SYN_FLAG_CNT = 42
    RST_FLAG_CNT = 43
    PSH_FLAG_CNT = 44
    ACK_FLAG_CNT = 45
    URG_FLAG_CNT = 46
    DOWN_UP_RATIO = 47
    PKT_SIZE_AVG = 48
    FWD_SEG_SIZE_AVG = 49
    BWD_SEG_SIZE_AVG = 50
    SUBFLOW_FWD_PKTS = 51
    SUBFLOW_FWD_BYTS = 52
    SUBFLOW_BWD_PKTS = 53
    SUBFLOW_BWD_BYTS = 54
    INIT_WIN_BYTS_FWD = 55
    INIT_WIN_BYTS_BWD = 56
    ACT_DATA_PKT_FWD = 57
    MIN_SEG_SIZE_FWD = 58
    ACTIVE_MEAN = 59
    ACTIVE_STD = 60
    ACTIVE_MAX = 61
    ACTIVE_MIN = 62
    IDLE_MEAN = 63
    IDLE_STD = 64
    IDLE_MAX = 65
    IDLE_MIN = 66
    # Indices 67-70 are dataset-specific
    # Check preprocess.py output

    @classmethod
    def max_index(cls) -> int:
        """Return the highest integer index defined in this class."""
        return max(
            v for v in vars(cls).values()
            if isinstance(v, int) and v >= 0
        )


###
# Abstract base class
###
class ConstraintValidator(ABC):
    """
    Abstract base class for all problem-space constraint validators.

    Subclasses implement protocol-specific or category-specific constraint logic. 
    All adversarial sample generators call validate() before accepting a perturbed sample.
    """

    def _check_inputs(
        self, original: np.ndarray, perturbed: np.ndarray
    ) -> None:
        """
        Verify that both arrays are 1D, have matching shapes, and are long enough to contain the highest feature index defined in CICIDSFeatures.
        Raises ValueError with a descriptive message if any check fails.
        """
        if original.ndim != 1 or perturbed.ndim != 1:
            raise ValueError(
                f"Expected 1D feature vectors, got shapes "
                f"{original.shape} and {perturbed.shape}"
            )
        if original.shape != perturbed.shape:
            raise ValueError(
                f"original and perturbed must have the same shape, "
                f"got {original.shape} and {perturbed.shape}"
            )
        min_length = CICIDSFeatures.max_index() + 1
        if original.shape[0] < min_length:
            raise ValueError(
                f"Feature vector length {original.shape[0]} is too short - "
                f"expected at least {min_length} features to cover all "
                f"CICIDSFeatures indices"
            )

    @abstractmethod
    def validate(self, original: np.ndarray, perturbed: np.ndarray) -> bool:
        """
        Return True if the perturbed sample satisfies all constraints enforced by this validator.

        Args:
            original:  The unmodified feature vector (1D numpy array)
            perturbed: The perturbed feature vector (1D numpy array)

        Returns:
            True if all constraints pass, False if any constraint is violated
        """
        pass

    @abstractmethod
    def describe_violations(
        self, original: np.ndarray, perturbed: np.ndarray
    ) -> List[str]:
        """
        Return a list of human-readable strings describing each constraint violation found in the perturbed sample. 
        Returns an empty list if no violations are found.

        Args:
            original:  The unmodified feature vector (1D numpy array).
            perturbed: The perturbed feature vector (1D numpy array).

        Returns:
            List of violation description strings (empty if valid).
        """
        pass

    def validate_batch(
        self, originals: np.ndarray, perturbed: np.ndarray
    ) -> np.ndarray:
        """
        Validate a batch of (original, perturbed) pairs.

        Args:
            originals: 2D array of shape (n_samples, n_features).
            perturbed: 2D array of shape (n_samples, n_features).

        Returns:
            Boolean array of shape (n_samples,) - True where valid.
        """
        return np.array([
            self.validate(orig, pert)
            for orig, pert in zip(originals, perturbed)
        ])


###
# TCP Constraint Validator
###
class TCPConstraintValidator(ConstraintValidator):
    """
    Validates that a perturbed TCP flow sample is internally consistent with respect to flow-level protocol rules.

    These checks operate on aggregated CICFlowMeter statistics, not raw packets. 
    Per-packet checksum and TCP sequence number validity cannot be verified from flow features alone.
    CICFlowMeter has already processed the raw packets to produce these values. 
    What is enforced here are structural consistency rules: counts, ratios, header lengths, and derived rate features must be mutually consistent.

    Enforces:
    - Forward packet count is strictly positive
    - Average packet size falls within physically plausible bounds
    - Per-direction flag counts do not exceed the corresponding packet count
    - Header lengths are consistent with packet counts
    - Flow duration is non-negative
    - Derived rate features are consistent with byte and duration values

    See docs/constraint_spec.md - Constraint Type 1: Protocol Validity.
    """

    MIN_PKT_BYTES = 20      # Min IP header size (bytes)
    MAX_PKT_BYTES = 65535   # Max IP payload (bytes)
    MIN_TCP_HEADER = 20      # Min TCP header size (bytes)

    def validate(self, original: np.ndarray, perturbed: np.ndarray) -> bool:
        return len(self.describe_violations(original, perturbed)) == 0

    def describe_violations(
        self, original: np.ndarray, perturbed: np.ndarray
    ) -> List[str]:
        self._check_inputs(original, perturbed)

        f = CICIDSFeatures
        violations = []
        p = perturbed

        fwd_pkts = p[f.TOT_FWD_PKTS]
        bwd_pkts = p[f.TOT_BWD_PKTS]

        # Forward packet count must be strictly positive
        # Backward may be zero for unidirectional flows such as SYN floods
        if fwd_pkts <= 0:
            violations.append(
                f"tot_fwd_pkts={fwd_pkts:.1f} must be > 0 "
                f"(original={original[f.TOT_FWD_PKTS]:.1f})"
            )
        if bwd_pkts < 0:
            violations.append(
                f"tot_bwd_pkts={bwd_pkts:.1f} must be >= 0 "
                f"(original={original[f.TOT_BWD_PKTS]:.1f})"
            )

        # Average forward packet size must fall within IP and Ethernet bounds
        if fwd_pkts > 0:
            fwd_len = p[f.TOT_LEN_FWD_PKTS]
            avg_fwd = fwd_len / fwd_pkts
            if avg_fwd < self.MIN_PKT_BYTES:
                violations.append(
                    f"avg fwd packet size={avg_fwd:.1f} bytes < "
                    f"minimum {self.MIN_PKT_BYTES} bytes "
                    f"(tot_len={fwd_len:.1f}, pkts={fwd_pkts:.1f})"
                )
            if avg_fwd > self.MAX_PKT_BYTES:
                violations.append(
                    f"avg fwd packet size={avg_fwd:.1f} bytes > "
                    f"MTU {self.MAX_PKT_BYTES} bytes "
                    f"(tot_len={fwd_len:.1f}, pkts={fwd_pkts:.1f})"
                )
        
        # SYN+FIN cannot co-occur
        # The only per-packet flag combination reliably enforceable at the flow level 
        # (unlike SYN+RST, which can legitimately appear across separate packets in a single flow)
        if p[f.SYN_FLAG_CNT] > 0 and p[f.FIN_FLAG_CNT] > 0:
            violations.append(
                f"syn_flag_cnt={p[f.SYN_FLAG_CNT]:.0f} and "
                f"fin_flag_cnt={p[f.FIN_FLAG_CNT]:.0f} are both non-zero - "
                f"SYN+FIN cannot co-occur in any TCP flow"
            )

        # Each flag count is a flow-level total (number of packets in that direction that had the flag set) 
        # A flag count greater than the total packet count is impossible since each packet sets a flag at most once
        total_pkts = fwd_pkts + bwd_pkts
        flag_checks = [
            (f.SYN_FLAG_CNT, "syn_flag_cnt"),
            (f.FIN_FLAG_CNT, "fin_flag_cnt"),
            (f.RST_FLAG_CNT, "rst_flag_cnt"),
            (f.PSH_FLAG_CNT, "psh_flag_cnt"),
            (f.ACK_FLAG_CNT, "ack_flag_cnt"),
            (f.URG_FLAG_CNT, "urg_flag_cnt"),
        ]
        for idx, name in flag_checks:
            if p[idx] > total_pkts:
                violations.append(
                    f"{name}={p[idx]:.1f} exceeds total_pkts={total_pkts:.1f} "
                    f"- flag count cannot exceed packet count"
                )

        # Flow duration must be non-negative
        duration = p[f.FLOW_DURATION]
        if duration < 0:
            violations.append(
                f"flow_duration={duration:.1f} must be >= 0"
            )

        # Average header length per forward packet must be at least the min TCP header size
        if fwd_pkts > 0:
            fwd_hdr = p[f.FWD_HEADER_LEN]
            avg_hdr = fwd_hdr / fwd_pkts
            if avg_hdr < self.MIN_TCP_HEADER:
                violations.append(
                    f"avg fwd header length={avg_hdr:.1f} bytes < "
                    f"minimum TCP header {self.MIN_TCP_HEADER} bytes "
                    f"(fwd_header_len={fwd_hdr:.1f}, pkts={fwd_pkts:.1f})"
                )

        # flow_byts_s must be consistent with the total bytes transferred and the flow duration 
        # A 10% tolerance is applied to account for floating point rounding in CICFlowMeter 
        # Duration is stored in microseconds in CICIDS2017
        if duration > 0:
            total_bytes = p[f.TOT_LEN_FWD_PKTS] + p[f.TOT_LEN_BWD_PKTS]
            expected_rate = total_bytes / (duration / 1e6)
            actual_rate = p[f.FLOW_BYTS_S]
            if actual_rate > 0 and abs(actual_rate - expected_rate) / max(expected_rate, 1) > 0.10:
                violations.append(
                    f"flow_byts_s={actual_rate:.2f} inconsistent with "
                    f"computed rate={expected_rate:.2f} "
                    f"(total_bytes={total_bytes:.1f}, duration={duration:.1f}us) "
                    f"- update flow_byts_s after modifying byte counts or duration"
                )

        return violations


###
# DNS Constraint Validator
###
class DNSConstraintValidator(ConstraintValidator):
    """
    Validates that a perturbed DNS flow sample is consistent with DNS protocol characteristics.

    DNS flows use UDP on port 53. The size limits here use the EDNS0 max (4096 bytes) rather than the legacy 512-byte limit, 
    since modern resolvers routinely negotiate EDNS extensions. 
    If this validator is applied to flows known to be legacy DNS (no EDNS), tighten MAX_DNS_PAYLOAD_BYTES to 512.

    Enforces:
    - Destination port is 53
    - Average query and response sizes do not exceed the EDNS0 payload limit
    - Flow duration does not exceed the standard DNS resolver timeout
    - Forward packet count is within normal DNS query/response range

    See docs/constraint_spec.md - Constraint Type 1: Protocol Validity.
    """

    DNS_PORT = 53
    MAX_DNS_PAYLOAD_BYTES = 4096        # EDNS0 max for both query and response
    MAX_DNS_DURATION_US = 5_000_000   # 5 seconds - standard resolver timeout
    MAX_DNS_FWD_PKTS = 10          # More than ~10 query packets per flow is anomalous

    def validate(self, original: np.ndarray, perturbed: np.ndarray) -> bool:
        return len(self.describe_violations(original, perturbed)) == 0

    def describe_violations(
        self, original: np.ndarray, perturbed: np.ndarray
    ) -> List[str]:
        self._check_inputs(original, perturbed)

        f = CICIDSFeatures
        violations = []
        p = perturbed

        # All remaining checks are only meaningful for DNS flows
        if p[f.DEST_PORT] != self.DNS_PORT:
            violations.append(
                f"dest_port={p[f.DEST_PORT]:.0f} is not DNS port 53 - "
                f"DNSConstraintValidator only applies to DNS flows"
            )
            return violations

        fwd_pkts = p[f.TOT_FWD_PKTS]
        if fwd_pkts > 0:
            avg_query_size = p[f.TOT_LEN_FWD_PKTS] / fwd_pkts
            if avg_query_size > self.MAX_DNS_PAYLOAD_BYTES:
                violations.append(
                    f"avg DNS query size={avg_query_size:.1f} bytes > "
                    f"EDNS0 maximum {self.MAX_DNS_PAYLOAD_BYTES} bytes"
                )

        bwd_pkts = p[f.TOT_BWD_PKTS]
        if bwd_pkts > 0:
            avg_response_size = p[f.TOT_LEN_BWD_PKTS] / bwd_pkts
            if avg_response_size > self.MAX_DNS_PAYLOAD_BYTES:
                violations.append(
                    f"avg DNS response size={avg_response_size:.1f} bytes > "
                    f"EDNS0 maximum {self.MAX_DNS_PAYLOAD_BYTES} bytes"
                )

        # Flows that outlast the resolver timeout would have been closed before completing, making them anomalous in captured traffic
        if p[f.FLOW_DURATION] > self.MAX_DNS_DURATION_US:
            violations.append(
                f"flow_duration={p[f.FLOW_DURATION]:.0f}us exceeds "
                f"DNS timeout threshold {self.MAX_DNS_DURATION_US}us (5s)"
            )

        # Standard DNS exchanges involve 1-2 query packets per direction
        # A high forward packet count suggests tunneling or amplification behavior
        if fwd_pkts > self.MAX_DNS_FWD_PKTS:
            violations.append(
                f"tot_fwd_pkts={fwd_pkts:.0f} > {self.MAX_DNS_FWD_PKTS} - "
                f"normal DNS flows have 1-2 query packets per exchange"
            )

        return violations


###
# Functional Preservation Validator
###
class FunctionalConstraintValidator(ConstraintValidator):
    """
    Validates that a perturbed attack flow retains sufficient attack functionality. 
    Perturbations that reduce attack effectiveness below a threshold are discarded (they have defeated the attack itself).

    Packet rate is recomputed from TOT_FWD_PKTS, TOT_BWD_PKTS, and FLOW_DURATION rather than read from the stored FLOW_PKTS_S field.
    FLOW_PKTS_S is a derived feature written by CICFlowMeter and will not reflect mutations to the base packet count or duration fields.

    Packet rate ratio is used as a flow-level proxy for attack effectiveness.
    For port scans, this approximates port coverage (the more precise metric defined in constraint_spec.md) 
    since CICIDS2017 does not expose per-port probe counts in the aggregated feature vector. 
    For C2 flows, duration expansion is used instead since callback timing is the critical constraint.

    Thresholds are defined per attack class. The attack_class parameter selects which thresholds apply at instantiation time.

    See docs/constraint_spec.md - Constraint Type 2: Functional Preservation.
    """

    THRESHOLDS = {
        "dos":        {"pkt_rate_min_ratio": 0.70},
        "portscan":   {"pkt_rate_min_ratio": 0.60},
        "c2":         {"duration_max_ratio": 2.0},
        "bruteforce": {"pkt_rate_min_ratio": 0.60},
        "default":    {"pkt_rate_min_ratio": 0.50},
    }

    def __init__(self, attack_class: str = "default"):
        """
        Args:
            attack_class: One of 'dos', 'portscan', 'c2', 'bruteforce',
                          or 'default'. Determines which thresholds apply.
        """
        self.attack_class = attack_class.lower()
        self.thresholds = self.THRESHOLDS.get(
            self.attack_class, self.THRESHOLDS["default"]
        )

    def _compute_pkt_rate(self, sample: np.ndarray) -> float:
        """
        Compute packets per second from the mutable base features.
        FLOW_DURATION is stored in microseconds in CICIDS2017.
        Returns inf if duration is zero to avoid division by zero.
        """
        f = CICIDSFeatures
        total_pkts = sample[f.TOT_FWD_PKTS] + sample[f.TOT_BWD_PKTS]
        duration_sec = sample[f.FLOW_DURATION] / 1e6
        return total_pkts / duration_sec if duration_sec > 0 else float('inf')

    def validate(self, original: np.ndarray, perturbed: np.ndarray) -> bool:
        return len(self.describe_violations(original, perturbed)) == 0

    def describe_violations(
        self, original: np.ndarray, perturbed: np.ndarray
    ) -> List[str]:
        self._check_inputs(original, perturbed)

        f = CICIDSFeatures
        violations = []
        p = perturbed
        o = original

        if "pkt_rate_min_ratio" in self.thresholds:
            threshold = self.thresholds["pkt_rate_min_ratio"]
            orig_rate = self._compute_pkt_rate(o)
            pert_rate = self._compute_pkt_rate(p)
            if orig_rate > 0 and orig_rate != float('inf'):
                ratio = pert_rate / orig_rate
                if ratio < threshold:
                    violations.append(
                        f"packet rate (recomputed) degraded to {ratio*100:.1f}% of original "
                        f"({pert_rate:.2f} vs {orig_rate:.2f} pkts/s) - "
                        f"functional preservation threshold for "
                        f"'{self.attack_class}' is {threshold*100:.0f}%"
                    )

        if "duration_max_ratio" in self.thresholds:
            threshold = self.thresholds["duration_max_ratio"]
            orig_dur = o[f.FLOW_DURATION]
            pert_dur = p[f.FLOW_DURATION]
            if orig_dur > 0:
                ratio = pert_dur / orig_dur
                if ratio > threshold:
                    violations.append(
                        f"flow_duration expanded to {ratio:.2f}x original "
                        f"({pert_dur:.0f}us vs {orig_dur:.0f}us) - "
                        f"functional preservation threshold for "
                        f"'{self.attack_class}' is {threshold:.1f}x max"
                    )

        return violations


###
# Behavioral Plausibility Validator
###
class PlausibilityConstraintValidator(ConstraintValidator):
    """
    Validates that a perturbed flow does not exhibit patterns that would be flagged by simple rule-based or threshold detectors, 
    independent of any ML classifier.

    See docs/constraint_spec.md - Constraint Type 3: Behavioral Plausibility.
    """

    MAX_IAT_MEAN_US = get_env_int("MAX_IAT_MEAN_US")
    MAX_FLOW_PKTS_S = get_env_int("MAX_FLOW_PKTS_S")
    MIN_AVG_PKT_BYTES = get_env_int("MIN_AVG_PKT_BYTES")
    MAX_AVG_PKT_BYTES = get_env_int("MAX_AVG_PKT_BYTES")

    def validate(self, original: np.ndarray, perturbed: np.ndarray) -> bool:
        return len(self.describe_violations(original, perturbed)) == 0

    def describe_violations(
        self, original: np.ndarray, perturbed: np.ndarray
    ) -> List[str]:
        self._check_inputs(original, perturbed)

        f = CICIDSFeatures
        violations = []
        p = perturbed

        # Flows with mean IAT above a stateful firewall's idle timeout would have been terminated before completing 
        # CICIDS2017 duration is stored in microseconds
        iat_mean = p[f.FLOW_IAT_MEAN]
        if iat_mean > self.MAX_IAT_MEAN_US:
            violations.append(
                f"flow_iat_mean={iat_mean:.0f}us ({iat_mean/1e6:.1f}s) > "
                f"max {self.MAX_IAT_MEAN_US/1e6:.0f}s - "
                f"flows with IAT above firewall timeout threshold are anomalous"
            )

        pkt_rate = p[f.FLOW_PKTS_S]
        if pkt_rate > self.MAX_FLOW_PKTS_S:
            violations.append(
                f"flow_pkts_s={pkt_rate:.0f} > {self.MAX_FLOW_PKTS_S:.0f} - "
                f"exceeds physical NIC limit; rate-based detectors would flag this"
            )

        # Average packet size is computed across both directions
        total_pkts = p[f.TOT_FWD_PKTS] + p[f.TOT_BWD_PKTS]
        if total_pkts > 0:
            total_bytes = p[f.TOT_LEN_FWD_PKTS] + p[f.TOT_LEN_BWD_PKTS]
            avg_pkt_size = total_bytes / total_pkts
            if avg_pkt_size < self.MIN_AVG_PKT_BYTES:
                violations.append(
                    f"avg_pkt_size={avg_pkt_size:.1f} bytes < "
                    f"minimum {self.MIN_AVG_PKT_BYTES} bytes - "
                    f"sub-IP-header-sized packets are implausible"
                )
            if avg_pkt_size > self.MAX_AVG_PKT_BYTES:
                violations.append(
                    f"avg_pkt_size={avg_pkt_size:.1f} bytes > "
                    f"MTU {self.MAX_AVG_PKT_BYTES} bytes - "
                    f"flows with jumbo-frame-sized packets are anomalous "
                    f"on standard Ethernet"
                )

        return violations


###
# Composite validator: run multiple validators in sequence
###
class CompositeConstraintValidator(ConstraintValidator):
    """
    Runs a list of validators in sequence. 
    A sample is valid only if it passes all validators. 
    validate() short-circuits on the first failure for efficiency in training loops. 
    describe_violations() runs all validators and returns the union of violations for debugging.

    Example:
        validator = CompositeConstraintValidator([
            TCPConstraintValidator(),
            FunctionalConstraintValidator(attack_class="dos"),
            PlausibilityConstraintValidator(),
        ])
    """

    def __init__(self, validators: List[ConstraintValidator]):
        self.validators = validators

    def validate(self, original: np.ndarray, perturbed: np.ndarray) -> bool:
        return all(v.validate(original, perturbed) for v in self.validators)

    def describe_violations(
        self, original: np.ndarray, perturbed: np.ndarray
    ) -> List[str]:
        all_violations = []
        for v in self.validators:
            all_violations.extend(v.describe_violations(original, perturbed))
        return all_violations