"""
Behavioral Mimicry Attacks

These functions help camouflage malicious network flows by making their timing (IAT) 
and packet sizes match the statistical profiles of normal (benign) traffic.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict

from src.constraints import CICIDSFeatures as Features
from src.dotenv_utils import get_env_float, get_env_int

MAX_TRANSMISSION_UNIT_BYTES = get_env_int("MAX_TRANSMISSION_UNIT_BYTES")
MINIMUM_HEADER_BYTES = get_env_int("MINIMUM_HEADER_BYTES")
SECONDS_TO_MICROSECONDS = get_env_float("SECONDS_TO_MICROSECONDS")


# Helper function
def _copy(sample: np.ndarray) -> np.ndarray:
    return sample.astype(np.float64).copy()

def load_benign_profile(app_name: str) -> Dict:
    json_path = Path(__file__).resolve().parents[2] / "data" / "benign_profiles.json"
    
    with open(json_path, "r") as file:
        data = json.load(file)
        
    if app_name not in data["applications"]:
        raise ValueError(f"Application '{app_name}' not found in benign profiles.")
        
    return data["applications"][app_name]


def mimic_timing(
    malicious_sample: np.ndarray,
    target_profile: Dict,
    max_delay_ms: float = 500.0,
    maximum_duration_ratio: float = 2.0,
) -> np.ndarray:
    """
    IAT-based mimicry: rescale flow timing to minimize Wasserstein distance to target profile.
    """
    flow = _copy(malicious_sample)

    original_duration = flow[Features.FLOW_DURATION]
    total_packets = flow[Features.TOT_FWD_PKTS] + flow[Features.TOT_BWD_PKTS]
    n_gaps = max(total_packets - 1, 1)

    target_mean_us = target_profile["flow_iat_mean"]["mean_us"]
    target_std_us = target_profile["flow_iat_std"]["mean_us"]

    desired_duration = target_mean_us * n_gaps
    max_duration_us = original_duration * maximum_duration_ratio
    max_iat_us = max_delay_ms * 1000.0

    if desired_duration > max_duration_us:
        scale = max_duration_us / max(desired_duration, 1)
        final_mean = target_mean_us * scale
        final_std = target_std_us * scale
    else:
        final_mean = target_mean_us
        final_std = target_std_us

    # Cap IAT so no delay exceeds max_delay_ms
    proposed_max_iat = final_mean + (2.0 * final_std)
    if proposed_max_iat > max_iat_us and final_std > 0:
        final_std = min(final_std, (max_iat_us - final_mean) / 2.0)
        final_std = max(final_std, 0.0)
        
    # If the mean itself is bigger than the max delay, clamp it.
    if final_mean > max_iat_us:
        final_mean = max_iat_us
        final_std = 0.0

    # Recalculate duration; enforce 2x cap (mean clamp can make duration exceed it)
    final_duration = final_mean * n_gaps
    if final_duration > max_duration_us:
        final_duration = max_duration_us
        final_mean = max_duration_us / n_gaps
        final_std = 0.0

    # Overwrite Overall Flow Timing
    flow[Features.FLOW_IAT_MEAN] = final_mean
    flow[Features.FLOW_IAT_STD] = final_std
    flow[Features.FLOW_IAT_MAX] = min(final_mean + (2.0 * final_std), max_iat_us)
    flow[Features.FLOW_IAT_MIN] = max(0.0, final_mean - (2.0 * final_std))
    flow[Features.FLOW_DURATION] = final_duration

    # Overwrite Directional Timing to maintain dataset consistency
    flow[Features.FWD_IAT_MEAN] = final_mean
    flow[Features.FWD_IAT_STD] = final_std
    flow[Features.FWD_IAT_MAX] = flow[Features.FLOW_IAT_MAX]
    flow[Features.FWD_IAT_MIN] = flow[Features.FLOW_IAT_MIN]
    flow[Features.FWD_IAT_TOT] = final_mean * max(flow[Features.TOT_FWD_PKTS] - 1, 1)

    flow[Features.BWD_IAT_MEAN] = final_mean
    flow[Features.BWD_IAT_STD] = final_std
    flow[Features.BWD_IAT_MAX] = flow[Features.FLOW_IAT_MAX]
    flow[Features.BWD_IAT_MIN] = flow[Features.FLOW_IAT_MIN]
    flow[Features.BWD_IAT_TOT] = final_mean * max(flow[Features.TOT_BWD_PKTS] - 1, 1)

    # Fix Rates
    duration_seconds = final_duration / SECONDS_TO_MICROSECONDS
    if duration_seconds > 0:
        flow[Features.FLOW_BYTS_S] = (flow[Features.TOT_LEN_FWD_PKTS] + flow[Features.TOT_LEN_BWD_PKTS]) / duration_seconds
        flow[Features.FLOW_PKTS_S] = total_packets / duration_seconds

    return flow


def mimic_packet_size(malicious_sample: np.ndarray, target_profile: Dict) -> np.ndarray:
    """
    Packet-size distribution mimicry: shift flow packet-length distribution toward benign profile.
    """
    flow = _copy(malicious_sample)

    target_mean_bytes = float(target_profile["pkt_len_mean"]["mean_us"])
    target_mean_bytes = min(target_mean_bytes, MAX_TRANSMISSION_UNIT_BYTES)

    fwd_pkts = flow[Features.TOT_FWD_PKTS]
    bwd_pkts = flow[Features.TOT_BWD_PKTS]

    # Forward direction
    if fwd_pkts > 0:
        current_fwd_bytes = flow[Features.TOT_LEN_FWD_PKTS]
        current_fwd_mean = current_fwd_bytes / fwd_pkts
        if current_fwd_mean < target_mean_bytes:
            extra_fwd = (target_mean_bytes * fwd_pkts) - current_fwd_bytes
            flow[Features.TOT_LEN_FWD_PKTS] += extra_fwd
            flow[Features.SUBFLOW_FWD_BYTS] += extra_fwd
            flow[Features.FWD_PKT_LEN_MAX] = max(flow[Features.FWD_PKT_LEN_MAX], target_mean_bytes)

    # Backward direction
    if bwd_pkts > 0:
        current_bwd_bytes = flow[Features.TOT_LEN_BWD_PKTS]
        current_bwd_mean = current_bwd_bytes / bwd_pkts
        if current_bwd_mean < target_mean_bytes:
            extra_bwd = (target_mean_bytes * bwd_pkts) - current_bwd_bytes
            flow[Features.TOT_LEN_BWD_PKTS] += extra_bwd
            flow[Features.SUBFLOW_BWD_BYTS] += extra_bwd
            flow[Features.BWD_PKT_LEN_MAX] = max(flow[Features.BWD_PKT_LEN_MAX], target_mean_bytes)

    # Recompute flow-level size features
    total_pkts = fwd_pkts + bwd_pkts
    if total_pkts > 0:
        total_bytes = flow[Features.TOT_LEN_FWD_PKTS] + flow[Features.TOT_LEN_BWD_PKTS]
        new_mean = total_bytes / total_pkts
        flow[Features.PKT_LEN_MEAN] = new_mean
        flow[Features.PKT_SIZE_AVG] = new_mean
        flow[Features.PKT_LEN_MAX] = max(flow[Features.PKT_LEN_MAX], target_mean_bytes)
        
    if fwd_pkts > 0:
        flow[Features.FWD_PKT_LEN_MEAN] = flow[Features.TOT_LEN_FWD_PKTS] / fwd_pkts
    if bwd_pkts > 0:
        flow[Features.BWD_PKT_LEN_MEAN] = flow[Features.TOT_LEN_BWD_PKTS] / bwd_pkts

    return flow