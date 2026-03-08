"""
Adversarial mutation functions for network flows.

Rules:
- Input: 1D numpy array (single sample)
- Output: New mutated array (never modify in-place)
"""

import numpy as np
from enum import Enum
from typing import Dict, Any, Optional, Callable

from src.constraints import (
    CICIDSFeatures as Features,
    TCPConstraintValidator,
    FunctionalConstraintValidator,
    PlausibilityConstraintValidator,
    CompositeConstraintValidator,
)

# Constants
SECONDS_TO_MICROSECONDS = 1_000_000.0
MILLISECONDS_TO_MICROSECONDS = 1000.0
IP_HEADER_LENGTH_BYTES = 20
MAX_TCP_HEADER_BYTES = 60
DEFAULT_TCP_OPTION_BYTES = 12

class Direction(str, Enum):
    FORWARD = "fwd"
    BACKWARD = "bwd"
    BOTH = "both"


# Registry setup
REGISTRY: Dict[str, Dict[str, Any]] = {}

def register(description: str, validator: Callable, parameters: dict):
    """Decorator to auto-register new attacks."""
    def wrapper(func):
        REGISTRY[func.__name__] = {
            "run": func,
            "validator": validator,
            "description": description,
            "parameters": parameters,
        }
        return func
    return wrapper


# --- Helpers ---

def _copy(sample: np.ndarray) -> np.ndarray:
    return sample.astype(np.float64).copy()

def _safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    return numerator / denominator if denominator != 0 else fallback

def _recalculate_rates(flow: np.ndarray) -> np.ndarray:
    """Recalculate rates (bytes per second, packets per second) after modifying totals."""
    duration_seconds = flow[Features.FLOW_DURATION] / SECONDS_TO_MICROSECONDS
    total_bytes = flow[Features.TOT_LEN_FWD_PKTS] + flow[Features.TOT_LEN_BWD_PKTS]
    total_packets = flow[Features.TOT_FWD_PKTS] + flow[Features.TOT_BWD_PKTS]

    flow[Features.FLOW_BYTS_S] = _safe_divide(total_bytes, duration_seconds)
    flow[Features.FLOW_PKTS_S] = _safe_divide(total_packets, duration_seconds)
    flow[Features.FWD_PKTS_S] = _safe_divide(flow[Features.TOT_FWD_PKTS], duration_seconds)
    flow[Features.BWD_PKTS_S] = _safe_divide(flow[Features.TOT_BWD_PKTS], duration_seconds)
    return flow

def _update_packet_sizes(flow: np.ndarray, minimum_value: Optional[float] = None, maximum_value: Optional[float] = None):
    """Update average packet sizes after adding or splitting packets."""
    total_packets = flow[Features.TOT_FWD_PKTS] + flow[Features.TOT_BWD_PKTS]
    total_bytes = flow[Features.TOT_LEN_FWD_PKTS] + flow[Features.TOT_LEN_BWD_PKTS]
    
    flow[Features.PKT_LEN_MEAN] = _safe_divide(total_bytes, total_packets)
    flow[Features.PKT_SIZE_AVG] = flow[Features.PKT_LEN_MEAN]
    
    if minimum_value is not None:
        flow[Features.PKT_LEN_MIN] = min(flow[Features.PKT_LEN_MIN], minimum_value)
    if maximum_value is not None:
        flow[Features.PKT_LEN_MAX] = max(flow[Features.PKT_LEN_MAX], maximum_value)

# Lookup maps for direction features
_FORWARD_INDICES = (
    Features.TOT_FWD_PKTS, Features.TOT_LEN_FWD_PKTS, Features.FWD_HEADER_LEN,
    Features.SUBFLOW_FWD_PKTS, Features.SUBFLOW_FWD_BYTS,
    Features.FWD_PKT_LEN_MEAN, Features.FWD_PKT_LEN_MIN, Features.FWD_SEG_SIZE_AVG,
)
_BACKWARD_INDICES = (
    Features.TOT_BWD_PKTS, Features.TOT_LEN_BWD_PKTS, Features.BWD_HEADER_LEN,
    Features.SUBFLOW_BWD_PKTS, Features.SUBFLOW_BWD_BYTS,
    Features.BWD_PKT_LEN_MEAN, Features.BWD_PKT_LEN_MIN, Features.BWD_SEG_SIZE_AVG,
)

def _add_packets(flow: np.ndarray, direction: Direction, number_of_packets: int, padding_bytes: int):
    """Add dummy packets and update the counters."""
    indices = _FORWARD_INDICES if direction == Direction.FORWARD else _BACKWARD_INDICES
    
    total_packets_index, total_bytes_index, header_length_index, subflow_packets_index, subflow_bytes_index, mean_index, minimum_index, segment_index = indices

    flow[total_packets_index] += number_of_packets
    flow[total_bytes_index] += number_of_packets * padding_bytes
    flow[header_length_index] += number_of_packets * IP_HEADER_LENGTH_BYTES
    flow[subflow_packets_index] += number_of_packets
    flow[subflow_bytes_index] += number_of_packets * padding_bytes

    flow[mean_index] = _safe_divide(flow[total_bytes_index], flow[total_packets_index])
    flow[minimum_index] = min(flow[minimum_index], padding_bytes)
    flow[segment_index] = flow[mean_index]

def _set_timing(flow: np.ndarray, mean_microseconds: float, standard_deviation_microseconds: float, duration_microseconds: float):
    """Overwrite all inter-arrival time statistics."""
    flow[Features.FLOW_IAT_MEAN] = mean_microseconds
    flow[Features.FLOW_IAT_STD] = standard_deviation_microseconds
    flow[Features.FLOW_IAT_MAX] = mean_microseconds + 2 * standard_deviation_microseconds
    flow[Features.FLOW_IAT_MIN] = max(0.0, mean_microseconds - 2 * standard_deviation_microseconds)
    flow[Features.FLOW_DURATION] = duration_microseconds

    flow[Features.FWD_IAT_MEAN] = mean_microseconds
    flow[Features.FWD_IAT_STD] = standard_deviation_microseconds
    flow[Features.FWD_IAT_MAX] = flow[Features.FLOW_IAT_MAX]
    flow[Features.FWD_IAT_MIN] = flow[Features.FLOW_IAT_MIN]
    flow[Features.FWD_IAT_TOT] = mean_microseconds * max(flow[Features.TOT_FWD_PKTS] - 1, 1)

    flow[Features.BWD_IAT_MEAN] = mean_microseconds
    flow[Features.BWD_IAT_STD] = standard_deviation_microseconds
    flow[Features.BWD_IAT_MAX] = flow[Features.FLOW_IAT_MAX]
    flow[Features.BWD_IAT_MIN] = flow[Features.FLOW_IAT_MIN]
    flow[Features.BWD_IAT_TOT] = mean_microseconds * max(flow[Features.TOT_BWD_PKTS] - 1, 1)

def _tcp_validator():
    return CompositeConstraintValidator([
        TCPConstraintValidator(),
        PlausibilityConstraintValidator(),
    ])

def _timing_validator():
    return CompositeConstraintValidator([
        TCPConstraintValidator(),
        FunctionalConstraintValidator(attack_class="c2"),
        PlausibilityConstraintValidator(),
    ])


# --- Mutations ---

@register(
    description="Adds dummy packets to inflate packet count and lower average size.",
    validator=_tcp_validator,
    parameters={"number_of_packets": int, "direction": Direction, "padding_bytes": int}
)
def add_padding(sample: np.ndarray, number_of_packets: int, direction: Direction = Direction.FORWARD, padding_bytes: int = 40) -> np.ndarray:
    if number_of_packets <= 0:
        raise ValueError("number_of_packets must be > 0")
    if padding_bytes < IP_HEADER_LENGTH_BYTES:
        raise ValueError(f"padding_bytes must be >= {IP_HEADER_LENGTH_BYTES}")

    flow = _copy(sample)
    _add_packets(flow, direction, number_of_packets, padding_bytes)
    _update_packet_sizes(flow, minimum_value=padding_bytes)
    return _recalculate_rates(flow)


@register(
    description="Increases time between packets to make the flow look slower.",
    validator=_tcp_validator,
    parameters={"delta_milliseconds": float, "direction": Direction}
)
def delay_packets(sample: np.ndarray, delta_milliseconds: float, direction: Direction = Direction.BOTH) -> np.ndarray:
    flow = _copy(sample)
    delta_microseconds = delta_milliseconds * MILLISECONDS_TO_MICROSECONDS

    def _shift(mean_index, standard_deviation_index, maximum_index, minimum_index, total_index, packet_count):
        original_mean = flow[mean_index]
        new_mean = max(0.0, original_mean + delta_microseconds)
        if original_mean > 0:
            scale = new_mean / original_mean
            flow[standard_deviation_index] *= scale
            flow[maximum_index] = max(new_mean, flow[maximum_index] * scale)
            flow[minimum_index] = max(0.0, flow[minimum_index] * scale)
            flow[total_index] = new_mean * max(packet_count - 1, 1)
        flow[mean_index] = new_mean

    forward_packets = flow[Features.TOT_FWD_PKTS]
    backward_packets = flow[Features.TOT_BWD_PKTS]

    if direction in (Direction.FORWARD, Direction.BOTH) and forward_packets > 1:
        _shift(Features.FWD_IAT_MEAN, Features.FWD_IAT_STD, Features.FWD_IAT_MAX, Features.FWD_IAT_MIN, Features.FWD_IAT_TOT, forward_packets)

    if direction in (Direction.BACKWARD, Direction.BOTH) and backward_packets > 1:
        _shift(Features.BWD_IAT_MEAN, Features.BWD_IAT_STD, Features.BWD_IAT_MAX, Features.BWD_IAT_MIN, Features.BWD_IAT_TOT, backward_packets)

    total_packets = forward_packets + backward_packets
    if total_packets > 1:
        flow[Features.FLOW_IAT_MEAN] = (flow[Features.FWD_IAT_MEAN] * forward_packets + flow[Features.BWD_IAT_MEAN] * backward_packets) / total_packets
        flow[Features.FLOW_IAT_MAX] = max(flow[Features.FWD_IAT_MAX], flow[Features.BWD_IAT_MAX])
        flow[Features.FLOW_IAT_MIN] = min(flow[Features.FWD_IAT_MIN], flow[Features.BWD_IAT_MIN])

    if flow[Features.FLOW_IAT_MEAN] > 0:
        flow[Features.FLOW_DURATION] = flow[Features.FLOW_IAT_MEAN] * max(total_packets - 1, 1)

    return _recalculate_rates(flow)


@register(
    description="Chops forward packets into smaller pieces to bypass size limits.",
    validator=_tcp_validator,
    parameters={"number_of_fragments": int}
)
def split_packets(sample: np.ndarray, number_of_fragments: int) -> np.ndarray:
    if number_of_fragments < 2:
        raise ValueError("number_of_fragments must be >= 2")

    flow = _copy(sample)
    new_forward_packets = flow[Features.TOT_FWD_PKTS] * number_of_fragments
    flow[Features.TOT_FWD_PKTS] = new_forward_packets
    flow[Features.SUBFLOW_FWD_PKTS] = new_forward_packets

    new_average = _safe_divide(flow[Features.TOT_LEN_FWD_PKTS], new_forward_packets)
    flow[Features.FWD_PKT_LEN_MEAN] = flow[Features.FWD_PKT_LEN_MAX] = flow[Features.FWD_PKT_LEN_MIN] = new_average
    flow[Features.FWD_PKT_LEN_STD] = 0.0
    flow[Features.FWD_SEG_SIZE_AVG] = new_average
    flow[Features.FWD_HEADER_LEN] *= number_of_fragments

    _update_packet_sizes(flow, minimum_value=new_average, maximum_value=new_average)
    return _recalculate_rates(flow)


@register(
    description="Protocol exploitation: split one forward payload packet into N fragments.",
    validator=_tcp_validator,
    parameters={"n_fragments": int}
)
def fragment_payload(sample: np.ndarray, n_fragments: int) -> np.ndarray:
    """
    Increase packet_count and reduce average packet size while preserving total bytes.
    """
    if n_fragments < 2:
        raise ValueError("n_fragments must be >= 2")

    flow = _copy(sample)
    fwd_pkts_before = flow[Features.TOT_FWD_PKTS]
    if fwd_pkts_before < 1:
        raise ValueError("Cannot fragment payload when TOT_FWD_PKTS < 1")

    additional_packets = float(n_fragments - 1)
    flow[Features.TOT_FWD_PKTS] = fwd_pkts_before + additional_packets
    flow[Features.SUBFLOW_FWD_PKTS] = max(flow[Features.SUBFLOW_FWD_PKTS], fwd_pkts_before) + additional_packets

    # Fragmentation keeps payload bytes unchanged but adds one base header per added packet.
    flow[Features.FWD_HEADER_LEN] += additional_packets * IP_HEADER_LENGTH_BYTES

    previous_average = _safe_divide(flow[Features.TOT_LEN_FWD_PKTS], fwd_pkts_before)
    fragment_size = _safe_divide(previous_average, float(n_fragments))

    flow[Features.FWD_PKT_LEN_MEAN] = _safe_divide(flow[Features.TOT_LEN_FWD_PKTS], flow[Features.TOT_FWD_PKTS])
    flow[Features.FWD_SEG_SIZE_AVG] = flow[Features.FWD_PKT_LEN_MEAN]
    flow[Features.FWD_PKT_LEN_MIN] = min(flow[Features.FWD_PKT_LEN_MIN], fragment_size)
    flow[Features.PKT_LEN_MIN] = min(flow[Features.PKT_LEN_MIN], fragment_size)

    if fwd_pkts_before <= 1:
        flow[Features.FWD_PKT_LEN_MAX] = fragment_size
        flow[Features.FWD_PKT_LEN_STD] = 0.0
        flow[Features.PKT_LEN_MAX] = fragment_size
        flow[Features.PKT_LEN_STD] = 0.0

    flow[Features.PKT_LEN_VAR] = flow[Features.PKT_LEN_STD] ** 2
    _update_packet_sizes(flow, minimum_value=fragment_size)
    return _recalculate_rates(flow)


@register(
    description="Protocol exploitation: add benign TCP options to inflate forward header length.",
    validator=_tcp_validator,
    parameters={"option_bytes_per_packet": int}
)
def add_tcp_options(sample: np.ndarray, option_bytes_per_packet: int = DEFAULT_TCP_OPTION_BYTES) -> np.ndarray:
    """
    Inflate fwd_header_length (e.g., NOP/window scaling options) without changing payload bytes.
    """
    if option_bytes_per_packet < 0:
        raise ValueError("option_bytes_per_packet must be >= 0")

    flow = _copy(sample)
    fwd_packets = flow[Features.TOT_FWD_PKTS]
    if fwd_packets <= 0:
        raise ValueError("Cannot add TCP options when TOT_FWD_PKTS <= 0")

    avg_header = _safe_divide(flow[Features.FWD_HEADER_LEN], fwd_packets)
    new_avg_header = min(MAX_TCP_HEADER_BYTES, avg_header + float(option_bytes_per_packet))
    new_avg_header = max(new_avg_header, float(IP_HEADER_LENGTH_BYTES))
    flow[Features.FWD_HEADER_LEN] = new_avg_header * fwd_packets

    return _recalculate_rates(flow)


@register(
    description="Protocol exploitation: shift ACK timing toward a target mean IAT profile.",
    validator=_timing_validator,
    parameters={"target_iat_ms": float}
)
def shift_ack_timing(sample: np.ndarray, target_iat_ms: float) -> np.ndarray:
    """
    Shift inter-arrival timing so FLOW_IAT_MEAN moves toward target_iat_ms.
    """
    if target_iat_ms <= 0:
        raise ValueError("target_iat_ms must be > 0")

    current_iat_ms = _safe_divide(float(sample[Features.FLOW_IAT_MEAN]), MILLISECONDS_TO_MICROSECONDS)
    delta_ms = target_iat_ms - current_iat_ms

    # Reuse existing timing mutation path for consistency.
    flow = delay_packets(sample, delta_milliseconds=delta_ms, direction=Direction.BOTH)

    # Keep timing statistics centered at target profile when possible.
    total_packets = flow[Features.TOT_FWD_PKTS] + flow[Features.TOT_BWD_PKTS]
    target_iat_us = target_iat_ms * MILLISECONDS_TO_MICROSECONDS
    old_mean = max(float(sample[Features.FLOW_IAT_MEAN]), 1.0)
    old_std = max(float(sample[Features.FLOW_IAT_STD]), 0.0)
    std_ratio = np.clip(old_std / old_mean, 0.05, 0.50)
    target_std_us = target_iat_us * std_ratio
    target_duration_us = target_iat_us * max(total_packets - 1, 1)
    _set_timing(flow, target_iat_us, target_std_us, target_duration_us)

    return _recalculate_rates(flow)


@register(
    description="Mixes the attack with normal traffic data to hide it.",
    validator=lambda: CompositeConstraintValidator([PlausibilityConstraintValidator()]),
    parameters={"k_samples": int, "benign_pool": "np.ndarray"}
)
def blend_with_benign(sample: np.ndarray, benign_pool: np.ndarray, k_samples: int, random_number_generator: Optional[np.random.Generator] = None) -> np.ndarray:
    if k_samples <= 0:
        raise ValueError("k_samples must be > 0")
    
    random_number_generator = random_number_generator or np.random.default_rng()
    selected_indices = random_number_generator.choice(len(benign_pool), size=min(k_samples, len(benign_pool)), replace=False)
    benign_mean = benign_pool[selected_indices].mean(axis=0)

    weight_attack = 1.0 / (k_samples + 1)
    weight_benign = k_samples / (k_samples + 1)

    return (weight_attack * sample.astype(np.float64)) + (weight_benign * benign_mean.astype(np.float64))


@register(
    description="Forces the attack to match the timing of normal traffic.",
    validator=_timing_validator,
    parameters={"target_mean_microseconds": float, "target_standard_deviation_microseconds": float, "maximum_ratio": float}
)
def mimic_timing(sample: np.ndarray, target_mean_microseconds: float, target_standard_deviation_microseconds: float, maximum_ratio: float = 2.0) -> np.ndarray:
    flow = _copy(sample)
    total_packets = flow[Features.TOT_FWD_PKTS] + flow[Features.TOT_BWD_PKTS]
    desired_duration_microseconds = target_mean_microseconds * max(total_packets - 1, 1)
    maximum_allowed_duration = flow[Features.FLOW_DURATION] * maximum_ratio

    if desired_duration_microseconds > maximum_allowed_duration:
        scale = maximum_allowed_duration / max(desired_duration_microseconds, 1)
        _set_timing(flow, target_mean_microseconds * scale, target_standard_deviation_microseconds * scale, maximum_allowed_duration)
    else:
        _set_timing(flow, target_mean_microseconds, target_standard_deviation_microseconds, desired_duration_microseconds)

    return _recalculate_rates(flow)


def list_mutations():
    """Prints available attacks."""
    print(f"{'Mutation':<20} {'Description'}")
    print("-" * 70)
    for name, data in REGISTRY.items():
        print(f"{name:<20} {data['description']}")
