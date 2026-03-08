"""
Per-slot distance functions and value conversion utilities.

Distance functions d_j(a, b) per context doc section 8.2:
- Binary: |a - b| (0 or 1)
- Categorical: 1 - (a == b) (Hamming)
- Continuous: |a - b| (L1 in [0,1])
"""

from typing import Union

import numpy as np

from certvla.slots.schema import SlotDomain, SlotMeta, SlotName, SLOT_REGISTRY, get_slot_meta


def slot_distance(slot: SlotName, a: Union[int, float, str], b: Union[int, float, str]) -> float:
    """Compute per-slot distance d_j(a, b).

    Args:
        slot: The slot name.
        a: First value.
        b: Second value.

    Returns:
        Distance in [0, 1].
    """
    meta = get_slot_meta(slot)
    return _distance_by_domain(meta, a, b)


def _distance_by_domain(meta: SlotMeta, a: Union[int, float, str], b: Union[int, float, str]) -> float:
    if meta.domain == SlotDomain.BINARY:
        return abs(float(a) - float(b))
    elif meta.domain == SlotDomain.CATEGORICAL:
        return 0.0 if a == b else 1.0
    elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        return abs(float(a) - float(b))
    else:
        raise ValueError(f"Unknown domain {meta.domain}")


def slot_value_to_tensor(slot: SlotName, value: Union[int, float, str]) -> np.ndarray:
    """Convert a slot value to its tensor representation.

    - Binary: [float] (0.0 or 1.0)
    - Categorical: one-hot vector of length num_categories
    - Continuous/Confidence: [float]
    """
    meta = get_slot_meta(slot)

    if meta.domain == SlotDomain.BINARY:
        return np.array([float(value)], dtype=np.float32)
    elif meta.domain == SlotDomain.CATEGORICAL:
        idx = meta.categories.index(value)
        one_hot = np.zeros(len(meta.categories), dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot
    elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        return np.array([float(value)], dtype=np.float32)
    else:
        raise ValueError(f"Unknown domain {meta.domain}")


def tensor_to_slot_value(slot: SlotName, tensor: np.ndarray) -> Union[int, float, str]:
    """Convert tensor representation back to slot value.

    - Binary: round to 0 or 1
    - Categorical: argmax -> category string
    - Continuous/Confidence: clamp to valid_range
    """
    meta = get_slot_meta(slot)

    if meta.domain == SlotDomain.BINARY:
        return int(round(float(tensor[0])))
    elif meta.domain == SlotDomain.CATEGORICAL:
        idx = int(np.argmax(tensor))
        return meta.categories[idx]
    elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        v = float(tensor[0])
        lo, hi = meta.valid_range
        return max(lo, min(hi, v))
    else:
        raise ValueError(f"Unknown domain {meta.domain}")


def slot_state_to_flat_tensor(values: dict) -> np.ndarray:
    """Convert a dict of {SlotName: value} to a flat numpy array.

    Ordering follows SlotName enum order. Binary/continuous slots contribute 1 dim;
    categorical slots contribute num_categories dims (one-hot).
    """
    parts = []
    for slot_name in SlotName:
        if slot_name in values:
            parts.append(slot_value_to_tensor(slot_name, values[slot_name]))
        else:
            meta = get_slot_meta(slot_name)
            # Missing slot: zero vector of appropriate size
            if meta.domain == SlotDomain.CATEGORICAL:
                parts.append(np.zeros(len(meta.categories), dtype=np.float32))
            else:
                parts.append(np.zeros(1, dtype=np.float32))
    return np.concatenate(parts)


def flat_tensor_dim() -> int:
    """Return the total dimensionality of the flat slot state tensor."""
    dim = 0
    for slot_name in SlotName:
        meta = get_slot_meta(slot_name)
        if meta.domain == SlotDomain.CATEGORICAL:
            dim += len(meta.categories)
        else:
            dim += 1
    return dim
