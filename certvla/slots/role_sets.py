"""
Slot family (role set) definitions.

J_E: enabling / transit slots
J_R: result / latch slots
J_C: confidence / observability slots
J_CERT = J_E | J_R (slots that participate in certificates)
"""

from typing import FrozenSet

from certvla.slots.schema import SlotFamily, SlotName, SLOT_REGISTRY


# === Family sets ===

J_E: FrozenSet[SlotName] = frozenset(
    name for name, meta in SLOT_REGISTRY.items() if meta.family == SlotFamily.ENABLING
)

J_R: FrozenSet[SlotName] = frozenset(
    name for name, meta in SLOT_REGISTRY.items() if meta.family == SlotFamily.RESULT
)

J_C: FrozenSet[SlotName] = frozenset(
    name for name, meta in SLOT_REGISTRY.items() if meta.family == SlotFamily.CONFIDENCE
)

# Certificate-participating slots
J_CERT: FrozenSet[SlotName] = J_E | J_R


def get_family(slot_name: SlotName) -> SlotFamily:
    """Return the family of a slot."""
    return SLOT_REGISTRY[slot_name].family


# Sanity checks
assert SlotName.EE_TARGET_PROXIMITY in J_E
assert SlotName.HAND_OCCUPANCY in J_E
assert SlotName.TARGET_CONTACT in J_E
assert SlotName.ARTICULATION_PROGRESS in J_E
assert SlotName.ORIENTATION_ALIGNMENT in J_E
assert len(J_E) == 5, f"Expected 5 enabling slots, got {len(J_E)}"

assert SlotName.TARGET_GOAL_PROXIMITY in J_R
assert SlotName.SUPPORT_RELATION in J_R
assert SlotName.CONTAINMENT_RELATION in J_R
assert SlotName.COMPLETION_LATCH in J_R
assert len(J_R) == 4, f"Expected 4 result slots, got {len(J_R)}"

assert SlotName.TASK_VISIBLE_CONFIDENCE in J_C
assert len(J_C) == 1, f"Expected 1 confidence slot, got {len(J_C)}"

assert len(J_CERT) == 9, f"Expected 9 cert slots (5+4), got {len(J_CERT)}"
