"""Shared fixtures for CertVLA tests."""

import pytest
from certvla.slots.schema import SlotName, SLOT_REGISTRY
from certvla.data.chunk_sample import SlotState


@pytest.fixture
def slot_registry():
    return SLOT_REGISTRY


@pytest.fixture
def make_slot_state():
    """Factory fixture for creating SlotState with defaults."""

    def _make(
        values=None,
        validity_mask=None,
        confidence=None,
    ):
        if values is None:
            values = {
                SlotName.EE_TARGET_PROXIMITY: 0.5,
                SlotName.HAND_OCCUPANCY: "empty",
                SlotName.TARGET_CONTACT: 0,
                SlotName.TARGET_GOAL_PROXIMITY: 0.5,
                SlotName.SUPPORT_RELATION: "none",
                SlotName.CONTAINMENT_RELATION: "none",
                SlotName.ARTICULATION_PROGRESS: 0.0,
                SlotName.ORIENTATION_ALIGNMENT: 0.5,
                SlotName.COMPLETION_LATCH: 0,
                SlotName.TASK_VISIBLE_CONFIDENCE: 1.0,
            }
        if validity_mask is None:
            validity_mask = {s: True for s in SlotName}
        if confidence is None:
            confidence = {s: 1.0 for s in SlotName}
        return SlotState(values=values, validity_mask=validity_mask, confidence=confidence)

    return _make
