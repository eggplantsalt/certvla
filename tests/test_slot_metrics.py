"""Tests for certvla.slots.metrics -- distance functions and value conversion."""

import numpy as np
import pytest

from certvla.slots.schema import SlotName
from certvla.slots.metrics import (
    slot_distance,
    slot_value_to_tensor,
    tensor_to_slot_value,
    slot_state_to_flat_tensor,
    flat_tensor_dim,
)


class TestSlotDistance:
    # --- Binary distance ---
    def test_binary_same(self):
        assert slot_distance(SlotName.TARGET_CONTACT, 0, 0) == 0.0
        assert slot_distance(SlotName.TARGET_CONTACT, 1, 1) == 0.0

    def test_binary_different(self):
        assert slot_distance(SlotName.TARGET_CONTACT, 0, 1) == 1.0
        assert slot_distance(SlotName.TARGET_CONTACT, 1, 0) == 1.0

    # --- Categorical distance ---
    def test_categorical_same(self):
        assert slot_distance(SlotName.HAND_OCCUPANCY, "empty", "empty") == 0.0
        assert slot_distance(SlotName.HAND_OCCUPANCY, "target", "target") == 0.0

    def test_categorical_different(self):
        assert slot_distance(SlotName.HAND_OCCUPANCY, "empty", "target") == 1.0
        assert slot_distance(SlotName.HAND_OCCUPANCY, "target", "other") == 1.0

    # --- Continuous distance ---
    def test_continuous_same(self):
        assert slot_distance(SlotName.EE_TARGET_PROXIMITY, 0.5, 0.5) == 0.0

    def test_continuous_different(self):
        assert abs(slot_distance(SlotName.EE_TARGET_PROXIMITY, 0.2, 0.8) - 0.6) < 1e-6

    def test_continuous_range(self):
        assert slot_distance(SlotName.EE_TARGET_PROXIMITY, 0.0, 1.0) == 1.0

    # --- Confidence distance ---
    def test_confidence_distance(self):
        assert slot_distance(SlotName.TASK_VISIBLE_CONFIDENCE, 0.3, 0.7) == pytest.approx(0.4)


class TestSlotValueToTensor:
    def test_binary_to_tensor(self):
        t = slot_value_to_tensor(SlotName.TARGET_CONTACT, 1)
        assert t.shape == (1,)
        assert t[0] == 1.0

    def test_binary_zero_to_tensor(self):
        t = slot_value_to_tensor(SlotName.TARGET_CONTACT, 0)
        assert t[0] == 0.0

    def test_categorical_to_tensor(self):
        t = slot_value_to_tensor(SlotName.HAND_OCCUPANCY, "target")
        assert t.shape == (3,)  # ("empty", "target", "other")
        assert t[0] == 0.0  # empty
        assert t[1] == 1.0  # target
        assert t[2] == 0.0  # other

    def test_continuous_to_tensor(self):
        t = slot_value_to_tensor(SlotName.EE_TARGET_PROXIMITY, 0.75)
        assert t.shape == (1,)
        assert t[0] == pytest.approx(0.75)


class TestTensorToSlotValue:
    def test_binary_roundtrip(self):
        for v in [0, 1]:
            t = slot_value_to_tensor(SlotName.TARGET_CONTACT, v)
            back = tensor_to_slot_value(SlotName.TARGET_CONTACT, t)
            assert back == v

    def test_categorical_roundtrip(self):
        for cat in ("empty", "target", "other"):
            t = slot_value_to_tensor(SlotName.HAND_OCCUPANCY, cat)
            back = tensor_to_slot_value(SlotName.HAND_OCCUPANCY, t)
            assert back == cat

    def test_continuous_roundtrip(self):
        for v in [0.0, 0.5, 1.0, 0.123]:
            t = slot_value_to_tensor(SlotName.EE_TARGET_PROXIMITY, v)
            back = tensor_to_slot_value(SlotName.EE_TARGET_PROXIMITY, t)
            assert abs(back - v) < 1e-6

    def test_continuous_clamping(self):
        # Out of range values should be clamped
        t = np.array([1.5], dtype=np.float32)
        back = tensor_to_slot_value(SlotName.EE_TARGET_PROXIMITY, t)
        assert back == 1.0

        t = np.array([-0.3], dtype=np.float32)
        back = tensor_to_slot_value(SlotName.EE_TARGET_PROXIMITY, t)
        assert back == 0.0


class TestFlatTensor:
    def test_flat_dim(self):
        # 10 slots: 4 continuous (1 each) + 2 binary (1 each) + 3 categorical (3+3+3) + 1 confidence (1)
        # = 4 + 2 + 9 + 1 = 16
        assert flat_tensor_dim() == 16

    def test_slot_state_to_flat(self):
        values = {
            SlotName.EE_TARGET_PROXIMITY: 0.5,
            SlotName.HAND_OCCUPANCY: "target",
            SlotName.TARGET_CONTACT: 1,
            SlotName.TARGET_GOAL_PROXIMITY: 0.3,
            SlotName.SUPPORT_RELATION: "none",
            SlotName.CONTAINMENT_RELATION: "in_goal",
            SlotName.ARTICULATION_PROGRESS: 0.0,
            SlotName.ORIENTATION_ALIGNMENT: 0.8,
            SlotName.COMPLETION_LATCH: 0,
            SlotName.TASK_VISIBLE_CONFIDENCE: 1.0,
        }
        flat = slot_state_to_flat_tensor(values)
        assert flat.shape == (16,)
        assert flat.dtype == np.float32
