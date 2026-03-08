"""Tests for certvla.slots.schema -- slot vocabulary definition."""

import pytest
from certvla.slots.schema import (
    SlotName, SlotDomain, SlotFamily, SlotMeta,
    SLOT_REGISTRY, SLOT_VOCAB_SIZE, get_slot_meta,
)


class TestSlotRegistry:
    def test_vocab_size(self):
        assert SLOT_VOCAB_SIZE == 10

    def test_all_slots_in_registry(self):
        for name in SlotName:
            assert name in SLOT_REGISTRY, f"Slot {name} missing from registry"

    def test_registry_size_matches_enum(self):
        assert len(SLOT_REGISTRY) == len(SlotName)

    def test_get_slot_meta(self):
        meta = get_slot_meta(SlotName.EE_TARGET_PROXIMITY)
        assert meta.name == SlotName.EE_TARGET_PROXIMITY
        assert meta.domain == SlotDomain.CONTINUOUS
        assert meta.family == SlotFamily.ENABLING


class TestSlotDomains:
    """Verify each slot has the correct domain per the v1 spec."""

    def test_binary_slots(self):
        binary_slots = [SlotName.TARGET_CONTACT, SlotName.COMPLETION_LATCH]
        for s in binary_slots:
            assert SLOT_REGISTRY[s].domain == SlotDomain.BINARY

    def test_categorical_slots(self):
        cat_slots = [SlotName.HAND_OCCUPANCY, SlotName.SUPPORT_RELATION, SlotName.CONTAINMENT_RELATION]
        for s in cat_slots:
            assert SLOT_REGISTRY[s].domain == SlotDomain.CATEGORICAL

    def test_continuous_slots(self):
        cont_slots = [
            SlotName.EE_TARGET_PROXIMITY,
            SlotName.TARGET_GOAL_PROXIMITY,
            SlotName.ARTICULATION_PROGRESS,
            SlotName.ORIENTATION_ALIGNMENT,
        ]
        for s in cont_slots:
            assert SLOT_REGISTRY[s].domain == SlotDomain.CONTINUOUS

    def test_confidence_slot(self):
        assert SLOT_REGISTRY[SlotName.TASK_VISIBLE_CONFIDENCE].domain == SlotDomain.CONFIDENCE


class TestSlotFamilies:
    def test_enabling_slots(self):
        enabling = [
            SlotName.EE_TARGET_PROXIMITY,
            SlotName.HAND_OCCUPANCY,
            SlotName.TARGET_CONTACT,
            SlotName.ARTICULATION_PROGRESS,
            SlotName.ORIENTATION_ALIGNMENT,
        ]
        for s in enabling:
            assert SLOT_REGISTRY[s].family == SlotFamily.ENABLING

    def test_result_slots(self):
        result = [
            SlotName.TARGET_GOAL_PROXIMITY,
            SlotName.SUPPORT_RELATION,
            SlotName.CONTAINMENT_RELATION,
            SlotName.COMPLETION_LATCH,
        ]
        for s in result:
            assert SLOT_REGISTRY[s].family == SlotFamily.RESULT

    def test_confidence_slots(self):
        assert SLOT_REGISTRY[SlotName.TASK_VISIBLE_CONFIDENCE].family == SlotFamily.CONFIDENCE


class TestSlotCategories:
    def test_hand_occupancy_categories(self):
        meta = SLOT_REGISTRY[SlotName.HAND_OCCUPANCY]
        assert meta.categories == ("empty", "target", "other")

    def test_support_relation_categories(self):
        meta = SLOT_REGISTRY[SlotName.SUPPORT_RELATION]
        assert meta.categories == ("none", "on_goal", "on_other")

    def test_containment_relation_categories(self):
        meta = SLOT_REGISTRY[SlotName.CONTAINMENT_RELATION]
        assert meta.categories == ("none", "in_goal", "in_other")

    def test_non_categorical_has_no_categories(self):
        meta = SLOT_REGISTRY[SlotName.EE_TARGET_PROXIMITY]
        assert meta.categories is None


class TestValueValidation:
    def test_binary_valid(self):
        meta = SLOT_REGISTRY[SlotName.TARGET_CONTACT]
        assert meta.validate_value(0)
        assert meta.validate_value(1)
        assert meta.validate_value(0.0)
        assert meta.validate_value(1.0)

    def test_binary_invalid(self):
        meta = SLOT_REGISTRY[SlotName.TARGET_CONTACT]
        assert not meta.validate_value(0.5)
        assert not meta.validate_value(2)
        assert not meta.validate_value("yes")

    def test_categorical_valid(self):
        meta = SLOT_REGISTRY[SlotName.HAND_OCCUPANCY]
        assert meta.validate_value("empty")
        assert meta.validate_value("target")
        assert meta.validate_value("other")

    def test_categorical_invalid(self):
        meta = SLOT_REGISTRY[SlotName.HAND_OCCUPANCY]
        assert not meta.validate_value("full")
        assert not meta.validate_value(0)

    def test_continuous_valid(self):
        meta = SLOT_REGISTRY[SlotName.EE_TARGET_PROXIMITY]
        assert meta.validate_value(0.0)
        assert meta.validate_value(0.5)
        assert meta.validate_value(1.0)

    def test_continuous_invalid(self):
        meta = SLOT_REGISTRY[SlotName.EE_TARGET_PROXIMITY]
        assert not meta.validate_value(-0.1)
        assert not meta.validate_value(1.1)
        assert not meta.validate_value("high")

    def test_confidence_valid(self):
        meta = SLOT_REGISTRY[SlotName.TASK_VISIBLE_CONFIDENCE]
        assert meta.validate_value(0.0)
        assert meta.validate_value(1.0)
        assert meta.validate_value(0.75)


class TestSlotMetaCreation:
    def test_categorical_requires_categories(self):
        with pytest.raises(ValueError):
            SlotMeta(
                name=SlotName.HAND_OCCUPANCY,
                domain=SlotDomain.CATEGORICAL,
                family=SlotFamily.ENABLING,
                categories=None,
            )

    def test_frozen(self):
        meta = SLOT_REGISTRY[SlotName.EE_TARGET_PROXIMITY]
        with pytest.raises(AttributeError):
            meta.name = SlotName.TARGET_CONTACT

    def test_num_categories(self):
        assert SLOT_REGISTRY[SlotName.HAND_OCCUPANCY].num_categories == 3
        assert SLOT_REGISTRY[SlotName.TARGET_CONTACT].num_categories == 2
        assert SLOT_REGISTRY[SlotName.EE_TARGET_PROXIMITY].num_categories == 1
