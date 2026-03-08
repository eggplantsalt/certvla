"""Tests for certvla.slots.preserve_rules -- latch-preserve + support-preserve."""

import pytest
from certvla.slots.schema import SlotName
from certvla.slots.preserve_rules import (
    latch_preserve,
    support_preserve,
    compute_preserve_set,
)


class TestLatchPreserve:
    def test_latch_preserves_result_slots_when_latched(self):
        """If completion_latch=1, result slots not in advance should be preserved."""
        state = {
            SlotName.COMPLETION_LATCH: 1,
            SlotName.TARGET_GOAL_PROXIMITY: 0.0,
            SlotName.SUPPORT_RELATION: "on_goal",
        }
        advance_set = set()  # nothing advancing
        preserve = latch_preserve(state, advance_set)

        assert SlotName.TARGET_GOAL_PROXIMITY in preserve
        assert SlotName.SUPPORT_RELATION in preserve
        assert SlotName.CONTAINMENT_RELATION in preserve
        assert SlotName.COMPLETION_LATCH in preserve  # latch itself is J_R

    def test_latch_excludes_advancing_slots(self):
        """Slots in the advance set are not latch-preserved."""
        state = {SlotName.COMPLETION_LATCH: 1}
        advance_set = {SlotName.SUPPORT_RELATION}
        preserve = latch_preserve(state, advance_set)

        assert SlotName.SUPPORT_RELATION not in preserve
        assert SlotName.TARGET_GOAL_PROXIMITY in preserve

    def test_no_latch_no_preserve(self):
        """If completion_latch=0, no latch-preserve."""
        state = {SlotName.COMPLETION_LATCH: 0}
        advance_set = set()
        preserve = latch_preserve(state, advance_set)
        assert len(preserve) == 0

    def test_missing_latch_defaults_no_preserve(self):
        """If completion_latch not in state, default to 0 -> no latch-preserve."""
        state = {}
        preserve = latch_preserve(state, set())
        assert len(preserve) == 0


class TestSupportPreserve:
    def test_transport_preserves_hand_occupancy(self):
        """During transport (target_goal_proximity advancing with target held), preserve hand_occupancy."""
        state = {
            SlotName.HAND_OCCUPANCY: "target",
            SlotName.TARGET_CONTACT: 1,
        }
        advance_set = {SlotName.TARGET_GOAL_PROXIMITY}
        preserve = support_preserve(state, advance_set)

        assert SlotName.HAND_OCCUPANCY in preserve

    def test_transport_no_preserve_when_empty(self):
        """If hand is empty, support-preserve for hand_occupancy should not trigger."""
        state = {SlotName.HAND_OCCUPANCY: "empty"}
        advance_set = {SlotName.TARGET_GOAL_PROXIMITY}
        preserve = support_preserve(state, advance_set)

        assert SlotName.HAND_OCCUPANCY not in preserve

    def test_containment_preserves_articulation(self):
        """When advancing containment with container open, preserve articulation_progress."""
        state = {
            SlotName.ARTICULATION_PROGRESS: 0.8,
            SlotName.TARGET_CONTACT: 1,
        }
        advance_set = {SlotName.CONTAINMENT_RELATION}
        preserve = support_preserve(state, advance_set)

        assert SlotName.ARTICULATION_PROGRESS in preserve

    def test_containment_preserves_contact(self):
        """When advancing containment and target in contact, preserve contact."""
        state = {SlotName.TARGET_CONTACT: 1}
        advance_set = {SlotName.CONTAINMENT_RELATION}
        preserve = support_preserve(state, advance_set)

        assert SlotName.TARGET_CONTACT in preserve

    def test_support_preserves_contact(self):
        """When advancing support_relation and target in contact, preserve contact."""
        state = {SlotName.TARGET_CONTACT: 1}
        advance_set = {SlotName.SUPPORT_RELATION}
        preserve = support_preserve(state, advance_set)

        assert SlotName.TARGET_CONTACT in preserve

    def test_no_trigger_no_preserve(self):
        """If no trigger slots are advancing, no support-preserve."""
        state = {SlotName.HAND_OCCUPANCY: "target", SlotName.TARGET_CONTACT: 1}
        advance_set = {SlotName.EE_TARGET_PROXIMITY}  # not a trigger for any rule
        preserve = support_preserve(state, advance_set)

        assert len(preserve) == 0

    def test_preserved_slot_not_in_advance(self):
        """A slot cannot be both advance and support-preserved."""
        state = {
            SlotName.HAND_OCCUPANCY: "target",
            SlotName.TARGET_CONTACT: 1,
        }
        # Both target_goal_proximity and target_contact advancing
        advance_set = {SlotName.TARGET_GOAL_PROXIMITY, SlotName.TARGET_CONTACT}
        preserve = support_preserve(state, advance_set)

        # target_contact should NOT be in preserve since it's in advance_set
        assert SlotName.TARGET_CONTACT not in preserve


class TestComputePreserveSet:
    def test_combines_latch_and_support(self):
        state = {
            SlotName.COMPLETION_LATCH: 1,
            SlotName.HAND_OCCUPANCY: "target",
            SlotName.TARGET_CONTACT: 1,
        }
        advance_set = {SlotName.TARGET_GOAL_PROXIMITY}

        preserve = compute_preserve_set(state, advance_set)

        # Latch should preserve other J_R slots not in advance
        assert SlotName.SUPPORT_RELATION in preserve
        assert SlotName.CONTAINMENT_RELATION in preserve
        # Support should preserve hand_occupancy and target_contact (during transport)
        assert SlotName.HAND_OCCUPANCY in preserve
        assert SlotName.TARGET_CONTACT in preserve
        # Advance set is excluded
        assert SlotName.TARGET_GOAL_PROXIMITY not in preserve

    def test_advance_never_in_preserve(self):
        """The final preserve set must exclude all advance slots."""
        state = {
            SlotName.COMPLETION_LATCH: 1,
            SlotName.HAND_OCCUPANCY: "target",
        }
        advance_set = {SlotName.COMPLETION_LATCH, SlotName.TARGET_GOAL_PROXIMITY}

        preserve = compute_preserve_set(state, advance_set)

        for s in advance_set:
            assert s not in preserve
