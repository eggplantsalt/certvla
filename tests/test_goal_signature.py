"""Tests for certvla.data.goal_signature -- goal signature aggregation."""

import pytest
from certvla.data.chunk_sample import SlotState
from certvla.data.goal_signature import compute_goal_signature
from certvla.slots.schema import SlotName


def _make_state(overrides=None, all_valid=True):
    defaults = {
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
    if overrides:
        defaults.update(overrides)
    return SlotState(
        values=defaults,
        validity_mask={s: all_valid for s in SlotName},
        confidence={s: 1.0 for s in SlotName},
    )


class TestGoalSignature:
    def test_successful_episode_terminal_state(self):
        """Goal signature should reflect terminal state for a successful episode."""
        # 10-step episode, last 5 steps show task completion
        states = []
        for i in range(5):
            states.append(_make_state({
                SlotName.TARGET_GOAL_PROXIMITY: 0.8 - i * 0.1,
                SlotName.SUPPORT_RELATION: "none",
                SlotName.COMPLETION_LATCH: 0,
            }))
        for i in range(5):
            states.append(_make_state({
                SlotName.TARGET_GOAL_PROXIMITY: 0.0,
                SlotName.SUPPORT_RELATION: "on_goal",
                SlotName.COMPLETION_LATCH: 1,
            }))

        goal = compute_goal_signature(states, K=5)

        assert goal.get(SlotName.TARGET_GOAL_PROXIMITY) == pytest.approx(0.0)
        assert goal.get(SlotName.SUPPORT_RELATION) == "on_goal"
        assert goal.get(SlotName.COMPLETION_LATCH) == 1

    def test_continuous_averaging(self):
        """Continuous slots should be averaged over terminal K steps."""
        states = [
            _make_state({SlotName.EE_TARGET_PROXIMITY: 0.1}),
            _make_state({SlotName.EE_TARGET_PROXIMITY: 0.2}),
            _make_state({SlotName.EE_TARGET_PROXIMITY: 0.3}),
        ]
        goal = compute_goal_signature(states, K=3)
        assert goal.get(SlotName.EE_TARGET_PROXIMITY) == pytest.approx(0.2)

    def test_binary_majority_vote(self):
        """Binary slots should use majority vote."""
        states = [
            _make_state({SlotName.COMPLETION_LATCH: 0}),
            _make_state({SlotName.COMPLETION_LATCH: 1}),
            _make_state({SlotName.COMPLETION_LATCH: 1}),
        ]
        goal = compute_goal_signature(states, K=3)
        assert goal.get(SlotName.COMPLETION_LATCH) == 1

    def test_categorical_mode(self):
        """Categorical slots should use mode (most common)."""
        states = [
            _make_state({SlotName.SUPPORT_RELATION: "none"}),
            _make_state({SlotName.SUPPORT_RELATION: "on_goal"}),
            _make_state({SlotName.SUPPORT_RELATION: "on_goal"}),
            _make_state({SlotName.SUPPORT_RELATION: "on_goal"}),
        ]
        goal = compute_goal_signature(states, K=4)
        assert goal.get(SlotName.SUPPORT_RELATION) == "on_goal"

    def test_K_larger_than_episode(self):
        """If K > len(episode), use all available states."""
        states = [
            _make_state({SlotName.COMPLETION_LATCH: 1}),
            _make_state({SlotName.COMPLETION_LATCH: 1}),
        ]
        goal = compute_goal_signature(states, K=10)
        assert goal.get(SlotName.COMPLETION_LATCH) == 1

    def test_empty_episode_raises(self):
        with pytest.raises(ValueError):
            compute_goal_signature([], K=5)

    def test_all_validity_true(self):
        """Goal signature should have all validity=True for valid slots."""
        states = [_make_state()] * 3
        goal = compute_goal_signature(states, K=3)
        for slot_name in SlotName:
            assert goal.is_valid(slot_name)

    def test_invalid_slots_excluded(self):
        """If a slot is invalid in all terminal states, goal validity should be False."""
        state = _make_state()
        state.validity_mask[SlotName.ARTICULATION_PROGRESS] = False
        states = [state] * 3
        goal = compute_goal_signature(states, K=3)
        assert not goal.is_valid(SlotName.ARTICULATION_PROGRESS)

    def test_mixed_slot_types(self):
        """Goal signature handles all domain types correctly in one call."""
        states = [
            _make_state({
                SlotName.EE_TARGET_PROXIMITY: 0.1,      # continuous
                SlotName.HAND_OCCUPANCY: "empty",        # categorical
                SlotName.TARGET_CONTACT: 0,              # binary
                SlotName.TASK_VISIBLE_CONFIDENCE: 0.9,   # confidence
            }),
        ] * 5
        goal = compute_goal_signature(states, K=5)
        assert isinstance(goal.get(SlotName.EE_TARGET_PROXIMITY), float)
        assert isinstance(goal.get(SlotName.HAND_OCCUPANCY), str)
        assert isinstance(goal.get(SlotName.TARGET_CONTACT), int)
        assert isinstance(goal.get(SlotName.TASK_VISIBLE_CONFIDENCE), float)
