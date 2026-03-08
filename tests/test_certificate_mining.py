"""Tests for certvla.data.certificate_mining -- advance/preserve/ignore mining."""

import pytest
from certvla.data.chunk_sample import SlotState, CertificateLabel
from certvla.data.certificate_mining import mine_certificate, MiningThresholds
from certvla.slots.schema import SlotName


def _make_state(overrides=None, all_valid=True):
    """Create a SlotState with defaults, optionally overriding specific values."""
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


# === Synthetic grasp-and-place scenario ===
# Episode: approach -> grasp -> transport -> place
# Per plan Task 1.10, test these four chunks.


class TestGraspAndPlaceScenario:
    """Test certificate mining on a synthetic 4-chunk pick-and-place episode."""

    def setup_method(self):
        self.thresholds = MiningThresholds(
            tau_delta=0.1, tau_rho=0.6, tau_upsilon=0.05,
            tau_R=0.1, L_future=5, epsilon_j=0.05,
        )
        # Goal signature: object on goal, latch=1
        self.goal = _make_state({
            SlotName.EE_TARGET_PROXIMITY: 0.1,
            SlotName.HAND_OCCUPANCY: "empty",
            SlotName.TARGET_CONTACT: 0,
            SlotName.TARGET_GOAL_PROXIMITY: 0.0,
            SlotName.SUPPORT_RELATION: "on_goal",
            SlotName.CONTAINMENT_RELATION: "none",
            SlotName.COMPLETION_LATCH: 1,
            SlotName.ORIENTATION_ALIGNMENT: 0.9,
        })

    def test_chunk1_approach(self):
        """Chunk 1: approach target. ee_target_proximity decreases (advances)."""
        state_t = _make_state({SlotName.EE_TARGET_PROXIMITY: 0.9})
        state_tH = _make_state({SlotName.EE_TARGET_PROXIMITY: 0.3})

        # Future: grasp happens, then result slots advance
        future = [
            _make_state({
                SlotName.EE_TARGET_PROXIMITY: 0.2,
                SlotName.TARGET_CONTACT: 1,
                SlotName.TARGET_GOAL_PROXIMITY: 0.4,
            }),
            _make_state({
                SlotName.EE_TARGET_PROXIMITY: 0.1,
                SlotName.TARGET_CONTACT: 1,
                SlotName.HAND_OCCUPANCY: "target",
                SlotName.TARGET_GOAL_PROXIMITY: 0.2,  # result advancing
            }),
        ]

        cert = mine_certificate(state_t, state_tH, self.goal, future, self.thresholds)

        assert cert.roles[SlotName.EE_TARGET_PROXIMITY] == "advance"
        # hand_occupancy didn't change -> ignore
        assert cert.roles[SlotName.HAND_OCCUPANCY] == "ignore"

    def test_chunk2_grasp(self):
        """Chunk 2: grasp target. target_contact, hand_occupancy advance."""
        state_t = _make_state({
            SlotName.EE_TARGET_PROXIMITY: 0.2,
            SlotName.TARGET_CONTACT: 0,
            SlotName.HAND_OCCUPANCY: "empty",
        })
        state_tH = _make_state({
            SlotName.EE_TARGET_PROXIMITY: 0.1,
            SlotName.TARGET_CONTACT: 1,
            SlotName.HAND_OCCUPANCY: "target",
        })

        # Future: transport then place (result slots advance)
        future = [
            _make_state({
                SlotName.TARGET_CONTACT: 1,
                SlotName.HAND_OCCUPANCY: "target",
                SlotName.TARGET_GOAL_PROXIMITY: 0.3,
            }),
            _make_state({
                SlotName.TARGET_CONTACT: 1,
                SlotName.HAND_OCCUPANCY: "target",
                SlotName.TARGET_GOAL_PROXIMITY: 0.1,  # result advancing
            }),
        ]

        cert = mine_certificate(state_t, state_tH, self.goal, future, self.thresholds)

        # target_contact: binary, changed 0->1, enabling slot
        # Need eta > 0 (future result advance) for enabling slots
        assert cert.roles[SlotName.TARGET_CONTACT] == "advance"
        # hand_occupancy: categorical, changed empty->target, enabling slot
        assert cert.roles[SlotName.HAND_OCCUPANCY] == "advance"

    def test_chunk3_transport(self):
        """Chunk 3: transport to goal. target_goal_proximity advances, hand_occupancy preserved."""
        state_t = _make_state({
            SlotName.TARGET_CONTACT: 1,
            SlotName.HAND_OCCUPANCY: "target",
            SlotName.TARGET_GOAL_PROXIMITY: 0.8,
        })
        state_tH = _make_state({
            SlotName.TARGET_CONTACT: 1,
            SlotName.HAND_OCCUPANCY: "target",
            SlotName.TARGET_GOAL_PROXIMITY: 0.2,
        })

        # Future: placement and latch
        future = [
            _make_state({
                SlotName.HAND_OCCUPANCY: "target",
                SlotName.TARGET_GOAL_PROXIMITY: 0.1,
                SlotName.SUPPORT_RELATION: "none",
            }),
            _make_state({
                SlotName.HAND_OCCUPANCY: "empty",
                SlotName.TARGET_GOAL_PROXIMITY: 0.0,
                SlotName.SUPPORT_RELATION: "on_goal",
            }),
        ]

        cert = mine_certificate(state_t, state_tH, self.goal, future, self.thresholds)

        # target_goal_proximity is a result slot, big delta + closer to goal
        assert cert.roles[SlotName.TARGET_GOAL_PROXIMITY] == "advance"
        # hand_occupancy should be support-preserved during transport
        assert cert.roles[SlotName.HAND_OCCUPANCY] == "preserve"

    def test_chunk4_place(self):
        """Chunk 4: place on goal. support_relation, completion_latch advance."""
        state_t = _make_state({
            SlotName.TARGET_CONTACT: 1,
            SlotName.HAND_OCCUPANCY: "target",
            SlotName.TARGET_GOAL_PROXIMITY: 0.1,
            SlotName.SUPPORT_RELATION: "none",
            SlotName.COMPLETION_LATCH: 0,
        })
        state_tH = _make_state({
            SlotName.TARGET_CONTACT: 0,
            SlotName.HAND_OCCUPANCY: "empty",
            SlotName.TARGET_GOAL_PROXIMITY: 0.0,
            SlotName.SUPPORT_RELATION: "on_goal",
            SlotName.COMPLETION_LATCH: 1,
        })

        # Future: task done, values persist
        future = [
            _make_state({
                SlotName.SUPPORT_RELATION: "on_goal",
                SlotName.COMPLETION_LATCH: 1,
                SlotName.TARGET_GOAL_PROXIMITY: 0.0,
            }),
        ] * 3

        cert = mine_certificate(state_t, state_tH, self.goal, future, self.thresholds)

        assert cert.roles[SlotName.SUPPORT_RELATION] == "advance"
        assert cert.roles[SlotName.COMPLETION_LATCH] == "advance"
        # Goal value for advance slots = s_{t+H}^j
        assert cert.goal_values[SlotName.SUPPORT_RELATION] == "on_goal"
        assert cert.goal_values[SlotName.COMPLETION_LATCH] == 1


class TestMiningEdgeCases:
    def setup_method(self):
        self.thresholds = MiningThresholds()

    def test_no_change_all_ignore(self):
        """No state change -> all slots should be ignore."""
        state = _make_state()
        goal = _make_state()
        cert = mine_certificate(state, state, goal, [], self.thresholds)
        for slot, role in cert.roles.items():
            assert role == "ignore", f"Slot {slot} should be ignore, got {role}"

    def test_no_future_states(self):
        """Mining without future states should still work."""
        state_t = _make_state({SlotName.TARGET_GOAL_PROXIMITY: 0.8})
        state_tH = _make_state({SlotName.TARGET_GOAL_PROXIMITY: 0.2})
        goal = _make_state({SlotName.TARGET_GOAL_PROXIMITY: 0.0})
        cert = mine_certificate(state_t, state_tH, goal, None, self.thresholds)
        # Result slot with big delta + big upsilon -> should advance
        assert cert.roles[SlotName.TARGET_GOAL_PROXIMITY] == "advance"

    def test_advance_has_goal_value(self):
        """Every advance slot must have a goal_value."""
        state_t = _make_state({SlotName.TARGET_GOAL_PROXIMITY: 0.8})
        state_tH = _make_state({SlotName.TARGET_GOAL_PROXIMITY: 0.1})
        goal = _make_state({SlotName.TARGET_GOAL_PROXIMITY: 0.0})
        cert = mine_certificate(state_t, state_tH, goal, [], self.thresholds)
        for slot in cert.advance_slots():
            assert slot in cert.goal_values

    def test_advance_and_preserve_disjoint(self):
        """Advance and preserve sets must never overlap."""
        state_t = _make_state({
            SlotName.TARGET_CONTACT: 1,
            SlotName.HAND_OCCUPANCY: "target",
            SlotName.TARGET_GOAL_PROXIMITY: 0.8,
        })
        state_tH = _make_state({
            SlotName.TARGET_CONTACT: 1,
            SlotName.HAND_OCCUPANCY: "target",
            SlotName.TARGET_GOAL_PROXIMITY: 0.2,
        })
        goal = _make_state({SlotName.TARGET_GOAL_PROXIMITY: 0.0})
        future = [_make_state({SlotName.TARGET_GOAL_PROXIMITY: 0.1})]

        cert = mine_certificate(state_t, state_tH, goal, future, self.thresholds)
        adv_set = set(cert.advance_slots())
        pre_set = set(cert.preserve_slots())
        assert adv_set & pre_set == set(), "Advance and preserve must be disjoint"

    def test_invalid_slot_excluded(self):
        """Slots with validity=False should not advance."""
        state_t = _make_state({SlotName.ARTICULATION_PROGRESS: 0.0})
        state_tH = _make_state({SlotName.ARTICULATION_PROGRESS: 0.9})
        # Mark articulation_progress as invalid
        state_t.validity_mask[SlotName.ARTICULATION_PROGRESS] = False
        state_tH.validity_mask[SlotName.ARTICULATION_PROGRESS] = False

        goal = _make_state({SlotName.ARTICULATION_PROGRESS: 1.0})
        cert = mine_certificate(state_t, state_tH, goal, [], MiningThresholds())
        assert cert.roles[SlotName.ARTICULATION_PROGRESS] == "ignore"
