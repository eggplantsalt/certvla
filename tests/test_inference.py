"""Phase 4 tests: certificate gap, repair loop, inference logging.

Tests:
- Per-slot gap computation
- Aggregated gap computation
- Repair controller with fake model_fn
- InferenceLogger episode traces
- End-to-end fake rollout smoke test
"""

import torch
import pytest

from certvla.slots.schema import SlotName, SlotDomain, SLOT_REGISTRY
from certvla.slots.role_sets import J_CERT
from certvla.model.certificate_head import ROLE_ADVANCE, ROLE_PRESERVE, ROLE_IGNORE
from certvla.model.outputs import CertVLAOutput
from certvla.inference.gap import slot_gap, aggregate_certificate_gap, GapResult
from certvla.inference.repair import RepairController, RepairConfig
from certvla.inference.logging import InferenceLogger, StepRecord, EpisodeTrace


B = 2
LLM_DIM = 64
ACTION_DIM = 7
CHUNK = 8


# ── helpers ──────────────────────────────────────────────────

def _make_role_logits(batch=B):
    """Role logits for all J_CERT slots."""
    return {s: torch.randn(batch, 3) for s in J_CERT}


def _make_state_readout(batch=B):
    preds = {}
    for slot_name in SlotName:
        meta = SLOT_REGISTRY[slot_name]
        if meta.domain == SlotDomain.CATEGORICAL:
            preds[slot_name] = torch.randn(batch, len(meta.categories))
        else:
            preds[slot_name] = torch.sigmoid(torch.randn(batch, 1))
    return preds


def _make_goal_preds(batch=B):
    preds = {}
    for s in J_CERT:
        meta = SLOT_REGISTRY[s]
        if meta.domain == SlotDomain.CATEGORICAL:
            preds[s] = torch.randn(batch, len(meta.categories))
        else:
            preds[s] = torch.sigmoid(torch.randn(batch, 1))
    return preds


def _make_dummy_output(batch=B, gap_level="normal"):
    """Create a fake CertVLAOutput for testing."""
    state_readout = _make_state_readout(batch)
    role_logits = _make_role_logits(batch)
    goal_preds = _make_goal_preds(batch)

    return CertVLAOutput(
        z_t=torch.randn(batch, LLM_DIM),
        state_readout=state_readout,
        role_logits=role_logits,
        goal_preds=goal_preds,
        actions_coarse=torch.randn(batch, CHUNK, ACTION_DIM),
        actions_fine=torch.randn(batch, CHUNK, ACTION_DIM),
        actions=torch.randn(batch, CHUNK, ACTION_DIM),
        actions_hidden_states=torch.randn(batch, CHUNK * ACTION_DIM, LLM_DIM),
        gate_value=torch.sigmoid(torch.randn(batch, 1)),
    )


# ── Per-slot gap ─────────────────────────────────────────────

class TestSlotGap:
    def test_returns_dict_for_cert_slots(self):
        role_logits = _make_role_logits()
        state = _make_state_readout()
        goals = _make_goal_preds()
        state_tH = _make_goal_preds()  # same format works

        gaps = slot_gap(role_logits, state, goals, state_tH)
        assert isinstance(gaps, dict)
        assert all(s in J_CERT for s in gaps)

    def test_per_slot_shape(self):
        gaps = slot_gap(
            _make_role_logits(), _make_state_readout(),
            _make_goal_preds(), _make_goal_preds(),
        )
        for s, g in gaps.items():
            assert g.shape == (B,), f"Slot {s}: expected ({B},), got {g.shape}"

    def test_values_non_negative(self):
        gaps = slot_gap(
            _make_role_logits(), _make_state_readout(),
            _make_goal_preds(), _make_goal_preds(),
        )
        for s, g in gaps.items():
            assert (g >= -1e-6).all(), f"Slot {s} has negative gap"

    def test_identical_states_low_gap(self):
        """If state_readout == state_tH and goal == state_tH, gaps should be ~0."""
        state = _make_state_readout()
        # Use same tensors for state at t and t+H
        # Only cert slots matter for goals
        goals = {}
        state_tH = {}
        for s in J_CERT:
            if s in state:
                goals[s] = state[s].clone()
                state_tH[s] = state[s].clone()

        # Role logits don't matter for this test, use uniform
        role_logits = {s: torch.zeros(B, 3) for s in J_CERT}
        gaps = slot_gap(role_logits, state, goals, state_tH)
        for s, g in gaps.items():
            assert g.max().item() < 0.1, f"Slot {s} gap too high for identical states"


# ── Aggregated gap ───────────────────────────────────────────

class TestAggregatedGap:
    def test_returns_gap_result(self):
        per_slot = {s: torch.rand(B) * 0.5 for s in J_CERT}
        role_logits = _make_role_logits()
        result = aggregate_certificate_gap(per_slot, role_logits)
        assert isinstance(result, GapResult)
        assert result.aggregated.shape == (B,)

    def test_uniform_weights_is_mean(self):
        val = 0.3
        per_slot = {s: torch.full((B,), val) for s in J_CERT}
        result = aggregate_certificate_gap(per_slot, _make_role_logits())
        # With uniform weights (1.0) and confidence (1.0), aggregated = mean of gaps
        # sum(1.0 * 1.0 * 0.3) / sum(1.0 * 1.0) = 0.3
        assert result.aggregated[0].item() == pytest.approx(val, abs=1e-5)

    def test_custom_slot_weights(self):
        per_slot = {s: torch.ones(B) for s in J_CERT}
        slots = list(J_CERT)
        # Give first slot weight 10, rest weight 0
        weights = {s: 0.0 for s in J_CERT}
        weights[slots[0]] = 10.0
        result = aggregate_certificate_gap(per_slot, _make_role_logits(), slot_weights=weights)
        # Only first slot contributes, so aggregated = 1.0
        assert result.aggregated[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_empty_gaps(self):
        result = aggregate_certificate_gap({}, {})
        assert result.aggregated.shape == (1,)
        assert result.aggregated.item() == 0.0

    def test_role_probs_sum_to_one(self):
        per_slot = {s: torch.rand(B) for s in J_CERT}
        role_logits = _make_role_logits()
        result = aggregate_certificate_gap(per_slot, role_logits)
        for s, probs in result.role_probs.items():
            sums = probs.sum(dim=-1)
            assert torch.allclose(sums, torch.ones(B), atol=1e-5)


# ── Repair controller ───────────────────────────────────────

class TestRepairController:
    def _make_controller(self, threshold=0.3, max_steps=3):
        """Create repair controller with a fake model_fn."""
        call_count = [0]

        def model_fn(hidden, stp, asp, z_prev=None):
            call_count[0] += 1
            return _make_dummy_output()

        config = RepairConfig(gap_threshold=threshold, max_repair_steps=max_steps)
        ctrl = RepairController(config, model_fn)
        return ctrl, call_count

    def test_no_repair_below_threshold(self):
        """If gap is always below threshold, no repair should happen."""
        call_count = [0]

        def model_fn(hidden, stp, asp, z_prev=None):
            call_count[0] += 1
            out = _make_dummy_output()
            # Make goals identical to state -> zero gap
            for s in J_CERT:
                if s in out.state_readout:
                    out.goal_preds[s] = out.state_readout[s].clone()
            return out

        config = RepairConfig(gap_threshold=0.5, max_repair_steps=3)
        ctrl = RepairController(config, model_fn)
        hidden = torch.randn(B, 10, LLM_DIM)

        # Use identical state_tH to get zero gap
        actions, gap, n = ctrl.step(hidden, 0, 2)
        assert actions.shape == (B, CHUNK, ACTION_DIM)
        assert call_count[0] == 1  # Only initial forward, no repairs

    def test_repair_triggers_on_high_gap(self):
        ctrl, call_count = self._make_controller(threshold=0.001, max_steps=2)
        hidden = torch.randn(B, 10, LLM_DIM)
        actions, gap, n = ctrl.step(hidden, 0, 2)
        # Should have done initial + up to max_repair_steps
        assert call_count[0] >= 2
        assert actions.shape == (B, CHUNK, ACTION_DIM)

    def test_returns_correct_types(self):
        ctrl, _ = self._make_controller()
        hidden = torch.randn(B, 10, LLM_DIM)
        actions, gap, n_repairs = ctrl.step(hidden, 0, 2)
        assert isinstance(actions, torch.Tensor)
        assert isinstance(gap, GapResult)
        assert isinstance(n_repairs, int)


# ── Inference logger ─────────────────────────────────────────

class TestInferenceLogger:
    def test_episode_lifecycle(self):
        log = InferenceLogger(verbose=False)
        log.begin_episode(metadata={"task": "test"})
        log.log_step(StepRecord(accepted=True, repair_attempt=0))
        log.log_step(StepRecord(accepted=True, repair_attempt=0))
        log.end_episode()
        trace = log.get_last_trace()
        assert trace is not None
        assert trace.num_steps == 2

    def test_warnings_recorded(self):
        log = InferenceLogger()
        log.begin_episode()
        log.log_warning("test warning")
        log.end_episode()
        trace = log.get_last_trace()
        assert len(trace.warnings) == 1

    def test_summary(self):
        log = InferenceLogger()
        log.begin_episode()
        log.log_step(StepRecord(accepted=True, repair_attempt=0))
        log.log_step(StepRecord(accepted=True, repair_attempt=1))
        log.end_episode()
        s = log.get_last_trace().summary()
        assert s["num_steps"] == 2
        assert s["num_repairs"] == 1

    def test_max_episodes_pruning(self):
        log = InferenceLogger(max_episodes=3)
        for i in range(5):
            log.begin_episode(metadata={"i": i})
            log.log_step(StepRecord(accepted=True))
            log.end_episode()
        assert len(log.get_all_traces()) == 3

    def test_clear(self):
        log = InferenceLogger()
        log.begin_episode()
        log.log_step(StepRecord(accepted=True))
        log.end_episode()
        log.clear()
        assert log.get_last_trace() is None


# ── End-to-end fake rollout ──────────────────────────────────

class TestEndToEndRollout:
    """Smoke test: simulate a 5-step fake episode with repair."""

    def test_fake_rollout(self):
        from certvla.model.certvla_wrapper import CertVLAWrapper

        wrapper = CertVLAWrapper(
            llm_dim=LLM_DIM, action_dim=ACTION_DIM, num_actions_chunk=CHUNK,
        )
        wrapper.eval()

        def model_fn(hidden, stp, asp, z_prev=None):
            return wrapper(hidden, stp, asp, z_prev)

        config = RepairConfig(gap_threshold=0.5, max_repair_steps=2)
        log = InferenceLogger(verbose=False)
        ctrl = RepairController(config, model_fn, logger=log)

        log.begin_episode(metadata={"task": "fake_rollout"})

        seq_len = 70
        z_prev = None
        total_repairs = 0

        for t in range(5):
            hidden = torch.randn(1, seq_len, LLM_DIM)
            actions, gap, n_repairs = ctrl.step(
                hidden, state_token_pos=0, action_start_pos=2, z_prev=z_prev,
            )

            # Check outputs
            assert actions.shape == (1, CHUNK, ACTION_DIM)
            assert isinstance(gap, GapResult)
            assert gap.aggregated.shape == (1,)

            # Use z_t from last forward for next step (simulating recurrence)
            # We need to get z_t; since repair controller returns actions,
            # we get it from the logger's last step.
            total_repairs += n_repairs

        log.end_episode()
        trace = log.get_last_trace()

        # Verify trace
        assert trace is not None
        assert trace.num_steps >= 5  # At least 5 accepted steps
        summary = trace.summary()
        assert summary["num_steps"] >= 5
        assert "mean_gap" in summary
