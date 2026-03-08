"""Phase 3 smoke tests: losses, curriculum, scheduled sampling.

Tests:
- Each loss returns a finite scalar with gradients
- Focal CE down-weights easy examples
- Curriculum stages return correct configs
- Scheduled sampler produces valid probabilities
- Full smoke test: CertVLAWrapper forward -> all losses -> backward
"""

import torch
import pytest

from certvla.slots.schema import SlotName, SlotDomain, SLOT_REGISTRY
from certvla.slots.role_sets import J_CERT
from certvla.model.certificate_head import ROLE_ADVANCE, ROLE_PRESERVE, ROLE_IGNORE
from certvla.training.losses import (
    focal_cross_entropy,
    cert_state_loss,
    cert_role_loss,
    cert_goal_loss,
    cert_action_loss,
    cert_consistency_loss,
    cert_dependence_loss,
    cert_counterfactual_loss,
    cert_total_loss,
)
from certvla.training.curriculum import (
    TrainingStage,
    StageConfig,
    CurriculumScheduler,
    DEFAULT_STAGES,
)
from certvla.training.sched_sampling import ScheduledSampler, SamplingSchedule


B = 4
LLM_DIM = 64
ACTION_DIM = 7
CHUNK = 8


# ── helpers ──────────────────────────────────────────────────

def _make_state_preds(batch=B, requires_grad=True):
    """Create fake state_readout in model-output format."""
    preds = {}
    for slot_name in SlotName:
        meta = SLOT_REGISTRY[slot_name]
        if meta.domain == SlotDomain.CATEGORICAL:
            t = torch.randn(batch, len(meta.categories), requires_grad=requires_grad)
        else:
            t = torch.sigmoid(torch.randn(batch, 1, requires_grad=requires_grad))
        preds[slot_name] = t
    return preds


def _make_state_targets(batch=B):
    """Create fake ground-truth state targets."""
    targets = {}
    for slot_name in SlotName:
        meta = SLOT_REGISTRY[slot_name]
        if meta.domain == SlotDomain.BINARY:
            targets[slot_name] = torch.randint(0, 2, (batch, 1)).float()
        elif meta.domain == SlotDomain.CATEGORICAL:
            targets[slot_name] = torch.randint(0, len(meta.categories), (batch,))
        else:
            targets[slot_name] = torch.rand(batch, 1)
    return targets


def _make_role_logits(batch=B, requires_grad=True):
    out = {}
    for s in J_CERT:
        out[s] = torch.randn(batch, 3, requires_grad=requires_grad)
    return out


def _make_role_targets(batch=B):
    return {s: torch.randint(0, 3, (batch,)) for s in J_CERT}


def _make_mask(batch=B):
    return {s: torch.ones(batch) for s in SlotName}


def _make_confidence(batch=B):
    return {s: torch.ones(batch) for s in SlotName}


def _make_goal_preds(batch=B, requires_grad=True):
    preds = {}
    for s in J_CERT:
        meta = SLOT_REGISTRY[s]
        if meta.domain == SlotDomain.CATEGORICAL:
            preds[s] = torch.randn(batch, len(meta.categories), requires_grad=requires_grad)
        else:
            preds[s] = torch.sigmoid(torch.randn(batch, 1, requires_grad=requires_grad))
    return preds


def _make_goal_targets(batch=B):
    targets = {}
    for s in J_CERT:
        meta = SLOT_REGISTRY[s]
        if meta.domain == SlotDomain.BINARY:
            targets[s] = torch.randint(0, 2, (batch, 1)).float()
        elif meta.domain == SlotDomain.CATEGORICAL:
            targets[s] = torch.randint(0, len(meta.categories), (batch,))
        else:
            targets[s] = torch.rand(batch, 1)
    return targets


# ── Focal CE ────────────────────────────────────────────────

class TestFocalCE:
    def test_returns_per_sample(self):
        logits = torch.randn(B, 3)
        targets = torch.randint(0, 3, (B,))
        out = focal_cross_entropy(logits, targets, gamma=2.0)
        assert out.shape == (B,)

    def test_gamma_zero_equals_ce(self):
        logits = torch.randn(B, 5)
        targets = torch.randint(0, 5, (B,))
        focal = focal_cross_entropy(logits, targets, gamma=0.0)
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        assert torch.allclose(focal, ce, atol=1e-6)

    def test_high_gamma_downweights_easy(self):
        # Moderately easy example (not saturated, so CE > 0)
        logits = torch.tensor([[3.0, 0.0, 0.0]])
        targets = torch.tensor([0])
        low_g = focal_cross_entropy(logits, targets, gamma=0.0).item()
        high_g = focal_cross_entropy(logits, targets, gamma=5.0).item()
        assert low_g > 1e-6, "CE should be non-trivial for this logit"
        assert high_g < low_g


# ── L_state ──────────────────────────────────────────────────

class TestStateLoss:
    def test_scalar_finite(self):
        loss = cert_state_loss(
            _make_state_preds(), _make_state_targets(),
            _make_mask(), _make_confidence(),
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_has_grad(self):
        preds = _make_state_preds()
        loss = cert_state_loss(preds, _make_state_targets(), _make_mask(), _make_confidence())
        loss.backward()
        grads = [p.grad is not None for p in preds.values() if p.requires_grad]
        assert any(grads)

    def test_mask_zeros_out(self):
        mask = {s: torch.zeros(B) for s in SlotName}
        loss = cert_state_loss(
            _make_state_preds(), _make_state_targets(), mask, _make_confidence(),
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-7)


# ── L_role ───────────────────────────────────────────────────

class TestRoleLoss:
    def test_scalar_finite(self):
        loss = cert_role_loss(
            _make_role_logits(), _make_role_targets(),
            _make_mask(), _make_confidence(),
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_has_grad(self):
        logits = _make_role_logits()
        loss = cert_role_loss(logits, _make_role_targets(), _make_mask(), _make_confidence())
        loss.backward()
        assert any(v.grad is not None for v in logits.values())


# ── L_goal ───────────────────────────────────────────────────

class TestGoalLoss:
    def test_scalar_finite(self):
        loss = cert_goal_loss(
            _make_goal_preds(), _make_goal_targets(), _make_role_targets(),
            _make_mask(), _make_confidence(),
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_no_advance_zero(self):
        roles = {s: torch.full((B,), ROLE_IGNORE, dtype=torch.long) for s in J_CERT}
        loss = cert_goal_loss(
            _make_goal_preds(), _make_goal_targets(), roles,
            _make_mask(), _make_confidence(),
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ── L_act ────────────────────────────────────────────────────

class TestActionLoss:
    def test_scalar_finite(self):
        pred = torch.randn(B, CHUNK, ACTION_DIM, requires_grad=True)
        expert = torch.randn(B, CHUNK, ACTION_DIM)
        loss = cert_action_loss(pred, expert)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert pred.grad is not None


# ── L_cons ───────────────────────────────────────────────────

class TestConsistencyLoss:
    def test_scalar_finite(self):
        loss = cert_consistency_loss(
            _make_state_preds(), _make_goal_preds(),
            _make_role_targets(), _make_goal_targets(),
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_all_ignore_zero(self):
        roles = {s: torch.full((B,), ROLE_IGNORE, dtype=torch.long) for s in J_CERT}
        loss = cert_consistency_loss(
            _make_state_preds(), _make_goal_preds(), roles, _make_goal_targets(),
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ── L_dep ────────────────────────────────────────────────────

class TestDependenceLoss:
    def test_scalar_finite(self):
        expert = torch.randn(B, CHUNK, ACTION_DIM)
        pos = torch.randn(B, CHUNK, ACTION_DIM, requires_grad=True)
        neg = torch.randn(B, CHUNK, ACTION_DIM, requires_grad=True)
        loss = cert_dependence_loss(expert, pos, neg, margin=0.1)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert pos.grad is not None

    def test_neg_worse_than_pos_low_loss(self):
        expert = torch.zeros(B, CHUNK, ACTION_DIM)
        pos = torch.zeros(B, CHUNK, ACTION_DIM)       # perfect
        neg = torch.ones(B, CHUNK, ACTION_DIM) * 10.0  # terrible
        loss = cert_dependence_loss(expert, pos, neg, margin=0.1)
        # e_pos=0, e_neg=10 => max(0, 0.1+0-10) = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ── L_cf ─────────────────────────────────────────────────────

class TestCounterfactualLoss:
    def test_no_pairs_returns_zero(self):
        z = torch.randn(B, LLM_DIM)
        loss = cert_counterfactual_loss(z)
        assert loss.dim() == 0

    def test_invariance_term(self):
        z = torch.randn(B, LLM_DIM, requires_grad=True)
        z_pos = torch.randn(B, LLM_DIM)
        loss = cert_counterfactual_loss(z, z_pos=z_pos)
        assert torch.isfinite(loss)
        loss.backward()
        assert z.grad is not None

    def test_breaking_term(self):
        z = torch.randn(B, LLM_DIM, requires_grad=True)
        z_neg = z.detach().clone()  # identical => high breaking loss
        loss = cert_counterfactual_loss(z, z_neg=z_neg, mu=1.0)
        assert loss.item() > 0


# ── Total loss ───────────────────────────────────────────────

class TestTotalLoss:
    def test_weighted_sum(self):
        l_s = torch.tensor(1.0, requires_grad=True)
        l_a = torch.tensor(2.0, requires_grad=True)
        total, comp = cert_total_loss(
            {"state": l_s, "action": l_a},
            {"lambda_s": 0.5, "lambda_a": 1.0},
        )
        assert total.item() == pytest.approx(0.5 * 1.0 + 1.0 * 2.0, abs=1e-5)
        assert "state" in comp and "action" in comp

    def test_zero_weight_excluded(self):
        l = torch.tensor(5.0, requires_grad=True)
        total, _ = cert_total_loss({"state": l}, {"lambda_s": 0.0})
        assert total.item() == pytest.approx(0.0, abs=1e-7)


# ── Curriculum ───────────────────────────────────────────────

class TestCurriculum:
    def test_stage1_only_state_loss(self):
        cfg = DEFAULT_STAGES[TrainingStage.STAGE_1_STATE]
        assert cfg.lambda_s > 0
        assert cfg.lambda_r == 0 and cfg.lambda_a == 0

    def test_stage3_has_action_and_dep(self):
        cfg = DEFAULT_STAGES[TrainingStage.STAGE_3_POLICY]
        assert cfg.lambda_a > 0
        assert cfg.lambda_d > 0

    def test_scheduler_boundaries(self):
        sched = CurriculumScheduler()
        assert sched.get_stage(0) == TrainingStage.STAGE_1_STATE
        assert sched.get_stage(6000) == TrainingStage.STAGE_2_CERTIFICATE
        assert sched.get_stage(20000) == TrainingStage.STAGE_3_POLICY
        assert sched.get_stage(50000) == TrainingStage.STAGE_4_COUNTERFACTUAL

    def test_past_boundary_stays_last(self):
        sched = CurriculumScheduler()
        assert sched.get_stage(999999) == TrainingStage.STAGE_4_COUNTERFACTUAL

    def test_loss_weights_dict(self):
        sched = CurriculumScheduler()
        w = sched.get_loss_weights(0)
        assert "lambda_s" in w and "lambda_cf" in w

    def test_should_compute_dep(self):
        sched = CurriculumScheduler()
        assert not sched.should_compute_dep(0)       # stage 1
        assert sched.should_compute_dep(20000)       # stage 3


# ── Scheduled sampling ──────────────────────────────────────

class TestScheduledSampling:
    def test_constant(self):
        s = ScheduledSampler(schedule=SamplingSchedule.CONSTANT, start_prob=1.0)
        assert s.get_teacher_force_prob(0) == 1.0
        assert s.get_teacher_force_prob(99999) == 1.0

    def test_linear_decay(self):
        s = ScheduledSampler(
            schedule=SamplingSchedule.LINEAR,
            start_prob=1.0, end_prob=0.0,
            warmup_steps=0, total_steps=100,
        )
        assert s.get_teacher_force_prob(0) == pytest.approx(1.0)
        assert s.get_teacher_force_prob(50) == pytest.approx(0.5)
        assert s.get_teacher_force_prob(100) == pytest.approx(0.0)

    def test_cosine_endpoints(self):
        s = ScheduledSampler(
            schedule=SamplingSchedule.COSINE,
            start_prob=1.0, end_prob=0.0,
            warmup_steps=0, total_steps=100,
        )
        assert s.get_teacher_force_prob(0) == pytest.approx(1.0)
        assert s.get_teacher_force_prob(100) == pytest.approx(0.0, abs=1e-6)

    def test_warmup(self):
        s = ScheduledSampler(
            schedule=SamplingSchedule.LINEAR,
            start_prob=1.0, end_prob=0.0,
            warmup_steps=50, total_steps=100,
        )
        assert s.get_teacher_force_prob(25) == pytest.approx(1.0)
        assert s.get_teacher_force_prob(75) == pytest.approx(0.5)


# ── Full smoke: wrapper forward → losses → backward ─────────

class TestFullSmoke:
    """End-to-end: build CertVLAWrapper, forward, compute all losses, backward."""

    def test_full_forward_and_loss(self):
        from certvla.model.certvla_wrapper import CertVLAWrapper

        wrapper = CertVLAWrapper(llm_dim=LLM_DIM, action_dim=ACTION_DIM, num_actions_chunk=CHUNK)
        seq_len = 70
        state_pos = 0
        action_start = 2

        hidden = torch.randn(B, seq_len, LLM_DIM)
        out = wrapper(hidden, state_token_pos=state_pos, action_start_pos=action_start)

        # Build GT
        state_tgt = _make_state_targets()
        goal_tgt = _make_goal_targets()
        role_tgt = _make_role_targets()
        mask = _make_mask()
        conf = _make_confidence()
        expert = torch.randn(B, CHUNK, ACTION_DIM)

        # Compute each loss
        l_state = cert_state_loss(out.state_readout, state_tgt, mask, conf)
        l_role = cert_role_loss(out.role_logits, role_tgt, mask, conf)
        l_goal = cert_goal_loss(out.goal_preds, goal_tgt, role_tgt, mask, conf)
        l_act = cert_action_loss(out.actions, expert)
        l_cons = cert_consistency_loss(out.state_readout, out.goal_preds, role_tgt, goal_tgt)

        # L_dep: need a negative cert forward
        from certvla.model.certificate_head import ROLE_ADVANCE
        neg_role_logits = {s: torch.randn_like(v) for s, v in out.role_logits.items()}
        neg_goal_preds = {s: torch.randn_like(v) for s, v in out.goal_preds.items()}
        actions_neg, _, _ = wrapper.action_head(
            out.z_t, neg_role_logits, neg_goal_preds, out.actions_hidden_states,
        )
        l_dep = cert_dependence_loss(expert, out.actions, actions_neg, margin=0.1)

        # L_cf: minimal (no augmented pairs)
        l_cf = cert_counterfactual_loss(out.z_t)

        # Total
        losses = {
            "state": l_state, "role": l_role, "goal": l_goal,
            "action": l_act, "consistency": l_cons,
            "dependence": l_dep, "counterfactual": l_cf,
        }
        weights = DEFAULT_STAGES[TrainingStage.STAGE_3_POLICY].loss_weights()
        total, components = cert_total_loss(losses, weights)

        assert torch.isfinite(total)
        total.backward()

        # Verify gradient flow to wrapper parameters
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in wrapper.parameters()
        )
        assert has_grad, "No gradients flowed to wrapper parameters"
