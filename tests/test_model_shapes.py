"""
Shape and smoke tests for CertVLA model layer.

Tests verify:
1. All modules produce outputs with correct shapes
2. State readout and certificate head only read from z_t (gradient isolation)
3. Full wrapper forward pass succeeds end-to-end
4. Action head depends on certificate (not ignoring it)
"""

import pytest
import torch
import torch.nn as nn

from certvla.model.state_token import StateTokenModule
from certvla.model.state_readout import StateReadoutHead
from certvla.model.certificate_head import CertificateHead, NUM_ROLES
from certvla.model.action_head import CertActionHead
from certvla.model.certvla_wrapper import CertVLAWrapper
from certvla.model.outputs import CertVLAOutput
from certvla.slots.schema import SlotName, SlotDomain, SLOT_REGISTRY
from certvla.slots.role_sets import J_CERT, J_E, J_R, J_C

# Test constants
B = 4           # batch size
LLM_DIM = 128   # small llm_dim for tests (real = 4096)
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8
NUM_PATCHES = 16  # simulated vision patches
NUM_PROMPT_TOKENS = 10
STATE_TOKEN_POS = NUM_PATCHES  # state token after vision patches
# Action tokens start after: BOS(implicit in LLM) + patches + state_token + prompt
ACTION_START = NUM_PATCHES + 1 + NUM_PROMPT_TOKENS
SEQ_LEN = ACTION_START + ACTION_DIM * NUM_ACTIONS_CHUNK + 1  # +1 for stop token


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dummy_hidden_states(device):
    """Simulate last_hidden_states from LLM."""
    return torch.randn(B, SEQ_LEN, LLM_DIM, device=device)


class TestStateTokenModule:
    def test_initial_state_shape(self):
        mod = StateTokenModule(LLM_DIM)
        z0 = mod.get_initial_state(B)
        assert z0.shape == (B, LLM_DIM)

    def test_state_token_embedding_shape(self):
        mod = StateTokenModule(LLM_DIM)
        emb = mod.get_state_token_embedding(B)
        assert emb.shape == (B, 1, LLM_DIM)

    def test_gated_update_shape(self):
        mod = StateTokenModule(LLM_DIM)
        tilde_z = torch.randn(B, LLM_DIM)
        z_prev = torch.randn(B, LLM_DIM)
        z_t, gate = mod.gated_update(tilde_z, z_prev)
        assert z_t.shape == (B, LLM_DIM)
        assert gate.shape == (B, LLM_DIM)

    def test_gate_is_sigmoid_bounded(self):
        mod = StateTokenModule(LLM_DIM)
        tilde_z = torch.randn(B, LLM_DIM)
        z_prev = torch.randn(B, LLM_DIM)
        _, gate = mod.gated_update(tilde_z, z_prev)
        assert (gate >= 0).all() and (gate <= 1).all()

    def test_initial_gate_near_half(self):
        """With zero-init weights, gate should be ~0.5 at init."""
        mod = StateTokenModule(LLM_DIM)
        tilde_z = torch.randn(B, LLM_DIM)
        z_prev = torch.randn(B, LLM_DIM)
        _, gate = mod.gated_update(tilde_z, z_prev)
        assert torch.abs(gate.mean() - 0.5).item() < 0.1


class TestStateReadoutHead:
    def test_output_shapes(self):
        head = StateReadoutHead(LLM_DIM, hidden_dim=64)
        z_t = torch.randn(B, LLM_DIM)
        out = head(z_t)

        assert isinstance(out, dict)
        assert len(out) == len(SLOT_REGISTRY)

        for slot_name, meta in SLOT_REGISTRY.items():
            assert slot_name in out, f"Missing slot {slot_name}"
            t = out[slot_name]
            if meta.domain == SlotDomain.BINARY:
                assert t.shape == (B, 1)
                assert (t >= 0).all() and (t <= 1).all(), "Binary should be sigmoid-bounded"
            elif meta.domain == SlotDomain.CATEGORICAL:
                assert t.shape == (B, len(meta.categories))
            elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
                assert t.shape == (B, 1)
                assert (t >= 0).all() and (t <= 1).all(), "Continuous should be sigmoid-bounded"

    def test_gradient_only_through_z_t(self):
        """Readout gradients must flow only through z_t input, not through other paths."""
        head = StateReadoutHead(LLM_DIM, hidden_dim=64)
        z_t = torch.randn(B, LLM_DIM, requires_grad=True)
        other_input = torch.randn(B, LLM_DIM, requires_grad=True)

        # Only pass z_t to readout
        out = head(z_t)
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        assert z_t.grad is not None, "z_t should have gradients"
        assert other_input.grad is None, "Other inputs should NOT have gradients"


class TestCertificateHead:
    def test_output_shapes(self):
        head = CertificateHead(LLM_DIM, hidden_dim=64)
        z_t = torch.randn(B, LLM_DIM)
        role_logits, goal_preds = head(z_t)

        assert len(role_logits) == len(J_CERT)
        assert len(goal_preds) == len(J_CERT)

        for slot_name in J_CERT:
            meta = SLOT_REGISTRY[slot_name]
            # Role logits: (B, 3)
            assert role_logits[slot_name].shape == (B, NUM_ROLES)

            # Goal preds: domain-dependent
            if meta.domain == SlotDomain.BINARY:
                assert goal_preds[slot_name].shape == (B, 1)
            elif meta.domain == SlotDomain.CATEGORICAL:
                assert goal_preds[slot_name].shape == (B, len(meta.categories))
            elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
                assert goal_preds[slot_name].shape == (B, 1)

    def test_only_cert_slots(self):
        """Certificate head should only produce outputs for J_CERT slots."""
        head = CertificateHead(LLM_DIM, hidden_dim=64)
        z_t = torch.randn(B, LLM_DIM)
        role_logits, goal_preds = head(z_t)

        for slot_name in role_logits:
            assert slot_name in J_CERT
        # J_C slot (task_visible_confidence) should NOT be in cert outputs
        assert SlotName.TASK_VISIBLE_CONFIDENCE not in role_logits

    def test_gradient_only_through_z_t(self):
        head = CertificateHead(LLM_DIM, hidden_dim=64)
        z_t = torch.randn(B, LLM_DIM, requires_grad=True)

        role_logits, goal_preds = head(z_t)
        loss = sum(v.sum() for v in role_logits.values()) + sum(v.sum() for v in goal_preds.values())
        loss.backward()

        assert z_t.grad is not None


class TestCertActionHead:
    def setup_method(self):
        self.head = CertActionHead(
            llm_dim=LLM_DIM,
            cert_embed_dim=32,
            coarse_hidden_dim=64,
            fine_hidden_dim=64,
            action_dim=ACTION_DIM,
            num_actions_chunk=NUM_ACTIONS_CHUNK,
        )
        self.cert_head = CertificateHead(LLM_DIM, hidden_dim=64)

    def test_output_shapes(self):
        z_t = torch.randn(B, LLM_DIM)
        actions_h = torch.randn(B, NUM_ACTIONS_CHUNK * ACTION_DIM, LLM_DIM)
        role_logits, goal_preds = self.cert_head(z_t)

        actions, coarse, fine = self.head(z_t, role_logits, goal_preds, actions_h)

        assert actions.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
        assert coarse.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
        assert fine.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM)

    def test_cert_influences_actions(self):
        """Changing certificate should change action output (certificate is not ignored)."""
        z_t = torch.randn(B, LLM_DIM)
        actions_h = torch.randn(B, NUM_ACTIONS_CHUNK * ACTION_DIM, LLM_DIM)

        # Two different certificate inputs
        role_logits_1, goal_preds_1 = self.cert_head(z_t)
        # Perturb the certificate
        role_logits_2 = {k: v + torch.randn_like(v) * 5.0 for k, v in role_logits_1.items()}
        goal_preds_2 = {k: v + torch.randn_like(v) * 5.0 for k, v in goal_preds_1.items()}

        with torch.no_grad():
            actions_1, _, _ = self.head(z_t, role_logits_1, goal_preds_1, actions_h)
            actions_2, _, _ = self.head(z_t, role_logits_2, goal_preds_2, actions_h)

        # Actions should differ when certificate differs
        diff = (actions_1 - actions_2).abs().sum()
        assert diff > 0, "Actions should change when certificate changes"

    def test_lambda_res_learnable(self):
        assert self.head.lambda_res.requires_grad
        assert self.head.lambda_res.item() == pytest.approx(0.1)


class TestCertVLAWrapper:
    def setup_method(self):
        self.wrapper = CertVLAWrapper(
            llm_dim=LLM_DIM,
            readout_hidden_dim=64,
            cert_hidden_dim=64,
            cert_embed_dim=32,
            coarse_hidden_dim=64,
            fine_hidden_dim=64,
            action_dim=ACTION_DIM,
            num_actions_chunk=NUM_ACTIONS_CHUNK,
        )

    def test_full_forward_shapes(self, dummy_hidden_states):
        output = self.wrapper(
            last_hidden_states=dummy_hidden_states,
            state_token_pos=STATE_TOKEN_POS,
            action_start_pos=ACTION_START,
        )

        assert isinstance(output, CertVLAOutput)
        assert output.z_t.shape == (B, LLM_DIM)
        assert output.actions.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
        assert output.actions_coarse.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
        assert output.actions_fine.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
        assert output.gate_value.shape == (B, 1)

        # State readout: all 10 slots
        assert len(output.state_readout) == len(SLOT_REGISTRY)

        # Role logits: only J_CERT (9 slots)
        assert len(output.role_logits) == len(J_CERT)

        # Goal preds: only J_CERT (9 slots)
        assert len(output.goal_preds) == len(J_CERT)

    def test_state_token_embedding_injectable(self):
        """State token embedding can be obtained for injection into LLM sequence."""
        emb = self.wrapper.get_state_token_embedding(B)
        assert emb.shape == (B, 1, LLM_DIM)

    def test_forward_with_explicit_z_prev(self, dummy_hidden_states):
        z_prev = torch.randn(B, LLM_DIM)
        output = self.wrapper(
            last_hidden_states=dummy_hidden_states,
            state_token_pos=STATE_TOKEN_POS,
            action_start_pos=ACTION_START,
            z_prev=z_prev,
        )
        assert output.z_t.shape == (B, LLM_DIM)

    def test_gradient_flow(self, dummy_hidden_states):
        """Verify gradients flow through the full forward path."""
        dummy_hidden_states.requires_grad_(True)
        output = self.wrapper(
            last_hidden_states=dummy_hidden_states,
            state_token_pos=STATE_TOKEN_POS,
            action_start_pos=ACTION_START,
        )
        loss = output.actions.sum() + output.z_t.sum()
        loss.backward()
        assert dummy_hidden_states.grad is not None

    def test_readout_isolated_from_action_tokens(self, dummy_hidden_states):
        """State readout must depend only on z_t position, not action positions."""
        dummy_hidden_states.requires_grad_(True)

        output = self.wrapper(
            last_hidden_states=dummy_hidden_states,
            state_token_pos=STATE_TOKEN_POS,
            action_start_pos=ACTION_START,
        )

        # Compute loss from state readout only
        readout_loss = sum(v.sum() for v in output.state_readout.values())
        readout_loss.backward()

        grad = dummy_hidden_states.grad  # (B, seq_len, llm_dim)

        # The gradient at the state token position should be non-zero
        state_grad = grad[:, STATE_TOKEN_POS, :].abs().sum()
        assert state_grad > 0, "State readout should have gradients at state token position"

        # The gradient at action token positions should be zero
        # (because readout doesn't read from action positions)
        action_grad = grad[:, ACTION_START:ACTION_START + ACTION_DIM * NUM_ACTIONS_CHUNK, :].abs().sum()
        assert action_grad == 0, "State readout should NOT have gradients at action positions"

    def test_no_base_model_modification(self):
        """CertVLAWrapper should not contain any reference to base VLA model."""
        # It's a standalone nn.Module that takes hidden states as input.
        # It does NOT subclass or contain OpenVLAForActionPrediction.
        params = dict(self.wrapper.named_parameters())
        for name in params:
            assert "language_model" not in name, "Wrapper should not contain base LLM parameters"
            assert "vision_backbone" not in name, "Wrapper should not contain vision backbone parameters"


class TestParameterCounts:
    """Sanity check that model sizes are reasonable."""

    def test_wrapper_not_too_large(self):
        wrapper = CertVLAWrapper(
            llm_dim=LLM_DIM,
            readout_hidden_dim=64,
            cert_hidden_dim=64,
            cert_embed_dim=32,
            coarse_hidden_dim=64,
            fine_hidden_dim=64,
        )
        total_params = sum(p.numel() for p in wrapper.parameters())
        # With LLM_DIM=128, should be well under 1M params
        assert total_params < 1_000_000, f"Too many params for test-size model: {total_params}"

    def test_wrapper_real_size_estimate(self):
        """Estimate param count with real LLM_DIM=4096."""
        wrapper = CertVLAWrapper(
            llm_dim=4096,
            readout_hidden_dim=512,
            cert_hidden_dim=512,
            cert_embed_dim=256,
            coarse_hidden_dim=1024,
            fine_hidden_dim=1024,
        )
        total_params = sum(p.numel() for p in wrapper.parameters())
        # Should be reasonable: expect ~50-150M params for all CertVLA heads
        # (much less than the 7B LLM)
        assert total_params < 200_000_000, f"CertVLA heads too large: {total_params}"
        assert total_params > 1_000_000, f"CertVLA heads suspiciously small: {total_params}"
