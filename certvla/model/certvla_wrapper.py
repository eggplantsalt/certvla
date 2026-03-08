"""
CertVLA wrapper: composes state token, readout, certificate, and action heads
around the base OpenVLA/OFT model.

This is an add-on module (NOT a subclass of OpenVLAForActionPrediction).
It wraps around the base VLA forward pass and adds CertVLA heads on top.

Design principle: minimal invasion. The base VLA is unchanged. CertVLA injects
a state token into the LLM sequence and reads out from the state token position
in the last hidden states.

Phase 2 scope: forward path only. No losses, no training loop, no inference repair.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from certvla.model.state_token import StateTokenModule
from certvla.model.state_readout import StateReadoutHead
from certvla.model.certificate_head import CertificateHead
from certvla.model.action_head import CertActionHead, _DEFAULT_ACTION_DIM, _DEFAULT_NUM_ACTIONS_CHUNK
from certvla.model.outputs import CertVLAOutput
from certvla.slots.schema import SlotName


class CertVLAWrapper(nn.Module):
    """Wrapper that adds CertVLA heads around a base VLA's LLM hidden states.

    This module does NOT own or modify the base VLA model. Instead, it:
    1. Provides a state token embedding to be injected into the LLM sequence
    2. After the LLM forward pass, takes the full last_hidden_states and:
       a. Extracts the hidden state at the state token position -> tilde_z_t
       b. Applies gated update -> z_t
       c. Runs state readout: z_t -> s_t
       d. Runs certificate head: z_t -> (role_logits, goal_preds)
       e. Extracts actions_hidden_states from the action token positions
       f. Runs cert action head: (z_t, cert, actions_hidden_states) -> actions

    Integration with `modeling_prismatic.py` (Phase 3):
    - The state token embedding is appended to projected_patch_embeddings
      before _build_multimodal_attention()
    - The state token position in the sequence = NUM_PATCHES (after all vision tokens,
      before text tokens)

    Phase 2: this wrapper operates on pre-extracted hidden states.
    Phase 3: will integrate with the full VLA forward pass.
    """

    def __init__(
        self,
        llm_dim: int = 4096,
        readout_hidden_dim: int = 512,
        cert_hidden_dim: int = 512,
        cert_embed_dim: int = 256,
        coarse_hidden_dim: int = 1024,
        fine_hidden_dim: int = 1024,
        action_dim: int = _DEFAULT_ACTION_DIM,
        num_actions_chunk: int = _DEFAULT_NUM_ACTIONS_CHUNK,
        lambda_res_init: float = 0.1,
    ):
        """
        Args:
            llm_dim: LLM hidden dimension (4096 for Llama 2 7B).
            readout_hidden_dim: State readout shared trunk hidden dim.
            cert_hidden_dim: Certificate head shared trunk hidden dim.
            cert_embed_dim: Certificate embedding dim for action conditioning.
            coarse_hidden_dim: Coarse action branch hidden dim.
            fine_hidden_dim: Fine action branch hidden dim.
            action_dim: Action space dimension.
            num_actions_chunk: Number of actions per chunk.
            lambda_res_init: Initial residual scaling factor.
        """
        super().__init__()
        self.llm_dim = llm_dim
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk

        # State token (provides z_0 and gated update)
        self.state_token = StateTokenModule(llm_dim)

        # State readout head: z_t -> per-slot predictions
        self.state_readout = StateReadoutHead(llm_dim, readout_hidden_dim)

        # Certificate head: z_t -> (role_logits, goal_preds)
        self.certificate_head = CertificateHead(llm_dim, cert_hidden_dim)

        # Certificate-conditioned action head
        self.action_head = CertActionHead(
            llm_dim=llm_dim,
            cert_embed_dim=cert_embed_dim,
            coarse_hidden_dim=coarse_hidden_dim,
            fine_hidden_dim=fine_hidden_dim,
            action_dim=action_dim,
            num_actions_chunk=num_actions_chunk,
            lambda_res_init=lambda_res_init,
        )

    def get_state_token_embedding(self, batch_size: int) -> torch.Tensor:
        """Get state token embedding to inject into LLM sequence.

        Returns:
            State token embedding. Shape (B, 1, llm_dim).
        """
        return self.state_token.get_state_token_embedding(batch_size)

    def forward(
        self,
        last_hidden_states: torch.Tensor,
        state_token_pos: int,
        action_start_pos: int,
        z_prev: Optional[torch.Tensor] = None,
    ) -> CertVLAOutput:
        """CertVLA forward pass on LLM hidden states.

        This processes the hidden states AFTER the base VLA's LLM forward pass.

        Args:
            last_hidden_states: Full LLM last hidden states.
                Shape (B, seq_len, llm_dim).
            state_token_pos: Position index of the state token in the sequence.
                The state token is inserted after vision patches.
            action_start_pos: Position index where action tokens start.
                Used to extract actions_hidden_states.
            z_prev: Previous state for gated update. Shape (B, llm_dim).
                If None, uses z_0 (default for v1 training).

        Returns:
            CertVLAOutput with all head outputs.
        """
        B = last_hidden_states.shape[0]

        # --- Step 1: Extract tilde_z_t from state token position ---
        tilde_z_t = last_hidden_states[:, state_token_pos, :]  # (B, llm_dim)

        # --- Step 2: Gated update ---
        if z_prev is None:
            z_prev = self.state_token.get_initial_state(B).to(
                device=last_hidden_states.device,
                dtype=last_hidden_states.dtype,
            )
        z_t, gate = self.state_token.gated_update(tilde_z_t, z_prev)  # (B, llm_dim)

        # --- Step 3: State readout (only from z_t) ---
        state_readout = self.state_readout(z_t)

        # --- Step 4: Certificate head (only from z_t) ---
        role_logits, goal_preds = self.certificate_head(z_t)

        # --- Step 5: Extract action hidden states ---
        action_end_pos = action_start_pos + self.action_dim * self.num_actions_chunk
        actions_hidden_states = last_hidden_states[
            :, action_start_pos:action_end_pos, :
        ]  # (B, chunk*action_dim, llm_dim)

        # --- Step 6: Cert-conditioned action head ---
        actions, actions_coarse, actions_fine = self.action_head(
            z_t=z_t,
            role_logits=role_logits,
            goal_preds=goal_preds,
            actions_hidden_states=actions_hidden_states,
        )

        # --- Pack output ---
        return CertVLAOutput(
            z_t=z_t,
            state_readout=state_readout,
            role_logits=role_logits,
            goal_preds=goal_preds,
            actions_coarse=actions_coarse,
            actions_fine=actions_fine,
            actions=actions,
            actions_hidden_states=actions_hidden_states,
            gate_value=gate.mean(dim=-1, keepdim=True),  # (B, 1) mean gate for diagnostics
        )
