"""
Persistent state token and gated update module.

Per context doc section 4.1:
    z_t = Phi(z_{t-1}, o_t, l)

Gated update:
    tilde_z_t = f_theta(o_t, l, z_{t-1})   [here: the LLM hidden state at the state token position]
    g_t = sigma(W_g [tilde_z_t ; z_{t-1}] + b_g)
    z_t = g_t * tilde_z_t + (1 - g_t) * z_{t-1}

v1 decision: learnable z_0 for every sample (no episode-level recurrence at training time).
"""

import torch
import torch.nn as nn


class StateTokenModule(nn.Module):
    """Persistent state token with gated update.

    This module:
    1. Provides a learnable initial state embedding z_0
    2. After the LLM forward pass, applies a gated update to produce z_t
       from the LLM's hidden state at the state token position.

    The state token embedding is injected into the LLM input sequence.
    The LLM processes it together with vision+text tokens. The hidden
    state at the state token position (tilde_z_t) is then combined with
    the previous state (z_{t-1}) via a sigmoid gate.

    At training time (v1): z_{t-1} = z_0 for every sample (no recurrence).
    """

    def __init__(self, llm_dim: int):
        """
        Args:
            llm_dim: LLM hidden dimension (4096 for Llama 2 7B).
        """
        super().__init__()
        self.llm_dim = llm_dim

        # Learnable initial state embedding z_0
        # Initialized as zeros; the model learns what a "blank" task state looks like
        self.z_0 = nn.Parameter(torch.zeros(1, llm_dim))
        nn.init.normal_(self.z_0, mean=0.0, std=0.02)

        # Gating network: takes [tilde_z_t ; z_{t-1}] -> scalar gate
        self.gate_proj = nn.Linear(2 * llm_dim, llm_dim)
        # Initialize gate bias to positive values so initial gate is ~0.5
        # (balanced mix of new and old state at start of training)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def get_initial_state(self, batch_size: int) -> torch.Tensor:
        """Get initial state z_0, expanded for batch.

        Returns:
            z_0: Shape (B, llm_dim).
        """
        return self.z_0.expand(batch_size, -1)

    def get_state_token_embedding(self, batch_size: int) -> torch.Tensor:
        """Get the state token embedding to inject into the LLM sequence.

        This is the embedding that goes into the LLM as an additional input token.
        At v1: this is just z_0 (no recurrence from previous chunks).

        Returns:
            State token embedding: Shape (B, 1, llm_dim).
        """
        return self.z_0.expand(batch_size, -1).unsqueeze(1)

    def gated_update(
        self,
        tilde_z_t: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> tuple:
        """Apply gated update: z_t = g * tilde_z + (1-g) * z_prev.

        Args:
            tilde_z_t: LLM hidden state at state token position. Shape (B, llm_dim).
            z_prev: Previous state (at v1 training: z_0). Shape (B, llm_dim).

        Returns:
            z_t: Updated state. Shape (B, llm_dim).
            gate: Gate values for diagnostics. Shape (B, llm_dim).
        """
        concat = torch.cat([tilde_z_t, z_prev], dim=-1)  # (B, 2*llm_dim)
        gate = torch.sigmoid(self.gate_proj(concat))       # (B, llm_dim)
        z_t = gate * tilde_z_t + (1.0 - gate) * z_prev     # (B, llm_dim)
        return z_t, gate
