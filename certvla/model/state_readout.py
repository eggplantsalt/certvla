"""
State readout head: R_phi(z_t) -> s_t.

Per context doc section 4.2:
    hat_s_t = R_phi(z_t)

CRITICAL: Readout operates ONLY on z_t (the hidden state at the state token
position), NOT on the full LLM sequence. This prevents the readout from
bypassing z_t via direct access to vision tokens.

Output: per-slot predictions with domain-appropriate activations.
"""

import torch
import torch.nn as nn
from typing import Dict

from certvla.slots.schema import SlotDomain, SlotName, SlotMeta, SLOT_REGISTRY


class StateReadoutHead(nn.Module):
    """State readout: z_t -> per-slot predictions.

    Takes ONLY the z_t vector (shape [B, llm_dim]) and produces per-slot
    predictions with appropriate output dimensions:
    - Binary slots: (B, 1) with sigmoid
    - Categorical slots: (B, num_categories) with raw logits (softmax applied at loss)
    - Continuous/Confidence slots: (B, 1) with sigmoid to clamp to [0, 1]

    Architecture: shared hidden layer -> per-slot output heads.
    """

    def __init__(self, llm_dim: int, hidden_dim: int = 512):
        """
        Args:
            llm_dim: Input dimension (LLM hidden dim, e.g. 4096).
            hidden_dim: Shared hidden layer dimension.
        """
        super().__init__()
        self.llm_dim = llm_dim

        # Shared trunk: project z_t to a compact representation
        self.shared = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Per-slot output heads
        self.slot_heads = nn.ModuleDict()
        for slot_name, meta in SLOT_REGISTRY.items():
            out_dim = self._output_dim(meta)
            self.slot_heads[slot_name.value] = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def _output_dim(meta: SlotMeta) -> int:
        if meta.domain == SlotDomain.BINARY:
            return 1
        elif meta.domain == SlotDomain.CATEGORICAL:
            return len(meta.categories)
        elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
            return 1
        raise ValueError(f"Unknown domain {meta.domain}")

    def forward(self, z_t: torch.Tensor) -> Dict[SlotName, torch.Tensor]:
        """
        Args:
            z_t: State token hidden state. Shape (B, llm_dim).
                 MUST be the single z_t vector, not the full sequence.

        Returns:
            Dict[SlotName, Tensor]: Per-slot predictions.
                - Binary: (B, 1) after sigmoid
                - Categorical: (B, num_categories) raw logits
                - Continuous/Confidence: (B, 1) after sigmoid
        """
        h = self.shared(z_t)  # (B, hidden_dim)

        outputs = {}
        for slot_name, meta in SLOT_REGISTRY.items():
            raw = self.slot_heads[slot_name.value](h)  # (B, out_dim)
            if meta.domain == SlotDomain.BINARY:
                outputs[slot_name] = torch.sigmoid(raw)
            elif meta.domain == SlotDomain.CATEGORICAL:
                outputs[slot_name] = raw  # raw logits; softmax at loss
            elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
                outputs[slot_name] = torch.sigmoid(raw)
            else:
                raise ValueError(f"Unknown domain {meta.domain}")

        return outputs
