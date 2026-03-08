"""
Forward output dataclass for CertVLA.

Defines CertVLAOutput that bundles all outputs from a CertVLA forward pass.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from certvla.slots.schema import SlotName


@dataclass
class CertVLAOutput:
    """Outputs from a CertVLA forward pass.

    All tensors are on the same device and dtype as the model.

    Attributes:
        z_t: Persistent state token hidden state. Shape: (B, llm_dim).
        state_readout: Per-slot state predictions. Dict[SlotName, Tensor].
            - Binary/continuous: (B, 1)
            - Categorical: (B, num_categories)
        role_logits: Per-slot role logits. Dict[SlotName, Tensor(B, 3)].
            Columns: [advance, preserve, ignore].
        goal_preds: Per-slot advance goal predictions. Dict[SlotName, Tensor].
            Same shapes as state_readout. Only meaningful for advance slots.
        actions_coarse: Coarse action predictions from cert-conditioned branch.
            Shape: (B, NUM_ACTIONS_CHUNK, ACTION_DIM).
        actions_fine: Fine residual from observation-conditioned branch.
            Shape: (B, NUM_ACTIONS_CHUNK, ACTION_DIM).
        actions: Final combined actions (coarse + lambda_res * fine).
            Shape: (B, NUM_ACTIONS_CHUNK, ACTION_DIM).
        actions_hidden_states: Raw action hidden states from LLM.
            Shape: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, llm_dim).
        gate_value: Gating value from state token update. Shape: (B, 1). For diagnostics.
    """

    z_t: torch.Tensor
    state_readout: Dict[SlotName, torch.Tensor]
    role_logits: Dict[SlotName, torch.Tensor]
    goal_preds: Dict[SlotName, torch.Tensor]
    actions_coarse: torch.Tensor
    actions_fine: torch.Tensor
    actions: torch.Tensor
    actions_hidden_states: torch.Tensor
    gate_value: Optional[torch.Tensor] = None
