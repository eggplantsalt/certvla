"""
Certificate head: Q_psi(z_t) -> c_t.

Per context doc section 4.3:
    hat_c_t = Q_psi(z_t)
    c_t = {(u_t^j, g_t^j)} for j in J_cert

where:
    u_t^j in {advance, preserve, ignore}  (role classification)
    g_t^j = predicted chunk-end value if u_t^j = advance

CRITICAL: Certificate head operates ONLY on z_t, same constraint as state readout.

Output:
    role_logits: Dict[SlotName, Tensor(B, 3)] for each cert slot
    goal_preds: Dict[SlotName, Tensor(B, dim)] for each cert slot
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from certvla.slots.schema import SlotDomain, SlotMeta, SlotName, SLOT_REGISTRY
from certvla.slots.role_sets import J_CERT

# Role indices
ROLE_ADVANCE = 0
ROLE_PRESERVE = 1
ROLE_IGNORE = 2
NUM_ROLES = 3


class CertificateHead(nn.Module):
    """Certificate head: z_t -> per-slot (role_logits, goal_prediction).

    Architecture:
    - Shared trunk processes z_t
    - Per-slot role classifier: (B, 3) logits for [advance, preserve, ignore]
    - Per-slot goal predictor: predicts chunk-end target value for advance slots.
      Output dimensions match the slot domain (same as state readout).

    Only operates on cert slots (J_CERT = J_E | J_R, 9 slots).
    """

    def __init__(self, llm_dim: int, hidden_dim: int = 512):
        """
        Args:
            llm_dim: Input dimension (LLM hidden dim).
            hidden_dim: Shared hidden layer dimension.
        """
        super().__init__()
        self.llm_dim = llm_dim

        # Shared trunk
        self.shared = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Per-slot role classifiers (advance / preserve / ignore)
        self.role_heads = nn.ModuleDict()
        # Per-slot goal predictors (chunk-end target value for advance slots)
        self.goal_heads = nn.ModuleDict()

        for slot_name in J_CERT:
            meta = SLOT_REGISTRY[slot_name]
            # Role head: 3-way classification
            self.role_heads[slot_name.value] = nn.Linear(hidden_dim, NUM_ROLES)
            # Goal head: domain-dependent output
            goal_dim = self._goal_output_dim(meta)
            self.goal_heads[slot_name.value] = nn.Linear(hidden_dim, goal_dim)

    @staticmethod
    def _goal_output_dim(meta: SlotMeta) -> int:
        if meta.domain == SlotDomain.BINARY:
            return 1
        elif meta.domain == SlotDomain.CATEGORICAL:
            return len(meta.categories)
        elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
            return 1
        raise ValueError(f"Unknown domain {meta.domain}")

    def forward(
        self, z_t: torch.Tensor
    ) -> Tuple[Dict[SlotName, torch.Tensor], Dict[SlotName, torch.Tensor]]:
        """
        Args:
            z_t: State token hidden state. Shape (B, llm_dim).
                 MUST be the single z_t vector.

        Returns:
            role_logits: Dict[SlotName, Tensor(B, 3)]. Raw logits for
                [advance, preserve, ignore]. Softmax applied at loss time.
            goal_preds: Dict[SlotName, Tensor(B, dim)]. Predicted chunk-end
                target values. Activation applied per domain:
                - Binary: sigmoid -> (B, 1)
                - Categorical: raw logits -> (B, num_categories)
                - Continuous: sigmoid -> (B, 1)
        """
        h = self.shared(z_t)  # (B, hidden_dim)

        role_logits = {}
        goal_preds = {}

        for slot_name in J_CERT:
            meta = SLOT_REGISTRY[slot_name]

            # Role classification
            role_logits[slot_name] = self.role_heads[slot_name.value](h)  # (B, 3)

            # Goal prediction (with domain-appropriate activation)
            raw_goal = self.goal_heads[slot_name.value](h)
            if meta.domain == SlotDomain.BINARY:
                goal_preds[slot_name] = torch.sigmoid(raw_goal)
            elif meta.domain == SlotDomain.CATEGORICAL:
                goal_preds[slot_name] = raw_goal  # raw logits
            elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
                goal_preds[slot_name] = torch.sigmoid(raw_goal)
            else:
                raise ValueError(f"Unknown domain {meta.domain}")

        return role_logits, goal_preds
