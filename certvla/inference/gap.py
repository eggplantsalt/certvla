"""
Certificate gap computation.

Context doc section 10.2:

Per-slot gap:
    gamma_t^j = p_t^{adv,j} * d_j(hat_g_t^j, hat_s_{t+H}^j)
              + p_t^{pre,j} * d_j(hat_s_t^j,   hat_s_{t+H}^j)

Aggregated gap:
    Gamma_t = [ sum_{j in J_cert} omega_j * kappa_t^j * gamma_t^j ]
            / [ sum_{j in J_cert} omega_j * kappa_t^j + epsilon ]

where:
    p_t^{adv,j}, p_t^{pre,j} = softmax(role_logits)[:, ADVANCE/PRESERVE]
    d_j = per-slot distance (differentiable)
    omega_j = static slot importance weight (default 1.0)
    kappa_t^j = confidence / observability weight
    epsilon = small constant for numerical stability
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from certvla.model.certificate_head import ROLE_ADVANCE, ROLE_PRESERVE
from certvla.slots.schema import SlotDomain, SlotName, SLOT_REGISTRY
from certvla.slots.role_sets import J_CERT


@dataclass
class GapResult:
    """Result of certificate gap computation.

    Attributes:
        per_slot: Per-slot gap values. Dict[SlotName, Tensor(B,)].
        aggregated: Scalar aggregated gap. Tensor(B,).
        role_probs: Softmax role probabilities. Dict[SlotName, Tensor(B, 3)].
    """
    per_slot: Dict[SlotName, torch.Tensor]
    aggregated: torch.Tensor
    role_probs: Dict[SlotName, torch.Tensor]


def _slot_distance(a: torch.Tensor, b: torch.Tensor, meta) -> torch.Tensor:
    """Differentiable per-slot distance. Returns (B,) in [0, 1]."""
    if meta.domain in (SlotDomain.BINARY, SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        return torch.abs(a.view(-1) - b.view(-1))
    elif meta.domain == SlotDomain.CATEGORICAL:
        pa = F.softmax(a, dim=-1)
        pb = F.softmax(b, dim=-1)
        return torch.abs(pa - pb).sum(dim=-1) / 2.0
    raise ValueError(f"Unknown domain {meta.domain}")


def slot_gap(
    role_logits: Dict[SlotName, torch.Tensor],
    state_readout: Dict[SlotName, torch.Tensor],
    goal_preds: Dict[SlotName, torch.Tensor],
    state_readout_tH: Dict[SlotName, torch.Tensor],
) -> Dict[SlotName, torch.Tensor]:
    """Compute per-slot certificate gap gamma_t^j.

    gamma_t^j = p_adv^j * d_j(goal^j, s_{t+H}^j)
              + p_pre^j * d_j(s_t^j,   s_{t+H}^j)

    Args:
        role_logits: Dict[SlotName, Tensor(B, 3)].
        state_readout: Model state prediction at t. Dict[SlotName, Tensor].
        goal_preds: Model goal predictions. Dict[SlotName, Tensor].
        state_readout_tH: Model state prediction at t+H. Dict[SlotName, Tensor].
            In v1, this can be the ground-truth state at t+H or a second
            forward pass result.

    Returns:
        Dict[SlotName, Tensor(B,)] per-slot gap values.
    """
    gaps = {}
    for slot_name in J_CERT:
        if slot_name not in role_logits:
            continue
        meta = SLOT_REGISTRY[slot_name]
        probs = F.softmax(role_logits[slot_name], dim=-1)  # (B, 3)
        p_adv = probs[:, ROLE_ADVANCE]   # (B,)
        p_pre = probs[:, ROLE_PRESERVE]  # (B,)

        gap = torch.zeros_like(p_adv)

        # Advance term: distance between goal prediction and chunk-end state
        if slot_name in goal_preds and slot_name in state_readout_tH:
            d_adv = _slot_distance(goal_preds[slot_name], state_readout_tH[slot_name], meta)
            gap = gap + p_adv * d_adv

        # Preserve term: distance between current state and chunk-end state
        if slot_name in state_readout and slot_name in state_readout_tH:
            d_pre = _slot_distance(state_readout[slot_name], state_readout_tH[slot_name], meta)
            gap = gap + p_pre * d_pre

        gaps[slot_name] = gap
    return gaps


def aggregate_certificate_gap(
    per_slot_gaps: Dict[SlotName, torch.Tensor],
    role_logits: Dict[SlotName, torch.Tensor],
    slot_weights: Optional[Dict[SlotName, float]] = None,
    confidence_weights: Optional[Dict[SlotName, torch.Tensor]] = None,
    epsilon: float = 1e-8,
) -> GapResult:
    """Aggregate per-slot gaps into a scalar certificate gap Gamma_t.

    Gamma_t = [ sum_j omega_j * kappa_j * gamma_j ]
            / [ sum_j omega_j * kappa_j + epsilon ]

    Args:
        per_slot_gaps: Dict[SlotName, Tensor(B,)] from slot_gap().
        role_logits: Dict[SlotName, Tensor(B, 3)] for role_probs output.
        slot_weights: Static per-slot importance omega_j. Default 1.0 each.
        confidence_weights: Dynamic per-slot kappa_j. Dict[SlotName, Tensor(B,)].
            Default 1.0 each.
        epsilon: Numerical stability constant.

    Returns:
        GapResult with per_slot, aggregated, and role_probs.
    """
    if not per_slot_gaps:
        dummy = torch.tensor([0.0])
        return GapResult(
            per_slot={},
            aggregated=dummy,
            role_probs={},
        )

    # Determine device and batch size from first entry
    first_gap = next(iter(per_slot_gaps.values()))
    dev = first_gap.device
    B = first_gap.shape[0]

    numerator = torch.zeros(B, device=dev)
    denominator = torch.zeros(B, device=dev)

    for slot_name, gap in per_slot_gaps.items():
        omega = slot_weights.get(slot_name, 1.0) if slot_weights else 1.0
        if confidence_weights and slot_name in confidence_weights:
            kappa = confidence_weights[slot_name]  # (B,)
        else:
            kappa = torch.ones(B, device=dev)
        numerator = numerator + omega * kappa * gap
        denominator = denominator + omega * kappa

    aggregated = numerator / (denominator + epsilon)

    role_probs = {}
    for slot_name in per_slot_gaps:
        if slot_name in role_logits:
            role_probs[slot_name] = F.softmax(role_logits[slot_name], dim=-1)

    return GapResult(
        per_slot=per_slot_gaps,
        aggregated=aggregated,
        role_probs=role_probs,
    )
