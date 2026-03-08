"""
CertVLA loss functions (Phase 3).

Context doc section 9 defines 7 loss terms:
    L_state : State readout (per-slot, masked, confidence-weighted)
    L_role  : Certificate role classification (focal CE)
    L_goal  : Advance goal value prediction
    L_act   : Action chunk regression (L1)
    L_cons  : Structural consistency (advance + preserve)
    L_dep   : Certificate dependence (margin / triplet)
    L_cf    : Counterfactual (invariance + breaking, minimal v1)

Ground-truth tensor conventions (produced by data pipeline):
    state_target / goal_target -- Dict[SlotName, Tensor]:
        Binary/Continuous/Confidence : (B, 1) float
        Categorical                  : (B,)   long  (class index)
    role_target  -- Dict[SlotName, Tensor(B,)] long  {0=advance, 1=preserve, 2=ignore}
    mask         -- Dict[SlotName, Tensor(B,)] float {0.0 or 1.0}
    confidence   -- Dict[SlotName, Tensor(B,)] float [0, 1]
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from certvla.model.certificate_head import ROLE_ADVANCE, ROLE_PRESERVE, ROLE_IGNORE
from certvla.slots.schema import SlotDomain, SlotName, SLOT_REGISTRY
from certvla.slots.role_sets import J_CERT


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Focal cross-entropy loss.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Args:
        logits: (B, C) raw logits.
        targets: (B,) class indices (long).
        gamma: Focusing parameter.
        reduction: "none" -> (B,), "mean" -> scalar.
    """
    ce = F.cross_entropy(logits, targets, reduction="none")  # (B,)
    pt = torch.exp(-ce)
    focal = ((1.0 - pt) ** gamma) * ce
    if reduction == "mean":
        return focal.mean()
    return focal


def _per_slot_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    meta,
) -> torch.Tensor:
    """Per-sample loss for one slot.  Returns (B,) tensor.

    Args:
        pred -- model output (sigmoid-activated for binary/cont, raw logits for cat).
        target -- ground truth (float for binary/cont, long class index for cat).
    """
    if meta.domain == SlotDomain.BINARY:
        return F.binary_cross_entropy(
            pred.view(-1), target.view(-1).float(), reduction="none"
        )
    elif meta.domain == SlotDomain.CATEGORICAL:
        tgt = target.long()
        if tgt.dim() > 1:
            tgt = tgt.squeeze(-1)
        return F.cross_entropy(pred, tgt, reduction="none")
    elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        return F.l1_loss(
            pred.view(-1), target.view(-1).float(), reduction="none"
        )
    raise ValueError(f"Unknown domain {meta.domain}")


def _slot_pred_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    meta,
) -> torch.Tensor:
    """Differentiable distance between two model-format predictions.

    Returns (B,) tensor in [0, 1].
    Used by consistency loss when both sides are model outputs (v2+).
    """
    if meta.domain in (SlotDomain.BINARY, SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        return torch.abs(a.view(-1) - b.view(-1))
    elif meta.domain == SlotDomain.CATEGORICAL:
        pa = F.softmax(a, dim=-1)
        pb = F.softmax(b, dim=-1)
        return torch.abs(pa - pb).sum(dim=-1) / 2.0
    raise ValueError(f"Unknown domain {meta.domain}")


def _zero(device: Optional[torch.device] = None) -> torch.Tensor:
    """Zero scalar on *device* with requires_grad for safe backward."""
    t = torch.tensor(0.0, device=device)
    t.requires_grad_(True)
    return t


# ──────────────────────────────────────────────────────────────
# L_state  (Section 9.1)
# ──────────────────────────────────────────────────────────────

def cert_state_loss(
    state_readout: Dict[SlotName, torch.Tensor],
    state_target: Dict[SlotName, torch.Tensor],
    mask: Dict[SlotName, torch.Tensor],
    confidence: Dict[SlotName, torch.Tensor],
) -> torch.Tensor:
    """L_state = sum_j  m^j * alpha^j * ell_j(hat_s^j, s^j).

    Iterates over **all** slots present in both readout and target.
    """
    dev = next(iter(state_readout.values())).device
    losses = []
    for slot_name in SlotName:
        if slot_name not in state_readout or slot_name not in state_target:
            continue
        meta = SLOT_REGISTRY[slot_name]
        pred = state_readout[slot_name]
        tgt = state_target[slot_name]
        m = mask.get(slot_name, torch.ones(pred.shape[0], device=dev))
        alpha = confidence.get(slot_name, torch.ones(pred.shape[0], device=dev))
        sl = _per_slot_loss(pred, tgt, meta)       # (B,)
        losses.append((m * alpha * sl).mean())
    if not losses:
        return _zero(dev)
    return torch.stack(losses).sum()


# ──────────────────────────────────────────────────────────────
# L_role  (Section 9.2)
# ──────────────────────────────────────────────────────────────

def cert_role_loss(
    role_logits: Dict[SlotName, torch.Tensor],
    role_target: Dict[SlotName, torch.Tensor],
    mask: Dict[SlotName, torch.Tensor],
    confidence: Dict[SlotName, torch.Tensor],
    gamma: float = 2.0,
) -> torch.Tensor:
    """L_role = sum_{j in J_cert}  m^j * alpha^j * FocalCE(hat_u^j, u^j).

    Only iterates over J_CERT (9 cert slots).
    """
    dev = next(iter(role_logits.values())).device
    losses = []
    for slot_name in J_CERT:
        if slot_name not in role_logits or slot_name not in role_target:
            continue
        logits = role_logits[slot_name]               # (B, 3)
        tgt = role_target[slot_name].long()            # (B,)
        m = mask.get(slot_name, torch.ones(logits.shape[0], device=dev))
        alpha = confidence.get(slot_name, torch.ones(logits.shape[0], device=dev))
        fl = focal_cross_entropy(logits, tgt, gamma=gamma)  # (B,)
        losses.append((m * alpha * fl).mean())
    if not losses:
        return _zero(dev)
    return torch.stack(losses).sum()


# ──────────────────────────────────────────────────────────────
# L_goal  (Section 9.3)
# ──────────────────────────────────────────────────────────────

def cert_goal_loss(
    goal_preds: Dict[SlotName, torch.Tensor],
    goal_target: Dict[SlotName, torch.Tensor],
    role_target: Dict[SlotName, torch.Tensor],
    mask: Dict[SlotName, torch.Tensor],
    confidence: Dict[SlotName, torch.Tensor],
) -> torch.Tensor:
    """L_goal = sum_{j: u^j=advance} m^j * alpha^j * ell_j(hat_g^j, s_{t+H}^j).

    Only accumulates loss for samples where role == ROLE_ADVANCE.
    """
    dev = next(iter(goal_preds.values())).device
    losses = []
    for slot_name in J_CERT:
        if slot_name not in goal_preds or slot_name not in goal_target:
            continue
        if slot_name not in role_target:
            continue
        meta = SLOT_REGISTRY[slot_name]
        pred = goal_preds[slot_name]
        tgt = goal_target[slot_name]
        roles = role_target[slot_name].long()                     # (B,)
        m = mask.get(slot_name, torch.ones(pred.shape[0], device=dev))
        alpha = confidence.get(slot_name, torch.ones(pred.shape[0], device=dev))
        advance_mask = (roles == ROLE_ADVANCE).float()            # (B,)
        sl = _per_slot_loss(pred, tgt, meta)                      # (B,)
        losses.append((m * alpha * advance_mask * sl).mean())
    if not losses:
        return _zero(dev)
    return torch.stack(losses).sum()


# ──────────────────────────────────────────────────────────────
# L_act  (Section 9.4)
# ──────────────────────────────────────────────────────────────

def cert_action_loss(
    pred_actions: torch.Tensor,
    expert_actions: torch.Tensor,
) -> torch.Tensor:
    """L_act = (1/H) sum_k ||hat_a_{t+k} - a*_{t+k}||_1.

    Args:
        pred_actions:   (B, H, action_dim).
        expert_actions: (B, H, action_dim).
    """
    return F.l1_loss(pred_actions, expert_actions, reduction="mean")


# ──────────────────────────────────────────────────────────────
# L_cons  (Section 9.5)
# ──────────────────────────────────────────────────────────────

def cert_consistency_loss(
    state_readout: Dict[SlotName, torch.Tensor],
    goal_preds: Dict[SlotName, torch.Tensor],
    role_target: Dict[SlotName, torch.Tensor],
    state_target_tH: Dict[SlotName, torch.Tensor],
    lambda_pre: float = 1.0,
) -> torch.Tensor:
    """L_cons = L_adv_cons + lambda_pre * L_pre_cons.

    Advance consistency : ell_j(goal_preds^j, state_target_tH^j)
        for slots where role == advance.
    Preserve consistency: ell_j(state_readout^j, state_target_tH^j)
        for slots where role == preserve.

    In v1 single-chunk training, state_target_tH is the ground-truth state
    at t+H.  In future v2 with episode-sequential training it can be replaced
    by the model's own prediction at t+H.

    Args:
        state_readout: Model state predictions at t.
        goal_preds: Model goal predictions.
        role_target: Ground-truth roles per cert slot.
        state_target_tH: State values at chunk end (GT or model-predicted).
            Same tensor format as state_target (class index for categorical).
        lambda_pre: Weight for preserve term.
    """
    dev = next(iter(state_readout.values())).device
    adv_losses = []
    pre_losses = []

    for slot_name in J_CERT:
        if slot_name not in role_target:
            continue
        meta = SLOT_REGISTRY[slot_name]
        roles = role_target[slot_name].long()  # (B,)

        # Advance: goal prediction should match chunk-end state
        if slot_name in goal_preds and slot_name in state_target_tH:
            adv_mask = (roles == ROLE_ADVANCE).float()
            sl = _per_slot_loss(goal_preds[slot_name], state_target_tH[slot_name], meta)
            adv_losses.append((adv_mask * sl).mean())

        # Preserve: current state readout should match chunk-end state
        if slot_name in state_readout and slot_name in state_target_tH:
            pre_mask = (roles == ROLE_PRESERVE).float()
            sl = _per_slot_loss(state_readout[slot_name], state_target_tH[slot_name], meta)
            pre_losses.append((pre_mask * sl).mean())

    l_adv = torch.stack(adv_losses).sum() if adv_losses else torch.tensor(0.0, device=dev)
    l_pre = torch.stack(pre_losses).sum() if pre_losses else torch.tensor(0.0, device=dev)
    total = l_adv + lambda_pre * l_pre
    if not adv_losses and not pre_losses:
        return _zero(dev)
    return total


# ──────────────────────────────────────────────────────────────
# L_dep  (Section 9.6)
# ──────────────────────────────────────────────────────────────

def cert_dependence_loss(
    expert_actions: torch.Tensor,
    actions_pos: torch.Tensor,
    actions_neg: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """L_dep = mean max(0,  margin + e_pos - e_neg).

    Forces the action head to produce smaller error with the correct
    certificate than with a corrupted certificate.

    Args:
        expert_actions: (B, H, action_dim).
        actions_pos: Actions under correct certificate.  (B, H, action_dim).
        actions_neg: Actions under corrupted certificate. (B, H, action_dim).
        margin: Minimum gap between positive and negative error.
    """
    e_pos = F.l1_loss(actions_pos, expert_actions, reduction="none").mean(dim=(1, 2))  # (B,)
    e_neg = F.l1_loss(actions_neg, expert_actions, reduction="none").mean(dim=(1, 2))  # (B,)
    triplet = torch.clamp(margin + e_pos - e_neg, min=0.0)
    return triplet.mean()


# ──────────────────────────────────────────────────────────────
# L_cf  (Section 9.7 -- interface + minimal v1 implementation)
# ──────────────────────────────────────────────────────────────

def cert_counterfactual_loss(
    z_t: torch.Tensor,
    z_pos: Optional[torch.Tensor] = None,
    z_neg: Optional[torch.Tensor] = None,
    state_readout_pos: Optional[Dict[SlotName, torch.Tensor]] = None,
    state_readout_neg: Optional[Dict[SlotName, torch.Tensor]] = None,
    state_target: Optional[Dict[SlotName, torch.Tensor]] = None,
    state_target_neg: Optional[Dict[SlotName, torch.Tensor]] = None,
    mask: Optional[Dict[SlotName, torch.Tensor]] = None,
    confidence: Optional[Dict[SlotName, torch.Tensor]] = None,
    mu: float = 1.0,
) -> torch.Tensor:
    """L_cf = L_inv + L_brk.

    L_inv (nuisance-preserving): z_t should be close to z_pos
        (nuisance-only augmentation: background, lighting, distractors).
    L_brk (consequence-breaking): z_t should differ from z_neg
        (task-relevant change: target identity, goal receptacle, etc.).

    v1: operates on z_t embeddings.  State/cert consistency terms are
    included when the optional dicts are provided.  Full augmented-pair
    generation is deferred to Phase 4.

    Returns 0 if no augmented pairs are supplied (graceful no-op).
    """
    dev = z_t.device
    loss = torch.tensor(0.0, device=dev)

    # L_inv: nuisance-preserving invariance
    if z_pos is not None:
        l_inv = F.mse_loss(z_t, z_pos.detach())
        if (state_readout_pos is not None and state_target is not None
                and mask is not None and confidence is not None):
            l_inv = l_inv + cert_state_loss(
                state_readout_pos, state_target, mask, confidence,
            )
        loss = loss + l_inv

    # L_brk: consequence-breaking divergence
    if z_neg is not None:
        dist_sq = (z_t - z_neg).pow(2).sum(dim=-1)                 # (B,)
        l_brk = torch.clamp(mu - dist_sq, min=0.0).mean()
        if (state_readout_neg is not None and state_target_neg is not None
                and mask is not None and confidence is not None):
            l_brk = l_brk + cert_state_loss(
                state_readout_neg, state_target_neg, mask, confidence,
            )
        loss = loss + l_brk

    if z_pos is None and z_neg is None:
        return _zero(dev)
    return loss


# ──────────────────────────────────────────────────────────────
# Total loss combiner  (Section 9.8)
# ──────────────────────────────────────────────────────────────

def cert_total_loss(
    losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted sum of individual loss terms.

    Args:
        losses: {name -> scalar tensor}.
            Recognised names: state, role, goal, action,
            consistency, dependence, counterfactual.
        weights: {weight_key -> float}.
            Keys: lambda_s, lambda_r, lambda_g, lambda_a,
            lambda_c, lambda_d, lambda_cf.

    Returns:
        total: Scalar weighted loss (gradient-bearing).
        components: {name -> weighted float value} (detached, for logging).
    """
    _KEY_MAP = {
        "state": "lambda_s",
        "role": "lambda_r",
        "goal": "lambda_g",
        "action": "lambda_a",
        "consistency": "lambda_c",
        "dependence": "lambda_d",
        "counterfactual": "lambda_cf",
    }
    dev = None
    for v in losses.values():
        dev = v.device
        break

    total = torch.tensor(0.0, device=dev)
    components: Dict[str, float] = {}
    for name, loss_val in losses.items():
        w_key = _KEY_MAP.get(name, f"lambda_{name}")
        w = weights.get(w_key, 0.0)
        weighted = w * loss_val
        if w > 0.0:
            total = total + weighted
        components[name] = weighted.detach().item()
    return total, components
