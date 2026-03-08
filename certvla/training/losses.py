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

# ========================================================================
# 模块概述 (Module Overview)
# ========================================================================
# 本模块是 CertVLA 训练流水线的核心损失函数集合，定义了 7 个损失项。
# 这些损失项在训练的不同阶段 (Stage 1-4) 被逐步激活，通过课程学习
# (curriculum learning) 的方式引导模型逐步学习：
#   Stage 1: L_state        —— 先学会从隐状态 z_t 读出环境状态
#   Stage 2: + L_role, L_goal —— 再学会预测槽位角色和目标值
#   Stage 3: + L_act, L_cons, L_dep —— 最后学会基于证书生成动作
#   Stage 4: + L_cf          —— 引入反事实增强提升鲁棒性
#
# 设计哲学：
#   - 每个损失项都按 "逐槽位 (per-slot)" 计算，然后聚合
#   - 掩码 (mask) 控制哪些槽位参与训练（处理缺失数据）
#   - 置信度 (confidence) 对每个槽位的损失进行加权（不确定的标注给予较低权重）
#   - 这种 m^j * alpha^j * ell_j 的模式贯穿所有损失项
# ========================================================================
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# ROLE_ADVANCE=0, ROLE_PRESERVE=1, ROLE_IGNORE=2 -- 三种槽位角色的整数编码
from certvla.model.certificate_head import ROLE_ADVANCE, ROLE_PRESERVE, ROLE_IGNORE
# SlotDomain: 槽位数据类型枚举 (BINARY / CATEGORICAL / CONTINUOUS / CONFIDENCE)
# SLOT_REGISTRY: 全局槽位元信息注册表，记录每个槽位的 domain、维度等
from certvla.slots.schema import SlotDomain, SlotName, SLOT_REGISTRY
# J_CERT: 参与证书计算的 9 个核心槽位集合（不包含辅助槽位）
from certvla.slots.role_sets import J_CERT


# ──────────────────────────────────────────────────────────────
# 辅助函数 (Helpers)
# ──────────────────────────────────────────────────────────────

def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Focal 交叉熵损失函数。

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    # ── 设计动机 ──
    # 标准交叉熵对所有样本一视同仁。在角色分类任务中，"ignore" 类别通常
    # 占绝大多数（大部分槽位在大部分时间步都不需要关注），造成严重的类别
    # 不平衡。Focal Loss 通过 (1 - p_t)^gamma 因子降低"容易样本"的权重，
    # 使模型将训练精力集中在"困难样本"（如 advance/preserve 类别）上。
    #
    # gamma=2.0 是 Lin et al. (2017) 论文推荐的默认值：
    #   - gamma=0 退化为标准交叉熵
    #   - gamma 越大，对容易样本的抑制越强
    #   - 实践中 gamma=2.0 在大多数不平衡场景下表现良好

    Args:
        logits: (B, C) 原始 logits（未经 softmax），C 为类别数。
        targets: (B,) 类别索引 (long 类型)。
        gamma: 聚焦参数，控制对容易样本的抑制程度。
        reduction: "none" -> 返回 (B,) 逐样本损失，"mean" -> 返回标量。
    """
    ce = F.cross_entropy(logits, targets, reduction="none")  # (B,)  标准交叉熵
    pt = torch.exp(-ce)  # p_t = exp(-CE) 就是正确类别的预测概率
    focal = ((1.0 - pt) ** gamma) * ce  # 对容易样本 (p_t 大) 降权
    if reduction == "mean":
        return focal.mean()
    return focal


def _per_slot_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    meta,
) -> torch.Tensor:
    """计算单个槽位的逐样本损失，返回 (B,) 张量。

    # ── 设计说明 ──
    # CertVLA 的状态槽位 (slot) 涵盖多种数据类型，因此需要根据槽位的
    # domain 类型选择合适的损失函数：
    #   - BINARY (二值):   BCE (二元交叉熵)，如"抓手是否打开"
    #   - CATEGORICAL (分类): CE (交叉熵)，如"当前阶段是 reach/grasp/lift"
    #   - CONTINUOUS (连续):  L1 (绝对值损失)，如"物体离目标的距离"
    #   - CONFIDENCE (置信度): L1，如"检测器对物体存在性的置信度"
    #
    # meta 参数来自 SLOT_REGISTRY，包含该槽位的 domain 信息。

    Args:
        pred -- 模型输出（对 binary/continuous 已做 sigmoid 激活，对 categorical 是原始 logits）。
        target -- 真实标签（binary/continuous 为 float，categorical 为 long 类别索引）。
    """
    if meta.domain == SlotDomain.BINARY:
        # 二值槽位：使用二元交叉熵，pred 已经过 sigmoid 映射到 [0, 1]
        return F.binary_cross_entropy(
            pred.view(-1), target.view(-1).float(), reduction="none"
        )
    elif meta.domain == SlotDomain.CATEGORICAL:
        # 分类槽位：使用多类交叉熵，pred 为 (B, num_classes) 原始 logits
        tgt = target.long()
        if tgt.dim() > 1:
            tgt = tgt.squeeze(-1)  # 确保 target 是 (B,) 而非 (B, 1)
        return F.cross_entropy(pred, tgt, reduction="none")
    elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        # 连续/置信度槽位：使用 L1 损失，对异常值更鲁棒（相比 MSE）
        return F.l1_loss(
            pred.view(-1), target.view(-1).float(), reduction="none"
        )
    raise ValueError(f"Unknown domain {meta.domain}")


def _slot_pred_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    meta,
) -> torch.Tensor:
    """计算两个模型格式预测值之间的可微分距离。

    返回 (B,) 张量，值域在 [0, 1] 之间。

    # ── 用途 ──
    # 主要用于一致性损失 (L_cons) 和推理时的证书间隙 (gap) 计算。
    # 当两边都是模型输出（而非 ground truth）时使用此函数。
    #
    # 距离度量的选择：
    #   - 连续/二值/置信度槽位: 直接取绝对值差 |a - b|，简单高效
    #   - 分类槽位: 使用总变差距离 (Total Variation Distance)
    #     TV(p, q) = (1/2) * sum|p_i - q_i|，值域 [0, 1]
    #     先对 logits 做 softmax 转为概率分布，再计算 TV 距离
    """
    if meta.domain in (SlotDomain.BINARY, SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        return torch.abs(a.view(-1) - b.view(-1))
    elif meta.domain == SlotDomain.CATEGORICAL:
        pa = F.softmax(a, dim=-1)  # (B, num_classes) 概率分布
        pb = F.softmax(b, dim=-1)  # (B, num_classes) 概率分布
        # 总变差距离：两个分布逐类别差的绝对值之和除以 2
        return torch.abs(pa - pb).sum(dim=-1) / 2.0
    raise ValueError(f"Unknown domain {meta.domain}")


def _zero(device: Optional[torch.device] = None) -> torch.Tensor:
    """创建一个值为 0 的标量张量，并启用梯度追踪。

    # ── 设计说明 ──
    # 当某个损失项没有任何有效槽位参与计算时（例如所有槽位都被掩码屏蔽），
    # 需要返回一个"零损失"。但这个零必须 requires_grad=True，否则在
    # backward() 时会因为计算图断裂而报错。这个辅助函数确保返回的零
    # 张量能安全地参与梯度反传。
    """
    t = torch.tensor(0.0, device=device)
    t.requires_grad_(True)
    return t


# ──────────────────────────────────────────────────────────────
# L_state  (论文第 9.1 节)
# ──────────────────────────────────────────────────────────────
# 状态读出损失：训练模型从持久状态 token z_t 中准确读出环境状态。
# 这是 CertVLA 训练的第一个阶段 (Stage 1) 的唯一损失项，也是后续
# 所有证书机制的基础——如果状态读出不准确，证书的角色分类和目标
# 预测都会失去意义。
#
# 公式: L_state = sum_j [ m^j * alpha^j * ell_j(hat_s^j, s^j) ]
#   其中:
#     j       -- 槽位索引，遍历所有已定义的状态槽位
#     m^j     -- 掩码 (0 或 1)，标记该槽位在该样本中是否有有效标注
#     alpha^j -- 置信度权重 [0, 1]，反映标注的可靠程度
#     ell_j   -- 根据槽位 domain 选择的损失函数 (BCE / CE / L1)
#     hat_s^j -- 模型对槽位 j 的预测值
#     s^j     -- 槽位 j 的真实值
# ──────────────────────────────────────────────────────────────

def cert_state_loss(
    state_readout: Dict[SlotName, torch.Tensor],
    state_target: Dict[SlotName, torch.Tensor],
    mask: Dict[SlotName, torch.Tensor],
    confidence: Dict[SlotName, torch.Tensor],
) -> torch.Tensor:
    """L_state = sum_j  m^j * alpha^j * ell_j(hat_s^j, s^j).

    遍历所有在 readout 和 target 中都存在的槽位，计算加权损失。

    # ── 张量形状流 ──
    # state_readout[slot]: (B, 1) 或 (B, num_classes)，取决于 domain
    # state_target[slot]:  (B, 1) float 或 (B,) long
    # mask[slot]:          (B,) float, 值为 0.0 或 1.0
    # confidence[slot]:    (B,) float, 值在 [0, 1] 区间
    # 输出: 标量张量
    """
    dev = next(iter(state_readout.values())).device
    losses = []
    for slot_name in SlotName:
        if slot_name not in state_readout or slot_name not in state_target:
            continue
        meta = SLOT_REGISTRY[slot_name]
        pred = state_readout[slot_name]
        tgt = state_target[slot_name]
        # 如果掩码或置信度缺失，默认为全 1（即不屏蔽、完全置信）
        m = mask.get(slot_name, torch.ones(pred.shape[0], device=dev))
        alpha = confidence.get(slot_name, torch.ones(pred.shape[0], device=dev))
        sl = _per_slot_loss(pred, tgt, meta)       # (B,) 逐样本损失
        # m * alpha * sl: 先掩码、再加权、最后对 batch 求均值
        losses.append((m * alpha * sl).mean())
    if not losses:
        return _zero(dev)  # 无有效槽位时返回可导的零
    # 将所有槽位的平均损失求和，得到总状态损失
    return torch.stack(losses).sum()


# ──────────────────────────────────────────────────────────────
# L_role  (论文第 9.2 节)
# ──────────────────────────────────────────────────────────────
# 角色分类损失：训练证书头 (certificate head) 为每个槽位预测其"角色"。
# 三种角色:
#   - advance (前进, id=0): 该槽位是当前任务要改变的目标状态
#   - preserve (保持, id=1): 该槽位应在动作执行过程中保持不变
#   - ignore (忽略, id=2): 该槽位与当前任务无关
#
# 使用 Focal CE 而非标准 CE 的原因:
#   在典型的机器人操作场景中，大多数槽位在大多数时间步都是 "ignore"
#   类别（例如 9 个槽位中通常只有 1-2 个是 advance），导致极端的
#   类别不平衡。Focal Loss 通过降低容易分类样本的权重来缓解这一问题。
#
# 公式: L_role = sum_{j in J_cert} m^j * alpha^j * FocalCE(hat_u^j, u^j)
#   其中:
#     J_cert  -- 参与证书计算的 9 个核心槽位
#     hat_u^j -- 模型对槽位 j 角色的 logits (B, 3)
#     u^j     -- 真实角色标签 {0, 1, 2}
# ──────────────────────────────────────────────────────────────

def cert_role_loss(
    role_logits: Dict[SlotName, torch.Tensor],
    role_target: Dict[SlotName, torch.Tensor],
    mask: Dict[SlotName, torch.Tensor],
    confidence: Dict[SlotName, torch.Tensor],
    gamma: float = 2.0,
) -> torch.Tensor:
    """L_role = sum_{j in J_cert}  m^j * alpha^j * FocalCE(hat_u^j, u^j).

    仅遍历 J_CERT 中的 9 个证书槽位（非证书槽位不参与角色分类）。

    # ── 张量形状流 ──
    # role_logits[slot]:  (B, 3) 原始 logits，对应 advance/preserve/ignore
    # role_target[slot]:  (B,) long，值为 0/1/2
    # 输出: 标量张量
    """
    dev = next(iter(role_logits.values())).device
    losses = []
    for slot_name in J_CERT:
        if slot_name not in role_logits or slot_name not in role_target:
            continue
        logits = role_logits[slot_name]               # (B, 3) 三类 logits
        tgt = role_target[slot_name].long()            # (B,) 真实角色
        m = mask.get(slot_name, torch.ones(logits.shape[0], device=dev))
        alpha = confidence.get(slot_name, torch.ones(logits.shape[0], device=dev))
        fl = focal_cross_entropy(logits, tgt, gamma=gamma)  # (B,) focal 损失
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
