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

# =============================================================================
# 【模块概述 - 证书头 (Certificate Head)】
#
# 本模块是 CertVLA 可认证性 (certifiability) 的核心所在。
# 它从状态 z_t 中预测每个证书槽位的"角色"和"目标值"，
# 形成一个结构化的"进展证书" c_t = {(u_t^j, g_t^j)}。
#
# 在整体流水线中的位置:
#   LLM → z_t → StateReadout(z_t→s_t)
#             → **CertificateHead(z_t→c_t)**  ← 当前模块
#             → CertActionHead(z_t, c_t→动作)
#
# 证书的语义 - 三种角色:
#   对于每个证书槽位 j (共 9 个，来自 J_CERT = J_E ∪ J_R):
#   - advance (推进, 索引0): 该槽位预计在当前 chunk 内会发生变化
#     → 模型同时预测 chunk 结束时的目标值 g_t^j
#   - preserve (保持, 索引1): 该槽位应该在当前 chunk 内保持不变
#     → 用于安全约束，确保机器人不会意外改变不应改变的状态
#   - ignore (忽略, 索引2): 该槽位与当前任务阶段无关
#     → 不参与损失计算，减少噪声
#
# 为什么需要证书？
#   1. 可解释性: 通过检查 role 分布，可以理解模型认为当前在做什么
#   2. 安全性: preserve 角色可以检测模型是否打算做违反约束的事情
#   3. 动作条件化: 证书信息直接输入动作头，引导动作生成
#   4. 收敛引导: 目标预测提供了额外的监督信号
#
# 关键信息隔离约束:
#   与 StateReadoutHead 相同，输入 **仅为 z_t**，不接触完整序列。
#   这确保证书预测完全基于压缩的任务状态，而非"看图说话"。
#
# 关键张量形状:
#   - z_t: (B, 4096)
#   - h (共享主干输出): (B, 512)
#   - role_logits[slot]: (B, 3) — 三分类的原始 logits
#   - goal_preds[slot]: (B, dim) — dim 取决于槽位域类型（1 或 num_categories）
# =============================================================================
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from certvla.slots.schema import SlotDomain, SlotMeta, SlotName, SLOT_REGISTRY
from certvla.slots.role_sets import J_CERT

# 【角色常量定义】
# 三种角色的整数索引，对应 role_logits 的三列
# 在损失函数中使用 CrossEntropyLoss(input=(B,3), target=(B,)) 时，
# target 的值就是这些索引
# Role indices
ROLE_ADVANCE = 0   # 推进: 槽位将在 chunk 内变化，有目标预测
ROLE_PRESERVE = 1  # 保持: 槽位应保持不变，用于安全约束
ROLE_IGNORE = 2    # 忽略: 槽位与当前阶段无关，不参与损失
NUM_ROLES = 3


# =============================================================================
# 【CertificateHead - 证书头】
#
# 架构: 共享主干 + 每槽位双输出头（角色分类 + 目标预测）
#
# 共享主干: LayerNorm → Linear(4096→512) → ReLU → Linear(512→512) → ReLU
#   与 StateReadoutHead 结构相同，但参数独立（不共享权重）
#
# 每个证书槽位有两个输出头:
#   1. role_head: Linear(512→3) → 三分类 logits [advance, preserve, ignore]
#   2. goal_head: Linear(512→goal_dim) → 目标值预测
#      goal_dim 取决于域类型（同 StateReadoutHead 的 _output_dim）
#
# 只处理 J_CERT 中的槽位（9个），而非所有 SLOT_REGISTRY 中的槽位
# J_CERT = J_E ∪ J_R:
#   J_E: 效果槽位（endpoint-related），如物体位置变化
#   J_R: 机器人槽位（robot-related），如抓手状态
#
# 常见陷阱:
#   - goal_preds 的值只在 role=advance 时有意义
#     其他角色下 goal_preds 仍然有输出，但不应参与损失计算
#   - role_logits 是原始 logits，softmax 在损失函数中处理
# =============================================================================
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

        # 【共享主干】与 StateReadoutHead 结构相同但参数独立
        # 先 LayerNorm 归一化 z_t，再通过两层 MLP 提取特征
        # Shared trunk
        self.shared = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 【每槽位角色分类头】
        # 输入: h (B, 512)，输出: (B, 3) 对应 [advance, preserve, ignore]
        # Per-slot role classifiers (advance / preserve / ignore)
        self.role_heads = nn.ModuleDict()
        # 【每槽位目标预测头】
        # 输入: h (B, 512)，输出: (B, goal_dim) 预测 chunk 结束时的目标值
        # 当 role=advance 时，goal_preds 提供了预期的槽位终值
        # Per-slot goal predictors (chunk-end target value for advance slots)
        self.goal_heads = nn.ModuleDict()

        # 只为 J_CERT 中的槽位创建头（非全部 SLOT_REGISTRY）
        for slot_name in J_CERT:
            meta = SLOT_REGISTRY[slot_name]
            # Role head: 3-way classification
            self.role_heads[slot_name.value] = nn.Linear(hidden_dim, NUM_ROLES)
            # Goal head: domain-dependent output
            goal_dim = self._goal_output_dim(meta)
            self.goal_heads[slot_name.value] = nn.Linear(hidden_dim, goal_dim)

    @staticmethod
    def _goal_output_dim(meta: SlotMeta) -> int:
        """计算目标预测头的输出维度，逻辑与 StateReadoutHead._output_dim 一致。
        BINARY → 1, CATEGORICAL → num_categories, CONTINUOUS/CONFIDENCE → 1
        """
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
        # 【前向传播 - 证书生成】
        # 步骤 1: 共享主干提取特征 (B, 4096) → (B, 512)
        h = self.shared(z_t)  # (B, hidden_dim)

        role_logits = {}
        goal_preds = {}

        # 步骤 2: 遍历每个证书槽位，分别预测角色和目标
        for slot_name in J_CERT:
            meta = SLOT_REGISTRY[slot_name]

            # 角色分类: (B, 512) → (B, 3)，三列分别对应 advance/preserve/ignore
            # 训练时使用 CrossEntropyLoss，故此处输出原始 logits
            # Role classification
            role_logits[slot_name] = self.role_heads[slot_name.value](h)  # (B, 3)

            # 目标预测: 预测 chunk 结束时该槽位的期望值
            # 只有当 role=advance 时，这个预测才有监督信号
            # 激活函数与 StateReadoutHead 保持一致（按域类型选择）
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
