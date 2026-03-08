"""
State readout head: R_phi(z_t) -> s_t.

Per context doc section 4.2:
    hat_s_t = R_phi(z_t)

CRITICAL: Readout operates ONLY on z_t (the hidden state at the state token
position), NOT on the full LLM sequence. This prevents the readout from
bypassing z_t via direct access to vision tokens.

Output: per-slot predictions with domain-appropriate activations.

# =============================================================================
# 【模块概述 - 状态读出头 (State Readout Head)】
#
# 本模块是 CertVLA 流水线的第二步：从压缩状态 z_t 中读出结构化的任务状态估计 s_t。
#
# 在整体流水线中的位置:
#   LLM 前向传播 → StateTokenModule(门控更新→z_t) → **StateReadoutHead(z_t→s_t)**
#                                                  → CertificateHead(z_t→c_t)
#                                                  → CertActionHead(z_t,c_t→动作)
#
# 核心设计约束 - 信息隔离:
#   状态读出头的输入 **仅仅是 z_t 向量**（形状 (B, 4096)），
#   而 **不是** LLM 的完整序列隐藏状态 (B, seq_len, 4096)。
#   这一约束至关重要，因为:
#   1. 它确保所有任务状态信息必须经过 z_t 这个瓶颈
#   2. 防止读出头"偷看"视觉令牌来绕过状态压缩
#   3. 使得 z_t 的质量可以通过读出精度来评估
#   如果读出头能直接访问完整序列，z_t 就可能退化为无信息的占位符
#
# 输出结构 - 按槽位 (slot) 组织:
#   每个槽位代表任务状态的一个方面（如"抓手是否打开"、"物体位置"等），
#   不同域类型有不同的输出格式:
#   - BINARY（二值）:  (B, 1)，经 sigmoid 激活，表示概率值
#   - CATEGORICAL（类别）: (B, num_categories)，原始 logits（在损失函数中 softmax）
#   - CONTINUOUS（连续）: (B, 1)，经 sigmoid 激活，归一化到 [0, 1]
#   - CONFIDENCE（置信度）: (B, 1)，经 sigmoid 激活，归一化到 [0, 1]
#
# 架构选择:
#   共享主干 + 多头输出（shared trunk + per-slot heads）
#   - 共享主干: LayerNorm → Linear(4096→512) → ReLU → Linear(512→512) → ReLU
#   - 每个槽位有自己的输出线性层 Linear(512→out_dim)
#   这比为每个槽位使用独立网络更高效，同时仍允许不同槽位学到不同的映射
# =============================================================================
"""

import torch
import torch.nn as nn
from typing import Dict

from certvla.slots.schema import SlotDomain, SlotName, SlotMeta, SLOT_REGISTRY


# =============================================================================
# 【StateReadoutHead - 状态读出头】
#
# 将 z_t (B, 4096) 映射为结构化的任务状态估计，每个槽位一个预测张量。
#
# 关键张量形状流转:
#   z_t: (B, 4096) → shared trunk → h: (B, 512) → per-slot heads → 各槽位输出
#
# 注意事项:
#   - SLOT_REGISTRY 定义了所有可能的槽位及其域类型，在 certvla/slots/schema.py 中
#   - 分类 (CATEGORICAL) 槽位输出原始 logits，不在此处做 softmax，
#     因为 CrossEntropyLoss 内部会处理 softmax（数值更稳定）
#   - 共享主干的 LayerNorm 很重要：z_t 来自 LLM 的不同层/位置，
#     其数值范围可能波动较大，LayerNorm 起到稳定输入的作用
# =============================================================================
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

        # 【共享主干网络】
        # 将高维的 z_t (4096维) 压缩为紧凑的共享表示 (512维)
        # LayerNorm: 归一化 z_t，稳定后续层的输入分布
        # 两层 MLP + ReLU: 提供足够的非线性变换能力
        # 为什么用 512 而非更大的 hidden_dim？
        #   - z_t 已经是高度压缩的状态，512 维足以做槽位解码
        #   - 过大的 hidden_dim 会增加过拟合风险，且计算开销不值得
        # Shared trunk: project z_t to a compact representation
        self.shared = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 【每槽位独立输出头】
        # ModuleDict 的 key 是槽位名称的字符串值（如 "gripper_open"）
        # 每个头是一个简单的 Linear(512→out_dim)
        # out_dim 取决于槽位的域类型:
        #   BINARY → 1, CATEGORICAL → num_categories, CONTINUOUS → 1, CONFIDENCE → 1
        # Per-slot output heads
        self.slot_heads = nn.ModuleDict()
        for slot_name, meta in SLOT_REGISTRY.items():
            out_dim = self._output_dim(meta)
            self.slot_heads[slot_name.value] = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def _output_dim(meta: SlotMeta) -> int:
        """根据槽位的域类型计算输出维度。
        BINARY/CONTINUOUS/CONFIDENCE → 1 维输出
        CATEGORICAL → 类别数量维输出（用于 softmax 分类）
        """
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
        # 【前向传播流程】
        # 步骤 1: 通过共享主干将 z_t 投影到紧凑表示空间
        # (B, 4096) → (B, 512)
        h = self.shared(z_t)  # (B, hidden_dim)

        # 步骤 2: 为每个槽位生成预测
        # 遍历所有注册的槽位，使用对应的输出头 + 域特定的激活函数
        outputs = {}
        for slot_name, meta in SLOT_REGISTRY.items():
            raw = self.slot_heads[slot_name.value](h)  # (B, out_dim)
            if meta.domain == SlotDomain.BINARY:
                # 二值槽位: sigmoid 将输出压缩到 [0,1]，表示概率
                outputs[slot_name] = torch.sigmoid(raw)
            elif meta.domain == SlotDomain.CATEGORICAL:
                # 分类槽位: 保持原始 logits，softmax 在损失函数中处理
                # 这样做数值更稳定（log-sum-exp 技巧）
                outputs[slot_name] = raw  # raw logits; softmax at loss
            elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
                # 连续/置信度槽位: sigmoid 归一化到 [0,1]
                # 适用于归一化后的物理量（如归一化位置）和置信度分数
                outputs[slot_name] = torch.sigmoid(raw)
            else:
                raise ValueError(f"Unknown domain {meta.domain}")

        return outputs
