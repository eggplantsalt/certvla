"""
Persistent state token and gated update module.

Per context doc section 4.1:
    z_t = Phi(z_{t-1}, o_t, l)

Gated update:
    tilde_z_t = f_theta(o_t, l, z_{t-1})   [here: the LLM hidden state at the state token position]
    g_t = sigma(W_g [tilde_z_t ; z_{t-1}] + b_g)
    z_t = g_t * tilde_z_t + (1 - g_t) * z_{t-1}

v1 decision: learnable z_0 for every sample (no episode-level recurrence at training time).

# =============================================================================
# 【模块概述 - 持久化状态令牌 (Persistent State Token)】
#
# 本模块是 CertVLA 流水线的核心入口，负责维护一个"任务状态向量" z_t。
#
# 在 CertVLA 系统中，z_t 扮演着"递归可认证的任务状态"角色：
#   - z_t 汇聚了当前观测 o_t、语言指令 l 和历史状态 z_{t-1} 的信息
#   - 下游所有决策头（状态读出 R_phi、证书头 Q_psi、动作头）都从 z_t 出发
#   - z_t 是一个 **信息瓶颈**：强制模型将任务理解压缩到单一向量中
#
# 门控更新机制 (Gated Update)：
#   - tilde_z_t: LLM 在状态令牌位置输出的隐藏状态，包含当前时刻的全部上下文信息
#   - g_t = sigmoid(W_g [tilde_z_t ; z_{t-1}] + b_g): 门控值，决定新旧状态的混合比例
#   - z_t = g_t * tilde_z_t + (1 - g_t) * z_{t-1}: 加权融合
#   - 当 g_t ≈ 1 时，状态完全更新为新信息（任务状态发生大变化）
#   - 当 g_t ≈ 0 时，状态几乎不变（任务状态保持稳定）
#   - 这类似于 GRU/LSTM 的门控思想，但作用于整个任务状态
#
# v1 训练策略：
#   - 训练时每个样本独立，z_prev = z_0（可学习初始状态）
#   - 不进行 episode 级别的递归（避免训练时的序列依赖问题）
#   - 推理时可以选择递归传递 z_t 以支持多步推理
#
# 关键张量形状：
#   - z_0, z_t, tilde_z_t, z_prev: (B, llm_dim)，其中 llm_dim=4096 (Llama 2 7B)
#   - 状态令牌嵌入（输入到 LLM）: (B, 1, llm_dim)，1 表示单个额外令牌
#   - gate: (B, llm_dim)，每个维度独立门控（逐元素门控，非标量门控）
# =============================================================================
"""

import torch
import torch.nn as nn


# =============================================================================
# 【StateTokenModule - 状态令牌模块】
#
# 在 CertVLA 流水线中的位置：
#   输入序列: [视觉 patch 令牌] [状态令牌 z_0] [文本令牌] [动作令牌]
#                                    ↑ 本模块提供这个嵌入
#
# 工作流程：
#   1. 训练/推理前：调用 get_state_token_embedding() 获取 z_0 嵌入，
#      注入 LLM 输入序列（位于视觉令牌之后、文本令牌之前）
#   2. LLM 前向传播：状态令牌与其他令牌一起经过 Transformer 层，
#      通过自注意力机制聚合全序列信息
#   3. LLM 前向传播后：从最后一层隐藏状态中提取状态令牌位置的输出
#      作为 tilde_z_t，然后调用 gated_update() 得到最终状态 z_t
#
# 设计决策说明：
#   - z_0 初始化为零均值、小方差的正态分布（std=0.02），而非全零
#     这样可以在训练初期打破对称性，帮助不同维度学到不同的语义
#   - 门控网络的权重和偏置都初始化为零 → 初始 sigmoid 输出 ≈ 0.5
#     这意味着训练开始时新旧状态各占一半，是一个安全的中性起点
#   - 门控是逐元素的 (per-dimension)，而非标量门控
#     这允许模型对不同语义维度采用不同的更新速率
#
# 常见陷阱：
#   - z_0 是共享参数，expand() 不会复制数据（共享内存），
#     但梯度会正确传播到 z_0
#   - 推理时如果要做多步递归，需要在外部维护 z_prev 的状态
# =============================================================================
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

        # 【可学习初始状态 z_0】
        # 形状: (1, llm_dim) = (1, 4096)
        # 含义: 表示"空白任务状态"的初始嵌入，所有 batch 样本共享同一个 z_0
        # 初始化策略: 先置零再用小方差正态分布初始化（std=0.02）
        # 设计原因: 小方差初始化确保训练初期状态令牌不会产生过大的激活值，
        #           同时非零值打破对称性，让不同维度可以学到不同的语义表示
        # Learnable initial state embedding z_0
        # Initialized as zeros; the model learns what a "blank" task state looks like
        self.z_0 = nn.Parameter(torch.zeros(1, llm_dim))
        nn.init.normal_(self.z_0, mean=0.0, std=0.02)

        # 【门控投影层】
        # 输入: [tilde_z_t ; z_{t-1}] 的拼接，维度 = 2 * llm_dim = 8192
        # 输出: llm_dim = 4096，经过 sigmoid 后成为逐元素门控值
        # 初始化: 权重和偏置全部为零 → sigmoid(0) = 0.5
        #         即训练初期，新状态和旧状态各贡献 50%，是安全的中性起点
        # 注意: 这里不是标量门控，而是 per-dimension 门控，
        #       允许模型对 4096 维中的每一维采用不同的更新速率
        # Gating network: takes [tilde_z_t ; z_{t-1}] -> scalar gate
        self.gate_proj = nn.Linear(2 * llm_dim, llm_dim)
        # Initialize gate bias to positive values so initial gate is ~0.5
        # (balanced mix of new and old state at start of training)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def get_initial_state(self, batch_size: int) -> torch.Tensor:
        """Get initial state z_0, expanded for batch.

        # 获取批次化的初始状态 z_0
        # expand() 不会复制内存，而是创建一个视图 (view)
        # 梯度会正确地从所有 batch 样本聚合回 z_0 参数

        Returns:
            z_0: Shape (B, llm_dim).
        """
        return self.z_0.expand(batch_size, -1)

    def get_state_token_embedding(self, batch_size: int) -> torch.Tensor:
        """Get the state token embedding to inject into the LLM sequence.

        # 获取要注入 LLM 输入序列的状态令牌嵌入
        # 返回形状 (B, 1, llm_dim)：中间的维度 1 对应序列中的一个令牌位置
        # 在 Phase 3 集成中，这个嵌入会被追加到视觉 patch 嵌入之后，
        # 然后送入 LLM 的 _build_multimodal_attention()
        #
        # v1 训练：每次都返回 z_0（无递归），推理时也可返回 z_0
        # 未来版本可以考虑使用上一步的 z_t 作为输入嵌入，实现递归

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

        # 【门控更新 - CertVLA 状态递归的核心操作】
        #
        # 数学公式:
        #   concat = [tilde_z_t ; z_prev]          形状: (B, 2*4096) = (B, 8192)
        #   gate = sigmoid(W_g * concat + b_g)     形状: (B, 4096)，值域 [0, 1]
        #   z_t = gate * tilde_z_t + (1-gate) * z_prev   形状: (B, 4096)
        #
        # 直觉理解:
        #   - tilde_z_t 是 LLM 基于当前观测和历史上下文产生的"提议状态"
        #   - z_prev 是上一时刻的任务状态（v1 训练时就是 z_0）
        #   - gate 决定了每个维度上"接受提议"的程度
        #   - 这使得状态更新具有可控的惯性：对于已经确定的任务信息，
        #     模型可以选择保留（gate≈0）；对于需要更新的信息，选择接受（gate≈1）
        #
        # 为什么返回 gate？
        #   - gate 的均值可以用于诊断：如果 gate 值过于极端（全0或全1），
        #     可能说明门控机制退化了，需要调整初始化或学习率
        #   - 详见 CertVLAWrapper 中如何将 gate 打包到输出中

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
