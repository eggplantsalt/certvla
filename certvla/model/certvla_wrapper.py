"""
CertVLA wrapper: composes state token, readout, certificate, and action heads
around the base OpenVLA/OFT model.

This is an add-on module (NOT a subclass of OpenVLAForActionPrediction).
It wraps around the base VLA forward pass and adds CertVLA heads on top.

Design principle: minimal invasion. The base VLA is unchanged. CertVLA injects
a state token into the LLM sequence and reads out from the state token position
in the last hidden states.

Phase 2 scope: forward path only. No losses, no training loop, no inference repair.

# =============================================================================
# 【模块概述 - CertVLA 包装器 (CertVLA Wrapper)】
#
# 本模块是 CertVLA 系统的顶层编排器 (orchestrator)，负责:
#   1. 将所有子模块（状态令牌、状态读出、证书头、动作头）组合在一起
#   2. 定义完整的 CertVLA 前向传播流程
#   3. 从 LLM 的隐藏状态中提取所需信息并分发给各子模块
#
# 关键设计决策:
#   - 这是一个独立的 nn.Module，**不是** OpenVLAForActionPrediction 的子类
#   - 不拥有也不修改基础 VLA 模型
#   - 通过接收 LLM 的 last_hidden_states 来工作，而非直接调用 LLM
#   - 这种"附加式"设计最小化了对原始代码的侵入性
#
# 与基础 VLA 的集成方式:
#   Phase 2 (当前): 操作预提取的隐藏状态（用于独立测试）
#   Phase 3: 集成到完整的 VLA 前向传播中
#     - 状态令牌嵌入插入到投影后的视觉 patch 嵌入之后
#     - 状态令牌在序列中的位置 = NUM_PATCHES（视觉令牌之后，文本令牌之前）
#     - LLM 前向传播后，从 last_hidden_states 中提取各位置的信息
#
# 前向传播六步流程:
#   1. 从 last_hidden_states 的 state_token_pos 位置提取 tilde_z_t
#   2. 门控更新: z_t = gate * tilde_z_t + (1-gate) * z_prev
#   3. 状态读出: z_t → s_t (Dict[SlotName, Tensor])
#   4. 证书头: z_t → (role_logits, goal_preds)
#   5. 从 action_start_pos 位置提取动作隐藏状态
#   6. 动作头: (z_t, cert, actions_hidden_states) → 最终动作
#
# 输入序列中的位置索引示意:
#   [视觉patches, 0..N-1] [状态令牌, N] [文本令牌, N+1..M] [动作令牌, M+1..end]
#                           ↑                                  ↑
#                     state_token_pos                    action_start_pos
#
# 依赖隔离:
#   本文件及 certvla/model/ 下的所有文件都不导入 prismatic 相关代码
#   这避免了 prismatic 的重量级依赖链（HuggingFace Transformers, timm 等）
#   在 Phase 3 集成时，连接点在 modeling_prismatic.py 中，而非这里
# =============================================================================
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from certvla.model.state_token import StateTokenModule
from certvla.model.state_readout import StateReadoutHead
from certvla.model.certificate_head import CertificateHead
from certvla.model.action_head import CertActionHead, _DEFAULT_ACTION_DIM, _DEFAULT_NUM_ACTIONS_CHUNK
from certvla.model.outputs import CertVLAOutput
from certvla.slots.schema import SlotName


# =============================================================================
# 【CertVLAWrapper - CertVLA 包装器（顶层模块）】
#
# 包含四个子模块:
#   self.state_token:     StateTokenModule     — 提供 z_0 嵌入和门控更新
#   self.state_readout:   StateReadoutHead     — z_t → 结构化任务状态 s_t
#   self.certificate_head: CertificateHead     — z_t → 进展证书 c_t
#   self.action_head:     CertActionHead       — (z_t, c_t, act_hidden) → 动作
#
# 使用方式（Phase 3 集成后的伪代码）:
#   wrapper = CertVLAWrapper(llm_dim=4096)
#
#   # 1. 获取状态令牌嵌入，注入 LLM 输入序列
#   state_embed = wrapper.get_state_token_embedding(batch_size)
#   # 插入到视觉 patches 之后
#
#   # 2. 运行 LLM 前向传播
#   last_hidden_states = llm(input_embeds)  # (B, seq_len, 4096)
#
#   # 3. 运行 CertVLA 前向传播
#   output = wrapper(last_hidden_states, state_token_pos, action_start_pos)
#   # output: CertVLAOutput 包含所有子模块的输出
# =============================================================================
class CertVLAWrapper(nn.Module):
    """Wrapper that adds CertVLA heads around a base VLA's LLM hidden states.

    This module does NOT own or modify the base VLA model. Instead, it:
    1. Provides a state token embedding to be injected into the LLM sequence
    2. After the LLM forward pass, takes the full last_hidden_states and:
       a. Extracts the hidden state at the state token position -> tilde_z_t
       b. Applies gated update -> z_t
       c. Runs state readout: z_t -> s_t
       d. Runs certificate head: z_t -> (role_logits, goal_preds)
       e. Extracts actions_hidden_states from the action token positions
       f. Runs cert action head: (z_t, cert, actions_hidden_states) -> actions

    Integration with `modeling_prismatic.py` (Phase 3):
    - The state token embedding is appended to projected_patch_embeddings
      before _build_multimodal_attention()
    - The state token position in the sequence = NUM_PATCHES (after all vision tokens,
      before text tokens)

    Phase 2: this wrapper operates on pre-extracted hidden states.
    Phase 3: will integrate with the full VLA forward pass.
    """

    def __init__(
        self,
        llm_dim: int = 4096,
        readout_hidden_dim: int = 512,
        cert_hidden_dim: int = 512,
        cert_embed_dim: int = 256,
        coarse_hidden_dim: int = 1024,
        fine_hidden_dim: int = 1024,
        action_dim: int = _DEFAULT_ACTION_DIM,
        num_actions_chunk: int = _DEFAULT_NUM_ACTIONS_CHUNK,
        lambda_res_init: float = 0.1,
    ):
        """
        Args:
            llm_dim: LLM hidden dimension (4096 for Llama 2 7B).
            readout_hidden_dim: State readout shared trunk hidden dim.
            cert_hidden_dim: Certificate head shared trunk hidden dim.
            cert_embed_dim: Certificate embedding dim for action conditioning.
            coarse_hidden_dim: Coarse action branch hidden dim.
            fine_hidden_dim: Fine action branch hidden dim.
            action_dim: Action space dimension.
            num_actions_chunk: Number of actions per chunk.
            lambda_res_init: Initial residual scaling factor.
        """
        super().__init__()
        self.llm_dim = llm_dim
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk

        # 【子模块 1: 状态令牌模块】
        # 提供可学习的初始状态 z_0 (1, 4096) 和门控更新机制
        # State token (provides z_0 and gated update)
        self.state_token = StateTokenModule(llm_dim)

        # 【子模块 2: 状态读出头】
        # 将 z_t (B, 4096) 映射为结构化的槽位预测
        # 只从 z_t 读取，不接触 LLM 完整序列（信息隔离约束）
        # State readout head: z_t -> per-slot predictions
        self.state_readout = StateReadoutHead(llm_dim, readout_hidden_dim)

        # 【子模块 3: 证书头】
        # 将 z_t (B, 4096) 映射为每槽位的角色分类和目标预测
        # 同样只从 z_t 读取（信息隔离约束）
        # Certificate head: z_t -> (role_logits, goal_preds)
        self.certificate_head = CertificateHead(llm_dim, cert_hidden_dim)

        # 【子模块 4: 证书条件化动作头】
        # 包含粗分支 + 细分支，生成最终的动作序列 (B, 8, 7)
        # Certificate-conditioned action head
        self.action_head = CertActionHead(
            llm_dim=llm_dim,
            cert_embed_dim=cert_embed_dim,
            coarse_hidden_dim=coarse_hidden_dim,
            fine_hidden_dim=fine_hidden_dim,
            action_dim=action_dim,
            num_actions_chunk=num_actions_chunk,
            lambda_res_init=lambda_res_init,
        )

    def get_state_token_embedding(self, batch_size: int) -> torch.Tensor:
        """Get state token embedding to inject into LLM sequence.

        # 获取状态令牌嵌入，用于注入 LLM 输入序列
        # 这是 CertVLA 与基础 VLA 之间的接口点之一:
        #   在 Phase 3 集成中，这个嵌入被追加到 projected_patch_embeddings 之后
        #   然后一起送入 LLM 的 _build_multimodal_attention()

        Returns:
            State token embedding. Shape (B, 1, llm_dim).
        """
        return self.state_token.get_state_token_embedding(batch_size)

    def forward(
        self,
        last_hidden_states: torch.Tensor,
        state_token_pos: int,
        action_start_pos: int,
        z_prev: Optional[torch.Tensor] = None,
    ) -> CertVLAOutput:
        """CertVLA forward pass on LLM hidden states.

        # 【CertVLA 完整前向传播 - 六步流程】
        #
        # 本方法是 CertVLA 的核心计算路径，在 LLM 前向传播完成后执行。
        # 它从 LLM 的最后一层隐藏状态中提取信息，并通过各子模块生成:
        #   - z_t: 更新后的任务状态
        #   - s_t: 结构化的状态读出
        #   - c_t: 进展证书（角色 + 目标）
        #   - A_t: 最终动作（粗 + 细残差）
        #
        # 输入序列布局:
        #   位置: 0 ... state_token_pos ... action_start_pos ... action_end_pos ... seq_len
        #   内容: [视觉patches] [状态令牌] [文本令牌] [动作令牌(56个)] [其他]
        #
        # 关于 z_prev:
        #   - v1 训练: z_prev=None，自动使用 z_0（每个样本独立）
        #   - 推理递归: 传入上一时刻的 z_t，支持多步状态积累
        #   - 注意: z_prev 需要与 last_hidden_states 在同一设备和 dtype 上

        This processes the hidden states AFTER the base VLA's LLM forward pass.

        Args:
            last_hidden_states: Full LLM last hidden states.
                Shape (B, seq_len, llm_dim).
            state_token_pos: Position index of the state token in the sequence.
                The state token is inserted after vision patches.
            action_start_pos: Position index where action tokens start.
                Used to extract actions_hidden_states.
            z_prev: Previous state for gated update. Shape (B, llm_dim).
                If None, uses z_0 (default for v1 training).

        Returns:
            CertVLAOutput with all head outputs.
        """
        B = last_hidden_states.shape[0]

        # --- 步骤 1: 提取 tilde_z_t ---
        # 从 LLM 最后一层隐藏状态中，取状态令牌位置的向量
        # 这个向量经过了 LLM 全部 Transformer 层的自注意力处理，
        # 已经聚合了视觉、文本和历史上下文的信息
        # --- Step 1: Extract tilde_z_t from state token position ---
        tilde_z_t = last_hidden_states[:, state_token_pos, :]  # (B, llm_dim)

        # --- 步骤 2: 门控更新 ---
        # 如果没有提供 z_prev（v1 训练模式），使用可学习的 z_0
        # 门控更新: z_t = gate * tilde_z_t + (1-gate) * z_prev
        # --- Step 2: Gated update ---
        if z_prev is None:
            z_prev = self.state_token.get_initial_state(B).to(
                device=last_hidden_states.device,
                dtype=last_hidden_states.dtype,
            )
        z_t, gate = self.state_token.gated_update(tilde_z_t, z_prev)  # (B, llm_dim)

        # --- 步骤 3: 状态读出 (仅从 z_t 读取) ---
        # 关键隔离: state_readout 只接收 z_t，不接触 last_hidden_states
        # 输出: Dict[SlotName, Tensor]，每个槽位一个预测张量
        # --- Step 3: State readout (only from z_t) ---
        state_readout = self.state_readout(z_t)

        # --- 步骤 4: 证书头 (仅从 z_t 读取) ---
        # 同样只接收 z_t，保持信息隔离
        # 输出: role_logits (每槽位 (B,3)), goal_preds (每槽位 (B,dim))
        # --- Step 4: Certificate head (only from z_t) ---
        role_logits, goal_preds = self.certificate_head(z_t)

        # --- 步骤 5: 提取动作隐藏状态 ---
        # 从 last_hidden_states 中切出动作令牌对应的隐藏状态
        # 位置范围: [action_start_pos, action_start_pos + 56)
        # 其中 56 = action_dim(7) * num_actions_chunk(8)
        # 这些隐藏状态编码了观测信息 o_t，供细动作分支使用
        # --- Step 5: Extract action hidden states ---
        action_end_pos = action_start_pos + self.action_dim * self.num_actions_chunk
        actions_hidden_states = last_hidden_states[
            :, action_start_pos:action_end_pos, :
        ]  # (B, chunk*action_dim, llm_dim) = (B, 56, 4096)

        # --- 步骤 6: 证书条件化动作头 ---
        # 输入: z_t + 证书 + 动作隐藏状态
        # 输出: 最终动作 (B,8,7), 粗动作 (B,8,7), 细残差 (B,8,7)
        # --- Step 6: Cert-conditioned action head ---
        actions, actions_coarse, actions_fine = self.action_head(
            z_t=z_t,
            role_logits=role_logits,
            goal_preds=goal_preds,
            actions_hidden_states=actions_hidden_states,
        )

        # --- 打包输出 ---
        # 将所有子模块的输出打包为 CertVLAOutput 数据类
        # gate_value 取 gate 的均值 (B, llm_dim) → (B, 1)，用于训练监控/诊断
        # 如果 gate 均值持续接近 0 或 1，说明门控可能退化了
        # --- Pack output ---
        return CertVLAOutput(
            z_t=z_t,
            state_readout=state_readout,
            role_logits=role_logits,
            goal_preds=goal_preds,
            actions_coarse=actions_coarse,
            actions_fine=actions_fine,
            actions=actions,
            actions_hidden_states=actions_hidden_states,
            gate_value=gate.mean(dim=-1, keepdim=True),  # (B, 1) mean gate for diagnostics
        )
