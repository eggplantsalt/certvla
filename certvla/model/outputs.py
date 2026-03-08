"""
Forward output dataclass for CertVLA.

Defines CertVLAOutput that bundles all outputs from a CertVLA forward pass.

# =============================================================================
# 【模块概述 - CertVLA 输出数据类 (CertVLAOutput)】
#
# 本模块定义了 CertVLA 前向传播的标准输出格式。
# 使用 Python dataclass 将所有子模块的输出打包为一个结构化对象，
# 便于下游代码（损失计算、日志记录、推理）统一访问。
#
# 在整体流水线中的位置:
#   CertVLAWrapper.forward() → 返回 CertVLAOutput
#   CertVLAOutput → 传给 CertVLALoss 计算损失
#   CertVLAOutput → 传给推理/评估代码提取动作
#
# 设计说明:
#   - 使用 dataclass 而非 dict，提供类型检查和自动文档
#   - 所有张量保证在同一设备 (device) 和数据类型 (dtype) 上
#   - gate_value 是可选字段，用于诊断（训练监控时检查门控健康度）
#   - 保留中间结果（actions_coarse, actions_fine）便于分析粗细分支的贡献
#   - actions_hidden_states 保留原始 LLM 动作隐藏状态，
#     方便在损失计算时做额外的正则化或分析
# =============================================================================
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from certvla.slots.schema import SlotName


# =============================================================================
# 【CertVLAOutput - CertVLA 前向传播输出数据类】
#
# 包含 CertVLA 所有子模块的输出，按以下逻辑分组:
#
# 1. 状态相关:
#    - z_t: 门控更新后的任务状态向量 (B, 4096)
#    - state_readout: 结构化状态估计 Dict[SlotName, Tensor]
#    - gate_value: 门控诊断值 (B, 1)
#
# 2. 证书相关:
#    - role_logits: 角色分类 Dict[SlotName, (B, 3)]
#    - goal_preds: 目标预测 Dict[SlotName, (B, dim)]
#
# 3. 动作相关:
#    - actions: 最终合成动作 (B, 8, 7) = coarse + lambda_res * fine
#    - actions_coarse: 粗动作分支输出 (B, 8, 7)
#    - actions_fine: 细残差分支输出 (B, 8, 7)
#    - actions_hidden_states: 原始 LLM 动作隐藏状态 (B, 56, 4096)
#
# 数值说明:
#   - B = batch_size (批次大小)
#   - 4096 = llm_dim (Llama 2 7B 隐藏维度)
#   - 8 = NUM_ACTIONS_CHUNK (动作块长度)
#   - 7 = ACTION_DIM (动作空间维度: 6个关节 + 1个抓手)
#   - 56 = 8 * 7 (动作块中的总令牌数)
#   - 3 = 角色数量 (advance/preserve/ignore)
# =============================================================================
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

    # 【任务状态向量】经过门控更新后的 z_t，是下游所有决策的信息来源
    # 形状: (B, llm_dim)，即 (B, 4096)
    z_t: torch.Tensor

    # 【结构化状态读出】每个槽位一个张量
    # 例如: {SlotName.GRIPPER_OPEN: Tensor(B,1), SlotName.OBJECT_TYPE: Tensor(B,5), ...}
    state_readout: Dict[SlotName, torch.Tensor]

    # 【角色分类 logits】每个证书槽位的三分类原始 logits
    # 例如: {SlotName.GRIPPER_OPEN: Tensor(B,3), ...}
    # 三列含义: [advance(推进), preserve(保持), ignore(忽略)]
    role_logits: Dict[SlotName, torch.Tensor]

    # 【目标预测】每个证书槽位的 chunk 结束目标值
    # 只有 role=advance 的槽位的目标预测才有监督信号
    goal_preds: Dict[SlotName, torch.Tensor]

    # 【粗动作输出】仅由 z_t + cert 生成，不含观测信息
    # 形状: (B, 8, 7)，表示 8 步 7 维动作
    actions_coarse: torch.Tensor

    # 【细残差输出】由 LLM 动作隐藏状态 + z_t + cert 生成的几何校正
    # 形状: (B, 8, 7)
    actions_fine: torch.Tensor

    # 【最终合成动作】= actions_coarse + lambda_res * actions_fine
    # 形状: (B, 8, 7)，这是送给机器人执行的实际动作
    actions: torch.Tensor

    # 【LLM 动作隐藏状态】从 LLM 最后一层提取的原始隐藏状态
    # 形状: (B, 56, 4096)，保留以便损失计算或后续分析
    actions_hidden_states: torch.Tensor

    # 【门控诊断值】gate 在 llm_dim 上的均值，用于监控门控健康度
    # 形状: (B, 1)
    # 健康范围: 0.3~0.7 (正常波动), <0.1 或 >0.9 可能表示退化
    gate_value: Optional[torch.Tensor] = None
