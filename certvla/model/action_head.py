"""
Certificate-conditioned action head (coarse + fine residual).

Per context doc section 4.4:
    A_t^coarse = pi_c(z_t, hat_c_t)
    Delta_A_t^fine = pi_f(o_t, z_t, hat_c_t)
    hat_A_t = A_t^coarse + lambda_res * Delta_A_t^fine

Design:
- Coarse branch: takes z_t + flattened certificate embedding. Does NOT see o_t.
  This ensures the coarse action semantics depend on the certificate.
- Fine branch: takes the original actions_hidden_states from LLM (which see o_t)
  plus the certificate embedding. Produces a residual correction.
- lambda_res is a learnable scalar, initialized to 0.1.

The coarse branch replaces the original L1RegressionActionHead for CertVLA.
The fine branch wraps around it to add geometric refinement.

# =============================================================================
# 【模块概述 - 证书条件化动作头 (Certificate-Conditioned Action Head)】
#
# 本模块是 CertVLA 流水线的最终输出环节：将任务状态和证书信息转化为机器人动作。
#
# 在整体流水线中的位置:
#   LLM → z_t → StateReadout(z_t→s_t)
#             → CertificateHead(z_t→c_t)
#             → **CertActionHead(z_t, c_t, actions_hidden→动作)** ← 当前模块
#
# 核心设计: 粗细双分支 (Coarse + Fine)
#
#   粗分支 (Coarse Branch): pi_c(z_t, c_t) → A_coarse
#   - 输入: 仅 z_t 和证书嵌入，**完全不接触观测** o_t
#   - 意义: 迫使粗动作的语义完全由任务状态和证书决定
#     例如："向目标物体移动"这一语义动作完全由 z_t + cert 编码
#   - 输出: (B, NUM_ACTIONS_CHUNK=8, ACTION_DIM=7)
#
#   细分支 (Fine Branch): pi_f(o_t, z_t, c_t) → Delta_A_fine
#   - 输入: LLM 的动作隐藏状态（编码了观测 o_t）+ z_t + 证书嵌入
#   - 意义: 基于当前观测做几何精修（如精确对齐物体位置）
#   - 输出: (B, NUM_ACTIONS_CHUNK=8, ACTION_DIM=7)
#
#   最终动作: A = A_coarse + lambda_res * Delta_A_fine
#   - lambda_res: 可学习标量，初始值 0.1（小残差起步）
#   - 初始化为小值确保训练早期以粗动作为主，避免细分支的噪声干扰
#
# 为什么需要双分支？
#   1. 粗分支没有观测访问 → 强制证书携带足够的任务语义信息
#   2. 细分支有观测访问 → 可以做精确的几何校正（如补偿视觉感知误差）
#   3. 残差结构 → 训练更稳定，先学粗略策略再精细调整
#   4. lambda_res 可学习 → 模型自动调节粗细分支的贡献比例
#
# 动作空间说明:
#   - ACTION_DIM=7: 典型的 LIBERO 机械臂，6 个关节 + 1 个抓手
#   - NUM_ACTIONS_CHUNK=8: 每次预测 8 步动作（chunk-based 预测）
#   - 总输出维度: 8 * 7 = 56 个数值
#
# 本文件包含四个类:
#   1. CertificateEmbedding: 将证书头的结构化输出展平并嵌入
#   2. CoarseActionBranch: 粗动作分支（无观测访问）
#   3. FineActionBranch: 细动作残差分支（有观测访问）
#   4. CertActionHead: 组合以上三个组件的顶层模块
# =============================================================================
"""

import torch
import torch.nn as nn

# 【默认动作空间常量】
# 这些值对应 LIBERO 基准任务的机械臂配置
# 在运行时通过构造函数参数覆盖，此处定义仅为避免导入 prismatic（会触发重量级依赖链）
# 注意: certvla/model/ 中的所有文件都不导入 prismatic，这是刻意的设计决策
# Default LIBERO constants. At runtime these are overridden via constructor args.
# Defined here to avoid importing prismatic (which triggers heavy dependencies).
_DEFAULT_ACTION_DIM = 7            # 7 维动作: 6 个关节角度/速度 + 1 个抓手开合
_DEFAULT_NUM_ACTIONS_CHUNK = 8     # 每次预测 8 步动作（动作块大小）


# =============================================================================
# 【CertificateEmbedding - 证书嵌入模块】
#
# 功能: 将 CertificateHead 输出的字典形式的 role_logits 和 goal_preds
#       展平 (flatten) 为一个固定大小的向量，再通过 MLP 投影到紧凑的嵌入空间。
#
# 为什么需要这个模块？
#   - CertificateHead 的输出是两个字典（role_logits 和 goal_preds），
#     包含不同大小的张量（有的 (B,3)，有的 (B,1)，有的 (B, num_cat)）
#   - 动作分支需要一个固定大小的条件向量
#   - 通过 flatten + 投影，将异构的证书信息统一为 (B, embed_dim) 的密集表示
#
# 张量流转:
#   cert_flat: (B, cert_raw_dim) → Linear → ReLU → Linear → (B, embed_dim)
#   cert_raw_dim = sum(3 + goal_dim for each cert slot)
#   embed_dim 默认 256
# =============================================================================
class CertificateEmbedding(nn.Module):
    """Flatten certificate head outputs into a single vector.

    Takes role_logits and goal_preds dicts, concatenates them into
    a fixed-size vector suitable for action conditioning.
    """

    def __init__(self, cert_raw_dim: int, embed_dim: int):
        """
        Args:
            cert_raw_dim: Total dimension of flattened role_logits + goal_preds.
            embed_dim: Output embedding dimension.
        """
        super().__init__()
        # 两层 MLP 投影: cert_raw_dim → embed_dim → embed_dim
        # 中间用 ReLU 非线性，最后一层无激活（让下游自行处理）
        self.proj = nn.Sequential(
            nn.Linear(cert_raw_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, cert_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cert_flat: Flattened certificate vector. Shape (B, cert_raw_dim).
                       由 CertActionHead.flatten_certificate() 生成

        Returns:
            Certificate embedding. Shape (B, embed_dim).
        """
        return self.proj(cert_flat)


# =============================================================================
# 【CoarseActionBranch - 粗动作分支】
#
# 核心特性: **不接触观测 o_t**
# 输入仅为 z_t + cert_embed，这是一个关键的设计约束。
#
# 直觉: 粗分支回答的问题是"基于当前的任务理解（z_t）和进展证书（cert），
#       应该执行什么语义级别的动作？" 比如"向前推"、"抬起手臂"。
#       不需要知道当前视觉中物体的精确像素位置。
#
# 架构: LayerNorm → Linear(input→1024) → ReLU → Linear(1024→1024) → ReLU
#        → LayerNorm → Linear(1024→56)
# 输入维度: llm_dim + cert_embed_dim = 4096 + 256 = 4352
# 输出维度: num_actions_chunk * action_dim = 8 * 7 = 56
# 输出 reshape 为: (B, 8, 7)，即 8 步动作，每步 7 维
#
# 双 LayerNorm 设计:
#   - 输入 LayerNorm: 归一化 z_t 和 cert_embed 拼接后的向量
#   - 输出前 LayerNorm: 在最后一个线性层之前归一化，稳定输出分布
# =============================================================================
class CoarseActionBranch(nn.Module):
    """Coarse action branch: pi_c(z_t, c_t) -> A_t^coarse.

    Takes z_t and certificate embedding. Does NOT see observation tokens.
    This ensures the action semantics are certificate-dependent.
    """

    def __init__(
        self,
        llm_dim: int,
        cert_embed_dim: int,
        hidden_dim: int = 1024,
        action_dim: int = _DEFAULT_ACTION_DIM,
        num_actions_chunk: int = _DEFAULT_NUM_ACTIONS_CHUNK,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk

        # 输入维度 = z_t 的维度 + 证书嵌入维度 = 4096 + 256 = 4352
        input_dim = llm_dim + cert_embed_dim
        # MLP 网络: 输入 → 隐藏层 → 输出 (chunk_len * action_dim = 56)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_actions_chunk * action_dim),
        )

    def forward(self, z_t: torch.Tensor, cert_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: State token hidden state. Shape (B, llm_dim).
            cert_embed: Certificate embedding. Shape (B, cert_embed_dim).

        Returns:
            Coarse actions. Shape (B, num_actions_chunk, action_dim).
                           即 (B, 8, 7): 8 步动作序列，每步 7 维
        """
        # 拼接 z_t 和证书嵌入作为输入
        x = torch.cat([z_t, cert_embed], dim=-1)  # (B, llm_dim + cert_embed_dim)
        # MLP 输出扁平动作向量
        out = self.net(x)  # (B, num_actions_chunk * action_dim) = (B, 56)
        # reshape 为 chunk 格式: (B, 8, 7)
        return out.reshape(-1, self.num_actions_chunk, self.action_dim)


# =============================================================================
# 【FineActionBranch - 细动作残差分支】
#
# 核心特性: **可以访问观测 o_t**（通过 LLM 的 actions_hidden_states）
# 这与粗分支形成互补——细分支负责基于当前视觉观测做几何层面的精细校正。
#
# 输入来源:
#   - actions_hidden_states: LLM 在动作令牌位置的隐藏状态
#     形状 (B, num_actions_chunk * action_dim, llm_dim) = (B, 56, 4096)
#     这些隐藏状态已经通过 LLM 的自注意力机制"看到"了视觉和文本令牌
#   - z_t: 任务状态 (B, 4096)
#   - cert_embed: 证书嵌入 (B, 256)
#
# 输入 reshape 策略:
#   将 (B, 56, 4096) reshape 为 (B, 8, 7*4096) = (B, 8, 28672)
#   即每个 chunk 步有 7 个动作维度 * 4096 = 28672 维的输入特征
#   然后拼接 z_t (4096) 和 cert_embed (256)
#   每步总输入: 28672 + 4096 + 256 = 33024
#
# 架构: LayerNorm → Linear(33024→1024) → ReLU → Linear(1024→1024) → ReLU
#        → LayerNorm → Linear(1024→7)
# 输出: (B, 8, 7) — 每步一个 7 维的残差修正
#
# 设计说明:
#   - 所有 chunk 步共享相同的网络权重（通过 broadcasting 实现）
#   - z_t 和 cert_embed 在每步都相同（expand + broadcast）
#   - 这比为每步使用独立网络更参数高效
#
# 常见陷阱:
#   - per_step_input_dim 非常大 (33024)，第一个线性层的参数量 = 33024*1024 ≈ 33M
#     这是该模块的参数瓶颈，但对于精细动作控制是必要的
# =============================================================================
class FineActionBranch(nn.Module):
    """Fine residual branch: pi_f(o_t, z_t, c_t) -> Delta_A_t.

    Takes the original actions_hidden_states from LLM (which encode o_t)
    plus z_t and certificate. Produces a residual correction.

    Architecture mirrors the original L1RegressionActionHead's MLPResNet
    but adds cert conditioning.
    """

    def __init__(
        self,
        llm_dim: int,
        cert_embed_dim: int,
        hidden_dim: int = 1024,
        action_dim: int = _DEFAULT_ACTION_DIM,
        num_actions_chunk: int = _DEFAULT_NUM_ACTIONS_CHUNK,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk

        # 【每步输入维度计算】
        # actions_hidden_states reshape 后每步: action_dim * llm_dim = 7 * 4096 = 28672
        # 加上 z_t: llm_dim = 4096
        # 加上 cert_embed: cert_embed_dim = 256
        # 总计: 28672 + 4096 + 256 = 33024
        # Input: per-chunk action hidden states (reshaped) + z_t + cert_embed
        # The action hidden states are (B, chunk_len, action_dim * llm_dim).
        # We add z_t and cert per chunk step via broadcast.
        per_step_input_dim = action_dim * llm_dim + llm_dim + cert_embed_dim

        # MLP 网络: 输入 → 隐藏层 → 输出 (action_dim=7)
        # 注意: 输出是单步的残差修正 (7维)，而非整个 chunk
        # 通过对 (B, chunk, per_step_input_dim) 形状的输入做前向传播，
        # nn.Sequential 自动在前两维上 broadcast（PyTorch 线性层支持任意前导维度）
        self.net = nn.Sequential(
            nn.LayerNorm(per_step_input_dim),
            nn.Linear(per_step_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        actions_hidden_states: torch.Tensor,
        z_t: torch.Tensor,
        cert_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            actions_hidden_states: LLM hidden states at action positions.
                Shape (B, num_actions_chunk * action_dim, llm_dim).
                即 (B, 56, 4096): 56 个动作令牌位置的 LLM 隐藏状态
            z_t: State token hidden state. Shape (B, llm_dim).
            cert_embed: Certificate embedding. Shape (B, cert_embed_dim).

        Returns:
            Fine residual actions. Shape (B, num_actions_chunk, action_dim).
            即 (B, 8, 7): 8 步的残差修正
        """
        B = actions_hidden_states.shape[0]
        # 将 (B, 56, 4096) reshape 为 (B, 8, 7*4096) = (B, 8, 28672)
        # 每个 chunk 步聚合了 7 个动作维度的 LLM 隐藏表示
        # Reshape to per-chunk-step: (B, chunk_len, action_dim * llm_dim)
        act_h = actions_hidden_states.reshape(B, self.num_actions_chunk, -1)

        # 将 z_t 和 cert_embed 扩展到匹配 chunk 步数
        # (B, 4096) → (B, 8, 4096): 每步使用相同的状态信息
        # Expand z_t and cert_embed to match chunk steps
        z_expanded = z_t.unsqueeze(1).expand(-1, self.num_actions_chunk, -1)   # (B, chunk, llm_dim)
        # (B, 256) → (B, 8, 256): 每步使用相同的证书信息
        c_expanded = cert_embed.unsqueeze(1).expand(-1, self.num_actions_chunk, -1)  # (B, chunk, cert_embed)

        # 沿最后一维拼接: (B, 8, 28672+4096+256) = (B, 8, 33024)
        # Concatenate per step
        x = torch.cat([act_h, z_expanded, c_expanded], dim=-1)  # (B, chunk, per_step_input_dim)

        # MLP 前向传播，共享权重处理所有 8 个步骤
        # PyTorch 的 Linear 自动支持高维输入: (B, 8, 33024) → (B, 8, 7)
        # Process each step (shared weights across steps)
        out = self.net(x)  # (B, chunk, action_dim)
        return out


# =============================================================================
# 【CertActionHead - 证书条件化动作头（顶层组合模块）】
#
# 这是动作生成的入口，组合了:
#   1. CertificateEmbedding: 将结构化证书展平并嵌入
#   2. CoarseActionBranch: 粗动作（语义级别）
#   3. FineActionBranch: 细残差（几何级别）
#   4. lambda_res: 可学习残差缩放因子
#
# 前向传播流程:
#   cert_flat = flatten(role_logits, goal_preds)  → (B, cert_raw_dim)
#   cert_embed = CertificateEmbedding(cert_flat)  → (B, 256)
#   A_coarse = CoarseActionBranch(z_t, cert_embed) → (B, 8, 7)
#   Delta_A_fine = FineActionBranch(act_hidden, z_t, cert_embed) → (B, 8, 7)
#   A = A_coarse + lambda_res * Delta_A_fine       → (B, 8, 7)
#
# 替换关系:
#   CertActionHead 替换了 OpenVLA-OFT 原始的 L1RegressionActionHead
#   在 CertVLA 模式下，原始动作头不再使用
#
# lambda_res 设计:
#   - 可学习标量 nn.Parameter，初始值 0.1
#   - 初始小值确保训练初期粗分支主导（避免随机初始化的细分支干扰）
#   - 随着训练进行，模型自动学习最优的粗细比例
#   - 可以通过监控 lambda_res 的值来诊断双分支的贡献平衡
# =============================================================================
class CertActionHead(nn.Module):
    """Certificate-conditioned action head: coarse + fine residual.

    hat_A_t = A_t^coarse + lambda_res * Delta_A_t^fine

    This replaces the base L1RegressionActionHead when CertVLA is active.
    """

    def __init__(
        self,
        llm_dim: int = 4096,
        cert_raw_dim: int = None,
        cert_embed_dim: int = 256,
        coarse_hidden_dim: int = 1024,
        fine_hidden_dim: int = 1024,
        action_dim: int = _DEFAULT_ACTION_DIM,
        num_actions_chunk: int = _DEFAULT_NUM_ACTIONS_CHUNK,
        lambda_res_init: float = 0.1,
    ):
        """
        Args:
            llm_dim: LLM hidden dimension.
            cert_raw_dim: Total dimension of flattened cert outputs.
                If None, auto-computed from SLOT_REGISTRY.
            cert_embed_dim: Certificate embedding dimension.
            coarse_hidden_dim: Coarse branch hidden dim.
            fine_hidden_dim: Fine branch hidden dim.
            action_dim: Action space dimension.
            num_actions_chunk: Number of actions per chunk.
            lambda_res_init: Initial value for lambda_res.
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk

        # 【自动计算证书展平维度】
        # 如果未手动指定 cert_raw_dim，则从 SLOT_REGISTRY 自动计算:
        # cert_raw_dim = sum(3 + goal_dim for each slot in J_CERT)
        # 其中 3 是角色 logits 的维度，goal_dim 取决于域类型
        # Auto-compute cert_raw_dim if not provided
        if cert_raw_dim is None:
            cert_raw_dim = self._compute_cert_raw_dim()

        # 【证书嵌入模块】将展平的证书向量投影到紧凑空间
        # Certificate embedding
        self.cert_embedding = CertificateEmbedding(cert_raw_dim, cert_embed_dim)

        # 【粗动作分支】只看 z_t + cert，不看观测
        # Coarse branch (z_t + cert -> actions, no observation)
        self.coarse = CoarseActionBranch(
            llm_dim=llm_dim,
            cert_embed_dim=cert_embed_dim,
            hidden_dim=coarse_hidden_dim,
            action_dim=action_dim,
            num_actions_chunk=num_actions_chunk,
        )

        # 【细动作残差分支】可以访问 LLM 动作隐藏状态（编码了观测信息）
        # Fine residual branch (actions_hidden_states + z_t + cert -> delta actions)
        self.fine = FineActionBranch(
            llm_dim=llm_dim,
            cert_embed_dim=cert_embed_dim,
            hidden_dim=fine_hidden_dim,
            action_dim=action_dim,
            num_actions_chunk=num_actions_chunk,
        )

        # 【可学习残差缩放因子】
        # 初始值 0.1: 训练初期粗分支主导 (≈91%)，细分支贡献微小 (≈9%)
        # 这确保了训练早期的稳定性——先学好粗略策略，再逐步引入细节校正
        # Learnable residual scaling
        self.lambda_res = nn.Parameter(torch.tensor(lambda_res_init))

    @staticmethod
    def _compute_cert_raw_dim() -> int:
        """Compute total flattened dimension of certificate head outputs.

        【计算证书展平后的总维度】
        对于 J_CERT 中的每个槽位:
          - 角色 logits 贡献 3 维 (advance/preserve/ignore)
          - 目标预测贡献 goal_dim 维 (BINARY→1, CATEGORICAL→num_cat, 其他→1)
        所有槽位的贡献求和得到 cert_raw_dim

        For each cert slot: 3 role logits + goal_dim.
        """
        from certvla.slots.schema import SlotDomain, SLOT_REGISTRY
        from certvla.slots.role_sets import J_CERT

        total = 0
        for slot_name in J_CERT:
            meta = SLOT_REGISTRY[slot_name]
            # Role logits: 3
            total += 3
            # Goal dim
            if meta.domain == SlotDomain.BINARY:
                total += 1
            elif meta.domain == SlotDomain.CATEGORICAL:
                total += len(meta.categories)
            else:
                total += 1
        return total

    def flatten_certificate(self, role_logits: dict, goal_preds: dict) -> torch.Tensor:
        """Flatten certificate head outputs into a single vector.

        【将证书头的字典输出展平为一维向量】
        按槽位名称排序（确保顺序一致性），依次拼接每个槽位的:
          - role_logits: (B, 3)
          - goal_preds: (B, dim)
        最终输出: (B, cert_raw_dim)

        注意: 排序是按 slot_name.value 的字典序，确保每次展平的顺序相同。
        如果槽位顺序不一致，嵌入网络会收到错误的输入对齐，导致训练失败。

        Args:
            role_logits: Dict[SlotName, Tensor(B, 3)]
            goal_preds: Dict[SlotName, Tensor(B, dim)]

        Returns:
            Flattened certificate vector. Shape (B, cert_raw_dim).
        """
        from certvla.slots.role_sets import J_CERT

        parts = []
        for slot_name in sorted(J_CERT, key=lambda s: s.value):
            parts.append(role_logits[slot_name])  # (B, 3)
            parts.append(goal_preds[slot_name])    # (B, dim)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        z_t: torch.Tensor,
        role_logits: dict,
        goal_preds: dict,
        actions_hidden_states: torch.Tensor,
    ) -> tuple:
        """
        Args:
            z_t: State token hidden state. Shape (B, llm_dim).
            role_logits: From certificate head. Dict[SlotName, Tensor(B, 3)].
            goal_preds: From certificate head. Dict[SlotName, Tensor(B, dim)].
            actions_hidden_states: LLM action hidden states.
                Shape (B, num_actions_chunk * action_dim, llm_dim).
                即 (B, 56, 4096): 从 LLM 最后一层提取的动作令牌隐藏状态

        Returns:
            actions: Final combined actions. Shape (B, chunk_len, action_dim).
                     即 (B, 8, 7): 最终输出的动作序列
            actions_coarse: Coarse actions. Shape (B, chunk_len, action_dim).
                           用于损失计算和诊断
            actions_fine: Fine residual. Shape (B, chunk_len, action_dim).
                         用于损失计算和诊断
        """
        # 步骤 1: 展平并嵌入证书信息
        # flatten: Dict 结构 → (B, cert_raw_dim)
        # embed: (B, cert_raw_dim) → (B, cert_embed_dim=256)
        # Flatten and embed certificate
        cert_flat = self.flatten_certificate(role_logits, goal_preds)
        cert_embed = self.cert_embedding(cert_flat)  # (B, cert_embed_dim)

        # 步骤 2: 粗分支 — 仅从 z_t 和证书生成语义级动作
        # 输入: z_t (B, 4096) + cert_embed (B, 256)
        # 输出: (B, 8, 7) 粗略动作序列
        # Coarse: z_t + cert -> action semantics
        actions_coarse = self.coarse(z_t, cert_embed)  # (B, chunk, action_dim)

        # 步骤 3: 细分支 — 利用 LLM 动作隐藏状态做几何精修
        # 输入: actions_hidden_states (B, 56, 4096) + z_t + cert_embed
        # 输出: (B, 8, 7) 残差修正量
        # Fine: actions_hidden_states + z_t + cert -> geometric residual
        actions_fine = self.fine(actions_hidden_states, z_t, cert_embed)  # (B, chunk, action_dim)

        # 步骤 4: 粗细合成 — A = A_coarse + lambda_res * Delta_A_fine
        # lambda_res 是可学习标量，控制细分支的贡献比例
        # Combine
        actions = actions_coarse + self.lambda_res * actions_fine

        return actions, actions_coarse, actions_fine
