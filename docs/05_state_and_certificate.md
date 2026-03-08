# CertVLA 状态与证书机制详解

> 本文档深入解析 CertVLA 的核心组件：持久状态 token 和结果证书。
> 阅读本文档前建议先阅读 `docs/02_theory_and_core_idea.md` 了解整体思想。

---

## 1. 为什么需要持久状态 token

### 1.1 普通 VLA 的局限

普通 VLA（如 OpenVLA-OFT）在每个时间步独立处理当前观测 `o_t` 和语言指令 `l`，直接输出动作 `A_t`。这种"无状态"设计存在根本性问题：

- **无法跨 chunk 记忆**：每个 action chunk（如 8 步动作序列）的预测完全独立，模型无法记住"上一段动作完成到什么程度了"。
- **无法追踪任务进度**：对于"先拿起杯子，再放到盘子上"这种多阶段任务，模型无法显式表示"当前处于哪个阶段"。
- **无法认证动作后果**：无法预测"下一段动作预期会产生什么结果"，也就无法在执行后验证结果是否符合预期。

### 1.2 CertVLA 的解决方案

CertVLA 引入一个**持久递归状态向量** `z_t`，它不是简单的历史编码，而是"对动作后果负责的状态摘要"。`z_t` 通过门控更新机制在每个 chunk 之间递归传递，承担三重职责：

1. **状态读出的唯一信息源**：`z_t` 是下游状态读出头（`StateReadoutHead`）的唯一输入。读出头必须从 `z_t` 这单一向量中解码出全部 10 个结构化槽位值，不能"偷看"视觉令牌或文本令牌。
2. **证书预测的唯一信息源**：证书头（`CertificateHead`）同样只接收 `z_t`，用于预测每个槽位的角色（advance / preserve / ignore）和目标值。
3. **动作生成的条件之一**：动作头通过 `z_t` + 证书嵌入来生成粗略动作，再结合 LLM 隐状态做精细校正。

这种**信息瓶颈设计**迫使 LLM 在前向传播时将所有任务相关信息压缩到 `z_t` 这一个位置，使得 `z_t` 真正成为"充分且可认证的任务状态"。

---

## 2. 状态 Token 模块 (`StateTokenModule`)

> 源文件：`certvla/model/state_token.py`

### 2.1 `z_0`: 可学习初始状态

```python
self.z_0 = nn.Parameter(torch.zeros(1, llm_dim))
nn.init.normal_(self.z_0, mean=0.0, std=0.02)
```

- **形状**：`(1, llm_dim)` = `(1, 4096)`（对应 Llama 2 7B 的隐层维度）
- **初始化**：先置零再用小方差正态分布初始化（`std=0.02`），打破对称性，让不同维度可以学到不同语义
- **共享方式**：所有 batch 样本共享同一个 `z_0`，通过 `expand()` 扩展到 `(B, llm_dim)`，不复制内存
- **v1 训练策略**：每个训练样本都从 `z_0` 开始（无 episode 级递归），因为训练数据是 chunk 级独立采样

### 2.2 状态 Token 嵌入注入

```python
def get_state_token_embedding(self, batch_size: int) -> torch.Tensor:
    return self.z_0.expand(batch_size, -1).unsqueeze(1)  # (B, 1, llm_dim)
```

- **返回形状**：`(B, 1, llm_dim)`，中间的 `1` 对应序列中的一个额外令牌位置
- **注入位置**：在 LLM 输入序列中位于**视觉 patch 令牌之后、文本令牌之前**

输入序列结构示意：

```
[视觉 patch tokens, 0..N-1] [状态 token, N] [文本 tokens, N+1..M] [动作 tokens, M+1..end]
                                   ^
                          state_token_pos = N
```

状态 token 在这个位置可以通过 LLM 的自注意力机制同时关注视觉信息和文本指令，从而在前向传播过程中聚合全序列上下文。

### 2.3 门控更新机制

门控更新是 CertVLA 状态递归的核心操作。LLM 前向传播结束后，从最后一层隐藏状态中提取状态 token 位置的输出 `tilde_z_t`，然后与上一时刻状态 `z_{t-1}` 融合：

```python
def gated_update(self, tilde_z_t, z_prev):
    concat = torch.cat([tilde_z_t, z_prev], dim=-1)  # (B, 2*llm_dim) = (B, 8192)
    gate = torch.sigmoid(self.gate_proj(concat))       # (B, llm_dim)
    z_t = gate * tilde_z_t + (1.0 - gate) * z_prev     # (B, llm_dim)
    return z_t, gate
```

**数学公式**：

```
z_tilde_t = LLM_hidden_state[state_token_pos]      -- LLM 的"提议状态"
gate = sigmoid(W_g * [z_tilde_t ; z_{t-1}] + b_g)  -- 逐元素门控值
z_t  = gate * z_tilde_t + (1 - gate) * z_{t-1}     -- 加权融合
```

**关键设计细节**：

| 设计点 | 说明 |
|--------|------|
| 逐元素门控 | `gate` 形状为 `(B, llm_dim)` 而非标量，允许 4096 个维度各自决定更新速率 |
| 门控初始化 | `W_g` 权重和 `b_g` 偏置全部初始化为零 → `sigmoid(0) = 0.5`，训练开始时新旧状态各占 50% |
| 类 GRU 设计 | 类似于 GRU 的更新门，但作用于整个任务状态向量，而非 RNN 隐状态 |

**直觉理解**：

- 当 `gate ≈ 1` 时：状态几乎完全更新为新信息（任务状态发生大变化，如完成抓取进入搬运阶段）
- 当 `gate ≈ 0` 时：状态几乎不变（任务状态保持稳定，如持续搬运过程中）
- `gate ≈ 0.5`（初始值）：新旧各半，是安全的中性起点，让模型自由学习最优混合比例

---

## 3. 状态读出 (`StateReadoutHead`)

> 源文件：`certvla/model/state_readout.py`

### 3.1 信息瓶颈设计

```
z_t (B, 4096) ──→ StateReadoutHead ──→ s_t = {slot_1, slot_2, ..., slot_10}
     唯一输入             ↑
              不可访问 LLM 完整序列
```

这是 CertVLA 最关键的架构约束之一：

- 读出头的输入**仅仅是 `z_t` 向量**（形状 `(B, 4096)`），而**不是** LLM 的完整序列隐藏状态 `(B, seq_len, 4096)`。
- 如果读出头能直接访问完整序列，它可以"偷看"视觉令牌绕过状态压缩，导致 `z_t` 退化为无信息的占位符。
- 这一约束使得 `z_t` 的质量可以通过读出精度来量化评估。

### 3.2 `StateReadoutHead` 架构

```
z_t (B, 4096) → LayerNorm → Linear(4096, 512) → ReLU → Linear(512, 512) → ReLU → h (B, 512)
                                                                                     │
                ┌────────────────────────────────────────────────────────────────────┘
                │
    ┌───────────┼───────────┬───────────┬───────────┬─── ... ───┐
    ↓           ↓           ↓           ↓           ↓           ↓
 slot_1       slot_2      slot_3      slot_4      ...       slot_10
Linear(512,1) Linear(512,3) Linear(512,1)  ...              Linear(512,1)
  sigmoid      (raw logits)  sigmoid                         sigmoid
```

**共享主干（Shared Trunk）**：

- `LayerNorm(llm_dim)`：归一化 `z_t`，稳定输入分布（`z_t` 来自 LLM 不同层，数值范围波动较大）
- 两层 MLP + ReLU：`Linear(4096→512) → ReLU → Linear(512→512) → ReLU`
- 为什么选 `hidden_dim=512`：`z_t` 已是高度压缩的状态，512 维足以做槽位解码，更大会增加过拟合风险

**每槽位输出头（Per-Slot Heads）**：

每个槽位有一个独立的 `Linear(512, out_dim)` 层，`out_dim` 取决于槽位的域类型：

| 域类型 | 输出维度 | 激活函数 | 输出形状 |
|--------|----------|----------|----------|
| `BINARY`（二值） | 1 | `sigmoid` | `(B, 1)` |
| `CONTINUOUS`（连续） | 1 | `sigmoid` | `(B, 1)` |
| `CONFIDENCE`（置信度） | 1 | `sigmoid` | `(B, 1)` |
| `CATEGORICAL`（分类） | `num_categories` | 无（原始 logits） | `(B, num_categories)` |

分类槽位不在此处做 `softmax`，因为 `CrossEntropyLoss` 内部会处理（数值更稳定，利用 log-sum-exp 技巧）。

### 3.3 10 个 Slot 详解

CertVLA v1 定义了 10 个固定的结构化槽位，构成"slot 词汇表"。所有槽位都是**任务相对的**（task-relative），即同一物理状态在不同任务指令下可能映射为不同的槽位值。

| # | 槽位名称 | 域类型 | 族 | 语义含义 |
|---|----------|--------|-----|----------|
| 1 | `ee_target_proximity` | `CONTINUOUS [0,1]` | `J_E` | 末端执行器到目标物体的归一化距离；0=接触，1=最远 |
| 2 | `hand_occupancy` | `CATEGORICAL {empty, target, other}` | `J_E` | 手爪抓取状态：空 / 抓着目标 / 抓着其他物体 |
| 3 | `target_contact` | `BINARY {0, 1}` | `J_E` | 末端执行器是否与目标物体接触 |
| 4 | `articulation_progress` | `CONTINUOUS [0,1]` | `J_E` | 铰接体（门/抽屉）打开进度；0=完全关闭，1=完全打开 |
| 5 | `orientation_alignment` | `CONTINUOUS [0,1]` | `J_E` | 目标物体朝向与期望朝向的对齐度；0=完全错位，1=完美对齐 |
| 6 | `target_goal_proximity` | `CONTINUOUS [0,1]` | `J_R` | 目标物体到目标位置的归一化距离；0=已到达，1=最远 |
| 7 | `support_relation` | `CATEGORICAL {none, on_goal, on_other}` | `J_R` | 目标物体的支撑关系：无 / 在目标表面上 / 在其他物体上 |
| 8 | `containment_relation` | `CATEGORICAL {none, in_goal, in_other}` | `J_R` | 目标物体的包含关系：无 / 在目标容器中 / 在其他容器中 |
| 9 | `completion_latch` | `BINARY {0, 1}` | `J_R` | 任务完成锁存位；一旦置 1 语义上不应回退 |
| 10 | `task_visible_confidence` | `CONFIDENCE [0,1]` | `J_C` | 任务相关物体是否可见的置信度；低值表示观测不可靠 |

> 源文件：`certvla/slots/schema.py` — 定义 `SlotName`, `SlotDomain`, `SlotFamily`, `SlotMeta`, `SLOT_REGISTRY`

---

## 4. 证书机制 (`Certificate`)

### 4.1 什么是结果证书

证书是 CertVLA 对"下一段动作会产生什么结构化后果"的**显式预测**。数学上定义为：

```
c_t = {(role_j, goal_j)}  对于 j ∈ J_CERT
```

其中 `J_CERT = J_E ∪ J_R` 包含 9 个参与证书判定的槽位（排除了置信度槽位 `task_visible_confidence`）。

证书的核心价值在于：

1. **可解释性**：通过查看 `role` 分布，可以理解模型认为"当前正在做什么"
2. **安全性**：`preserve` 角色可以检测模型是否打算做违反约束的事情
3. **动作条件化**：证书信息直接输入动作头，引导动作生成
4. **执行后验证**：执行动作后可以比较实际状态与证书预测，计算"证书间隙"（gap）

### 4.2 三种角色

对于 `J_CERT` 中的每个槽位 `j`，证书头预测一个**三分类角色** `role_j`：

| 角色 | 索引 | 含义 | `goal_j` 是否有意义 |
|------|------|------|---------------------|
| `advance`（推进） | 0 | 该槽位在当前 chunk 内**预计会发生变化**，朝着目标值推进 | 有意义：预测 chunk 结束时的目标值 |
| `preserve`（保持） | 1 | 该槽位在当前 chunk 内**必须保持不变**，是安全约束 | 无意义：保持当前值 |
| `ignore`（忽略） | 2 | 该槽位**与当前任务阶段无关** | 无意义：不参与损失计算 |

角色常量定义在 `certvla/model/certificate_head.py`：

```python
ROLE_ADVANCE = 0    # 推进
ROLE_PRESERVE = 1   # 保持
ROLE_IGNORE = 2     # 忽略
NUM_ROLES = 3
```

**典型场景举例**（抓取并放置任务）：

| 阶段 | `ee_target_proximity` | `hand_occupancy` | `target_goal_proximity` | `support_relation` |
|------|----------------------|-------------------|------------------------|--------------------|
| 靠近目标 | **advance** (缩小距离) | ignore | ignore | ignore |
| 抓取 | **advance** (接触) | **advance** (→target) | ignore | ignore |
| 搬运 | ignore | **preserve** (保持抓住) | **advance** (接近目标位) | ignore |
| 放置 | ignore | **advance** (→empty) | **advance** (到达) | **advance** (→on_goal) |

### 4.3 `CertificateHead` 架构

> 源文件：`certvla/model/certificate_head.py`

```
z_t (B, 4096) → LayerNorm → Linear(4096, 512) → ReLU → Linear(512, 512) → ReLU → h (B, 512)
                                                                                     │
                ┌────────────────────────────────────────────────────────────────────┘
                │
    ┌───────────┼──────────────────────────────────── × 9 个证书槽位 ───┐
    │           │                                                       │
    ├─ role_head: Linear(512, 3) ─→ role_logits (B, 3)                 │
    │   三分类: [advance, preserve, ignore]                             │
    │                                                                   │
    └─ goal_head: Linear(512, goal_dim) ─→ goal_preds (B, dim)        │
        目标值预测（激活函数同 StateReadoutHead）                        │
                                                                        │
        ────────────────────────────────────────────────────────────────┘
```

**与 `StateReadoutHead` 的对比**：

| 特性 | `StateReadoutHead` | `CertificateHead` |
|------|--------------------|--------------------|
| 处理的槽位数 | 全部 10 个 | 仅 `J_CERT` 中的 9 个 |
| 输出内容 | 每槽位 1 个预测值 | 每槽位 2 个输出（角色 + 目标值） |
| 共享主干结构 | `LayerNorm→Linear→ReLU→Linear→ReLU` | 结构相同但**参数独立** |
| 信息隔离 | 仅接收 `z_t` | 仅接收 `z_t` |

**`goal_head` 输出维度**：与 `StateReadoutHead` 的 `_output_dim` 逻辑一致：

- `BINARY` → 1 维，经 `sigmoid`
- `CATEGORICAL` → `num_categories` 维，原始 logits
- `CONTINUOUS` / `CONFIDENCE` → 1 维，经 `sigmoid`

### 4.4 `CertificateEmbedding`: 角色到动作的条件传递

> 源文件：`certvla/model/action_head.py` 中的 `CertificateEmbedding` 类

证书头的输出是两个字典（`role_logits` 和 `goal_preds`），包含异构形状的张量。为了将证书信息传递给动作头，需要先将其展平并嵌入为固定大小的向量：

```python
# 步骤 1: 展平 — 按槽位名称排序，拼接 role_logits 和 goal_preds
cert_flat = flatten(role_logits, goal_preds)  # (B, cert_raw_dim)
# cert_raw_dim = sum(3 + goal_dim for each slot in J_CERT)

# 步骤 2: 嵌入 — 通过 MLP 投影到紧凑空间
cert_embed = CertificateEmbedding(cert_flat)  # (B, cert_embed_dim=256)
```

`CertificateEmbedding` 的内部结构：

```python
self.proj = nn.Sequential(
    nn.Linear(cert_raw_dim, embed_dim),    # cert_raw_dim → 256
    nn.ReLU(),
    nn.Linear(embed_dim, embed_dim),       # 256 → 256
)
```

**排序一致性**：展平时按 `slot_name.value` 的字典序排列，确保每次展平的顺序相同。如果槽位顺序不一致，嵌入网络会收到错误的输入对齐，导致训练失败。

---

## 5. 证书-动作耦合 (`CertActionHead`)

> 源文件：`certvla/model/action_head.py`

### 5.1 粗-细双分支设计

```
                     ┌────────────────────────────┐
    z_t ─────────────┤                            │
                     │  CoarseActionBranch        │──→ A_coarse (B, 8, 7)
    cert_embed ──────┤  pi_c(z_t, cert_embed)     │
                     │  ** 不接收观测 **           │
                     └────────────────────────────┘
                                                          │
                     ┌────────────────────────────┐       │
    actions_hidden ──┤                            │       │
    (来自 LLM,      │  FineActionBranch          │       │
     编码了观测 o_t) │  pi_f(act_h, z_t, cert)   │──→ Delta_A_fine (B, 8, 7)
    z_t ─────────────┤                            │       │
    cert_embed ──────┤                            │       │
                     └────────────────────────────┘       │
                                                          ↓
                          A = A_coarse + lambda_res * Delta_A_fine
                          最终动作输出 (B, 8, 7)
```

**公式**：

```
A_coarse    = pi_c(z_t, cert_embed)                     -- 粗分支：语义级动作
Delta_A_fine = pi_f(actions_hidden_states, z_t, cert_embed)  -- 细分支：几何残差
A_t         = A_coarse + lambda_res * Delta_A_fine        -- 粗细合成
```

**`lambda_res` 参数**：

```python
self.lambda_res = nn.Parameter(torch.tensor(0.1))
```

- 可学习标量，初始化为 `0.1`
- 训练初期粗分支主导（约 91%），细分支贡献微小（约 9%）
- 随训练进行，模型自动学习最优的粗细比例
- 可通过监控 `lambda_res` 值来诊断双分支贡献平衡

### 5.2 粗分支 (`CoarseActionBranch`) — 不接收观测

```python
# 输入维度: llm_dim + cert_embed_dim = 4096 + 256 = 4352
# 输出维度: num_actions_chunk * action_dim = 8 * 7 = 56
self.net = nn.Sequential(
    nn.LayerNorm(input_dim),        # 归一化拼接后的输入
    nn.Linear(input_dim, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.LayerNorm(1024),             # 输出前归一化，稳定输出分布
    nn.Linear(1024, 56),            # 扁平动作向量 → reshape 为 (B, 8, 7)
)
```

**为什么粗分支不接收观测？** 这是证书-动作耦合的核心设计约束：

1. **强制证书携带语义**：粗分支只能从 `z_t + cert_embed` 生成动作，这迫使证书必须包含足够的任务语义信息。如果粗分支也能看到观测，模型可能绕过证书，直接从视觉生成动作，导致证书退化为摆设。
2. **依赖性保证**：证书变化 → 动作必然变化。这是"证书可认证动作"的前提。
3. **可解释性**：粗动作完全由任务状态和进展计划决定，回答"基于当前理解，应该做什么"这一语义层面的问题。

### 5.3 细分支 (`FineActionBranch`) — 接收观测

```python
# 输入 reshape 策略:
# actions_hidden_states: (B, 56, 4096) → reshape → (B, 8, 7*4096) = (B, 8, 28672)
# 每步拼接 z_t (4096) 和 cert_embed (256)
# 每步总输入: 28672 + 4096 + 256 = 33024

self.net = nn.Sequential(
    nn.LayerNorm(per_step_input_dim),  # 33024
    nn.Linear(33024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.LayerNorm(1024),
    nn.Linear(1024, 7),   # 每步输出 7 维残差修正
)
```

**设计要点**：

- `actions_hidden_states` 来自 LLM 在动作令牌位置的隐藏状态，已通过自注意力"看到"了视觉和文本令牌
- 所有 chunk 步共享相同的网络权重（PyTorch `Linear` 支持任意前导维度的 broadcasting）
- `z_t` 和 `cert_embed` 在每步都相同（通过 `unsqueeze + expand` 复制）
- 输出 `(B, 8, 7)` 是每步的残差修正量

### 5.4 动作空间说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `ACTION_DIM` | 7 | 6 个关节角度/速度 + 1 个抓手开合 |
| `NUM_ACTIONS_CHUNK` | 8 | 每次预测 8 步动作（chunk-based 预测） |
| 总输出维度 | 56 | `8 * 7 = 56` 个数值 |

---

## 6. Slot 家族体系

> 源文件：`certvla/slots/role_sets.py`

CertVLA 将 10 个 slot 分为三个家族（family），每个家族在证书系统中扮演不同的角色。家族集合通过 `SLOT_REGISTRY` 的 `family` 字段自动推导，使用 `frozenset` 保证不可变。

### 6.1 `J_E`: 使能/过渡 slot（5 个）

```python
J_E = frozenset({
    SlotName.EE_TARGET_PROXIMITY,      # 末端到目标的距离
    SlotName.HAND_OCCUPANCY,           # 手爪抓取状态
    SlotName.TARGET_CONTACT,           # 是否接触目标
    SlotName.ARTICULATION_PROGRESS,    # 铰接体打开进度
    SlotName.ORIENTATION_ALIGNMENT,    # 朝向对齐度
})
```

**语义**：描述"动作执行的前提条件是否满足"。这些 slot 在执行 action chunk 的**过程中**会发生变化，是任务推进的"过渡状态"。例如：

- 要抓住物体，首先 `ee_target_proximity` 必须减小到接触距离
- 要搬运物体，`hand_occupancy` 必须是 `target`
- 要放入容器，`articulation_progress`（容器的开合程度）必须达到打开状态

### 6.2 `J_R`: 结果/锁存 slot（4 个）

```python
J_R = frozenset({
    SlotName.TARGET_GOAL_PROXIMITY,    # 目标物体到目标位置的距离
    SlotName.SUPPORT_RELATION,         # 支撑关系
    SlotName.CONTAINMENT_RELATION,     # 包含关系
    SlotName.COMPLETION_LATCH,         # 任务完成锁存
})
```

**语义**：描述"动作执行后的目标结果"。这些 slot 在 action chunk **完成后**应达到期望值，且具有**锁存特性**：一旦达成就不应回退。例如：

- 放置完成后 `support_relation` 应变为 `on_goal`，不应再掉落
- 放入容器后 `containment_relation` 应变为 `in_goal`
- `completion_latch` 一旦置 1 则整个任务视为完成

### 6.3 `J_C`: 置信度 slot（1 个）

```python
J_C = frozenset({
    SlotName.TASK_VISIBLE_CONFIDENCE,  # 任务相关物体可见性置信度
})
```

**语义**：描述"当前观测是否足够可靠"。
- **不参与证书角色判定**（不属于 `J_CERT`）
- 用途：在间隙（gap）聚合时作为权重，低置信度的观测对间隙计算贡献更小
- 仿真环境中通常恒为 `1.0`（始终完全可见），在真实机器人场景中会变化

### 6.4 `J_CERT = J_E ∪ J_R`（9 个证书 slot）

```python
J_CERT = J_E | J_R  # 5 + 4 = 9 个槽位
```

这 9 个槽位参与完整的证书机制：

1. **角色分类**：证书头为每个 `J_CERT` 槽位预测 `advance / preserve / ignore`
2. **目标预测**：对 `advance` 角色的槽位预测 chunk 结束时的目标值
3. **间隙计算**：推理时比较证书预测与实际观测状态的差异

---

## 7. 端到端数据流

### 7.1 完整前向传播流程

> 源文件：`certvla/model/certvla_wrapper.py` — `CertVLAWrapper.forward()`

`CertVLAWrapper` 是 CertVLA 的顶层编排器，组合四个子模块并定义完整的前向传播流程。它**不是** `OpenVLAForActionPrediction` 的子类，而是一个独立的附加模块，接收 LLM 的 `last_hidden_states` 进行后处理。

```
        ┌──────────────────────────────────────────────────────┐
        │              LLM 前向传播                             │
        │  输入: [视觉patches] [状态token z_0] [文本] [动作]   │
        │  输出: last_hidden_states (B, seq_len, 4096)         │
        └──────────────────────┬───────────────────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
    步骤 1: 提取 state_token_pos      步骤 5: 提取 action_start_pos
    tilde_z_t = h[:, pos, :]         actions_hidden = h[:, start:end, :]
    (B, 4096)                        (B, 56, 4096)
                 │                           │
    步骤 2: 门控更新                         │
    z_t, gate = gated_update(                │
        tilde_z_t, z_prev)                   │
    (B, 4096)                                │
                 │                           │
         ┌───────┼───────────┐               │
         │       │           │               │
         ↓       ↓           ↓               │
     步骤 3   步骤 4       步骤 6             │
    readout  cert_head   action_head  ←──────┘
    R_phi(z_t) Q_psi(z_t)  pi(z_t, c_t, act_h)
         │       │              │
         ↓       ↓              ↓
        s_t    c_t = {         A_t = A_coarse
    (10个slot  (role_j,            + lambda_res
     预测值)    goal_j)}           * Delta_A_fine
                                (B, 8, 7)
```

### 7.2 六步流程详解

**步骤 1**：从 `last_hidden_states` 的 `state_token_pos` 位置提取 `tilde_z_t`

```python
tilde_z_t = last_hidden_states[:, state_token_pos, :]  # (B, llm_dim)
```

**步骤 2**：门控更新，融合 LLM 提议状态和上一时刻状态

```python
if z_prev is None:
    z_prev = self.state_token.get_initial_state(B)  # v1: z_0
z_t, gate = self.state_token.gated_update(tilde_z_t, z_prev)
```

**步骤 3**：状态读出（仅从 `z_t`）

```python
state_readout = self.state_readout(z_t)  # Dict[SlotName, Tensor]
```

**步骤 4**：证书头（仅从 `z_t`）

```python
role_logits, goal_preds = self.certificate_head(z_t)
```

**步骤 5**：提取动作隐藏状态

```python
action_end_pos = action_start_pos + action_dim * num_actions_chunk
actions_hidden_states = last_hidden_states[:, action_start_pos:action_end_pos, :]
```

**步骤 6**：证书条件化动作生成

```python
actions, actions_coarse, actions_fine = self.action_head(
    z_t=z_t,
    role_logits=role_logits,
    goal_preds=goal_preds,
    actions_hidden_states=actions_hidden_states,
)
```

### 7.3 输出数据结构

> 源文件：`certvla/model/outputs.py`

所有输出打包为 `CertVLAOutput` 数据类：

```python
@dataclass
class CertVLAOutput:
    z_t: torch.Tensor                          # (B, llm_dim) 持久状态
    state_readout: Dict[SlotName, Tensor]      # 10 个槽位的当前状态预测
    role_logits: Dict[SlotName, Tensor]        # 9 个证书槽位的角色 logits (B, 3)
    goal_preds: Dict[SlotName, Tensor]         # 9 个证书槽位的目标预测
    actions_coarse: Tensor                     # (B, 8, 7) 粗动作
    actions_fine: Tensor                       # (B, 8, 7) 细残差
    actions: Tensor                            # (B, 8, 7) 最终合成动作
    actions_hidden_states: Tensor              # (B, 56, 4096) LLM 动作隐状态
    gate_value: Optional[Tensor] = None        # (B, 1) 门控均值（诊断用）
```

---

## 8. 训练损失体系

> 源文件：`certvla/training/losses.py`

CertVLA 定义了 7 个损失项，通过课程学习（curriculum learning）逐阶段激活：

| 阶段 | 激活的损失项 | 训练目标 |
|------|-------------|----------|
| Stage 1 | `L_state` | 学会从 `z_t` 读出环境状态 |
| Stage 2 | + `L_role`, `L_goal` | 学会预测槽位角色和目标值 |
| Stage 3 | + `L_act`, `L_cons`, `L_dep` | 学会基于证书生成动作 |
| Stage 4 | + `L_cf` | 引入反事实增强提升鲁棒性 |

### 8.1 `L_state`（状态读出损失）

```
L_state = sum_j [ m^j * alpha^j * ell_j(hat_s^j, s^j) ]
```

- 遍历全部 10 个槽位，根据域类型选择损失函数：`BCE` / `CE` / `L1`
- `m^j`：掩码（是否有有效标注），`alpha^j`：置信度权重

### 8.2 `L_role`（角色分类损失）

```
L_role = sum_{j ∈ J_CERT} m^j * alpha^j * FocalCE(hat_u^j, u^j)
```

- 使用 **Focal Cross Entropy**（`gamma=2.0`）而非标准 CE
- 原因：大多数槽位在大多数时间步都是 `ignore`，存在极端类别不平衡
- Focal Loss 通过 `(1 - p_t)^gamma` 因子降低"容易样本"的权重

### 8.3 `L_goal`（目标预测损失）

```
L_goal = sum_{j: role^j = advance} m^j * alpha^j * ell_j(hat_g^j, s_{t+H}^j)
```

- 仅对 `role = advance` 的槽位计算（`preserve` 和 `ignore` 不参与）
- 监督信号来自 chunk 结束时的真实状态值

### 8.4 `L_act`（动作回归损失）

```
L_act = (1/H) * sum_k ||hat_a_{t+k} - a*_{t+k}||_1
```

- 标准 L1 动作回归损失，与 OpenVLA-OFT 原始损失形式一致

### 8.5 `L_cons`（结构一致性损失）

```
L_cons = L_adv_cons + lambda_pre * L_pre_cons
```

- **Advance 一致性**：目标预测 `goal^j` 应与 chunk 结束时的真实状态 `s_{t+H}^j` 一致
- **Preserve 一致性**：当前状态读出 `s_t^j` 应与 chunk 结束时的真实状态 `s_{t+H}^j` 一致

### 8.6 `L_dep`（证书依赖损失）

```
L_dep = mean max(0, margin + e_pos - e_neg)
```

- **三元组损失**：正样本（正确证书）的动作误差应小于负样本（被篡改证书）的动作误差
- 防止动作头忽略证书信息直接从 `z_t` 生成动作

### 8.7 `L_cf`（反事实损失）

```
L_cf = L_inv + L_brk
```

- `L_inv`：干扰不变性 — 仅改变无关因素（背景、光照），`z_t` 应保持不变
- `L_brk`：后果敏感性 — 改变任务相关因素（目标物体），`z_t` 应发生变化
- v1 为最小实现，完整增强对生成延后到后续版本

### 8.8 总损失加权

```python
L_total = lambda_s * L_state + lambda_r * L_role + lambda_g * L_goal
        + lambda_a * L_act   + lambda_c * L_cons + lambda_d * L_dep
        + lambda_cf * L_cf
```

权重通过 `cert_total_loss()` 函数组合，不同训练阶段通过将对应权重设为 `0.0` 来禁用相应损失项。

---

## 9. 推理时的证书间隙与修复

> 源文件：`certvla/inference/gap.py`, `certvla/inference/repair.py`

### 9.1 证书间隙计算

推理时，CertVLA 可以计算"证书间隙"（certificate gap）来检测动作执行是否偏离预期：

**逐槽位间隙**：

```
gamma_t^j = p_adv^j * d_j(goal^j, s_{t+H}^j)    -- advance 项: 目标预测 vs 实际结果
           + p_pre^j * d_j(s_t^j, s_{t+H}^j)      -- preserve 项: 当前状态 vs 结果状态
```

其中 `p_adv^j` 和 `p_pre^j` 是角色 logits 经 softmax 后的概率值。

**聚合间隙**：

```
Gamma_t = [ sum_{j ∈ J_CERT} omega_j * kappa_j * gamma_j ]
        / [ sum_{j ∈ J_CERT} omega_j * kappa_j + epsilon ]
```

- `omega_j`：静态槽位重要性权重（默认 `1.0`）
- `kappa_j`：动态置信度权重（可来自 `task_visible_confidence`）
- `epsilon`：数值稳定项

### 9.2 修复控制器

当聚合间隙 `Gamma_t` 超过阈值时，修复控制器（`RepairController`）触发重新预测：

```python
@dataclass
class RepairConfig:
    gap_threshold: float = 0.3       # 间隙阈值
    max_repair_steps: int = 3        # 最大重试次数
    use_best_of_n: bool = True       # 是否总是选最低间隙的尝试
```

**修复流程**：

1. 正常前向传播 → 计算间隙
2. 如果间隙 > 阈值 → 重新前向传播（最多 `max_repair_steps` 次）
3. 选择间隙最小的动作 chunk
4. 如果所有尝试都超过阈值 → 接受最低间隙的尝试并记录警告

这**不是**完整的重规划器，而是一个局部的、无状态的重试循环，依赖模型中的随机性（dropout、temperature）或门控状态更新提供略有不同的 `z_t`。

---

## 10. 关键文件索引

| 文件路径 | 职责 |
|----------|------|
| `certvla/slots/schema.py` | Slot 词汇表定义：`SlotName`, `SlotDomain`, `SlotFamily`, `SlotMeta`, `SLOT_REGISTRY` |
| `certvla/slots/role_sets.py` | 家族集合：`J_E`, `J_R`, `J_C`, `J_CERT` |
| `certvla/slots/metrics.py` | 逐 slot 距离函数 `d_j(a, b)` 和张量编码转换 |
| `certvla/slots/preserve_rules.py` | 锁存保持和支撑保持的结构化规则 |
| `certvla/model/state_token.py` | `StateTokenModule`：`z_0` 嵌入 + 门控更新 |
| `certvla/model/state_readout.py` | `StateReadoutHead`：`z_t → s_t` 逐 slot 读出 |
| `certvla/model/certificate_head.py` | `CertificateHead`：`z_t → (role_logits, goal_preds)` |
| `certvla/model/action_head.py` | `CertActionHead`：粗细双分支动作生成 + `CertificateEmbedding` |
| `certvla/model/certvla_wrapper.py` | `CertVLAWrapper`：顶层编排器，组合所有子模块 |
| `certvla/model/outputs.py` | `CertVLAOutput` 数据类 |
| `certvla/training/losses.py` | 7 个损失函数 + 总损失组合器 |
| `certvla/inference/gap.py` | 证书间隙计算 |
| `certvla/inference/repair.py` | 短时局部修复控制器 |
