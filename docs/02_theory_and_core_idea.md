# CertVLA 核心理论与代码对照

> 本文档解释 CertVLA 的核心数学对象，以及它们在代码中的精确对应关系。

## 1. 核心问题

现有 VLA（Vision-Language-Action）模型在长程操作任务中表现脆弱。不是因为缺少 planner 或 memory，而是因为：

> **VLA 的内部隐状态没有被定义为一个"对局部动作后果负责的对象"。**

大多数 VLA 把内部状态当作"编码了更多历史信息"的向量，但从未显式要求：这个状态应该能够**预测和验证**即将执行的动作将产生什么结构化后果。

## 2. 核心洞察

一个好的 VLA 内部状态，应该是：

> **对下一段动作的结构化后果可证实的最小充分状态（the minimal state sufficient to certify the structured consequence of the next action chunk）**

这不是：
- ❌ "记住更多历史"的 memory
- ❌ "更容易预测动作"的 latent
- ❌ "外挂一个 verifier"的系统拼装

而是：
- ✅ 一个**递归的、可读出的、可验证的**任务状态抽象

## 3. 数学对象与代码对照

### 3.1 输入与输出

每个 action chunk 开始时，模型接收：

```
输入：(o_t, l, z_{t-1})
  o_t     = 当前观测（RGB 图像）
  l       = 语言指令
  z_{t-1} = 上一步的递归任务状态
```

模型输出：

```
输出：(z_t, ŝ_t, ĉ_t, Â_t)
  z_t  = 更新后的递归 certifiable task state
  ŝ_t  = 当前任务状态读数（state readout）
  ĉ_t  = 下一段动作的 semi-structured consequence certificate
  Â_t  = 连续动作 chunk（8 步 × 7 维）
```

### 3.2 z_t — 递归 Certifiable Task State

**数学定义**：
```
z̃_t = f_θ(o_t, l, z_{t-1})    ← LLM 在 state token 位置的隐状态
g_t  = σ(W_g[z̃_t ; z_{t-1}] + b_g)   ← 门控值
z_t  = g_t · z̃_t + (1 - g_t) · z_{t-1}  ← 门控更新
```

**代码对应**：
| 符号 | 代码位置 | 变量名 |
|------|---------|--------|
| z_0 | `certvla/model/state_token.py:StateTokenModule.z_0` | `self.z_0` — nn.Parameter(1, llm_dim) |
| z̃_t | `state_token.py:gated_update()` 的参数 | `tilde_z_t` — 从 LLM hidden states 提取 |
| g_t | `state_token.py:gated_update()` | `gate` = sigmoid(gate_proj([z̃; z_prev])) |
| z_t | `state_token.py:gated_update()` 的返回值 | `z_t` = gate * tilde_z + (1-gate) * z_prev |

**关键设计**：
- v1 训练时，每个样本都从 z_0 开始（无跨 chunk 递归）
- 门控初始化为 ~0.5（权重全零），让模型自由学习新旧状态的混合比例
- z_t 的维度 = LLM hidden dim = 4096（Llama 2 7B）

### 3.3 ŝ_t — State Readout

**数学定义**：
```
ŝ_t = R_φ(z_t)
```

R_φ 从 z_t **单独**读出 10 个 slot 的预测值。

**⚠️ 关键约束**：readout **只能**读 z_t 这一个向量，**不能**访问 LLM 的完整隐状态序列。这防止了 readout 绕过 z_t 直接从视觉 token 读信息。

**代码对应**：`certvla/model/state_readout.py:StateReadoutHead`
- 共享 trunk：LayerNorm → Linear → ReLU → Linear → ReLU
- 每个 slot 一个输出头
- Binary/Continuous 用 sigmoid 激活，Categorical 输出原始 logits

### 3.4 10 个 Task-Relative Slot

Slot 不是全局 scene graph，而是**围绕任务目标**定义的关系量。

| # | Slot 名称 | 域 | 家族 | 语义 |
|---|-----------|-----|------|------|
| 1 | `ee_target_proximity` | continuous [0,1] | J_E | 末端执行器与目标物距离 |
| 2 | `hand_occupancy` | categorical {empty, target, other} | J_E | 手中抓持物 |
| 3 | `target_contact` | binary {0,1} | J_E | 是否接触目标物 |
| 4 | `target_goal_proximity` | continuous [0,1] | J_R | 目标物离目标位置的距离 |
| 5 | `support_relation` | categorical {none, on_goal, on_other} | J_R | 支撑关系 |
| 6 | `containment_relation` | categorical {none, in_goal, in_other} | J_R | 包含关系 |
| 7 | `articulation_progress` | continuous [0,1] | J_E | 关节进度（开关门等） |
| 8 | `orientation_alignment` | continuous [0,1] | J_E | 姿态对齐度 |
| 9 | `completion_latch` | binary {0,1} | J_R | 任务完成锁存 |
| 10 | `task_visible_confidence` | confidence [0,1] | J_C | 任务可观测置信度 |

**代码对应**：`certvla/slots/schema.py:SLOT_REGISTRY`

**Slot 家族**（`certvla/slots/role_sets.py`）：
- **J_E**（5 个 enabling/transit slot）：推进任务进程的前置条件
- **J_R**（4 个 result/latch slot）：任务完成的结果标志
- **J_C**（1 个 confidence slot）：观测置信度
- **J_CERT = J_E ∪ J_R**（9 个 cert slot）：参与证书的 slot

### 3.5 ĉ_t — Consequence Certificate

**数学定义**：
```
ĉ_t = {(u_t^j, g_t^j)}  for j ∈ J_CERT
  u_t^j ∈ {advance, preserve, ignore}  — 角色分类
  g_t^j = 目标值（仅当 u_t^j = advance 时有意义）
```

**三种角色**：
- **advance**：这个 slot 在这段动作中会发生有意义的变化，朝目标推进
- **preserve**：这个 slot 必须保持不变（比如抓着的物体不能掉）
- **ignore**：这个 slot 在当前动作中不相关

**代码对应**：`certvla/model/certificate_head.py:CertificateHead`
- 共享 trunk + 每个 cert slot 的 role_head（3-way 分类器）和 goal_head
- 角色常量：`ROLE_ADVANCE=0, ROLE_PRESERVE=1, ROLE_IGNORE=2`

**证书挖掘**（离线标注数据用）：`certvla/data/certificate_mining.py:mine_certificate()`
- advance：数据驱动挖掘（delta 变化量、rho 持续性、upsilon 目标效用 等阈值）
- preserve：结构规则（latch-preserve、support-preserve），见 `certvla/slots/preserve_rules.py`

### 3.6 Â_t — 动作 Chunk

**数学定义**：
```
A_coarse = π_c(z_t, ĉ_t)              ← 粗动作（无观测输入！）
ΔA_fine  = π_f(o_t, z_t, ĉ_t)        ← 精细残差（有观测输入）
Â_t = A_coarse + λ_res · ΔA_fine     ← 最终动作
```

**⚠️ 关键设计**：粗动作分支**不接收观测**。这确保了动作语义必须依赖证书。精细分支通过 LLM 的 action hidden states（编码了视觉信息）添加几何修正。

**代码对应**：`certvla/model/action_head.py`
- `CoarseActionBranch`：输入 z_t + cert_embed → 输出 (B, 8, 7)
- `FineActionBranch`：输入 actions_hidden_states + z_t + cert_embed → 输出 (B, 8, 7)
- `CertActionHead`：组合以上，λ_res 初始化为 0.1

### 3.7 Γ_t — Certificate Gap

**数学定义**：
```
γ_t^j = p_adv^j · d_j(ĝ_t^j, ŝ_{t+H}^j) + p_pre^j · d_j(ŝ_t^j, ŝ_{t+H}^j)

Γ_t = [Σ_j ω_j · κ_t^j · γ_t^j] / [Σ_j ω_j · κ_t^j + ε]
```

执行完一个 action chunk 后，观测新的状态 ŝ_{t+H}，计算证书预测与实际结果之间的 gap。

- 如果 Γ_t 小 → 证书预测正确 → **继续执行**
- 如果 Γ_t 大 → 证书预测失败 → **触发 repair**

**代码对应**：
- 逐 slot gap：`certvla/inference/gap.py:slot_gap()`
- 聚合 gap：`certvla/inference/gap.py:aggregate_certificate_gap()`
- Repair 控制器：`certvla/inference/repair.py:RepairController`

## 4. 总体前向流程

```
观测 o_t + 指令 l
        ↓
   [LLM 前向传播]  ← state token 被注入到输入序列中
        ↓
   last_hidden_states (B, seq_len, 4096)
        ↓
   提取 state token 位置 → z̃_t
        ↓
   门控更新 → z_t
        ↓
   ┌────────────┬──────────────┐
   ↓            ↓              ↓
  R_φ(z_t)   Q_ψ(z_t)    π(z_t, ĉ_t, o_t)
   ↓            ↓              ↓
  ŝ_t        ĉ_t           Â_t
  (10 slots)  (9 roles+goals) (8×7 actions)
```

**代码对应**：`certvla/model/certvla_wrapper.py:CertVLAWrapper.forward()`

## 5. 闭环执行流程

```
loop:
  1. 观测 o_t → LLM 前向 → z_t, ŝ_t, ĉ_t, Â_t
  2. 执行 Â_t（8 步）
  3. 观测 o_{t+H} → 计算 ŝ_{t+H}
  4. 计算 gap: Γ_t = Gap(ĉ_t, ŝ_t, ŝ_{t+H})
  5. 如果 Γ_t ≤ threshold → continue
     如果 Γ_t > threshold → repair（重试最多 N 次，取最低 gap）
  6. z_{t-1} ← z_t，继续下一个 chunk
```

**代码对应**：`certvla/inference/repair.py:RepairController.step()`
