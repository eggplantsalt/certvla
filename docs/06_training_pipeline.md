# CertVLA 训练流水线

> 本文档详细介绍 CertVLA 的训练流程、损失函数、课程学习策略和调度采样。

---

## 1. 训练概览

CertVLA 的训练分为 **4 个阶段**，逐步解冻模块、逐步添加损失项。

核心设计理念：**先学会"看"（状态读出），再学会"说"（角色分类），最后学会"做"（动作生成）。**

训练流水线的总步数为 **60,000 步**，各阶段边界如下：

| 阶段 | 名称 | 步数范围 | 新增损失项 |
|------|------|----------|-----------|
| Stage 1 | state | 0 - 5,000 | `L_state` |
| Stage 2 | certificate | 5,000 - 15,000 | + `L_role`, `L_goal` |
| Stage 3 | policy | 15,000 - 40,000 | + `L_act`, `L_cons`, `L_dep` |
| Stage 4 | counterfactual | 40,000 - 60,000 | + `L_cf` |

所有损失项都遵循统一的"逐槽位 (per-slot)"计算模式：

```
loss_j = m^j * alpha^j * ell_j(pred, target)
```

其中 `m^j` 是掩码（0 或 1，处理缺失数据），`alpha^j` 是置信度权重（[0, 1]，不确定的标注给予较低权重），`ell_j` 是根据槽位类型 (domain) 选择的基础损失函数。

**相关代码文件：**

| 文件 | 职责 |
|------|------|
| `certvla/training/losses.py` | 7 个损失函数的定义与总损失组合 |
| `certvla/training/curriculum.py` | 课程学习调度器 (`CurriculumScheduler`) |
| `certvla/training/sched_sampling.py` | 调度采样 (`ScheduledSampler`) |
| `certvla/training/__init__.py` | 统一导出训练层的所有公开接口 |
| `certvla/slots/role_sets.py` | 槽位族集合定义 (`J_E`, `J_R`, `J_CERT`) |
| `scripts/certvla/train_baseline.sh` | 基线训练启动脚本 |

---

## 2. 七个损失函数

CertVLA 定义了 7 个损失项，在训练的不同阶段被逐步激活。下面逐一详细说明。

### 2.1 `L_state` -- 状态读出损失

**公式：**

```
L_state = sum_j [ m^j * alpha^j * ell_j(s_hat^j, s^j) ]
```

**作用：** 训练模型从持久状态 token `z_t` 中准确读出环境状态。这是整个证书机制的基础——如果状态读出不准确，证书的角色分类和目标预测都会失去意义。

**代码入口：** `certvla/training/losses.py` 中的 `cert_state_loss()` 函数。

**计算细节：**

- 遍历所有在 `state_readout` 和 `state_target` 中都存在的槽位
- 根据槽位的 `domain` 类型选择不同的损失函数：
  - `BINARY`（二值，如"抓手是否打开"）：二元交叉熵 (BCE)
  - `CATEGORICAL`（分类，如"当前阶段是 reach/grasp/lift"）：交叉熵 (CE)
  - `CONTINUOUS`（连续，如"物体离目标的距离"）：L1 损失
  - `CONFIDENCE`（置信度）：L1 损失
- 逐样本计算后，应用掩码 `m` 和置信度 `alpha` 加权，再对 batch 求均值
- 所有槽位的平均损失求和得到总状态损失
- 若无有效槽位，返回可导的零张量 (`_zero()`)

**激活时机：** Stage 1 唯一激活的损失，是训练流水线的第一个目标。

**张量形状流：**

```
state_readout[slot]:  (B, 1) 或 (B, num_classes)  -- 取决于 domain
state_target[slot]:   (B, 1) float 或 (B,) long
mask[slot]:           (B,) float, 值为 0.0 或 1.0
confidence[slot]:     (B,) float, 值在 [0, 1]
输出:                 标量张量
```

---

### 2.2 `L_role` -- 角色分类损失

**公式：**

```
L_role = sum_{j in J_CERT} [ m^j * alpha^j * FocalCE(u_hat^j, u^j) ]
```

**作用：** 训练证书头 (certificate head) 为每个槽位预测其"角色"。

**代码入口：** `certvla/training/losses.py` 中的 `cert_role_loss()` 函数。

**三种角色：**

| 角色 | 编码 | 含义 |
|------|------|------|
| advance（前进） | 0 | 该槽位是当前任务要改变的目标状态 |
| preserve（保持） | 1 | 该槽位应在动作执行过程中保持不变 |
| ignore（忽略） | 2 | 该槽位与当前任务无关 |

**为什么使用 Focal CE 而非标准 CE：**

在典型的机器人操作场景中，大多数槽位在大多数时间步都是 `ignore` 类别（例如 9 个证书槽位中通常只有 1-2 个是 `advance`），导致极端的类别不平衡。Focal Loss 通过 `(1 - p_t)^gamma` 因子降低"容易样本"（如轻松分类的 `ignore` 样本）的权重，使模型将训练精力集中在"困难样本"（如 `advance` / `preserve` 类别）上。

```
FL(p_t) = -(1 - p_t)^gamma * log(p_t)
```

- `gamma=0` 退化为标准交叉熵
- `gamma=2.0`（默认值，来自 Lin et al. 2017）在大多数不平衡场景下表现良好
- `gamma` 越大，对容易样本的抑制越强

**计算细节：**

- 仅遍历 `J_CERT` 中的 9 个证书槽位（`J_E` 的 5 个使能槽位 + `J_R` 的 4 个结果槽位）
- 非证书槽位（如 `task_visible_confidence`）不参与角色分类

**激活时机：** Stage 2 开始激活。

---

### 2.3 `L_goal` -- 目标值损失

**公式：**

```
L_goal = sum_{j: u^j=advance} [ m^j * alpha^j * ell_j(g_hat^j, s_{t+H}^j) ]
```

**作用：** 训练证书头预测 advance 槽位在动作块 (action chunk) 执行完毕后的目标状态值。

**代码入口：** `certvla/training/losses.py` 中的 `cert_goal_loss()` 函数。

**计算细节：**

- 仅遍历 `J_CERT` 中的 9 个证书槽位
- 通过 `advance_mask = (roles == ROLE_ADVANCE).float()` 过滤，**只对 `role=advance` 的样本计算损失**
- 对于 `preserve` 和 `ignore` 槽位，损失贡献为零
- 最终将 `m * alpha * advance_mask * sl` 对 batch 求均值后累加

**激活时机：** Stage 2 开始激活。

---

### 2.4 `L_act` -- 动作损失

**公式：**

```
L_act = (1/H) * sum_k ||a_hat_{t+k} - a*_{t+k}||_1
```

**作用：** 使用 L1 回归损失训练动作头生成动作块。

**代码入口：** `certvla/training/losses.py` 中的 `cert_action_loss()` 函数。

**计算细节：**

- `pred_actions` 形状为 `(B, H, action_dim)`，`H` 为 chunk 长度
- `expert_actions` 形状为 `(B, H, action_dim)`，来自专家演示
- 使用 `F.l1_loss(..., reduction="mean")` 直接计算，自动包含对 H 维度的平均

**激活时机：** Stage 3 开始激活。

---

### 2.5 `L_cons` -- 一致性损失

**公式：**

```
L_cons = L_adv_cons + lambda_pre * L_pre_cons
```

**作用：** 确保模型预测的结构一致性。由两部分组成：

1. **`L_adv_cons`（advance 一致性）：** 对于 `role=advance` 的槽位，目标预测 (`goal_preds`) 应匹配 chunk 结束时刻 `t+H` 的真实状态值
2. **`L_pre_cons`（preserve 一致性）：** 对于 `role=preserve` 的槽位，当前时刻的状态读出 (`state_readout`) 应匹配 chunk 结束时刻 `t+H` 的真实状态值（因为 preserve 意味着"不变"）

**代码入口：** `certvla/training/losses.py` 中的 `cert_consistency_loss()` 函数。

**参数说明：**

- `lambda_pre`：preserve 项的内部权重，默认 `1.0`
- `state_target_tH`：chunk 结束时刻的状态真值（v1 使用 ground truth；v2 可替换为模型自身预测）

**激活时机：** Stage 3 开始激活。

---

### 2.6 `L_dep` -- 依赖性损失

**公式：**

```
L_dep = mean[ max(0, margin + e_pos - e_neg) ]
```

**作用：** 三元组 (triplet) 损失，强制模型在使用正确证书时产生更小的动作误差，在使用错误（被篡改的）证书时产生更大的动作误差。这确保了动作头对证书信息的依赖性。

**代码入口：** `certvla/training/losses.py` 中的 `cert_dependence_loss()` 函数。

**计算细节：**

- `e_pos`：正确证书下的动作误差 `||actions_pos - expert_actions||_1`
- `e_neg`：错误证书下的动作误差 `||actions_neg - expert_actions||_1`
- `margin`：最小间隔，默认 `0.1`
- 需要额外的 negative-cert forward pass（课程调度器通过 `should_compute_dep()` 控制是否执行）

**张量形状流：**

```
expert_actions: (B, H, action_dim)
actions_pos:    (B, H, action_dim)  -- 正确证书下的动作输出
actions_neg:    (B, H, action_dim)  -- 错误证书下的动作输出
e_pos, e_neg:   (B,)               -- 每样本误差
输出:           标量张量
```

**激活时机：** Stage 3 开始激活。

---

### 2.7 `L_cf` -- 反事实损失

**公式：**

```
L_cf = L_inv + L_brk
```

**作用：** 通过反事实数据增强提升模型的鲁棒性，由两部分组成：

1. **`L_inv`（不变性项）：** `z_t` 应接近 `z_pos`（仅改变与任务无关的因素，如背景、光照、干扰物）。使用 MSE 损失 `||z_t - z_pos||_2^2`。可选地加上增强观测的状态读出损失。
2. **`L_brk`（打破项）：** `z_t` 应远离 `z_neg`（改变了任务关键因素，如目标物身份、目标容器等）。使用铰链损失 `max(0, mu - ||z_t - z_neg||_2^2)`，其中 `mu` 是最小距离边距，默认 `1.0`。

**代码入口：** `certvla/training/losses.py` 中的 `cert_counterfactual_loss()` 函数。

**计算细节：**

- v1 实现在 `z_t` 嵌入空间上操作
- 当提供 `state_readout_pos` / `state_readout_neg` 等可选参数时，会额外加入状态/证书一致性项
- 若未提供任何增强对 (`z_pos` 和 `z_neg` 都为 `None`)，优雅降级返回可导的零
- 完整的增强对生成逻辑推迟到 Phase 4 实现

**激活时机：** Stage 4 激活。

---

## 3. 总损失组合

**代码入口：** `certvla/training/losses.py` 中的 `cert_total_loss()` 函数。

**公式：**

```
L_total = lambda_s * L_state
        + lambda_r * L_role
        + lambda_g * L_goal
        + lambda_a * L_act
        + lambda_c * L_cons
        + lambda_d * L_dep
        + lambda_cf * L_cf
```

**接口设计：**

```python
def cert_total_loss(
    losses: Dict[str, torch.Tensor],    # {name -> 标量张量}
    weights: Dict[str, float],           # {weight_key -> float}
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Args:
        losses: 名称到标量损失张量的映射。
            支持的名称: state, role, goal, action,
            consistency, dependence, counterfactual
        weights: 权重键到浮点数的映射。
            键: lambda_s, lambda_r, lambda_g, lambda_a,
            lambda_c, lambda_d, lambda_cf

    Returns:
        total: 加权总损失（保留梯度）
        components: {name -> 加权后的浮点值}（已 detach，用于日志记录）
    """
```

**内部映射表 (`_KEY_MAP`)：**

| 损失名称 | 权重键 |
|----------|--------|
| `"state"` | `"lambda_s"` |
| `"role"` | `"lambda_r"` |
| `"goal"` | `"lambda_g"` |
| `"action"` | `"lambda_a"` |
| `"consistency"` | `"lambda_c"` |
| `"dependence"` | `"lambda_d"` |
| `"counterfactual"` | `"lambda_cf"` |

当某个权重为 `0.0` 时，对应的损失项不会被加入计算图，避免不必要的梯度计算。

---

## 4. 四阶段课程学习

课程学习 (curriculum learning) 是 CertVLA 训练的核心策略。通过逐步解冻模块和激活损失项，避免模型在早期被复杂的多任务学习信号淹没。

**代码入口：** `certvla/training/curriculum.py`

### 4.1 Stage 1: state（步骤 0 - 5,000）

| 参数 | 值 |
|------|-----|
| 阶段枚举 | `TrainingStage.STAGE_1_STATE` |
| 激活损失 | `L_state` (`lambda_s=1.0`) |
| 解冻模块 | state token + readout head |
| 冻结模块 | backbone, certificate head, action head |
| 目标 | 学会从 `z_t` 准确读出环境状态 |
| 描述 | `"Learn state readout from z_t"` |

**权重配置：**

```python
lambda_s=1.0, lambda_r=0.0, lambda_g=0.0,
lambda_a=0.0, lambda_c=0.0, lambda_d=0.0, lambda_cf=0.0
```

**设计意图：** 这是最简单的目标——只需让 readout head 学会从隐状态中提取信息。状态读出是所有后续机制的基础，必须先稳定下来。

---

### 4.2 Stage 2: certificate（步骤 5,000 - 15,000）

| 参数 | 值 |
|------|-----|
| 阶段枚举 | `TrainingStage.STAGE_2_CERTIFICATE` |
| 激活损失 | `L_state` + `L_role` + `L_goal`（各 `lambda=1.0`） |
| 解冻模块 | state token + readout head + certificate head |
| 冻结模块 | backbone, action head |
| 目标 | 学会预测槽位角色和 advance 槽位的目标值 |
| 描述 | `"Learn certificate role + goal prediction"` |

**权重配置：**

```python
lambda_s=1.0, lambda_r=1.0, lambda_g=1.0,
lambda_a=0.0, lambda_c=0.0, lambda_d=0.0, lambda_cf=0.0
```

**设计意图：** 在状态读出已稳定的基础上，解冻证书头，让模型学会"理解当前任务"：哪些槽位需要改变 (advance)，哪些需要保持 (preserve)，目标值是什么。

---

### 4.3 Stage 3: policy（步骤 15,000 - 40,000）

| 参数 | 值 |
|------|-----|
| 阶段枚举 | `TrainingStage.STAGE_3_POLICY` |
| 激活损失 | 全部（除 `L_cf`）|
| 关键权重 | `lambda_c=0.5`, `lambda_d=0.5` |
| 解冻模块 | state token + readout + certificate head + action head |
| 冻结模块 | backbone (VLA 基础模型) |
| 目标 | 训练证书条件动作头 |
| 描述 | `"Train cert-conditioned action head"` |

**权重配置：**

```python
lambda_s=1.0, lambda_r=1.0, lambda_g=1.0,
lambda_a=1.0, lambda_c=0.5, lambda_d=0.5, lambda_cf=0.0
```

**设计意图：** 这是训练时间最长的阶段（25,000 步）。解冻动作头，让模型学会根据证书信息生成动作。一致性损失 (`L_cons`) 确保预测的结构一致性，依赖性损失 (`L_dep`) 确保动作头真正依赖证书而非忽略它。辅助损失的权重设为 0.5，避免对主目标（动作回归）产生过大干扰。

> **注意：** 此阶段需要额外的 negative-cert forward pass（由 `CurriculumScheduler.should_compute_dep()` 控制）。

---

### 4.4 Stage 4: counterfactual（步骤 40,000 - 60,000）

| 参数 | 值 |
|------|-----|
| 阶段枚举 | `TrainingStage.STAGE_4_COUNTERFACTUAL` |
| 激活损失 | 全部 7 个损失 |
| 关键权重 | `lambda_cf=0.5` |
| 解冻模块 | state token + readout + certificate head + action head |
| 冻结模块 | backbone |
| 目标 | 反事实增强，提升表征鲁棒性 |
| 描述 | `"Full training with counterfactual augmentation"` |

**权重配置：**

```python
lambda_s=1.0, lambda_r=1.0, lambda_g=1.0,
lambda_a=1.0, lambda_c=0.5, lambda_d=0.5, lambda_cf=0.5
```

**设计意图：** 在模型已学会基本能力后，引入反事实数据增强。通过不变性约束和打破约束，使模型的表征空间更加鲁棒：对无关扰动不敏感，对任务关键变化高度敏感。

> **注意：** 此阶段需要增强数据对的生成（由 `CurriculumScheduler.should_compute_cf()` 控制）。

---

## 5. `CurriculumScheduler` 使用方式

`CurriculumScheduler` 是基于训练步数的阶段调度器，根据当前步数自动返回对应的阶段配置。

**代码入口：** `certvla/training/curriculum.py`

### 5.1 核心类

**`TrainingStage` 枚举：**

```python
class TrainingStage(str, Enum):
    STAGE_1_STATE = "stage1_state"
    STAGE_2_CERTIFICATE = "stage2_certificate"
    STAGE_3_POLICY = "stage3_policy"
    STAGE_4_COUNTERFACTUAL = "stage4_counterfactual"
```

**`StageConfig` 数据类：**

包含以下属性：
- 7 个损失权重 (`lambda_s` 至 `lambda_cf`)
- 4 个损失超参数 (`lambda_pre`, `dep_margin`, `focal_gamma`, `cf_mu`)
- 4 个模块冻结标志 (`freeze_backbone`, `freeze_state`, `freeze_certificate`, `freeze_action`)
- `loss_weights()` 方法：返回适用于 `cert_total_loss()` 的权重字典

### 5.2 典型使用模式

```python
from certvla.training import CurriculumScheduler

# 使用默认阶段边界初始化
scheduler = CurriculumScheduler()

for step in range(60_000):
    # 获取当前阶段配置
    config = scheduler.get_config(step)

    # 获取损失权重
    weights = config.loss_weights()

    # 根据配置冻结/解冻模块
    freeze_module(backbone, config.freeze_backbone)
    freeze_module(state_head, config.freeze_state)
    freeze_module(cert_head, config.freeze_certificate)
    freeze_module(action_head, config.freeze_action)

    # 判断是否需要额外前向传播
    if scheduler.should_compute_dep(step):
        # 执行 negative-cert forward pass (用于 L_dep)
        actions_neg = model.forward_with_corrupted_cert(...)

    if scheduler.should_compute_cf(step):
        # 生成增强数据对 (用于 L_cf)
        z_pos, z_neg = augmentation_pipeline(...)

    # 计算总损失
    total, components = cert_total_loss(losses, weights)
    total.backward()
```

### 5.3 自定义阶段边界

```python
# 自定义阶段边界（例如缩短训练）
custom_boundaries = {
    TrainingStage.STAGE_1_STATE:          (0,     2_000),
    TrainingStage.STAGE_2_CERTIFICATE:    (2_000, 8_000),
    TrainingStage.STAGE_3_POLICY:         (8_000, 25_000),
    TrainingStage.STAGE_4_COUNTERFACTUAL: (25_000, 35_000),
}
scheduler = CurriculumScheduler(stage_boundaries=custom_boundaries)
```

### 5.4 辅助查询方法

| 方法 | 返回值 | 用途 |
|------|--------|------|
| `get_stage(step)` | `TrainingStage` 枚举 | 获取当前阶段 |
| `get_config(step)` | `StageConfig` 实例 | 获取完整阶段配置 |
| `get_loss_weights(step)` | `Dict[str, float]` | 直接获取损失权重字典 |
| `should_compute_dep(step)` | `bool` | 是否需要 negative-cert forward pass |
| `should_compute_cf(step)` | `bool` | 是否需要生成增强数据对 |

超出所有阶段边界的步数会停留在最后一个阶段 (Stage 4)。

---

## 6. 调度采样 (Scheduled Sampling)

调度采样控制训练过程中是否使用"教师强制" (teacher forcing)，即是否用真实初始状态 `z_0` 还是模型自身的前一步预测 `z_{t-1}` 作为输入。

**代码入口：** `certvla/training/sched_sampling.py`

### 6.1 v1 策略（当前实现）

- 始终使用 `z_0` 作为初始状态（`teacher_force_prob = 1.0`，即 `schedule="constant"`）
- 每个 chunk 独立训练，不跨 chunk 递归
- 这是最稳定的训练方式，但可能导致模型在推理时产生误差累积

### 6.2 v2 计划（未来扩展）

- **逐步退火：** `start_prob=1.0` 逐渐衰减至 `end_prob=0.0`
- **Warmup 阶段：** 在 `warmup_steps` 之前，概率保持为 `start_prob`
- **三种退火策略：**

| 策略 | 公式 | 特点 |
|------|------|------|
| `constant` | `p = start_prob` | 不衰减（v1 默认） |
| `linear` | `p = start + (end - start) * progress` | 匀速衰减 |
| `cosine` | `p = end + (start - end) * (1 + cos(pi * progress)) / 2` | 前期衰减慢，后期衰减快 |

其中 `progress = (step - warmup_steps) / (total_steps - warmup_steps)`，取值范围 `[0, 1]`。

**推荐使用 cosine 策略：** 前期保留较高的教师强制概率以稳定训练，后期快速衰减以让模型学会自主递推。

### 6.3 `ScheduledSampler` 接口

```python
from certvla.training import ScheduledSampler

# v1: 始终教师强制（默认配置）
sampler_v1 = ScheduledSampler(schedule="constant", start_prob=1.0)

# v2: cosine 退火
sampler_v2 = ScheduledSampler(
    schedule="cosine",
    start_prob=1.0,
    end_prob=0.0,
    warmup_steps=5_000,
    total_steps=60_000,
)

# 在训练循环中使用
for step in range(60_000):
    if sampler.should_use_teacher(step):
        z_prev = z_0          # 教师强制：使用可学习初始状态
    else:
        z_prev = model_z_prev  # 自由运行：使用模型自身的前一步预测
```

**`ScheduledSampler` 属性说明：**

| 属性 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `schedule` | `SamplingSchedule` | `"constant"` | 退火策略 |
| `start_prob` | `float` | `1.0` | 初始教师强制概率 |
| `end_prob` | `float` | `0.0` | 终止教师强制概率 |
| `warmup_steps` | `int` | `0` | 衰减开始前的预热步数 |
| `total_steps` | `int` | `10_000` | 衰减结束的总步数 |

**方法说明：**

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `get_teacher_force_prob(step)` | `float` | 返回当前步的教师强制概率 |
| `should_use_teacher(step)` | `bool` | 随机决定是否使用教师强制（基于概率） |

---

## 7. 训练脚本

### 7.1 基线训练

参见 `scripts/certvla/train_baseline.sh`，这是在 LIBERO 数据集上微调 OpenVLA-OFT 基线模型的脚本（不包含 CertVLA 模块）。

**关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_GPUS` | `1` | GPU 数量 |
| `DATA_ROOT` | -- | RLDS 数据集根目录（用户必改） |
| `RUN_ROOT` | -- | checkpoint 和日志输出目录（用户必改） |
| `DATASET` | `"libero_spatial_no_noops"` | 数据集名称 |
| `VLA_PATH` | `"openvla/openvla-7b"` | 预训练 VLA 模型路径 |
| `BATCH_SIZE` | `8` | 每卡 batch size（8 需要约 62GB 显存，1 需要约 25GB） |
| `LR` | `5e-4` | 学习率 |
| `MAX_STEPS` | `150005` | 总训练步数 |
| `SAVE_FREQ` | `10000` | checkpoint 保存频率 |
| `LORA_RANK` | `32` | LoRA rank |

**可选数据集：**

- `libero_spatial_no_noops`
- `libero_object_no_noops`
- `libero_goal_no_noops`
- `libero_10_no_noops`

**使用方法：**

```bash
# 1. 修改脚本中的用户必改参数
# 2. 添加执行权限
chmod +x scripts/certvla/train_baseline.sh
# 3. 运行
bash scripts/certvla/train_baseline.sh
```

**底层调用：**

```bash
torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} \
    vla-scripts/finetune.py \
    --vla_path ${VLA_PATH} \
    --data_root_dir ${DATA_ROOT} \
    --dataset_name ${DATASET} \
    --run_root_dir ${RUN_ROOT} \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --num_steps_before_decay ${DECAY_STEPS} \
    --max_steps ${MAX_STEPS} \
    --save_freq ${SAVE_FREQ} \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank ${LORA_RANK}
```

---

## 8. 槽位族 (Slot Family) 与证书参与集合

训练流水线中的损失函数依赖于槽位的分族信息，这些信息定义在 `certvla/slots/role_sets.py` 中。

### 8.1 四个核心集合

| 集合 | 成员数 | 含义 | 成员 |
|------|--------|------|------|
| `J_E` | 5 | 使能/过渡槽位 | `ee_target_proximity`, `hand_occupancy`, `target_contact`, `articulation_progress`, `orientation_alignment` |
| `J_R` | 4 | 结果/锁存槽位 | `target_goal_proximity`, `support_relation`, `containment_relation`, `completion_latch` |
| `J_C` | 1 | 置信度槽位 | `task_visible_confidence` |
| `J_CERT` | 9 | 证书参与槽位 (`J_E` + `J_R`) | 上述 9 个槽位的并集 |

### 8.2 各集合在损失计算中的用途

- **`J_CERT`：** `L_role` 和 `L_goal` 仅遍历此集合中的槽位；`L_cons` 同样仅在此集合上计算
- **`J_E`（使能槽位）：** 描述动作执行的前提条件和过渡中间状态，在 chunk 执行过程中会发生变化
- **`J_R`（结果槽位）：** 描述动作的目标后果，具有"锁存"语义（一旦达到期望值就不应回退）
- **`J_C`（置信度槽位）：** 不参与证书判定，但可用于 `L_state` 的训练和低质量数据过滤

---

## 9. 训练流程总结

```
训练步骤 0                                                          60,000
  |                                                                    |
  | Stage 1: state  | Stage 2: cert  |  Stage 3: policy  | Stage 4: cf |
  |    (5k steps)   |  (10k steps)   |   (25k steps)     | (20k steps) |
  |                 |                |                    |             |
  | L_state         | + L_role       | + L_act            | + L_cf      |
  |                 | + L_goal       | + L_cons            |             |
  |                 |                | + L_dep             |             |
  |                 |                |                    |             |
  | 解冻: state     | 解冻: + cert   | 解冻: + action     | (同 Stage 3)|
  | 冻结: 其他全部  | 冻结: bb, act  | 冻结: backbone     | 冻结: bb    |
```

**关键设计原则：**

1. **渐进式解冻：** backbone 始终冻结，CertVLA 新增模块按复杂度逐步解冻
2. **损失项递增：** 每个阶段在前一阶段的基础上添加新的损失项，而非替换
3. **掩码与置信度：** 所有逐槽位损失都通过 `m^j * alpha^j` 模式处理缺失数据和不确定标注
4. **辅助损失降权：** 一致性、依赖性、反事实损失使用 `0.5` 的权重，避免喧宾夺主
5. **安全降级：** 各损失函数在无有效数据时返回可导的零张量，不会导致训练中断
