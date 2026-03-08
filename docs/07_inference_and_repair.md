# CertVLA 推理与修复机制

> 本文档详细介绍 CertVLA 在推理时的证书间隙计算和短程修复循环。
> 涵盖从观测到动作输出的完整闭环流程，以及当间隙超标时如何通过局部重试恢复。

---

## 1. 推理闭环概览

CertVLA 的推理以 **chunk** 为单位运行。每个 chunk 包含 H 步动作 (即 `NUM_ACTIONS_CHUNK`，默认 8)。
模型在每个 chunk 的起点接收观测、产生动作序列，并通过证书间隙判断执行效果是否符合预期。

整体闭环流程如下：

```
loop:
  1. 观测 o_t --> LLM 前向 --> z_t, s_t, c_t, A_t
  2. 执行 A_t (H 步动作 chunk)
  3. 观测 o_{t+H} --> 计算 s_{t+H}
  4. 计算间隙: Gamma_t = Gap(c_t, s_t, s_{t+H})
  5. 如果 Gamma_t <= threshold --> 继续下一个 chunk
     如果 Gamma_t > threshold  --> 触发修复
  6. z_{t-1} <-- z_t，进入下一个 chunk
```

其中各符号含义：

| 符号 | 含义 | 来源 |
|------|------|------|
| `o_t` | 时刻 t 的视觉观测 | 环境 |
| `z_t` | 持久状态 token 的隐藏向量 | `CertVLAOutput.z_t`，形状 `(B, llm_dim)` |
| `s_t` | 结构化状态读出 | `CertVLAOutput.state_readout`，`Dict[SlotName, Tensor]` |
| `c_t` | 进展证书 `{(u_t^j, g_t^j)}` | `CertVLAOutput.role_logits` + `goal_preds` |
| `A_t` | 动作 chunk | `CertVLAOutput.actions`，形状 `(B, H, action_dim)` |
| `s_{t+H}` | chunk 结束时的状态读出 | 下一次前向传播或环境反馈 |
| `Gamma_t` | 聚合证书间隙 | `GapResult.aggregated`，标量 |

相关代码入口：

- 间隙计算: `certvla/inference/gap.py`
- 修复控制: `certvla/inference/repair.py`
- 推理日志: `certvla/inference/logging.py`
- 模型输出: `certvla/model/outputs.py`

---

## 2. 证书间隙 (Certificate Gap)

证书间隙是 CertVLA 推理时的核心自检信号。它衡量模型对 chunk 执行效果的预测与实际结果之间的偏差。
间隙值越大，说明模型的预测越不准确，或者执行结果越偏离预期。

### 2.1 逐 Slot 间隙 gamma_j

对于 `J_CERT` 中的每个 slot `j`（共 9 个：5 个使能 slot + 4 个结果 slot），计算：

```
gamma_t^j = p_adv^j * d_j(g_t^j, s_{t+H}^j)
           + p_pre^j * d_j(s_t^j, s_{t+H}^j)
```

各项解释：

- **`p_adv^j`, `p_pre^j`**: 来自 `softmax(role_logits^j)` 的前两个分量。
  `role_logits` 是 `(B, 3)` 的张量，三列分别对应 advance (索引 0)、preserve (索引 1)、ignore (索引 2)。
  通过 softmax 转化为概率后，`p_adv = probs[:, ROLE_ADVANCE]`，`p_pre = probs[:, ROLE_PRESERVE]`。

- **`d_j`**: 逐 slot 距离函数，根据 slot 的 domain 类型选择不同的距离度量：
  - `BINARY` / `CONTINUOUS` / `CONFIDENCE`: 使用 L1 距离 `|a - b|`
  - `CATEGORICAL`: 使用全变差距离 (Total Variation) `sum(|softmax(a) - softmax(b)|) / 2`

- **advance 项** (`p_adv * d(goal, s_{t+H})`):
  目标预测 `g_t^j` 与 chunk 结束时实际状态 `s_{t+H}^j` 的偏差，
  以 advance 概率加权。如果模型预测这个 slot 应该"推进"到某个目标值，
  但实际结果偏离了，这一项就会增大。

- **preserve 项** (`p_pre * d(s_t, s_{t+H})`):
  当前状态 `s_t^j` 与 chunk 结束时状态 `s_{t+H}^j` 的偏差，
  以 preserve 概率加权。如果模型预测这个 slot 应该"保持不变"，
  但实际上发生了变化，这一项就会增大。

实现要点（参见 `certvla/inference/gap.py:slot_gap()`）：

```python
def slot_gap(
    role_logits: Dict[SlotName, torch.Tensor],    # (B, 3)
    state_readout: Dict[SlotName, torch.Tensor],   # s_t
    goal_preds: Dict[SlotName, torch.Tensor],      # g_t
    state_readout_tH: Dict[SlotName, torch.Tensor], # s_{t+H}
) -> Dict[SlotName, torch.Tensor]:                 # gamma^j, (B,)
```

该函数遍历 `J_CERT` 中的所有 slot，对每个 slot 计算 advance 项和 preserve 项之和。
如果某个 slot 在输入字典中缺失，则对应项被跳过（安全降级）。

### 2.2 聚合间隙 Gamma_t

将所有 slot 的间隙汇总为一个标量：

```
Gamma_t = [sum_j omega_j * kappa_j * gamma_j] / [sum_j omega_j * kappa_j + epsilon]
```

各参数含义：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `omega_j` | 静态 slot 重要性权重 | 1.0 |
| `kappa_j` | 动态置信度权重 (每步可变) | 1.0 |
| `epsilon` | 数值稳定常数 | 1e-8 |

- **`omega_j`** (static slot weights): 在 `RepairConfig.slot_weights` 中配置。
  可以对关键 slot（如 `target_goal_proximity`）赋予更高权重。
  如果未配置，所有 slot 默认权重为 1.0。

- **`kappa_j`** (confidence weights): 作为参数传入 `aggregate_certificate_gap()`。
  通常来自模型的置信度估计，允许在低置信度时降低某些 slot 对聚合间隙的贡献。

- **`epsilon`**: 防止除零错误的小常数。

实现函数签名（参见 `certvla/inference/gap.py:aggregate_certificate_gap()`）：

```python
def aggregate_certificate_gap(
    per_slot_gaps: Dict[SlotName, torch.Tensor],       # gamma^j
    role_logits: Dict[SlotName, torch.Tensor],          # 用于输出 role_probs
    slot_weights: Optional[Dict[SlotName, float]] = None,  # omega
    confidence_weights: Optional[Dict[SlotName, torch.Tensor]] = None,  # kappa
    epsilon: float = 1e-8,
) -> GapResult:
```

当 `per_slot_gaps` 为空时，函数返回一个间隙值为 0 的 `GapResult`（安全降级）。

### 2.3 GapResult 数据结构

`GapResult` 是间隙计算的完整输出，定义在 `certvla/inference/gap.py` 中：

```python
@dataclass
class GapResult:
    per_slot: Dict[SlotName, torch.Tensor]    # 逐 slot 间隙, 每个 Tensor 形状 (B,)
    aggregated: torch.Tensor                   # 聚合间隙, 形状 (B,)
    role_probs: Dict[SlotName, torch.Tensor]  # softmax 角色概率, 每个 Tensor 形状 (B, 3)
```

字段说明：

- **`per_slot`**: 每个 cert slot 的独立间隙值。可用于诊断哪个 slot 贡献了最大间隙。
- **`aggregated`**: 加权聚合后的标量间隙。修复控制器根据此值的均值 (`aggregated.mean().item()`) 决定是否触发修复。
- **`role_probs`**: softmax 后的角色概率分布。三列分别对应 advance、preserve、ignore。可用于可视化模型对每个 slot 角色的判断。

### 2.4 距离函数细节

`_slot_distance()` 是一个内部函数，根据 slot 的 domain 类型选择距离度量：

```python
def _slot_distance(a: torch.Tensor, b: torch.Tensor, meta) -> torch.Tensor:
    """可微的逐 slot 距离，返回 (B,)，值域 [0, 1]。"""
    if meta.domain in (SlotDomain.BINARY, SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        return torch.abs(a.view(-1) - b.view(-1))  # L1 距离
    elif meta.domain == SlotDomain.CATEGORICAL:
        pa = F.softmax(a, dim=-1)
        pb = F.softmax(b, dim=-1)
        return torch.abs(pa - pb).sum(dim=-1) / 2.0  # 全变差距离
```

设计要点：
- 所有距离值归一化到 [0, 1] 区间，使得不同 domain 的 slot 间隙具有可比性。
- 函数是可微的 (differentiable)，理论上支持通过间隙反传梯度（虽然推理时通常不需要）。
- 对于 categorical slot，先将 logits 转换为概率分布再计算 TV 距离，避免了 logits 尺度不一致的问题。

---

## 3. v1 Proxy 间隙

### 3.1 问题背景

在完整的闭环推理中，计算间隙需要 chunk 结束后的状态读出 `s_{t+H}`。
但在 v1 实现中，模型在产生动作 chunk 后尚未收到 `o_{t+H}` 的观测，
因此无法获得真正的 `s_{t+H}`。

### 3.2 Fallback 策略

v1 的 `RepairController._compute_gap()` 采用 **代理间隙 (proxy gap)** 策略：
当 `state_readout_tH` 参数为 `None` 时，使用 `goal_preds` 作为 `s_{t+H}` 的代理值。

```python
def _compute_gap(self, output, state_readout_tH=None, confidence_weights=None):
    if state_readout_tH is None:
        # v1 fallback: 用 goal_preds 作为预期的 chunk 结束状态
        state_readout_tH = output.goal_preds
    # ...后续正常计算
```

### 3.3 语义含义

这意味着 v1 的间隙实质上是一个 **自一致性间隙 (self-consistency gap)**，而非基于真实观测的间隙：

- **advance 项**: `d(goal_preds, goal_preds) = 0`，总是为零。
  因为"目标预测"与"chunk 结束预测"是同一个值。
- **preserve 项**: `d(state_readout, goal_preds)`，当 preserve 角色的 slot 的状态读出与目标预测不一致时产生间隙。

实际上，v1 proxy 间隙主要检测的是：
1. 模型内部预测的 **一致性** ——如果 role 分配说某个 slot 应该 preserve，
   但 goal prediction 却预测了一个不同的值，说明模型内部存在矛盾。
2. 这种矛盾信号虽然不如真实间隙准确，但仍然能捕捉到部分"模型不确定"的情况。

### 3.4 向 v2 的演进方向

未来版本可以通过以下方式获得更准确的间隙：
- 在 chunk 执行完毕后，使用下一帧观测 `o_{t+H}` 做一次额外的 state readout 前向传播
- 使用环境提供的真实状态信息（如果可用）
- 引入状态预测网络单独估计 `s_{t+H}`

---

## 4. 修复控制器 (RepairController)

修复控制器是 CertVLA 推理时的自我纠错机制。当证书间隙超过阈值时，
控制器会重新调用模型，尝试产生间隙更低的动作 chunk。

代码位于 `certvla/inference/repair.py`。

### 4.1 设计原则

1. **短程局部修复，非全局重规划**:
   修复只是在同一观测下重新前向传播，尝试产生更好的动作。
   它不会回退到更早的时间步，也不会修改任务目标。
   这与传统的 replanner (如 task-level replanning) 有本质区别。

2. **无状态重试**:
   每次重试的输入完全相同 (`last_hidden_states`, `state_token_pos`, `action_start_pos`, `z_prev`)。
   依赖模型内部的随机性（如 dropout、temperature 采样）来产生不同的输出。
   因此修复在推理时需要启用 dropout 或使用非零 temperature。

3. **Best-of-N 策略**:
   在 `max_repair_steps` 次尝试中，选择聚合间隙最低的那次作为最终输出。
   即使所有尝试都超过阈值，也返回最优的那次（并记录警告）。

### 4.2 RepairConfig

```python
@dataclass
class RepairConfig:
    gap_threshold: float = 0.3        # 触发修复的间隙阈值
    max_repair_steps: int = 3         # 最大重试次数
    slot_weights: Optional[Dict] = None  # 静态 slot 权重 (omega_j)
    use_best_of_n: bool = True        # 是否始终选择最优尝试
```

各参数详解：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gap_threshold` | 0.3 | `Gamma_t` 的均值超过此值则触发修复循环。过低会导致频繁不必要的修复；过高则放过错误预测。 |
| `max_repair_steps` | 3 | 修复循环的最大次数。超过后强制接受当前最优结果。对实时系统可降低到 1-2。 |
| `slot_weights` | None | 传递给 `aggregate_certificate_gap()` 的 `omega_j`。None 表示所有 slot 权重为 1.0。 |
| `use_best_of_n` | True | 若为 True，在所有尝试都超过阈值时仍返回最优尝试；若为 False，仅在找到低于阈值的尝试时才替换初始结果。 |

### 4.3 修复流程

`RepairController.step()` 方法的完整流程：

```
step(last_hidden_states, state_token_pos, action_start_pos, z_prev,
     confidence_weights, state_readout_tH):

  1. 初始前向:
     output = model_fn(last_hidden_states, stp, asp, z_prev)
     gap = _compute_gap(output, state_readout_tH, confidence_weights)
     best_output, best_gap, best_gap_val = output, gap, gap.aggregated.mean()

  2. 阈值检查:
     如果 best_gap_val <= gap_threshold:
       --> 记录日志 (accepted=True, repair_attempt=0)
       --> 返回 (output.actions, gap, n_repairs=0)

  3. 修复循环 (gap 超标时进入):
     for attempt in 1 .. max_repair_steps:
       a. output_r = model_fn(last_hidden_states, stp, asp, z_prev)
       b. gap_r = _compute_gap(output_r, ...)
       c. 记录日志 (accepted=False, repair_attempt=attempt)
       d. 如果 gap_r < best_gap_val: 更新 best_output, best_gap, best_gap_val
       e. 如果 best_gap_val <= gap_threshold: 提前退出循环

  4. 接受最优结果:
     --> 记录日志 (accepted=True, repair_attempt=n_repairs)
     --> 如果 best_gap_val 仍 > threshold: 记录警告
     --> 返回 (best_output.actions, best_gap, n_repairs)
```

流程图：

```
        +-----------+
        | 初始前向  |
        +-----+-----+
              |
              v
        +----------+    <= threshold     +----------+
        | 计算间隙 |--------------------->| 接受输出 |
        +-----+----+                     +----------+
              |
              | > threshold
              v
        +----------------+
        | 修复循环       |
        | (最多 N 次)    |
        |   重新前向     |
        |   计算间隙     |
        |   更新最优     |
        +-------+--------+
                |
                v
        +--------------+
        | 接受最优结果 |
        +--------------+
```

### 4.4 model_fn 解耦设计

`RepairController` 不直接依赖 `CertVLAWrapper` 或任何具体模型类。
它通过一个可调用的 `model_fn` 与模型交互：

```python
model_fn: Callable[..., CertVLAOutput]
# 签名: (last_hidden_states, state_token_pos, action_start_pos, z_prev) -> CertVLAOutput
```

这种设计的优势：

1. **可测试性**: 单元测试中可以传入 mock 函数，不需要实例化完整模型。
2. **灵活性**: 在集成时，`model_fn` 可以包装任意模型前向逻辑（例如包含预处理/后处理步骤）。
3. **关注点分离**: 修复控制器只关心"调用模型 -> 得到输出 -> 计算间隙"这一循环，
   不关心模型内部的架构细节。

在实际集成时，`model_fn` 通常是一个闭包，捕获 `CertVLAWrapper` 实例：

```python
def make_model_fn(wrapper: CertVLAWrapper):
    def model_fn(hidden, stp, asp, z_prev):
        return wrapper.forward(hidden, stp, asp, z_prev)
    return model_fn

controller = RepairController(config, make_model_fn(wrapper), logger)
```

### 4.5 @torch.no_grad() 装饰

`step()` 方法使用 `@torch.no_grad()` 装饰器，确保推理时不构建计算图，节省显存。
修复循环中的所有前向传播和间隙计算都在无梯度上下文中执行。

---

## 5. 推理日志 (InferenceLogger)

推理日志系统记录每一步的详细数据，用于调试、分析和可视化。
代码位于 `certvla/inference/logging.py`。

### 5.1 StepRecord

一条推理步骤记录：

```python
@dataclass
class StepRecord:
    output: Any = None         # CertVLAOutput (可选，用于详细诊断)
    gap: Any = None            # GapResult (可选)
    repair_attempt: int = 0    # 0 = 初始前向，1+ = 修复尝试序号
    accepted: bool = False     # 是否被接受为最终输出
```

注意事项：
- `output` 和 `gap` 类型标注为 `Any` 以避免循环导入，实际类型分别是 `CertVLAOutput` 和 `GapResult`。
- 每个 chunk 可能产生多条 `StepRecord`（初始 + 修复尝试），但只有一条 `accepted=True`。
- 建议在记录前对 tensor 执行 `detach()`，避免持有计算图引用导致显存泄漏。

### 5.2 EpisodeTrace

一个完整推理 episode 的追踪数据：

```python
@dataclass
class EpisodeTrace:
    steps: List[StepRecord]          # 已接受的步骤（每个 chunk 一条）
    all_attempts: List[StepRecord]   # 所有尝试（含被拒绝的修复）
    warnings: List[str]              # 警告消息（如修复耗尽）
    metadata: Dict[str, Any]         # 自由格式元数据（如任务名称）
```

`steps` 只包含 `accepted=True` 的记录，其长度等于实际执行的 chunk 数。
`all_attempts` 包含所有记录（含被拒绝的修复尝试），可用于分析修复行为。

### 5.3 统计摘要

`EpisodeTrace` 提供多个便捷属性和方法：

```python
# 属性
trace.num_steps      # 已接受的步数（即 chunk 数）
trace.num_repairs    # 触发过修复的步数（repair_attempt > 0 的已接受步骤数）
trace.gap_history    # 逐步聚合间隙值列表 List[float]

# 方法
trace.summary()      # 返回统计摘要字典
```

`summary()` 返回的字典格式：

```python
{
    "num_steps": 10,           # 总步数
    "num_repairs": 2,          # 触发修复的步数
    "total_attempts": 16,      # 总尝试次数（含修复重试）
    "mean_gap": 0.15,          # 平均间隙
    "max_gap": 0.28,           # 最大间隙
    "num_warnings": 0,         # 警告次数
}
```

关键指标解读：
- `num_repairs / num_steps` = **修复率**: 反映模型预测质量。理想情况下应低于 10%。
- `total_attempts / num_steps` = **平均尝试次数**: 如果远大于 1，说明模型频繁触发修复。
- `max_gap`: 如果远超阈值，说明存在修复无法解决的困难步骤。

### 5.4 InferenceLogger 使用方式

```python
from certvla.inference.logging import InferenceLogger, StepRecord

# 创建日志器
logger = InferenceLogger(verbose=True, max_episodes=100)

# 开始一个新 episode
logger.begin_episode(metadata={"task": "pick_and_place", "scene": "tabletop"})

# ... 推理循环中 ...
# RepairController 会自动调用 logger.log_step()
# 也可以手动记录:
logger.log_step(StepRecord(
    output=certvla_output,
    gap=gap_result,
    repair_attempt=0,
    accepted=True,
))

# 记录警告（如果需要）
logger.log_warning("Unexpected collision detected")

# 结束 episode
logger.end_episode()

# 获取追踪数据
trace = logger.get_last_trace()
print(trace.summary())
# {'num_steps': 10, 'num_repairs': 2, 'total_attempts': 16,
#  'mean_gap': 0.15, 'max_gap': 0.28, 'num_warnings': 0}

# 获取所有历史追踪
all_traces = logger.get_all_traces()

# 清除历史数据
logger.clear()
```

构造参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `verbose` | False | 为 True 时，每个被接受的步骤都会通过 Python `logging` 模块输出日志（logger 名称: `certvla.inference`）。 |
| `max_episodes` | 100 | 内存中保留的最大 episode 数量。超出后自动裁剪最旧的记录。 |

内存管理：
- `InferenceLogger` 内部使用列表存储 `EpisodeTrace`，当 trace 数量超过 `max_episodes` 时，保留最新的记录。
- 如果 `StepRecord.output` 包含完整的 `CertVLAOutput`（含 tensor），长时间运行可能占用大量显存。建议在不需要详细诊断时将 `output` 设为 `None`。

---

## 6. 端到端推理示例

以下伪代码展示一个包含 5 个 chunk 的完整推理 episode，其中第 3 个 chunk 触发修复：

```python
import torch
from certvla.inference.gap import slot_gap, aggregate_certificate_gap
from certvla.inference.repair import RepairController, RepairConfig
from certvla.inference.logging import InferenceLogger

# ============================================================
# 初始化
# ============================================================
config = RepairConfig(
    gap_threshold=0.3,
    max_repair_steps=3,
    slot_weights=None,       # 所有 slot 权重 1.0
    use_best_of_n=True,
)

logger = InferenceLogger(verbose=True, max_episodes=50)

# model_fn 包装实际模型的前向传播
def model_fn(hidden, stp, asp, z_prev):
    return certvla_wrapper.forward(hidden, stp, asp, z_prev)

controller = RepairController(config, model_fn, logger)

# ============================================================
# 推理循环
# ============================================================
logger.begin_episode(metadata={"task": "pick_cup", "episode_id": 42})

z_prev = None  # 初始状态：使用 z_0

for chunk_idx in range(5):
    # ----------------------------------------------------------
    # 步骤 1: 获取观测并通过 LLM 编码
    # ----------------------------------------------------------
    observation = env.get_observation()
    last_hidden_states = llm_backbone.forward(observation)  # (B, seq, llm_dim)
    state_token_pos = find_state_token(last_hidden_states)
    action_start_pos = find_action_start(last_hidden_states)

    # ----------------------------------------------------------
    # 步骤 2: 修复控制器执行前向 + 间隙检查 + 可选修复
    # ----------------------------------------------------------
    actions, gap_result, n_repairs = controller.step(
        last_hidden_states=last_hidden_states,
        state_token_pos=state_token_pos,
        action_start_pos=action_start_pos,
        z_prev=z_prev,
        confidence_weights=None,      # v1: 不使用动态置信度
        state_readout_tH=None,        # v1: 使用 proxy gap
    )

    # ----------------------------------------------------------
    # 步骤 3: 记录诊断信息
    # ----------------------------------------------------------
    gap_val = gap_result.aggregated.mean().item()
    print(f"Chunk {chunk_idx}: gap={gap_val:.4f}, repairs={n_repairs}")
    # 示例输出:
    #   Chunk 0: gap=0.12, repairs=0
    #   Chunk 1: gap=0.18, repairs=0
    #   Chunk 2: gap=0.42, repairs=2   <-- 触发修复，2 次重试后接受
    #   Chunk 3: gap=0.09, repairs=0
    #   Chunk 4: gap=0.15, repairs=0

    # ----------------------------------------------------------
    # 步骤 4: 执行动作 chunk
    # ----------------------------------------------------------
    for h in range(actions.shape[1]):  # H 步
        env.step(actions[:, h, :])

    # ----------------------------------------------------------
    # 步骤 5: 更新持久状态 z_prev
    # ----------------------------------------------------------
    # 从 controller 内部获取最新的 z_t
    # (实际实现中 z_t 从 CertVLAOutput.z_t 获取)
    z_prev = gap_result  # 简化示意; 实际需要从 output.z_t 获取

logger.end_episode()

# ============================================================
# 分析结果
# ============================================================
trace = logger.get_last_trace()
summary = trace.summary()
print(f"Episode 完成: {summary}")
# Episode 完成: {'num_steps': 5, 'num_repairs': 1, 'total_attempts': 8,
#                'mean_gap': 0.192, 'max_gap': 0.42, 'num_warnings': 0}

# 检查间隙历史
print(f"间隙趋势: {trace.gap_history}")
# 间隙趋势: [0.12, 0.18, 0.25, 0.09, 0.15]
# (注: 第 3 步显示的是修复后被接受的间隙值 0.25，而非初始的 0.42)
```

### 6.1 示例中的第 3 个 Chunk 详细分析

```
Chunk 2 详细过程:
  初始前向: gap = 0.42 > threshold 0.3 --> 触发修复
  修复尝试 1: gap = 0.35 > threshold 0.3 --> 继续尝试
  修复尝试 2: gap = 0.25 <= threshold 0.3 --> 接受!
  最终: actions 来自尝试 2, n_repairs = 2
```

---

## 7. 关键参数调优建议

### 7.1 gap_threshold (间隙阈值)

这是最重要的超参数，直接影响推理的行为模式。

| 取值范围 | 效果 | 适用场景 |
|----------|------|----------|
| 0.1 - 0.2 | 非常严格，频繁触发修复 | 高精度要求任务（如手术机器人） |
| 0.2 - 0.4 | 适中，偶尔修复 | 一般操控任务（推荐起始值: 0.3） |
| 0.4 - 0.6 | 宽松，很少修复 | 快速执行优先、容错性高的任务 |
| > 0.6 | 基本不修复 | 等同于关闭修复机制 |

调优建议：
- 先用默认值 0.3 运行一批 episode，观察修复率。
- 如果修复率 > 30%，说明阈值过低或模型本身预测质量不足。
- 如果修复率 < 5% 但任务成功率不理想，可尝试降低阈值。
- v1 的 proxy 间隙通常比真实间隙偏小，因此 v1 阈值可适当设低一些。

### 7.2 max_repair_steps (最大修复次数)

| 取值 | 效果 | 适用场景 |
|------|------|----------|
| 1 | 一次机会，快速决策 | 实时控制 (< 10ms 预算) |
| 2-3 | 平衡修复效果与延迟 | 一般操控 (推荐默认: 3) |
| 5+ | 更多尝试空间 | 离线评估、不关心延迟 |

注意事项：
- 每次修复尝试意味着一次完整的 `model_fn` 前向传播，计算开销与初始前向相同。
- 如果模型的随机性很低（如 temperature=0），多次重试可能产生几乎相同的结果，此时增加修复次数无意义。

### 7.3 slot_weights (Slot 权重)

默认情况下所有 cert slot 权重相等 (1.0)。根据任务特性可以调整：

```python
# 示例: 对物体位置相关的 slot 赋予更高权重
slot_weights = {
    SlotName.TARGET_GOAL_PROXIMITY: 2.0,   # 关键: 物体是否到达目标
    SlotName.EE_TARGET_PROXIMITY: 1.5,     # 重要: 末端执行器是否接近目标
    SlotName.HAND_OCCUPANCY: 1.0,          # 标准
    SlotName.COMPLETION_LATCH: 1.5,        # 重要: 任务是否完成
    # ... 其余 slot 未指定则默认 1.0
}
config = RepairConfig(slot_weights=slot_weights)
```

权重调优策略：
- 对 **结果 slot** (`J_R`) 赋予较高权重——这些 slot 直接反映任务是否成功。
- 对 **使能 slot** (`J_E`) 使用标准权重——这些 slot 反映中间过程，偏差有时是正常的。
- 可以通过分析 `GapResult.per_slot` 在多个 episode 上的分布来识别哪些 slot 间隙方差最大，并据此调整权重。

### 7.4 模型随机性配置

修复机制依赖模型在相同输入下产生不同输出。需要确保：

- **Dropout**: 推理时保持一定的 dropout 概率（如 0.1），而非完全关闭。
  可以使用 `model.train()` 模式或显式控制 dropout 层。
- **Temperature**: 如果动作头使用采样，设置非零 temperature（如 0.5-1.0）。
- 如果模型完全确定性（dropout=0, temperature=0），修复循环将退化为无效重复，
  此时应设置 `max_repair_steps=0` 以关闭修复，避免浪费计算。

---

## 8. 文件索引

| 文件路径 | 职责 |
|----------|------|
| `certvla/inference/__init__.py` | 推理层公共接口，导出 `slot_gap`, `aggregate_certificate_gap`, `GapResult`, `RepairController`, `InferenceLogger` |
| `certvla/inference/gap.py` | 逐 slot 间隙计算 (`slot_gap`)、聚合间隙计算 (`aggregate_certificate_gap`)、`GapResult` 数据结构 |
| `certvla/inference/repair.py` | 修复控制器 `RepairController` 和配置 `RepairConfig` |
| `certvla/inference/logging.py` | 推理日志 `InferenceLogger`、步骤记录 `StepRecord`、episode 追踪 `EpisodeTrace` |
| `certvla/model/outputs.py` | 模型前向输出 `CertVLAOutput` 数据结构 |
| `certvla/model/certificate_head.py` | 证书头，产生 `role_logits` 和 `goal_preds` |
| `certvla/slots/role_sets.py` | Slot 族定义: `J_E`, `J_R`, `J_C`, `J_CERT` |
