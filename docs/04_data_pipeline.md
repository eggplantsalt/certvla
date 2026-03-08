# CertVLA 数据管线与证书挖掘

> 本文档详细介绍 CertVLA 的数据处理流程：从原始轨迹到训练样本。
> 所有代码均位于 `certvla/data/` 和 `certvla/slots/` 目录中。

---

## 1. 数据流概览

CertVLA 的数据管线将原始 LIBERO 演示轨迹转化为带有证书监督信号的训练样本。
完整流程如下：

```
原始轨迹 (HDF5)
    │
    ▼
仿真器回放 + slot 标注  ← StateLabeler / LiberoOracleLabeler
    │                      (certvla/data/state_labels.py)
    ▼
SlotState 时间序列
    │
    ├──→ 目标签名计算 (compute_goal_signature)
    │      (certvla/data/goal_signature.py)
    │
    ▼
chunk 切分 (每 H=8 步为一个 chunk)
    │
    ▼
证书挖掘 (mine_certificate)  ← 核心算法
    │  (certvla/data/certificate_mining.py)
    │
    ▼
CertChunkSample  ← 完整训练样本
    │  (certvla/data/chunk_sample.py)
    │
    ├──→ 张量编码 (slot_value_to_tensor, slot_state_to_flat_tensor)
    │      (certvla/slots/metrics.py)
    │
    ├──→ 数据增强 (corrupt_certificate, 反事实增强)
    │      (certvla/data/counterfactuals.py)
    │
    ▼
sidecar .npz 文件 / batch 整理
    (certvla/data/label_episodes.py)
```

### 1.1 离线标注的必要性

RLDS 训练数据集中只包含 `(观测图像, 动作)` 对，不包含仿真器内部状态。然而 CertVLA 的证书标签需要知道每个时间步的 slot 状态值（如物体位置、接触状态等），这些只能通过访问 MuJoCo 仿真器获得。因此，必须在**离线阶段**通过回放演示轨迹提取标签，并以 sidecar `.npz` 文件的形式保存。

### 1.2 Sidecar 文件格式

每个回合对应一个 `.npz` 文件，命名规则为 `{task_name}_demo_{i}_labels.npz`，内容包含：

| 键名 | 形状 | 类型 | 含义 |
|------|------|------|------|
| `slot_states` | `(T, flat_dim)` | float32 | 每个时间步的 slot 值向量 |
| `validity_masks` | `(T, num_slots)` | bool | 每个时间步的 slot 有效性 |
| `confidences` | `(T, num_slots)` | float32 | 每个时间步的标签置信度 |
| `chunk_certificates` | `(num_chunks, num_cert_slots)` | int8 | 每个 chunk 的角色编码 (0=ignore, 1=advance, 2=preserve) |
| `goal_signature` | `(flat_dim,)` | float32 | 回合级目标签名 |
| `metadata_json` | JSON 字符串 | str | 回合元信息 (episode_id, task_name 等) |

其中 `flat_dim = 16`（v1 的 10 个 slot 展平后的维度），`num_slots = 10`，`num_cert_slots = 9`（`J_CERT` 的大小）。

---

## 2. SlotState 与 CertificateLabel

> 源码位置：`certvla/data/chunk_sample.py`

CertVLA 数据层的三个核心数据结构构成层次关系：

```
SlotState         -- 单个时间步的所有 slot 值（最底层）
CertificateLabel  -- 单个 chunk 的角色分配和目标值（中间层）
CertChunkSample   -- 完整的训练样本（最顶层）
```

### 2.1 SlotState

`SlotState` 是 CertVLA 中最基础的数据结构，记录了某一时间步 `t` 下所有语义 slot 的取值、有效性和置信度。

```python
@dataclass
class SlotState:
    values: Dict[SlotName, Union[int, float, str]]     # {slot名: 值}
    validity_mask: Dict[SlotName, bool]                 # {slot名: 是否适用于当前任务}
    confidence: Dict[SlotName, float]                   # {slot名: 标签置信度}
```

**字段说明：**

- **`values`** — 字典结构 `{SlotName: value}`，值类型取决于 slot 的域类型：
  - `int` (0 或 1)：二值域 slot，如 `target_contact`、`completion_latch`
  - `str`：类别域 slot，如 `hand_occupancy` 取值 `"empty"` / `"target"` / `"other"`
  - `float` (0.0~1.0)：连续域/置信度域 slot，如 `ee_target_proximity`

- **`validity_mask`** — 标记每个 slot 在当前任务中是否适用。例如 pick-and-place 任务不涉及 `articulation_progress`（铰接操作），因此该 slot 的 `validity_mask` 为 `False`。证书挖掘时无效 slot 会被自动跳过。

- **`confidence`** — 标签来源的置信度。仿真器 oracle 标签 = 1.0（精确），伪标签 < 1.0（推断得到，有不确定性）。训练时低置信度标签可以降权。

**关键方法：**

```python
def get(self, slot: SlotName, default=None):
    """获取指定 slot 的值，若不存在则返回 default"""
    return self.values.get(slot, default)

def is_valid(self, slot: SlotName) -> bool:
    """检查指定 slot 是否有效（适用于当前任务）"""
    return self.validity_mask.get(slot, False)
```

### 2.2 CertificateLabel

`CertificateLabel` 是 CertVLA 的核心创新之一。它为每个 chunk 中的每个 slot 分配一个**角色 (role)**，形成所谓的"证书"：

$$c_t = \{(u_t^j, g_t^j)\} \quad \text{for } j \in J_{\text{cert}}$$

```python
@dataclass
class CertificateLabel:
    roles: Dict[SlotName, str]           # {slot名: "advance"/"preserve"/"ignore"}
    goal_values: Dict[SlotName, Any]     # 仅 advance slot 有目标值
```

**三种角色的含义：**

| 角色 | 含义 | 目标值 | 示例 |
|------|------|--------|------|
| `advance` | 该 slot 在本 chunk 中发生了向目标方向的有意义变化 | `g_t^j = s_{t+H}^j` (chunk 结束时的值) | `ee_target_proximity` 从 0.8 减至 0.3 |
| `preserve` | 该 slot 的值在本 chunk 中不应改变 | 无 | `completion_latch` 已经为 1，需保持 |
| `ignore` | 该 slot 与当前 chunk 的动作无关 | 无 | 抓取阶段的 `containment_relation` |

**便捷查询方法：**

```python
cert.advance_slots()   # 返回所有 advance 角色的 slot 列表
cert.preserve_slots()  # 返回所有 preserve 角色的 slot 列表
cert.ignore_slots()    # 返回所有 ignore 角色的 slot 列表
```

### 2.3 CertChunkSample

`CertChunkSample` 是送入 CertVLA 模型训练的**最终数据单元**，将基础 chunk 数据与证书监督信号组合在一起。

```python
@dataclass
class CertChunkSample:
    observation: np.ndarray               # RGB 图像 (H, W, 3), 时间步 t
    instruction: str                      # 自然语言指令, 如 "put the bowl on the plate"
    actions: np.ndarray                   # 动作序列 (H, action_dim), 从 t 到 t+H-1
    state_t: SlotState                    # chunk 起始时刻的 slot 状态
    state_t_H: SlotState                  # chunk 结束时刻的 slot 状态
    certificate: CertificateLabel         # 该 chunk 的证书标签
    goal_signature: Optional[SlotState]   # 回合级目标签名 s*
    episode_id: Optional[str]             # 回合标识符
    timestep: Optional[int]               # chunk 起始时间步索引
```

其中：
- `observation` — 形状 `(H, W, 3)`，uint8 类型的 RGB 图像
- `actions` — 形状 `(H, action_dim)`，对于 LIBERO，`H=8`、`action_dim=7`（6D 末端执行器增量 + 1D 夹爪）
- `goal_signature` — 由回合终端 K 步聚合得到（详见第 3.4 节）

---

## 3. 证书挖掘算法

> 源码位置：`certvla/data/certificate_mining.py`

证书挖掘是 CertVLA 数据管线中**最关键的模块**。它从离线回放的 slot 状态序列中自动挖掘出每个 chunk 的证书标签（advance / preserve / ignore）。

核心思想：给定一个 chunk 的起始状态 `s_t` 和结束状态 `s_{t+H}`，以及回合级目标签名 `s*`，算法判断每个 slot 在该 chunk 中应该扮演什么角色。

### 3.1 Slot 族与角色集

> 源码位置：`certvla/slots/role_sets.py`、`certvla/slots/schema.py`

在介绍挖掘算法前，需要理解 CertVLA 的 10 个 slot 被分为三个族：

| 族 | 含义 | 成员 | 数量 |
|----|------|------|------|
| **J_E** (使能/过渡) | 描述动作执行的前提条件 | `ee_target_proximity`, `hand_occupancy`, `target_contact`, `articulation_progress`, `orientation_alignment` | 5 |
| **J_R** (结果/锁存) | 描述动作执行的目标后果 | `target_goal_proximity`, `support_relation`, `containment_relation`, `completion_latch` | 4 |
| **J_C** (置信度) | 描述观测可靠性 | `task_visible_confidence` | 1 |

其中 `J_CERT = J_E | J_R`（9 个 slot）参与证书判定，`J_C` 不参与证书但用于过滤低置信度样本。

对结果 slot (J_R) 和使能 slot (J_E) 使用**不同的挖掘策略**：
- J_R：直接反映任务完成度，使用 delta + (rho OR upsilon) 判定
- J_E：为结果 slot 创造条件，使用 delta + eta 判定

### 3.2 四个度量指标

证书挖掘依赖四个度量指标来判断 slot 的变化是否有意义。

#### 3.2.1 delta（变化量）

```
delta_t^j = d_j(s_t^j, s_{t+H}^j)
```

**含义：** chunk 执行前后，某个 slot 的值发生了多大的变化。使用 `slot_distance()` 函数计算，根据 slot 的域类型选择合适的距离度量：

| 域类型 | 距离度量 | 结果范围 |
|--------|----------|----------|
| binary | `|a - b|` | 0 或 1 |
| categorical | Hamming (相同=0, 不同=1) | 0 或 1 |
| continuous / confidence | L1 距离 `|a - b|` | [0, 1] |

```python
def _compute_delta(slot, state_t, state_t_H) -> float:
    val_t = state_t.get(slot)       # chunk 起始时刻的 slot 值
    val_tH = state_t_H.get(slot)    # chunk 结束时刻的 slot 值
    if val_t is None or val_tH is None:
        return 0.0                   # 值缺失时视为无变化
    return slot_distance(slot, val_t, val_tH)
```

#### 3.2.2 rho（持久性）

```
rho_t^j = (1/L) * sum_{ell=1}^{L} 1[d_j(s_{t+H}^j, s_{t+H+ell}^j) < epsilon_j]
```

**含义：** chunk 结束时 slot 的新值在未来是否能稳定保持。检查未来 L 步中有多大比例的时间步，slot 值与 chunk 结束时的值保持接近（距离小于 `epsilon_j`）。

**为什么需要持久性？** 考虑这样一个场景：机器人短暂接触了目标物体（`target_contact` 变为 1），但随即松开（变回 0）。虽然 delta 很大，但这个变化没有持续，不应被标记为 advance。rho 阈值过滤掉这类短暂变化。

**只用于结果 slot (J_R)。**

返回值在 [0, 1] 之间。若没有未来数据，乐观地返回 1.0。

#### 3.2.3 upsilon（目标效用）

```
upsilon_t^j = d_j(s_t^j, s*^j) - d_j(s_{t+H}^j, s*^j)
```

**含义：** chunk 执行使得该 slot 向最终目标 `s*` 靠近了多少。

- **正值** = 朝目标靠近（距离减小）
- **负值** = 远离目标（距离增大）
- **零** = 无变化

**为什么 upsilon 与 rho 形成"或"关系？** 有些 slot 变化虽然最终不持久（rho 低），但确实朝目标方向推进了（upsilon 高），这种情况也应标记为 advance。反过来，有些变化持久但方向不完全朝向最终目标（如中间过渡状态），此时 rho 高就足以标记为 advance。两者取"或"覆盖更多有意义的变化。

**只用于结果 slot (J_R)。**

#### 3.2.4 eta（未来支持度）

```
eta_t^j = max_{ell} 1[exists k in J_R: d_k(s_{ell-1}^k, s*^k) - d_k(s_{ell}^k, s*^k) > tau_R]
```

**含义：** 使能 slot 变化后，未来是否有结果 slot 因此而推进。

**直觉理解：** 如果末端执行器接近了目标物体（使能 slot 变化），但之后物体并没有被成功抓取或移动（没有结果 slot 推进），那这个接近动作可能是无效的。eta 确保使能 slot 的变化确实在未来"支持"了任务进展。

**只用于使能 slot (J_E)。**

返回值为 0.0 或 1.0（二值指标）。至少需要两步未来数据才能计算。

### 3.3 挖掘流程 5 步

> 对应代码中的 `mine_certificate()` 函数。

```python
def mine_certificate(
    state_t: SlotState,          # chunk 起始状态
    state_t_H: SlotState,        # chunk 结束状态
    goal_signature: SlotState,   # 回合级目标签名 s*
    future_states: List[SlotState],  # chunk 之后的状态序列
    thresholds: MiningThresholds,    # 挖掘超参数
) -> CertificateLabel:
```

**第 1 步：结果 slot 挖掘（J_R）**

对每个结果 slot `j in J_R`：
1. 计算 `delta`（变化量）
2. 若 `delta <= tau_delta` → 跳过（变化量太小）
3. 计算 `rho`（持久性）和 `upsilon`（目标效用）
4. 若 `rho > tau_rho` **或** `upsilon > tau_upsilon` → 标记为 **advance**

```python
for j in J_R:
    delta = _compute_delta(j, state_t, state_t_H)
    if delta <= thresholds.tau_delta:
        continue
    rho = _compute_rho(j, state_t_H, future_window, thresholds.epsilon_j)
    upsilon = _compute_upsilon(j, state_t, state_t_H, goal_signature)
    if rho > thresholds.tau_rho or upsilon > thresholds.tau_upsilon:
        advance_set.add(j)
```

**第 2 步：使能 slot 挖掘（J_E）**

对每个使能 slot `j in J_E`：
1. 计算 `delta`（变化量）
2. 若 `delta <= tau_delta` → 跳过
3. 计算 `eta`（未来支持度）
4. 若 `eta > 0` → 标记为 **advance**

```python
for j in J_E:
    delta = _compute_delta(j, state_t, state_t_H)
    if delta <= thresholds.tau_delta:
        continue
    eta = _compute_eta(j, future_window, goal_signature, thresholds.tau_R)
    if eta > 0:
        advance_set.add(j)
```

**第 3 步：结构化规则推导 preserve 集合**

使用 `compute_preserve_set()` 合并两类保持规则（详见第 4 节）：

```python
preserve_set = compute_preserve_set(state_t.values, advance_set)
```

**第 4 步：分配角色**

- 在 `advance_set` 中 → `"advance"`
- 在 `preserve_set` 中 → `"preserve"`
- 其余 → `"ignore"`

```python
for j in J_CERT:
    if j in advance_set:
        roles[j] = "advance"
        goal_values[j] = state_t_H.get(j)   # 记录目标值
    elif j in preserve_set:
        roles[j] = "preserve"
    else:
        roles[j] = "ignore"
```

**第 5 步：记录 advance slot 的目标值**

对每个 advance slot，目标值 `g_t^j = s_{t+H}^j`（即 chunk 结束时的 slot 值）。preserve 和 ignore slot 不需要目标值。

### 3.4 目标签名计算

> 源码位置：`certvla/data/goal_signature.py`

目标签名 `s*` 是对成功回合终端状态的聚合表示，描述了"任务成功完成时，各个 slot 应处于什么值"。

```
s* = Aggregate(s_{T-K:T})
```

取回合最后 K 步（默认 K=5）的 slot 状态，按域类型聚合：

| 域类型 | 聚合方式 | 示例 |
|--------|----------|------|
| 连续域 / 置信度域 | 取均值 (mean) | `[0.12, 0.10, 0.11, 0.09, 0.10]` → `0.104` |
| 二值域 | 多数投票 = `round(mean)` | `[1, 1, 1, 0, 1]` → `mean=0.8` → `1` |
| 类别域 | 众数 (mode) | `["on_goal", "on_goal", "none", "on_goal", "on_goal"]` → `"on_goal"` |

**为什么取最后 K 步而不是最后一步？**
1. **鲁棒性**：最后一步可能有噪声（如物体轻微滑动），取多步平均更稳定
2. **一致性**：对于二值/类别 slot，多步投票可以消除瞬时错误
3. **代表性**：终端 K 步反映了"稳定完成"状态，而非瞬间状态

### 3.5 MiningThresholds 超参数

> 源码位置：`certvla/data/certificate_mining.py`

```python
@dataclass
class MiningThresholds:
    tau_delta: float = 0.1       # 最小变化量阈值
    tau_rho: float = 0.6         # 持久性阈值
    tau_upsilon: float = 0.05    # 目标效用阈值
    tau_R: float = 0.1           # 使能 slot 的未来结果推进阈值
    L_future: int = 5            # 前瞻窗口步数
    epsilon_j: float = 0.05      # 持久性判断的容忍度
```

各超参数的设计意图：

| 超参数 | 默认值 | 作用 |
|--------|--------|------|
| `tau_delta` | 0.1 | slot 变化量必须超过此值才有资格成为 advance。过小的变化视为噪声。 |
| `tau_rho` | 0.6 | 变化在未来 `L_future` 步中保持不变的比例必须超过 60%。防止短暂的、会被撤销的变化被错误标记。 |
| `tau_upsilon` | 0.05 | chunk 使 slot 向目标靠近的净推进量必须超过 0.05。确保变化方向是朝向任务目标的。 |
| `tau_R` | 0.1 | 用于 eta 计算：未来某个结果 slot 向目标推进的最小量。 |
| `L_future` | 5 | 前瞻窗口大小。用于计算 rho（持久性）和 eta（未来支持度）时向前看多少步。 |
| `epsilon_j` | 0.05 | 在判断 slot 值是否"保持不变"时允许的最大距离。处理浮点精度和微小波动。 |

---

## 4. Preserve 规则

> 源码位置：`certvla/slots/preserve_rules.py`

CertVLA 的核心设计决策之一是：**"数据挖 advance，结构先验定 preserve"**。

advance 的判定是数据驱动的（通过上述四个度量指标和阈值），而 preserve 的判定是规则驱动的（基于物理世界的结构性不变量）。这种混合策略既保证了 advance 标签的准确性（基于实际变化），又保证了 preserve 标签的完备性（基于结构化知识）。

最终 preserve 集合的计算公式：

```
P_t = P_t^latch ∪ P_t^support - A_t
```

即合并两类保持规则的结果后，移除 advance 集合中的 slot，确保 advance 和 preserve **严格互斥**。

### 4.1 Latch-preserve（锁存保持）

```python
def latch_preserve(state, advance_set) -> Set[SlotName]:
```

**条件：** `completion_latch` 在当前状态 `state_t` 中的值为 1（任务已完成）。

**效果：** 所有不在 advance 集合中的结果 slot (`J_R`) 都被标记为 preserve。

**语义：** 一旦某个子任务已经完成（锁存位置 1），不能让后续动作"撤销"这个成果。例如，碗已经放到了盘子上，后续的动作不应该把碗移走。

```
P_t^latch = { j in J_R | completion_latch = 1 且 j not in A_t }
```

### 4.2 Support-preserve（支撑保持）

```python
def support_preserve(state, advance_set) -> Set[SlotName]:
```

**条件和效果由规则表定义。** 每条规则编码一个物理世界的结构性不变量：

> "要成功推进 X slot，你必须保持 Y slot 不变。"

规则格式为 `(trigger_slots, preserved_slot, state_condition)` 三元组。只有当 `advance_set` 包含某个 `trigger_slot` **且**当前状态满足 `state_condition` 时，`preserved_slot` 才被标记为 preserve。

**v1 版本的 5 条规则：**

| 编号 | 触发条件 (advance 中包含) | 状态条件 | 被保持的 slot | 语义 |
|------|---------------------------|----------|---------------|------|
| 1 | `target_goal_proximity` | `hand_occupancy == "target"` | `hand_occupancy` | 搬运过程中必须保持抓握 |
| 2 | `containment_relation` | `target_contact == 1` | `target_contact` | 放入容器时必须保持接触 |
| 3 | `support_relation` | `target_contact == 1` | `target_contact` | 放到支撑面时必须保持接触 |
| 4 | `containment_relation` | `articulation_progress > 0.5` | `articulation_progress` | 放入容器时必须保持容器打开 |
| 5 | `target_goal_proximity` | `target_contact == 1` | `target_contact` | 搬运过程中必须保持接触 |

**互斥约束：** 同一个 slot 不能同时被标记为 advance 和 preserve。如果规则推导出的 preserve 包含 advance 中的 slot，advance 优先。

---

## 5. 张量编码 (Label Codec)

> 源码位置：`certvla/slots/metrics.py`

CertVLA 需要将人类可读的 slot 值（`int` / `float` / `str`）转为神经网络可处理的张量表示。本节描述编码约定。

### 5.1 slot_value_to_tensor

将单个 slot 值编码为 numpy 向量：

| 域类型 | 编码方式 | 输出形状 | 示例 |
|--------|----------|----------|------|
| binary | 直接转 float | `(1,)` | `1 → [1.0]` |
| categorical | one-hot 编码 | `(num_categories,)` | `"target" → [0, 1, 0]` |
| continuous / confidence | 直接转 float | `(1,)` | `0.35 → [0.35]` |

类别到索引的映射由 `SlotMeta.categories` 的声明顺序决定。例如 `hand_occupancy` 的 `("empty", "target", "other")` 对应索引 `(0, 1, 2)`。

### 5.2 tensor_to_slot_value

将张量还原为 slot 值：

| 域类型 | 解码方式 | 注意事项 |
|--------|----------|----------|
| binary | `round()` 取整为 0 或 1 | -- |
| categorical | `argmax()` 取最大分量对应的类别标签 | 全零向量会默认选第一个类别 |
| continuous / confidence | `clamp()` 截断到 `[0, 1]` | -- |

### 5.3 slot_state_to_flat_tensor

将完整的 `SlotState.values` 字典拼接为一个一维向量。拼接顺序按 `SlotName` 枚举顺序，总维度为 `flat_tensor_dim() = 16`。

```
[ee_target_proximity(1)] [hand_occupancy(3)] [target_contact(1)]
[target_goal_proximity(1)] [support_relation(3)] [containment_relation(3)]
[articulation_progress(1)] [orientation_alignment(1)] [completion_latch(1)]
[task_visible_confidence(1)]
```

**缺失 slot 处理：** 字典中缺少的 slot 用零向量填充。

### 5.4 role_to_tensor（设计规范）

角色标签的张量编码约定：

```
advance  → 0
preserve → 1
ignore   → 2
```

输出为 `(B,)` 形状的 long tensor，其中 B 为 batch size。

### 5.5 goal_to_tensor（设计规范）

目标值的张量编码根据 slot 域类型不同：
- **binary / continuous**：`(B, 1)` float tensor
- **categorical**：`(B,)` long tensor（类别索引）

### 5.6 mask 生成

掩码用于在训练时跳过无效 slot 的损失计算：

| 条件 | mask 值 |
|------|---------|
| slot 有效（`validity_mask = True`） | 1.0 |
| slot 无效（`validity_mask = False`） | 0.0 |

---

## 6. 数据增强

### 6.1 corrupt_certificate（设计规范）

**用途：** 随机替换 advance slot 的角色和/或目标值，用于生成依赖性损失 `L_dep` 的负样本。

通过将正确的证书标签"腐化"（corruption），训练模型学会区分正确的证书和错误的证书，增强模型对证书依赖关系的理解。

### 6.2 反事实增强

> 源码位置：`certvla/data/counterfactuals.py`

反事实训练是 CertVLA 的重要训练策略。核心思想是构造成对的训练样本，让模型学会区分"什么变化是重要的"与"什么变化是无关的"。

#### 6.2.1 CounterfactualPair

```python
@dataclass
class CounterfactualPair:
    anchor: CertChunkSample          # 原始样本
    augmented: CertChunkSample       # 反事实变换后的样本
    pair_type: str                   # "nuisance" 或 "breaking"
    modified_slots: Optional[List[str]]  # 仅 "breaking" 对需要
```

#### 6.2.2 干扰保持增强 (nuisance_augment)

对原始样本施加**不应改变** slot 状态/证书标签的变换：
- 背景颜色/纹理变化
- 光照条件变化（亮度、对比度）
- 添加无关干扰物
- 轻微视角变化

训练时，模型对 anchor 和 augmented 应预测**相同**的证书。用于 `L_cf` 损失的 `z_pos`（正样本对）。

#### 6.2.3 后果打破增强 (consequence_augment)

对原始样本施加**应该改变** slot 状态/证书标签的变换：
- 目标物体身份替换（如碗换成杯子）
- 目标容器替换（如盘子改为桌面）
- 支撑/包含关系改变
- 指令与目标对齐改变

训练时，模型对 anchor 和 augmented 应预测**不同**的证书。用于 `L_cf` 损失的 `z_neg`（负样本对）。

#### 6.2.4 当前实现状态

Phase 1 提供了 `IdentityCounterfactualBuilder` 占位实现（返回相同的样本对，不做任何实际变换），仅用于测试管线连通性。Phase 3+ 将实现基于图像增强管线的真正反事实构建器。

---

## 7. Batch Collation

> 设计规范：将 `CertChunkSample` 列表打包为 batch 级张量。

`CertCollator` 将 `List[CertChunkSample]` 打包为如下 batch 结构：

| 键名 | 类型 | 含义 |
|------|------|------|
| `state_target` | `Dict[SlotName, Tensor]` | 每个 slot 的目标状态张量 |
| `role_target` | `Dict[SlotName, Tensor]` | 每个 slot 的角色标签张量（0/1/2） |
| `goal_target` | `Dict[SlotName, Tensor]` | 每个 advance slot 的目标值张量 |
| `mask` | `Dict[SlotName, Tensor]` | 每个 slot 的有效性掩码 |
| `confidence` | `Dict[SlotName, Tensor]` | 每个 slot 的标签置信度 |
| `actions` | `Tensor (B, H, action_dim)` | 批量动作序列 |

其中 `B` 为 batch size，`H` 为 chunk 大小（默认 8），`action_dim` 为动作维度（LIBERO 为 7）。

`CertCollator` 的职责包括：
1. 将每个样本的 `SlotState` 按 slot 拆分为独立张量
2. 将角色字符串编码为整数 (`advance=0, preserve=1, ignore=2`)
3. 将目标值按域类型编码为适当的张量格式
4. 生成有效性掩码，无效 slot 的损失权重设为 0.0
5. 将动作序列堆叠为 `(B, H, action_dim)` 张量

---

## 8. 完整离线标注流程

> 源码位置：`certvla/data/label_episodes.py`

以下是从原始 LIBERO 演示数据到可训练样本的完整流程：

```
┌────────────────────────┐
│ LIBERO HDF5 演示文件     │  初始状态 + 动作序列
│ (初始状态 + 动作)        │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ 仿真器回放               │  env.reset() + env.step(action)
│ + extract_state() 标注  │  每步调用 LiberoOracleLabeler
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ SlotState 序列           │  T 个时间步的完整 slot 状态
│ [s_0, s_1, ..., s_T]   │
└──────────┬─────────────┘
           │
    ┌──────┴───────┐
    ▼              ▼
┌──────────┐  ┌────────────────┐
│ 目标签名   │  │ chunk 切分       │  每 H=8 步切一个 chunk
│ s* 计算   │  │ (t=0,8,16,...) │
└────┬─────┘  └──────┬─────────┘
     │               │
     └───────┬───────┘
             ▼
┌────────────────────────┐
│ mine_certificate()      │  对每个 chunk:
│ (s_t, s_{t+H}, s*,     │    state_t, state_t_H, goal_sig
│  future_states)         │    + 未来 L_future 步状态
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ CertificateLabel        │  roles + goal_values
│ (advance/preserve/ign) │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ save_episode_labels()   │  → {task}_demo_{i}_labels.npz
│ np.savez_compressed    │
└────────────────────────┘
```

**命令行用法（需要 LIBERO + MuJoCo 环境）：**

```bash
python -m certvla.data.label_episodes \
    --libero_task_suite libero_spatial \
    --hdf5_dir ./LIBERO/libero/datasets/libero_spatial_no_noops \
    --output_dir ./certvla_labels/libero_spatial \
    --chunk_size 8
```

---

## 9. 关键源码文件速查

| 文件路径 | 职责 |
|----------|------|
| `certvla/slots/schema.py` | Slot 词汇表定义：10 个 slot 的名称、域类型、族归属 |
| `certvla/slots/metrics.py` | 距离度量 `d_j(a,b)` 和值-张量互转工具 |
| `certvla/slots/role_sets.py` | 族集合定义：J_E、J_R、J_C、J_CERT |
| `certvla/slots/preserve_rules.py` | 结构性保持规则：latch-preserve、support-preserve |
| `certvla/data/chunk_sample.py` | 核心数据结构：SlotState、CertificateLabel、CertChunkSample |
| `certvla/data/certificate_mining.py` | 证书挖掘算法：四个度量指标 + 5 步流程 |
| `certvla/data/goal_signature.py` | 目标签名计算：终端 K 步聚合 |
| `certvla/data/state_labels.py` | 状态标注器接口：StateLabeler、LiberoOracleLabeler |
| `certvla/data/label_episodes.py` | 离线标注主脚本：回放 + 标注 + 保存 sidecar |
| `certvla/data/counterfactuals.py` | 反事实样本构建接口 |
