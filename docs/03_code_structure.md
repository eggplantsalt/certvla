# CertVLA 代码结构详解

> 本文档详细介绍 CertVLA 各模块的代码组织、文件职责和数据流。
> 所有解释性文字使用中文，代码标识符和文件路径保留英文原样。

---

## 1. 目录总览

```
certvla/
├── __init__.py                   # 包顶层入口
├── slots/                        # 状态槽位词汇表（最底层定义）
│   ├── __init__.py               # 导出 SlotName, SLOT_REGISTRY, J_E, J_R 等
│   ├── schema.py                 # v1 冻结词汇表: 10 个 slot 的枚举 + 元数据
│   ├── metrics.py                # 逐 slot 距离度量 + 值-张量互转
│   ├── role_sets.py              # J_E / J_R / J_C / J_CERT 族集合
│   └── preserve_rules.py         # 结构先验保持规则 (latch + support)
├── data/                         # 数据管线（标注、挖掘、样本结构）
│   ├── __init__.py               # 导出 SlotState, CertificateLabel, mine_certificate 等
│   ├── chunk_sample.py           # 训练样本核心数据结构
│   ├── certificate_mining.py     # 证书挖掘算法 (advance/preserve/ignore)
│   ├── goal_signature.py         # 目标签名 s* 计算
│   ├── state_labels.py           # 状态标注器抽象接口 + LIBERO oracle 桩
│   ├── counterfactuals.py        # 反事实样本构建接口
│   └── label_episodes.py         # 离线标注脚本: HDF5 -> sidecar .npz
├── model/                        # 模型层（前向传播子模块）
│   ├── __init__.py               # 导出 CertVLAOutput
│   ├── state_token.py            # 持久化状态令牌 z_0 + 门控更新
│   ├── state_readout.py          # 状态读出头 R_phi(z_t) -> s_t
│   ├── certificate_head.py       # 证书头 Q_psi(z_t) -> c_t
│   ├── action_head.py            # 证书条件化动作头 (粗 + 细残差)
│   ├── certvla_wrapper.py        # 顶层包装器: 编排所有子模块
│   └── outputs.py                # 前向传播输出数据类 CertVLAOutput
├── training/                     # 训练流水线
│   ├── __init__.py               # 导出所有损失函数 + 课程调度器
│   ├── losses.py                 # 7 个损失函数 + 加权组合
│   ├── curriculum.py             # 4 阶段课程学习调度
│   └── sched_sampling.py         # 状态令牌的教师强迫退火
└── inference/                    # 推理与修复
    ├── __init__.py               # 导出 GapResult, RepairController, InferenceLogger
    ├── gap.py                    # 证书间隙计算
    ├── repair.py                 # 短视野局部修复控制器
    └── logging.py                # 推理日志与调试追踪
```

---

## 2. `certvla/slots/` -- 状态槽位词汇表

本子包是整个 CertVLA 的**最底层定义**，定义了"slot"（状态槽位）这一核心抽象。所有下游模块都从此处导入 slot 相关的元数据。

### 2.1 `schema.py` -- 词汇表的唯一权威定义

本文件定义了 v1 版本全部 **10 个 slot** 的名称、值域类型和族归属。它是整个 CertVLA 的"真理之源"(single source of truth)。

**核心类型：**

| 类型 | 说明 |
|------|------|
| `SlotName(str, Enum)` | 10 个 slot 的枚举名称。枚举声明顺序决定了 flat tensor 的拼接顺序，**不可重排** |
| `SlotDomain(str, Enum)` | 4 种值域类型：`BINARY`（二值 {0,1}）、`CATEGORICAL`（有限字符串标签集）、`CONTINUOUS`（[0,1] 连续值）、`CONFIDENCE`（[0,1] 置信度，语义上区别于 CONTINUOUS） |
| `SlotFamily(str, Enum)` | 3 个族：`ENABLING`（J_E，使能/过渡）、`RESULT`（J_R，结果/锁存）、`CONFIDENCE`（J_C，置信度） |
| `SlotMeta` | 单个 slot 的元数据，`frozen=True` 的 dataclass。字段包括 `name`、`domain`、`family`、`categories`（仅 CATEGORICAL）、`valid_range` |

**全局注册表：**

```python
SLOT_REGISTRY: Dict[SlotName, SlotMeta]  # 10 个 slot 的完整注册表
SLOT_VOCAB_SIZE = 10                     # v1 断言: 恰好 10 个 slot
```

**v1 的 10 个 slot 速览：**

| Slot 名称 | 值域 | 族 | 语义 |
|-----------|------|-----|------|
| `ee_target_proximity` | continuous | J_E | 末端执行器到目标物体的归一化距离 |
| `hand_occupancy` | categorical(3) | J_E | 手爪状态: empty / target / other |
| `target_contact` | binary | J_E | 是否与目标物体接触 |
| `articulation_progress` | continuous | J_E | 铰接体的打开进度 [0,1] |
| `orientation_alignment` | continuous | J_E | 目标物体朝向对齐度 [0,1] |
| `target_goal_proximity` | continuous | J_R | 目标物体到目标位置的归一化距离 |
| `support_relation` | categorical(3) | J_R | 支撑关系: none / on_goal / on_other |
| `containment_relation` | categorical(3) | J_R | 包含关系: none / in_goal / in_other |
| `completion_latch` | binary | J_R | 任务完成锁存: 0 或 1，一旦置 1 不应回退 |
| `task_visible_confidence` | confidence | J_C | 任务相关物体是否可见的置信度 |

**辅助函数：**

- `get_slot_meta(name)`: 按名称查询 slot 元数据的统一入口
- `SlotMeta.validate_value(value)`: 校验给定值是否在该 slot 的合法域内
- `SlotMeta.num_categories`: 返回该 slot 在 flat tensor 中占据的维度数

**设计要点：**

- 所有 slot 是**任务相对**的（task-relative），而非场景图级别的绝对描述
- 连续量统一归一化到 [0, 1]，便于跨 slot 比较和加权
- 分类型 slot 使用字符串标签保持人类可读性，转张量时才做 one-hot 编码
- `frozen=True` 保证元数据一旦注册不可变

---

### 2.2 `metrics.py` -- 距离度量与值-张量转换

本文件提供两大能力：(1) 逐 slot 距离度量 d_j(a, b)；(2) slot 值与 numpy 张量的互转。

**距离函数：**

```python
def slot_distance(slot: SlotName, a, b) -> float:
    """计算单个 slot 的距离 d_j(a, b)，返回值在 [0, 1] 内"""
```

距离度量按值域类型分派：
- **BINARY / CONTINUOUS / CONFIDENCE**: L1 距离 `|a - b|`
- **CATEGORICAL**: Hamming 距离（相同 = 0，不同 = 1）

**值-张量转换：**

| 函数 | 方向 | 说明 |
|------|------|------|
| `slot_value_to_tensor(slot, value)` | 值 -> numpy | BINARY/CONTINUOUS/CONFIDENCE -> `shape (1,)`; CATEGORICAL -> one-hot `shape (num_cat,)` |
| `tensor_to_slot_value(slot, tensor)` | numpy -> 值 | BINARY: round; CATEGORICAL: argmax; CONTINUOUS: clamp |
| `slot_state_to_flat_tensor(values)` | dict -> 1d numpy | 按 `SlotName` 枚举顺序拼接所有 slot 的张量，缺失 slot 用零填充 |
| `flat_tensor_dim()` | -- | 返回 flat tensor 的总维度。v1 = 16 维 |

**v1 Flat Tensor 布局（共 16 维）：**

```
[ee_target_proximity(1)] [hand_occupancy(3)] [target_contact(1)]
[target_goal_proximity(1)] [support_relation(3)] [containment_relation(3)]
[articulation_progress(1)] [orientation_alignment(1)] [completion_latch(1)]
[task_visible_confidence(1)]
```

**常见陷阱：**
- 缺失 slot 在 flat tensor 中用零向量表示。对 binary slot 来说零等价于"否"，对 categorical slot 来说全零不是合法 one-hot
- `tensor_to_slot_value` 对全零 categorical 张量做 argmax 会返回 index=0（即第一个类别），这是默认行为但未必语义正确

---

### 2.3 `role_sets.py` -- 族集合定义

本文件基于 `SLOT_REGISTRY` 的 `family` 字段，自动构建四个关键的 slot 集合：

```python
J_E: FrozenSet[SlotName]    # 使能/过渡 slot (5 个): ee_target_proximity, hand_occupancy,
                             #   target_contact, articulation_progress, orientation_alignment
J_R: FrozenSet[SlotName]    # 结果/锁存 slot (4 个): target_goal_proximity, support_relation,
                             #   containment_relation, completion_latch
J_C: FrozenSet[SlotName]    # 置信度 slot (1 个): task_visible_confidence
J_CERT: FrozenSet[SlotName] # 证书参与 slot (9 个) = J_E | J_R
```

**族的语义：**
- **J_E（使能/过渡）**: 描述动作执行的前提条件，在 action chunk 执行过程中会发生变化
- **J_R（结果/锁存）**: 描述动作执行的目标后果，具有锁存特性（一旦达到期望值不应回退）
- **J_C（置信度）**: 描述观测可靠性，不参与证书判定

集合由 `SLOT_REGISTRY` 自动推导（不硬编码成员），使用 `frozenset` 保证不可变。文件末尾有 v1 完整性断言，确保族集合成员与设计文档一致。

辅助函数 `get_family(slot_name)` 返回给定 slot 的族归属。

---

### 2.4 `preserve_rules.py` -- 结构先验保持规则

CertVLA 的核心设计决策之一："数据挖 advance，结构先验定 preserve"。本文件实现了 preserve 集合的推导逻辑。

**两类保持规则：**

**1. Latch-preserve（锁存保持）：**

```python
def latch_preserve(state, advance_set) -> Set[SlotName]:
    """当 completion_latch=1 时，所有不在 advance_set 中的 J_R slot 都应保持"""
```
- 公式: `P_t^latch = { j in J_R | completion_latch = 1 and j not in A_t }`
- 直觉: 已经完成的子目标不应被后续动作"撤销"

**2. Support-preserve（支撑保持）：**

```python
def support_preserve(state, advance_set) -> Set[SlotName]:
    """基于规则表推导: 推进 X 时必须保持 Y"""
```

规则表（`_SUPPORT_RULES`）编码了 5 条物理世界的结构性不变量：

| 规则 | 触发条件 (trigger) | 被保持 slot | 状态条件 |
|------|-------------------|----|----------|
| 1 | 推进 `target_goal_proximity` | `hand_occupancy` | 手爪正抓着目标物 |
| 2 | 推进 `containment_relation` | `target_contact` | 已与目标接触 |
| 3 | 推进 `support_relation` | `target_contact` | 已与目标接触 |
| 4 | 推进 `containment_relation` | `articulation_progress` | 铰接体已打开超过 50% |
| 5 | 推进 `target_goal_proximity` | `target_contact` | 已与目标接触 |

**综合计算：**

```python
def compute_preserve_set(state, advance_set) -> Set[SlotName]:
    """P_t = P_t^latch | P_t^support - A_t"""
```

合并两类规则后移除 advance 集合，确保 advance 和 preserve 严格互斥。剩余 slot 自动归为 ignore。

---

## 3. `certvla/data/` -- 数据管线

本子包定义了训练数据的完整表示和生成流程：从原始仿真状态到带证书监督信号的训练样本。

### 3.1 `chunk_sample.py` -- 训练样本核心数据结构

定义了三层嵌套的数据结构，构成训练数据单元：

**`SlotState`（最底层）-- 单个时间步的 slot 状态：**

```python
@dataclass
class SlotState:
    values: Dict[SlotName, Union[int, float, str]]  # slot 值
    validity_mask: Dict[SlotName, bool]              # 有效性掩码
    confidence: Dict[SlotName, float]                # 标签置信度
```

- `values`: 键为 `SlotName`，值的类型取决于域（binary -> int, categorical -> str, continuous -> float）
- `validity_mask`: 标记 slot 在当前任务中是否适用（如 pick-and-place 任务中 `articulation_progress` 无效）
- `confidence`: 仿真器 oracle 标注 = 1.0，伪标签 < 1.0，用于训练时的损失加权

**`CertificateLabel`（中间层）-- 单个 chunk 的证书标签：**

```python
@dataclass
class CertificateLabel:
    roles: Dict[SlotName, str]          # 角色: "advance" / "preserve" / "ignore"
    goal_values: Dict[SlotName, Any]    # 仅 advance slot 有值: chunk 结束时的目标值
```

提供 `advance_slots()`、`preserve_slots()`、`ignore_slots()` 便捷查询方法。

**`CertChunkSample`（最顶层）-- 完整训练样本：**

```python
@dataclass
class CertChunkSample:
    observation: np.ndarray            # RGB 图像 (H, W, 3)
    instruction: str                   # 自然语言指令
    actions: np.ndarray                # 动作序列 (H, action_dim)
    state_t: SlotState                 # chunk 起始 slot 状态
    state_t_H: SlotState              # chunk 结束 slot 状态
    certificate: CertificateLabel      # 证书标签
    goal_signature: Optional[SlotState]  # 回合级目标签名 s*
    episode_id: Optional[str]          # 回合标识
    timestep: Optional[int]            # 起始时间步索引
```

对应论文中的 `tau_t = (o_t, l, A_{t:t+H-1}, o_{t+H})` 加上证书监督字段。

---

### 3.2 `certificate_mining.py` -- 证书挖掘算法

这是数据管线中最关键的模块，从离线回放的 slot 状态序列中自动挖掘每个 chunk 的证书标签。

**超参数：**

```python
@dataclass
class MiningThresholds:
    tau_delta: float = 0.1     # 最小变化量阈值
    tau_rho: float = 0.6       # 持久性阈值
    tau_upsilon: float = 0.05  # 目标效用阈值
    tau_R: float = 0.1         # 使能 slot 的未来结果阈值
    L_future: int = 5          # 前瞻窗口步数
    epsilon_j: float = 0.05    # 持久性容忍度
```

**四个度量指标：**

| 度量 | 公式 | 适用族 | 含义 |
|------|------|--------|------|
| delta | `d_j(s_t^j, s_{t+H}^j)` | J_E, J_R | chunk 前后的变化量 |
| rho | `(1/L) * sum 1[d < epsilon]` | J_R | 变化在未来是否稳定保持 |
| upsilon | `d(s_t, s*) - d(s_{t+H}, s*)` | J_R | 是否朝目标方向推进（正值 = 靠近） |
| eta | `max 1[exists k in J_R: ...]` | J_E | 未来是否有结果 slot 因此推进 |

**主入口函数 `mine_certificate()` 的 5 步流程：**

1. **结果 slot (J_R)**: 计算 delta + (rho **或** upsilon)。若 `delta > tau_delta` 且 `rho > tau_rho` 或 `upsilon > tau_upsilon`，标记为 advance
2. **使能 slot (J_E)**: 计算 delta + eta。若 `delta > tau_delta` 且 `eta > 0`，标记为 advance
3. **Preserve 集合**: 调用 `compute_preserve_set()` 基于结构规则推导
4. **角色分配**: advance / preserve / 其余为 ignore
5. **目标值记录**: advance slot 的 `goal_values[j] = s_{t+H}^j`

**关键设计：** advance 的判定是数据驱动的（通过阈值），preserve 是规则驱动的。这种混合策略保证了 advance 标签的准确性和 preserve 标签的完备性。

---

### 3.3 `goal_signature.py` -- 目标签名计算

```python
def compute_goal_signature(episode_states: List[SlotState], K: int = 5) -> SlotState:
```

将回合终端 K 步（默认 5 步）的 slot 状态聚合为一个目标签名 `s*`，作为证书挖掘中 upsilon 计算的参考点。

**聚合策略按域类型区分：**
- **连续域 / 置信度域**: 取均值 (mean)
- **二值域**: 多数投票 (majority vote) = `round(mean)`
- **类别域**: 众数 (mode)

取多步而非最后一步的原因：鲁棒性（消除噪声）、一致性（投票消除瞬时错误）、代表性（反映"稳定完成"状态）。

---

### 3.4 `state_labels.py` -- 状态标注器接口

定义了两层抽象：

**`StateLabeler`（抽象基类）：**

```python
class StateLabeler(ABC):
    def extract_state(self, env, obs, instruction) -> SlotState: ...
```

统一接口，任何环境都可通过实现此接口提供 slot 状态标签。`instruction` 参数用于角色绑定（确定哪个物体是目标、哪个是容器等）。

**`PseudoLabelInterface`（伪标签接口）：**

```python
class PseudoLabelInterface(ABC):
    def extract_state_from_obs(self, image, proprio, instruction) -> SlotState: ...
```

不需要仿真器，仅从图像和本体感觉推断。置信度 < 1.0。Phase 1 只定义接口，Phase 3+ 实现。

**`LiberoOracleLabeler`（LIBERO 仿真器 oracle 标注器）：**

`StateLabeler` 在 LIBERO 环境下的具体实现。通过 `env.sim.data` 精确计算所有 10 个 slot 的值。当前为桩实现（`NotImplementedError`），完整实现推迟到 LIBERO 环境 API 调查完成后。

---

### 3.5 `counterfactuals.py` -- 反事实样本构建接口

定义了用于对比学习的反事实训练机制：

**`CounterfactualPair`** -- 反事实样本对数据结构：

```python
@dataclass
class CounterfactualPair:
    anchor: CertChunkSample          # 原始样本
    augmented: CertChunkSample       # 变换后样本
    pair_type: str                   # "nuisance" 或 "breaking"
    modified_slots: Optional[List[str]]  # 仅 breaking 对需要
```

**两种反事实对：**
- **干扰保持对 (Nuisance-preserving)**: 施加无关变换（背景、光照、干扰物），结果不应改变证书 -- 训练不变性
- **后果打破对 (Consequence-breaking)**: 施加有意义的变换（目标替换、容器替换），结果应改变证书 -- 训练敏感性

**`CounterfactualBuilder`** -- 抽象构建器接口：
- `build_nuisance_pair(sample)`: 构建干扰保持对
- `build_breaking_pair(sample)`: 构建后果打破对

**`IdentityCounterfactualBuilder`** -- 恒等变换占位实现（返回相同样本），用于无图像增强依赖时的管线测试。

---

### 3.6 `label_episodes.py` -- 离线标注脚本

将 LIBERO HDF5 演示数据转换为 CertVLA 训练所需的 sidecar `.npz` 标签文件。

**命令行用法：**
```bash
python -m certvla.data.label_episodes \
    --libero_task_suite libero_spatial \
    --hdf5_dir ./LIBERO/libero/datasets/libero_spatial_no_noops \
    --output_dir ./certvla_labels/libero_spatial \
    --chunk_size 8
```

**核心函数：**

| 函数 | 说明 |
|------|------|
| `label_all_episodes(...)` | 主流程: 加载 HDF5 -> 仿真器回放 -> 逐步标注 -> 证书挖掘 -> 保存 .npz |
| `save_episode_labels(...)` | 将标签保存为 `.npz`，包含 slot_states、validity_masks、confidences、chunk_certificates、goal_signature、metadata |
| `load_episode_labels(path)` | 加载 `.npz` 标签文件 |

`.npz` 文件格式：
- `slot_states`: `(T, flat_dim)` float32
- `validity_masks`: `(T, num_slots)` bool
- `confidences`: `(T, num_slots)` float32
- `chunk_certificates`: `(num_chunks, num_cert_slots)` int8
- `goal_signature`: `(flat_dim,)` float32

---

## 4. `certvla/model/` -- 模型层

本子包定义了 CertVLA 的所有神经网络子模块。**重要设计约束**: 本包中的所有文件都不导入 `prismatic` 相关代码，避免触发重量级依赖链。

### 4.1 `state_token.py` -- 持久化状态令牌

```python
class StateTokenModule(nn.Module):
```

维护一个"任务状态向量" `z_t`，是 CertVLA 的信息瓶颈：下游所有决策头（状态读出、证书头、动作头）都仅从 `z_t` 出发。

**核心参数：**
- `z_0`: 可学习初始状态 `nn.Parameter`，形状 `(1, llm_dim)`，用小方差正态分布初始化 (std=0.02)
- `gate_proj`: 门控投影层 `Linear(2 * llm_dim, llm_dim)`，权重和偏置初始化为零

**门控更新公式：**

```
tilde_z_t = LLM 在状态令牌位置的隐藏状态
g_t = sigmoid(W_g [tilde_z_t ; z_{t-1}] + b_g)    # 逐元素门控, shape (B, llm_dim)
z_t = g_t * tilde_z_t + (1 - g_t) * z_{t-1}
```

- 当 `g_t` 接近 1 时，状态完全更新为新信息
- 当 `g_t` 接近 0 时，状态几乎不变
- 初始化偏置为零使得 `sigmoid(0) = 0.5`，训练初期新旧各占一半

**关键方法：**

| 方法 | 返回形状 | 说明 |
|------|---------|------|
| `get_initial_state(batch_size)` | `(B, llm_dim)` | 获取批次化的 z_0 |
| `get_state_token_embedding(batch_size)` | `(B, 1, llm_dim)` | 获取注入 LLM 输入序列的嵌入 |
| `gated_update(tilde_z_t, z_prev)` | `(z_t, gate)` | 门控融合，返回更新后状态和门控值 |

**v1 训练策略：** 每个样本独立，`z_prev = z_0`（不做 episode 级递归），推理时可选择递归传递 `z_t`。

---

### 4.2 `state_readout.py` -- 状态读出头

```python
class StateReadoutHead(nn.Module):
    """R_phi(z_t) -> hat_s_t"""
```

从压缩状态 `z_t` 中读出结构化的任务状态估计。

**核心信息隔离约束：** 输入**仅为 `z_t` 向量** `(B, llm_dim)`，而**不是** LLM 的完整序列隐藏状态。这确保所有任务状态信息必须经过 `z_t` 瓶颈，防止读出头"偷看"视觉令牌。

**架构：** 共享主干 + 每 slot 独立输出头

```
z_t (B, 4096) -> LayerNorm -> Linear(4096, 512) -> ReLU -> Linear(512, 512) -> ReLU
                                                                                |
                    ┌────────────────────────────────────────────────────────────┘
                    ↓
              per-slot Linear(512, out_dim)  -- 每个 slot 一个独立的输出头
```

**输出格式（按域类型）：**

| 域类型 | 输出形状 | 激活函数 |
|--------|---------|----------|
| BINARY | `(B, 1)` | sigmoid |
| CATEGORICAL | `(B, num_categories)` | 无（原始 logits，softmax 在损失函数中处理） |
| CONTINUOUS | `(B, 1)` | sigmoid |
| CONFIDENCE | `(B, 1)` | sigmoid |

---

### 4.3 `certificate_head.py` -- 证书头

```python
class CertificateHead(nn.Module):
    """Q_psi(z_t) -> hat_c_t = {(u_t^j, g_t^j)} for j in J_CERT"""
```

从 `z_t` 中预测每个证书 slot 的角色（advance/preserve/ignore）和目标值。

**角色常量定义：**

```python
ROLE_ADVANCE = 0   # 推进
ROLE_PRESERVE = 1  # 保持
ROLE_IGNORE = 2    # 忽略
NUM_ROLES = 3
```

**架构：** 共享主干 + 每 slot 双输出头（角色分类 + 目标预测），仅处理 J_CERT 中的 9 个 slot

```
z_t (B, 4096) -> 共享主干 (与 StateReadoutHead 结构相同但参数独立) -> h (B, 512)
                                                                       |
               ┌───────────────────────────────────────────────────────┘
               ↓
    per-slot role_head: Linear(512, 3)           -> role_logits (B, 3)
    per-slot goal_head: Linear(512, goal_dim)    -> goal_preds  (B, dim)
```

**输出：**
- `role_logits`: `Dict[SlotName, Tensor(B, 3)]` -- 三分类原始 logits
- `goal_preds`: `Dict[SlotName, Tensor(B, dim)]` -- 目标值预测，激活函数与 `StateReadoutHead` 一致

**注意：** `goal_preds` 的值只在 `role=advance` 时有意义。其他角色下仍有输出但不应参与损失计算。

---

### 4.4 `action_head.py` -- 证书条件化动作头

本文件包含四个类，实现了粗-细双分支动作生成架构。

**4.4.1 `CertificateEmbedding` -- 证书嵌入模块**

将 `CertificateHead` 输出的字典形式的 `role_logits` 和 `goal_preds` 展平并投影为固定大小的密集向量。

```
cert_flat (B, cert_raw_dim) -> Linear -> ReLU -> Linear -> cert_embed (B, embed_dim=256)
```

`cert_raw_dim = sum(3 + goal_dim for each slot in J_CERT)`

**4.4.2 `CoarseActionBranch` -- 粗动作分支**

```python
class CoarseActionBranch(nn.Module):
    """pi_c(z_t, hat_c_t) -> A_t^coarse   -- 不接触观测 o_t!"""
```

**核心特性：输入仅为 `z_t` + `cert_embed`，完全不接触观测**。这确保粗动作的语义完全由任务状态和证书决定。

```
[z_t (4096) ; cert_embed (256)] -> LayerNorm -> Linear(4352, 1024) -> ReLU
    -> Linear(1024, 1024) -> ReLU -> LayerNorm -> Linear(1024, 56) -> reshape -> (B, 8, 7)
```

**4.4.3 `FineActionBranch` -- 细动作残差分支**

```python
class FineActionBranch(nn.Module):
    """pi_f(o_t, z_t, hat_c_t) -> Delta_A_t   -- 可以访问观测"""
```

输入包含 LLM 的动作隐藏状态（编码了观测 `o_t`），负责基于当前视觉做几何精修。

```
actions_hidden_states (B, 56, 4096) -> reshape -> (B, 8, 28672)
concat [act_h (28672) ; z_t (4096) ; cert_embed (256)] = (B, 8, 33024)
    -> LayerNorm -> Linear -> ReLU -> Linear -> ReLU -> LayerNorm -> Linear -> (B, 8, 7)
```

所有 chunk 步共享相同的网络权重。

**4.4.4 `CertActionHead` -- 顶层组合模块**

```python
class CertActionHead(nn.Module):
    """hat_A_t = A_t^coarse + lambda_res * Delta_A_t^fine"""
```

组合以上三个组件，替换 OpenVLA-OFT 原始的 `L1RegressionActionHead`。

**前向传播流程：**

```
1. cert_flat = flatten(role_logits, goal_preds)      -> (B, cert_raw_dim)
2. cert_embed = CertificateEmbedding(cert_flat)      -> (B, 256)
3. A_coarse = CoarseActionBranch(z_t, cert_embed)    -> (B, 8, 7)
4. Delta_A = FineActionBranch(act_hidden, z_t, cert_embed) -> (B, 8, 7)
5. A = A_coarse + lambda_res * Delta_A               -> (B, 8, 7)
```

`lambda_res` 是可学习标量 `nn.Parameter`，初始值 0.1，确保训练初期粗分支主导（约 91%），细分支贡献微小（约 9%）。

---

### 4.5 `certvla_wrapper.py` -- 顶层包装器

```python
class CertVLAWrapper(nn.Module):
```

CertVLA 系统的顶层编排器，组合所有子模块。**设计原则：最小侵入**。它是独立的 `nn.Module`，不是 `OpenVLAForActionPrediction` 的子类，不拥有也不修改基础 VLA 模型。

**四个子模块：**

```python
self.state_token       = StateTokenModule(llm_dim)
self.state_readout     = StateReadoutHead(llm_dim, readout_hidden_dim)
self.certificate_head  = CertificateHead(llm_dim, cert_hidden_dim)
self.action_head       = CertActionHead(...)
```

**`forward()` 六步流程：**

```
输入: last_hidden_states (B, seq_len, llm_dim), state_token_pos, action_start_pos

Step 1: tilde_z_t = last_hidden_states[:, state_token_pos, :]     # 提取状态令牌位置
Step 2: z_t, gate = gated_update(tilde_z_t, z_prev)               # 门控更新
Step 3: state_readout = StateReadoutHead(z_t)                      # 状态读出
Step 4: role_logits, goal_preds = CertificateHead(z_t)             # 证书预测
Step 5: actions_hidden = last_hidden_states[:, action_start:end, :] # 提取动作隐藏
Step 6: actions = CertActionHead(z_t, role_logits, goal_preds, actions_hidden)

输出: CertVLAOutput
```

**输入序列中的位置索引：**

```
[视觉 patches, 0..N-1] [状态令牌, N] [文本令牌, N+1..M] [动作令牌, M+1..end]
                         ^                                 ^
                   state_token_pos                   action_start_pos
```

---

### 4.6 `outputs.py` -- 前向传播输出数据类

```python
@dataclass
class CertVLAOutput:
    z_t: torch.Tensor                          # (B, llm_dim) 持久状态
    state_readout: Dict[SlotName, Tensor]       # 逐 slot 状态预测
    role_logits: Dict[SlotName, Tensor]         # (B, 3) 角色 logits
    goal_preds: Dict[SlotName, Tensor]          # 目标值预测
    actions_coarse: torch.Tensor                # (B, H, action_dim) 粗动作
    actions_fine: torch.Tensor                  # (B, H, action_dim) 细残差
    actions: torch.Tensor                       # (B, H, action_dim) 最终动作
    actions_hidden_states: torch.Tensor         # (B, H*action_dim, llm_dim) LLM 动作隐藏
    gate_value: Optional[torch.Tensor] = None   # (B, 1) 门控均值 (诊断用)
```

打包了前向传播所有子模块的输出，供损失计算和推理使用。

---

## 5. `certvla/training/` -- 训练流水线

### 5.1 `losses.py` -- 7 个损失函数 + 辅助工具

本文件是训练流水线的核心，定义了 7 个损失项和加权组合逻辑。

**辅助函数：**

| 函数 | 说明 |
|------|------|
| `focal_cross_entropy(logits, targets, gamma=2.0)` | Focal Loss: `FL = -(1-p_t)^gamma * log(p_t)`，缓解 ignore 类别的严重不平衡 |
| `_per_slot_loss(pred, target, meta)` | 按域类型选择损失: BINARY -> BCE, CATEGORICAL -> CE, CONTINUOUS/CONFIDENCE -> L1 |
| `_slot_pred_distance(a, b, meta)` | 两个模型预测之间的可微距离: 连续/二值 -> \|a-b\|, 分类 -> 总变差距离 |
| `_zero(device)` | 创建 `requires_grad=True` 的零张量，避免计算图断裂 |

**7 个损失函数：**

| 损失 | 函数 | 公式 | 说明 |
|------|------|------|------|
| L_state | `cert_state_loss()` | `sum_j [m^j * alpha^j * ell_j(hat_s^j, s^j)]` | 状态读出损失，按域类型分派 BCE/CE/L1 |
| L_role | `cert_role_loss()` | `sum_{j in J_cert} [m^j * alpha^j * FocalCE(hat_u^j, u^j)]` | 角色分类损失，使用 Focal CE 处理类别不平衡 |
| L_goal | `cert_goal_loss()` | `sum_{j: u=adv} [m^j * alpha^j * ell_j(hat_g^j, s_{t+H}^j)]` | 目标预测损失，仅对 advance slot 计算 |
| L_act | `cert_action_loss()` | `(1/H) sum_k \|\|hat_a - a*\|\|_1` | 动作回归 L1 损失 |
| L_cons | `cert_consistency_loss()` | `L_adv_cons + lambda_pre * L_pre_cons` | 结构一致性: advance slot 目标 vs chunk 结束值 + preserve slot 维持 |
| L_dep | `cert_dependence_loss()` | `max(0, margin + e_pos - e_neg)` | 证书依赖性: 正确证书下的动作误差应小于腐蚀证书 |
| L_cf | `cert_counterfactual_loss()` | `L_inv + L_brk` | 反事实: z_t 对干扰不变 + 对后果敏感 |

**总损失加权组合：**

```python
def cert_total_loss(losses, weights) -> (total, components):
    """加权求和: total = sum(lambda_x * L_x)"""
```

权重映射: `state -> lambda_s, role -> lambda_r, goal -> lambda_g, action -> lambda_a, consistency -> lambda_c, dependence -> lambda_d, counterfactual -> lambda_cf`。

**贯穿所有损失项的设计模式：** `m^j * alpha^j * loss_j` -- 掩码控制哪些 slot 参与，置信度加权降低不可靠标注的影响。

---

### 5.2 `curriculum.py` -- 4 阶段课程学习

**`TrainingStage` -- 4 个阶段枚举：**

| 阶段 | 名称 | 激活的损失 | 冻结策略 |
|------|------|-----------|----------|
| Stage 1 | `state` | L_state | 冻结 backbone + cert + action，训练 state |
| Stage 2 | `certificate` | + L_role, L_goal | 解冻 cert head |
| Stage 3 | `policy` | + L_act, L_cons, L_dep | 解冻 action head |
| Stage 4 | `counterfactual` | + L_cf | 全部解冻（backbone 仍冻结） |

**`StageConfig` -- 单阶段配置 dataclass：**

包含所有损失权重（`lambda_s` 到 `lambda_cf`）、损失超参数（`lambda_pre`, `dep_margin`, `focal_gamma`, `cf_mu`）和冻结标记（`freeze_backbone`, `freeze_state`, `freeze_certificate`, `freeze_action`）。

**`CurriculumScheduler` -- 步数到阶段的调度器：**

```python
scheduler = CurriculumScheduler()
config = scheduler.get_config(step=8000)  # 返回 Stage 2 的配置
weights = scheduler.get_loss_weights(step=20000)  # 返回 Stage 3 的权重字典
```

默认阶段边界：

| 阶段 | 步数范围 |
|------|---------|
| Stage 1 | 0 -- 5,000 |
| Stage 2 | 5,000 -- 15,000 |
| Stage 3 | 15,000 -- 40,000 |
| Stage 4 | 40,000 -- 60,000 |

提供 `should_compute_dep(step)` 和 `should_compute_cf(step)` 便捷方法，判断当前阶段是否需要负证书前向传播或增强样本对。

---

### 5.3 `sched_sampling.py` -- 教师强迫退火

```python
@dataclass
class ScheduledSampler:
    schedule: SamplingSchedule   # constant / linear / cosine
    start_prob: float = 1.0
    end_prob: float = 0.0
    warmup_steps: int = 0
    total_steps: int = 10_000
```

控制状态令牌的教师强迫概率。决定每一步是使用真实 `z_{t-1}`（教师强迫）还是模型自身预测的 `z_{t-1}`（自回归）。

**三种退火策略：**

| 策略 | 行为 |
|------|------|
| `constant` | 始终返回 `start_prob` |
| `linear` | 从 `start_prob` 线性衰减到 `end_prob` |
| `cosine` | 余弦退火从 `start_prob` 到 `end_prob` |

**v1 默认配置：** `constant, prob=1.0`（始终从 `z_0` 开始，无 episode 递归）。

关键方法：
- `get_teacher_force_prob(step)`: 返回当前步的教师强迫概率
- `should_use_teacher(step)`: 随机采样决定是否使用教师强迫

---

## 6. `certvla/inference/` -- 推理与修复

### 6.1 `gap.py` -- 证书间隙计算

**逐 slot 间隙：**

```
gamma_t^j = p_t^{adv,j} * d_j(hat_g_t^j, hat_s_{t+H}^j)
           + p_t^{pre,j} * d_j(hat_s_t^j, hat_s_{t+H}^j)
```

其中 `p_adv, p_pre = softmax(role_logits)`。间隙衡量的是"如果按照模型预测的证书执行，预期的状态变化与实际是否一致"。

```python
def slot_gap(role_logits, state_readout, goal_preds, state_readout_tH) -> Dict[SlotName, Tensor(B,)]:
```

**聚合为标量：**

```
Gamma_t = [ sum_j omega_j * kappa_j * gamma_j ] / [ sum_j omega_j * kappa_j + epsilon ]
```

```python
def aggregate_certificate_gap(per_slot_gaps, role_logits, ...) -> GapResult:
```

**`GapResult` 数据类：**

```python
@dataclass
class GapResult:
    per_slot: Dict[SlotName, Tensor(B,)]   # 逐 slot 间隙
    aggregated: Tensor(B,)                 # 聚合标量间隙
    role_probs: Dict[SlotName, Tensor(B, 3)]  # softmax 后的角色概率
```

`_slot_distance` 使用可微分实现：连续/二值用绝对值差，分类用总变差距离 (TV distance)。

---

### 6.2 `repair.py` -- 短视野局部修复控制器

当聚合间隙 `Gamma_t` 超过阈值时，触发重新预测。

**`RepairConfig`：**

```python
@dataclass
class RepairConfig:
    gap_threshold: float = 0.3        # 触发修复的间隙阈值
    max_repair_steps: int = 3         # 最大重试次数
    slot_weights: Optional[Dict]      # 逐 slot 重要性权重
    use_best_of_n: bool = True        # 始终选择最低间隙的尝试
```

**`RepairController` -- 修复循环：**

```python
controller = RepairController(config, model_fn, logger)
actions, gap_result, n_repairs = controller.step(
    last_hidden_states, state_token_pos, action_start_pos, z_prev, ...
)
```

**`model_fn` 解耦设计：** `RepairController` 不依赖具体模型类，而是接收一个 `Callable[..., CertVLAOutput]`。集成时只需用 lambda 或 partial 包装 `CertVLAWrapper.forward`。

**修复流程：**

```
1. 正常前向传播 -> 计算间隙 Gamma_t
2. 若 Gamma_t > threshold -> 最多重试 max_repair_steps 次
3. 每次重试都重新前向传播，计算间隙
4. 接受间隙最小的尝试
5. 如果所有尝试都超阈值 -> 仍接受最佳尝试 + 记录警告
```

注意：这**不是**完整的重规划器，而是依赖模型随机性（dropout、温度）的局部重试。

---

### 6.3 `logging.py` -- 推理日志与调试追踪

提供三层嵌套的日志结构：

**`StepRecord` -- 单步记录：**

```python
@dataclass
class StepRecord:
    output: CertVLAOutput      # 前向传播输出
    gap: GapResult             # 间隙计算结果
    repair_attempt: int        # 第几次尝试 (0 = 初始)
    accepted: bool             # 是否被接受为最终结果
```

**`EpisodeTrace` -- 回合追踪：**

```python
@dataclass
class EpisodeTrace:
    steps: List[StepRecord]        # 被接受的步骤记录
    all_attempts: List[StepRecord]  # 所有尝试（含被拒绝的修复尝试）
    warnings: List[str]            # 警告消息
    metadata: Dict[str, Any]       # 自由格式元数据
```

提供属性: `num_steps`, `num_repairs`, `gap_history`，以及 `summary()` 方法返回统计摘要。

**`InferenceLogger` -- 日志管理器：**

```python
logger = InferenceLogger(verbose=True)
logger.begin_episode(metadata={"task": "pick_and_place"})
# ... 推理过程中 ...
logger.log_step(record)
logger.end_episode()
trace = logger.get_last_trace()
```

支持可选的 verbose 模式（输出到 Python logging），保留最近 `max_episodes` 条追踪记录。

---

## 7. 数据流总结

```
                         ┌──────────────────────────────────────────────┐
                         │           离线标注阶段 (Offline)              │
                         │                                              │
  LIBERO HDF5 demos ──→ │ 仿真器回放 ──→ LiberoOracleLabeler           │
                         │                    │                         │
                         │              SlotState 序列                  │
                         │                    │                         │
                         │     compute_goal_signature ──→ s*            │
                         │                    │                         │
                         │     mine_certificate(s_t, s_{t+H}, s*, ...)  │
                         │                    │                         │
                         │            CertificateLabel                  │
                         │                    │                         │
                         │     save_episode_labels ──→ sidecar .npz     │
                         └──────────────────────────────────────────────┘
                                              │
                                              ↓
                         ┌──────────────────────────────────────────────┐
                         │           训练阶段 (Training)                 │
                         │                                              │
  .npz + HDF5 ────────→ │  CertChunkSample (batch)                     │
                         │       │                                      │
                         │  ┌────┴────┐                                 │
                         │  │  o_t    │──→ Vision Encoder ──→ patches   │
                         │  │  l      │──→ Text Tokenizer ──→ tokens    │
                         │  └─────────┘                                 │
                         │       ↓                                      │
                         │  [patches] [z_0 嵌入] [text tokens] [action] │
                         │       ↓                                      │
                         │     LLM Forward Pass                         │
                         │       ↓                                      │
                         │  last_hidden_states (B, seq_len, 4096)       │
                         │       │                                      │
                         │  CertVLAWrapper.forward():                   │
                         │   Step 1: 提取 tilde_z_t                     │
                         │   Step 2: 门控更新 → z_t                     │
                         │   Step 3: StateReadoutHead(z_t) → hat_s_t    │
                         │   Step 4: CertificateHead(z_t) → hat_c_t    │
                         │   Step 5: 提取 actions_hidden_states         │
                         │   Step 6: CertActionHead → hat_A_t           │
                         │       │                                      │
                         │  CurriculumScheduler(step) → loss_weights    │
                         │       │                                      │
                         │  cert_total_loss():                          │
                         │   Stage 1: L_state                           │
                         │   Stage 2: + L_role + L_goal                 │
                         │   Stage 3: + L_act + L_cons + L_dep          │
                         │   Stage 4: + L_cf                            │
                         └──────────────────────────────────────────────┘
                                              │
                                              ↓
                         ┌──────────────────────────────────────────────┐
                         │           推理阶段 (Inference)                │
                         │                                              │
  o_t + l ─────────────→ │  LLM → CertVLAWrapper → CertVLAOutput       │
                         │       │                                      │
                         │  gap.py:                                     │
                         │   slot_gap() → gamma_t^j (逐 slot)          │
                         │   aggregate_certificate_gap() → Gamma_t      │
                         │       │                                      │
                         │  repair.py:                                  │
                         │   Gamma_t > threshold?                       │
                         │     是 → 重试 (最多 N 次)                     │
                         │     否 → 接受当前动作                         │
                         │       │                                      │
                         │  logging.py:                                 │
                         │   StepRecord → EpisodeTrace                  │
                         │       │                                      │
                         │  输出: actions (B, H, action_dim)            │
                         └──────────────────────────────────────────────┘
```

---

## 8. 关键设计模式

### 8.1 逐 slot (per-slot) 模式

CertVLA 中几乎所有计算都是逐 slot 进行的：

- 距离度量 `d_j(a, b)` 按域类型选择不同的距离函数
- 损失函数 `_per_slot_loss()` 按域类型选择 BCE / CE / L1
- 模型输出头（readout、certificate）每个 slot 有独立的线性层
- 间隙计算 `slot_gap()` 逐 slot 计算后再加权聚合

### 8.2 掩码 + 置信度加权: `m^j * alpha^j * loss`

这一模式贯穿所有损失函数：
- `m^j`（掩码）: 控制该 slot 在该样本中是否有有效标注（处理缺失数据和不适用的 slot）
- `alpha^j`（置信度）: 对该 slot 损失进行加权（oracle 标注 = 1.0，伪标签 < 1.0）
- 这使得系统能优雅地处理异构数据源和部分观测

### 8.3 信息瓶颈: 读出头只能通过 z_t 读信息

`StateReadoutHead` 和 `CertificateHead` 的输入**仅为 `z_t` 向量**（维度 4096），而非 LLM 的完整序列隐藏状态。这一约束至关重要：
- 强制模型将所有任务相关信息压缩到 `z_t` 中
- 防止读出头绕过状态瓶颈直接从视觉令牌获取信息
- 使得 `z_t` 的质量可以通过读出精度来量化评估

### 8.4 粗-细分支: 粗动作不接收观测

`CoarseActionBranch` 只接收 `z_t + cert_embed`，完全不接触观测 `o_t`：
- 迫使证书携带足够的任务语义信息（否则粗动作无法生成合理策略）
- 粗分支生成语义级动作（如"向前移动"、"抬起手臂"）
- `FineActionBranch` 接收 LLM 动作隐藏状态（编码了观测），做几何精修（如精确对齐位置）
- 残差结构 `A = A_coarse + lambda_res * Delta_A_fine` 确保训练稳定性

### 8.5 `model_fn` 解耦: 推理逻辑不依赖具体模型类

`RepairController` 通过接收一个 `Callable[..., CertVLAOutput]` 类型的 `model_fn` 实现与具体模型类的解耦：
- 修复循环只需要调用 `model_fn` 并检查返回的 `CertVLAOutput`
- 集成时只需用 lambda 或 `functools.partial` 包装 `CertVLAWrapper.forward`
- 同样的修复逻辑可以适配不同的模型架构或推理后端

### 8.6 依赖隔离

`certvla/model/` 下的所有文件都不导入 `prismatic`（OpenVLA 的基础框架），避免触发重量级依赖链（HuggingFace Transformers, timm 等）。连接点在 `modeling_prismatic.py` 中，而非 `certvla/` 内部。这使得 CertVLA 的核心逻辑可以独立测试。
