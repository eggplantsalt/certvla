# =============================================================================
# CertVLA v1 Slot 词汇表定义 (schema.py)
# =============================================================================
#
# 【模块在整体流水线中的位置】
#   CertVLA 将 VLA 的内部状态定义为"足以认证下一个动作块结构性后果的最小状态"。
#   本文件是该最小状态的 **唯一权威定义**：它声明了 v1 版本所有 10 个 slot 的名称、
#   值域 (domain) 和族 (family)，构成"slot 词汇表"。
#   下游所有模块——距离度量 (metrics.py)、角色集 (role_sets.py)、
#   保持规则 (preserve_rules.py) 以及训练 / 推理流程——都从此文件导入定义。
#
# 【设计决策与常见陷阱】
#   1. Slot 是 **任务相对** 的（task-relative），不是场景图（scene graph）。
#      例如 "ee_target_proximity" 指末端执行器到 *当前任务目标* 的距离，
#      而非到场景中某个绝对坐标的距离。这意味着同一物理状态在不同任务指令下
#      可映射为不同的 slot 值。
#   2. 所有连续量都被归一化到 [0, 1]，这样不同 slot 之间的距离可以直接比较，
#      也方便用 L1 距离衡量变化量。
#   3. 分类型 slot 使用字符串标签而非整数编码，保持人类可读性；
#      在转换为张量时才做 one-hot 编码（见 metrics.py）。
#   4. frozen=True 的 dataclass 保证元数据一旦注册就不可变，
#      防止运行时意外修改导致下游不一致。
#   5. 本文件定义的 10 个 slot 是 v1 "冻结"版本，不能随意增删——
#      如果需要扩展，应定义 v2 schema。
#
# 【10 个 v1 slot 速览】
#   ┌─────────────────────────┬────────────┬──────────┐
#   │ Slot 名称                │ 值域       │ 族       │
#   ├─────────────────────────┼────────────┼──────────┤
#   │ ee_target_proximity     │ continuous │ J_E      │
#   │ hand_occupancy          │ categorical│ J_E      │
#   │ target_contact          │ binary     │ J_E      │
#   │ articulation_progress   │ continuous │ J_E      │
#   │ orientation_alignment   │ continuous │ J_E      │
#   │ target_goal_proximity   │ continuous │ J_R      │
#   │ support_relation        │ categorical│ J_R      │
#   │ containment_relation    │ categorical│ J_R      │
#   │ completion_latch        │ binary     │ J_R      │
#   │ task_visible_confidence │ confidence │ J_C      │
#   └─────────────────────────┴────────────┴──────────┘
#   J_E (5 个): 使能/过渡 slot —— 描述动作执行的前提条件
#   J_R (4 个): 结果/锁存 slot —— 描述动作执行的目标后果
#   J_C (1 个): 置信度 slot    —— 描述观测质量
# =============================================================================

"""
CertVLA v1 slot vocabulary definition.

Frozen v1 schema: 10 slots with fixed names, domains, families.
This file is the single source of truth for the slot vocabulary.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# SlotName: 所有 v1 slot 的枚举名称
# ---------------------------------------------------------------------------
# 注意：枚举的声明顺序决定了下游 flat tensor 的拼接顺序
#       (见 metrics.py 中的 slot_state_to_flat_tensor)。
#       如果更改枚举顺序，已保存的张量就无法兼容，因此千万不要重排。
class SlotName(str, Enum):
    """Names of all slots in the v1 vocabulary."""

    # --- J_E (使能/过渡 slot) ---
    EE_TARGET_PROXIMITY = "ee_target_proximity"       # 末端执行器到目标物体的归一化距离 [0,1]
    HAND_OCCUPANCY = "hand_occupancy"                 # 手爪抓取状态: empty / target / other
    TARGET_CONTACT = "target_contact"                 # 是否与目标物体接触: 0 或 1
    # --- J_R (结果/锁存 slot) ---
    TARGET_GOAL_PROXIMITY = "target_goal_proximity"   # 目标物体到目标位置的归一化距离 [0,1]
    SUPPORT_RELATION = "support_relation"             # 支撑关系: none / on_goal / on_other
    CONTAINMENT_RELATION = "containment_relation"     # 包含关系: none / in_goal / in_other
    # --- J_E (使能/过渡 slot, 续) ---
    ARTICULATION_PROGRESS = "articulation_progress"   # 铰接体（如抽屉、门）的打开进度 [0,1]
    ORIENTATION_ALIGNMENT = "orientation_alignment"   # 目标物体朝向与目标朝向的对齐程度 [0,1]
    # --- J_R (结果/锁存 slot, 续) ---
    COMPLETION_LATCH = "completion_latch"             # 任务完成锁存: 0 或 1, 一旦置 1 不应回退
    # --- J_C (置信度 slot) ---
    TASK_VISIBLE_CONFIDENCE = "task_visible_confidence"  # 任务相关物体是否可见的置信度 [0,1]


# ---------------------------------------------------------------------------
# SlotDomain: 值域类型
# ---------------------------------------------------------------------------
# 每种值域类型决定了：
#   (1) 合法值的集合
#   (2) 距离度量 d_j(a, b) 的计算方式 (见 metrics.py)
#   (3) 张量编码方式：binary/continuous/confidence -> 1 维标量, categorical -> one-hot 向量
# 注意: CONFIDENCE 在数值上与 CONTINUOUS 完全相同（都是 [0,1] 标量），
#        但语义不同——CONFIDENCE 衡量的是 *观测质量*，不参与 certificate 判定。
class SlotDomain(str, Enum):
    """Value domain types for slots."""

    BINARY = "binary"                # {0, 1} —— 二值型
    CATEGORICAL = "categorical"      # finite set of string labels —— 分类型（有限字符串标签集）
    CONTINUOUS = "continuous"         # [0, 1] —— 连续型（归一化）
    CONFIDENCE = "confidence"        # [0, 1], semantically distinct from CONTINUOUS —— 置信度型


# ---------------------------------------------------------------------------
# SlotFamily: Slot 族归属
# ---------------------------------------------------------------------------
# CertVLA 将 10 个 slot 分为三个族，分别对应 certificate 中不同的角色：
#   J_E (使能/过渡): 描述 "动作能否成功执行" 的前提条件
#   J_R (结果/锁存): 描述 "动作执行后世界状态的期望变化"
#   J_C (置信度):    描述 "当前观测是否足够可靠"
# J_CERT = J_E | J_R 是参与 certificate 判定的 slot 集合，
# J_C 不参与 certificate 但用于过滤低置信度样本。
class SlotFamily(str, Enum):
    """Slot family membership (J_E, J_R, J_C)."""

    ENABLING = "enabling"      # J_E: transit / enabling slots —— 使能/过渡 slot
    RESULT = "result"          # J_R: result / latch slots —— 结果/锁存 slot
    CONFIDENCE = "confidence"  # J_C: observability / confidence slots —— 置信度 slot


# ---------------------------------------------------------------------------
# SlotMeta: 单个 slot 的元数据
# ---------------------------------------------------------------------------
# frozen=True 保证注册后不可变。
# 关键字段：
#   - name:        SlotName 枚举值
#   - domain:      值域类型，决定距离函数和张量编码方式
#   - family:      J_E / J_R / J_C 族归属
#   - categories:  仅 CATEGORICAL slot 需要，定义合法标签元组
#   - valid_range: 仅 CONTINUOUS / CONFIDENCE 有意义，默认 (0.0, 1.0)
@dataclass(frozen=True)
class SlotMeta:
    """Metadata for a single slot in the vocabulary."""

    name: SlotName
    domain: SlotDomain
    family: SlotFamily
    categories: Optional[Tuple[str, ...]] = None  # 仅 CATEGORICAL slot 需要此字段
    valid_range: Tuple[float, float] = (0.0, 1.0)  # 仅 CONTINUOUS/CONFIDENCE 有意义

    def __post_init__(self):
        # 构造后校验：分类型 slot 必须声明 categories；二值型 slot 强制归一化 valid_range
        if self.domain == SlotDomain.CATEGORICAL and not self.categories:
            raise ValueError(f"Categorical slot {self.name} must define categories")
        if self.domain == SlotDomain.BINARY:
            # 二值型的范围实际是 {0, 1}，这里存 (0, 1) 是为了与连续型保持接口统一
            # 注意：frozen dataclass 不能直接赋值，需用 object.__setattr__
            object.__setattr__(self, "valid_range", (0.0, 1.0))

    def validate_value(self, value: Union[int, float, str]) -> bool:
        """Check whether a value is in this slot's valid domain."""
        # 根据值域类型分派验证逻辑
        if self.domain == SlotDomain.BINARY:
            # 接受 int 0/1, float 0.0/1.0, bool True/False
            return value in (0, 1, 0.0, 1.0, True, False)
        elif self.domain == SlotDomain.CATEGORICAL:
            # 必须是声明过的合法标签之一
            return value in self.categories
        elif self.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
            # 必须为数值且在 [lo, hi] 范围内
            return isinstance(value, (int, float)) and self.valid_range[0] <= float(value) <= self.valid_range[1]
        return False

    @property
    def num_categories(self) -> int:
        """Number of categories for categorical slots, 2 for binary, 1 for continuous.

        该属性决定了此 slot 在 flat tensor 中占据的维度数：
        - CATEGORICAL: len(categories) 维 (one-hot 编码)
        - BINARY:      返回 2，但实际编码仍为 1 维标量 (注意区分！)
        - CONTINUOUS/CONFIDENCE: 1 维标量
        """
        if self.domain == SlotDomain.CATEGORICAL:
            return len(self.categories)
        elif self.domain == SlotDomain.BINARY:
            return 2
        else:
            return 1  # continuous / confidence: single scalar


# ===========================================================================
# 冻结的 v1 Slot 注册表
# ===========================================================================
# SLOT_REGISTRY 是从 SlotName -> SlotMeta 的字典，包含全部 10 个 slot。
# 这个字典是整个 CertVLA 的"真理之源"：
#   - role_sets.py 通过遍历此字典来构建 J_E, J_R, J_C 集合
#   - metrics.py 通过此字典查找域类型以决定距离/编码方式
#   - preserve_rules.py 间接通过 role_sets.py 使用此字典
#
# 【各 slot 语义详解】
#   ee_target_proximity:    末端执行器 → 目标物体的归一化距离; 0=接触, 1=最远
#   hand_occupancy:         手爪中物体类型; "empty"=空, "target"=抓着目标, "other"=抓着其他
#   target_contact:         末端执行器是否接触目标物体; 二值
#   target_goal_proximity:  目标物体 → 目标位置的归一化距离; 0=到达, 1=最远
#   support_relation:       目标物体的支撑关系; "none", "on_goal"(在目标上), "on_other"
#   containment_relation:   目标物体的包含关系; "none", "in_goal"(在目标容器中), "in_other"
#   articulation_progress:  铰接体（门/抽屉等）打开程度; 0=关闭, 1=完全打开
#   orientation_alignment:  目标物体朝向与期望朝向的对齐度; 0=完全错位, 1=完全对齐
#   completion_latch:       任务完成锁存位; 0=未完成, 1=已完成 (一旦置 1 语义上不应回退)
#   task_visible_confidence:模型对"任务相关物体可见"的置信度; 低值意味着观测不可靠

# === Frozen v1 slot registry ===

SLOT_REGISTRY: Dict[SlotName, SlotMeta] = {
    # ---- J_E: 使能/过渡 slot ----
    SlotName.EE_TARGET_PROXIMITY: SlotMeta(
        name=SlotName.EE_TARGET_PROXIMITY,
        domain=SlotDomain.CONTINUOUS,       # 归一化连续值 [0, 1]
        family=SlotFamily.ENABLING,
    ),
    SlotName.HAND_OCCUPANCY: SlotMeta(
        name=SlotName.HAND_OCCUPANCY,
        domain=SlotDomain.CATEGORICAL,      # 3 个类别 -> one-hot 维度 = 3
        family=SlotFamily.ENABLING,
        categories=("empty", "target", "other"),
    ),
    SlotName.TARGET_CONTACT: SlotMeta(
        name=SlotName.TARGET_CONTACT,
        domain=SlotDomain.BINARY,           # {0, 1}
        family=SlotFamily.ENABLING,
    ),
    # ---- J_R: 结果/锁存 slot ----
    SlotName.TARGET_GOAL_PROXIMITY: SlotMeta(
        name=SlotName.TARGET_GOAL_PROXIMITY,
        domain=SlotDomain.CONTINUOUS,       # 归一化连续值 [0, 1]
        family=SlotFamily.RESULT,
    ),
    SlotName.SUPPORT_RELATION: SlotMeta(
        name=SlotName.SUPPORT_RELATION,
        domain=SlotDomain.CATEGORICAL,      # 3 个类别 -> one-hot 维度 = 3
        family=SlotFamily.RESULT,
        categories=("none", "on_goal", "on_other"),
    ),
    SlotName.CONTAINMENT_RELATION: SlotMeta(
        name=SlotName.CONTAINMENT_RELATION,
        domain=SlotDomain.CATEGORICAL,      # 3 个类别 -> one-hot 维度 = 3
        family=SlotFamily.RESULT,
        categories=("none", "in_goal", "in_other"),
    ),
    # ---- J_E: 使能/过渡 slot (续) ----
    SlotName.ARTICULATION_PROGRESS: SlotMeta(
        name=SlotName.ARTICULATION_PROGRESS,
        domain=SlotDomain.CONTINUOUS,       # 归一化连续值 [0, 1]
        family=SlotFamily.ENABLING,
    ),
    SlotName.ORIENTATION_ALIGNMENT: SlotMeta(
        name=SlotName.ORIENTATION_ALIGNMENT,
        domain=SlotDomain.CONTINUOUS,       # 归一化连续值 [0, 1]
        family=SlotFamily.ENABLING,
    ),
    # ---- J_R: 结果/锁存 slot (续) ----
    SlotName.COMPLETION_LATCH: SlotMeta(
        name=SlotName.COMPLETION_LATCH,
        domain=SlotDomain.BINARY,           # {0, 1} —— 锁存特性: 一旦 =1 不应回退
        family=SlotFamily.RESULT,
    ),
    # ---- J_C: 置信度 slot ----
    SlotName.TASK_VISIBLE_CONFIDENCE: SlotMeta(
        name=SlotName.TASK_VISIBLE_CONFIDENCE,
        domain=SlotDomain.CONFIDENCE,       # [0, 1] 语义为置信度, 不参与 certificate
        family=SlotFamily.CONFIDENCE,
    ),
}

# v1 词汇表大小断言：确保恰好 10 个 slot, 防止意外增删
SLOT_VOCAB_SIZE: int = len(SLOT_REGISTRY)
assert SLOT_VOCAB_SIZE == 10, f"v1 vocabulary must have exactly 10 slots, got {SLOT_VOCAB_SIZE}"


def get_slot_meta(name: SlotName) -> SlotMeta:
    """Look up metadata for a slot by name.

    这是下游模块获取 slot 元数据的统一入口。
    如果传入的 name 不在注册表中会抛出 KeyError，
    这在开发阶段能帮助快速发现拼写错误或版本不匹配问题。
    """
    return SLOT_REGISTRY[name]
