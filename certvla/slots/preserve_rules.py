# =============================================================================
# CertVLA 结构性保持规则 (preserve_rules.py)
# =============================================================================
#
# 【模块在整体流水线中的位置】
#   在 CertVLA 的 certificate 机制中，每个 action chunk 对应三种 slot 角色：
#     - advance (推进): 该 slot 在此 chunk 中应该发生期望的变化
#     - preserve (保持): 该 slot 在此 chunk 中不应发生变化（维持当前值）
#     - ignore (忽略):   该 slot 与当前 chunk 无关，不约束
#
#   其中，advance 集合由数据挖掘得到（"数据挖 advance"）；
#   而 preserve 集合由本模块的 **结构先验规则** 推导（"结构先验定 preserve"）。
#   这是 CertVLA 的核心设计决策之一，见论文 8.5 节。
#
# 【为什么 preserve 用结构先验而非数据挖掘？】
#   如果用统计方法从数据中学习哪些 slot 应该被保持，容易出现两个问题：
#     1. 高频共现不等于因果关系——可能学到虚假的相关性
#     2. 遗漏低频但关键的不变量——如"搬运时必须保持抓握"这类物理约束
#   结构先验规则直接编码物理世界的不变量，虽然需要人工设计，但更可靠。
#
# 【两类保持规则】
#   1. Latch-preserve (锁存保持):
#      当 completion_latch = 1（任务已完成）时，所有不在 advance 集合中的结果 slot
#      都应被保持。直觉：已经完成的子目标不应被后续动作"撤销"。
#      公式: P_t^latch = { j in J_R | completion_latch = 1 and j not in A_t }
#
#   2. Support-preserve (支撑保持):
#      某些使能 slot 是当前 advance slot 的 "结构性前提"，必须在推进过程中保持。
#      例如：搬运物体（推进 target_goal_proximity）时必须保持手爪抓握（hand_occupancy）。
#      这些规则以规则表 (rule table) 的形式硬编码。
#
# 【常见陷阱】
#   1. advance 和 preserve 互斥：同一个 slot 不能同时被推进和保持。
#      如果规则推导出的 preserve 包含 advance 中的 slot，advance 优先。
#   2. preserve 规则中的状态条件 (state_condition) 是 lambda，
#      只有当前状态满足条件时才触发保持。例如，只有 hand_occupancy == "target" 时
#      才需要在搬运过程中保持它；如果手是空的，这条规则不触发。
#   3. 返回值是 Set[SlotName]，不是有序的——这是 OK 的，因为 preserve 集合
#      只用于成员检测，不需要确定顺序。
# =============================================================================

"""
Preserve rule framework.

Preserve labels are NOT mined from data statistics. They are defined by structural priors:
- Latch-preserve: completed result slots not in current advance set
- Support-preserve: enabling conditions required by the current advance set

Per context doc section 8.5: "数据挖 advance, 结构先验定 preserve"
"""

from typing import Dict, Set, Union

from certvla.slots.schema import SlotName
from certvla.slots.role_sets import J_E, J_R

# SlotStateDict: slot 状态字典的类型别名
# 键为 SlotName, 值为该 slot 的当前值 (int / float / str, 取决于域类型)
# Type alias for slot state dict
SlotStateDict = Dict[SlotName, Union[int, float, str]]


# ---------------------------------------------------------------------------
# 规则一: Latch-preserve (锁存保持)
# ---------------------------------------------------------------------------
# 当 completion_latch = 1 时，表示某个子任务已完成，
# 此时该子任务对应的结果 slot (J_R) 应被保持，以免后续动作"撤销"成果。
# 但如果某个结果 slot 恰好在当前 advance 集合中（即当前 chunk 要推进它），
# 则不纳入 preserve（因为 advance 优先级高于 preserve）。
def latch_preserve(state: SlotStateDict, advance_set: Set[SlotName]) -> Set[SlotName]:
    """Latch-preserve: result slots where completion_latch=1 and not advancing.

    Once a result slot's associated task sub-goal is completed (indicated by
    completion_latch=1), it should be preserved unless it is actively advancing
    in the current chunk.

    Per section 8.5:
      P_t^latch = { j in J_R | completion_latch = 1 and j not in A_t }

    【参数说明】
      state:       当前时刻的 slot 状态字典 {SlotName: value}
      advance_set: 当前 chunk 的 advance 集合（从数据挖掘得到的应推进的 slot）

    【返回值】
      需要 latch-preserve 的 slot 集合 (J_R 的子集)
    """
    result = set()
    # 读取 completion_latch 的当前值, 缺失时默认为 0 (未完成)
    latch_val = state.get(SlotName.COMPLETION_LATCH, 0)
    if latch_val in (1, 1.0, True):
        # 锁存已触发: 遍历所有结果 slot, 将不在 advance 集合中的加入 preserve
        for j in J_R:
            if j not in advance_set:
                result.add(j)
    return result


# ---------------------------------------------------------------------------
# 规则二: Support-preserve (支撑保持) —— 规则表
# ---------------------------------------------------------------------------
# 支撑保持规则编码了物理世界的结构性不变量：
# "要成功推进 X slot, 你必须保持 Y slot 不变。"
#
# 规则格式 (字典):
#   trigger_slots:    集合, 当 advance_set 与此集合有交集时触发
#   preserved_slot:   单个 SlotName, 触发后应被保持的 slot
#   state_condition:  lambda, 只有当前状态满足此条件时才实际触发
#
# 【设计说明】
#   - 规则表是手工编写的，反映操作任务中的常见物理约束。
#   - 每条规则可以理解为一个 "结构性因果假设"：
#     如果推进某个结果 slot，那么其执行所依赖的某个使能 slot 必须保持稳定。
#   - 规则中的 state_condition 确保只在 "已经满足前提" 时才触发保持。
#     例如，手里没有目标物时不需要 "保持抓握"。

# === Support-preserve rule table ===
# These capture structural invariants: "to successfully advance X, you must preserve Y."
# Rules are (trigger_condition, preserved_slot) pairs.

# Rule format: if any slot in `trigger_slots` is in advance_set AND the state condition holds,
# then `preserved_slot` must be preserved.

_SUPPORT_RULES = [
    # 规则 1: 搬运过程中保持抓握
    # 当推进 target_goal_proximity（搬运物体到目标位置）时,
    # 如果手爪正抓着目标物, 则 hand_occupancy 必须保持为 "target"
    # During transport (target_goal_proximity advancing), hand must hold target
    {
        "trigger_slots": {SlotName.TARGET_GOAL_PROXIMITY},
        "preserved_slot": SlotName.HAND_OCCUPANCY,
        "state_condition": lambda s: s.get(SlotName.HAND_OCCUPANCY) == "target",
    },
    # 规则 2: 放入容器时保持接触
    # 当推进 containment_relation（将物体放入容器）时,
    # 如果已与目标接触, 则 target_contact 必须保持为 1
    # During placement into container (containment_relation advancing), keep contact
    {
        "trigger_slots": {SlotName.CONTAINMENT_RELATION},
        "preserved_slot": SlotName.TARGET_CONTACT,
        "state_condition": lambda s: s.get(SlotName.TARGET_CONTACT) in (1, 1.0, True),
    },
    # 规则 3: 放到支撑面上时保持接触
    # 当推进 support_relation（将物体放到支撑面上）时,
    # 如果已与目标接触, 则 target_contact 必须保持为 1
    # During placement onto support (support_relation advancing), keep contact
    {
        "trigger_slots": {SlotName.SUPPORT_RELATION},
        "preserved_slot": SlotName.TARGET_CONTACT,
        "state_condition": lambda s: s.get(SlotName.TARGET_CONTACT) in (1, 1.0, True),
    },
    # 规则 4: 放入容器时保持容器打开状态
    # 当推进 containment_relation 时,
    # 如果铰接体已打开超过 50%, 则 articulation_progress 必须保持
    # (直觉: 你不能一边把东西放进抽屉, 一边把抽屉关上)
    # When advancing containment, keep container articulated (open)
    {
        "trigger_slots": {SlotName.CONTAINMENT_RELATION},
        "preserved_slot": SlotName.ARTICULATION_PROGRESS,
        "state_condition": lambda s: s.get(SlotName.ARTICULATION_PROGRESS, 0.0) > 0.5,
    },
    # 规则 5: 搬运过程中保持接触
    # 当推进 target_goal_proximity（搬运物体）时,
    # 如果已与目标接触, 则 target_contact 必须保持为 1
    # (与规则 1 互补: 规则 1 保持 hand_occupancy, 本规则保持 target_contact)
    # During transport, maintain target contact
    {
        "trigger_slots": {SlotName.TARGET_GOAL_PROXIMITY},
        "preserved_slot": SlotName.TARGET_CONTACT,
        "state_condition": lambda s: s.get(SlotName.TARGET_CONTACT) in (1, 1.0, True),
    },
]


# ---------------------------------------------------------------------------
# support_preserve: 遍历规则表, 推导 support-preserve 集合
# ---------------------------------------------------------------------------
def support_preserve(state: SlotStateDict, advance_set: Set[SlotName]) -> Set[SlotName]:
    """Support-preserve: enabling conditions required by current advance set.

    Uses a rule table of structural invariants. Each rule says:
    "If advancing slot X and condition Y holds, then preserve slot Z."

    【算法】
      1. 遍历 _SUPPORT_RULES 中的每条规则
      2. 如果该规则的 trigger_slots 与 advance_set 有交集（即触发条件满足）
      3. 且当前状态满足 state_condition
      4. 则将 preserved_slot 加入结果集
      5. 但如果 preserved_slot 已经在 advance_set 中，则跳过（advance 优先）

    【参数说明】
      state:       当前时刻的 slot 状态字典
      advance_set: 当前 chunk 的 advance 集合

    【返回值】
      需要 support-preserve 的 slot 集合
    """
    result = set()
    for rule in _SUPPORT_RULES:
        # 检查触发条件: advance_set 中是否有任何一个 trigger_slot
        if rule["trigger_slots"] & advance_set:  # any trigger in advance set
            # 检查状态条件: 当前状态是否满足 lambda 表达式
            if rule["state_condition"](state):
                preserved = rule["preserved_slot"]
                # 互斥约束: 同一 slot 不能既 advance 又 preserve
                # A slot cannot be simultaneously advance and preserve
                if preserved not in advance_set:
                    result.add(preserved)
    return result


# ---------------------------------------------------------------------------
# compute_preserve_set: 合并两类保持规则, 得到最终的 preserve 集合
# ---------------------------------------------------------------------------
def compute_preserve_set(state: SlotStateDict, advance_set: Set[SlotName]) -> Set[SlotName]:
    """Compute the full preserve set: latch-preserve | support-preserve.

    The result never overlaps with advance_set.

    【最终公式】
      P_t = P_t^latch ∪ P_t^support - A_t

    即: 合并两类保持规则的结果后，再移除 advance 集合中的 slot，
    确保 advance 和 preserve 严格互斥。

    剩余既不在 advance 也不在 preserve 中的 slot 自动归为 ignore。

    【参数说明】
      state:       当前时刻的 slot 状态字典
      advance_set: 当前 chunk 的 advance 集合 (由数据挖掘得到)

    【返回值】
      最终的 preserve 集合, 保证与 advance_set 不重叠
    """
    preserve = latch_preserve(state, advance_set) | support_preserve(state, advance_set)
    # 安全兜底: 移除任何意外落入 advance 集合的 slot (advance 优先级最高)
    # Safety: remove any slot that is in advance (advance takes priority)
    preserve -= advance_set
    return preserve
