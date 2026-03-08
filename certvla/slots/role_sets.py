# =============================================================================
# CertVLA Slot 族 (角色集) 定义 (role_sets.py)
# =============================================================================
#
# 【模块在整体流水线中的位置】
#   CertVLA 的 certificate 机制需要区分不同 slot 在 "认证下一个动作块后果" 时扮演的角色。
#   本模块基于 schema.py 的注册表，自动构建四个关键的 slot 集合：
#
#     J_E  —— 使能/过渡 slot (Enabling)
#              描述动作执行的 *前提条件*，如"手爪已抓住目标"、"末端已靠近目标"等。
#              这些 slot 在执行 action chunk 的 *过程中* 会发生变化。
#
#     J_R  —— 结果/锁存 slot (Result)
#              描述动作执行的 *目标后果*，如"目标已到达指定位置"、"目标已放入容器"等。
#              这些 slot 在 action chunk *完成后* 应达到期望值，且具有"锁存"特性：
#              一旦达到期望值就不应回退。
#
#     J_C  —— 置信度 slot (Confidence)
#              描述观测的 *可靠性*，目前只有 task_visible_confidence 一个。
#              不参与 certificate 判定，但可用于过滤低质量数据。
#
#     J_CERT = J_E | J_R —— Certificate 参与 slot
#              即所有参与 certificate 判定的 slot 的并集。
#              在 certificate 验证时，对 J_CERT 中每个 slot 检查其角色
#              (advance / preserve / ignore) 并计算是否满足约束。
#
# 【设计决策】
#   - 使用 frozenset 而非普通 set，保证集合不可变，防止运行时意外修改。
#   - 集合由 SLOT_REGISTRY 的 family 字段自动推导，不硬编码成员，
#     这样增删 slot 时只需改 schema.py，此文件自动同步（前提是族分类不变）。
#   - 文件末尾的 assert 语句是 v1 的完整性校验，确保 slot 分族结果与设计文档一致。
#
# 【与 preserve_rules.py 的关系】
#   preserve_rules.py 使用本模块的 J_E、J_R 来判定哪些 slot 可能被 latch-preserve
#   或 support-preserve。certificate 角色分配的逻辑顺序是：
#     1. 从数据中挖掘 advance 集合 A_t (哪些 slot 在当前 chunk 中应该变化)
#     2. 用结构先验规则推导 preserve 集合 P_t (哪些 slot 不应变化)
#     3. 剩余的 slot 标记为 ignore
# =============================================================================

"""
Slot family (role set) definitions.

J_E: enabling / transit slots
J_R: result / latch slots
J_C: confidence / observability slots
J_CERT = J_E | J_R (slots that participate in certificates)
"""

from typing import FrozenSet

from certvla.slots.schema import SlotFamily, SlotName, SLOT_REGISTRY


# === 族集合 (Family Sets) ===

# J_E: 使能/过渡 slot 集合 (5 个)
# 包含: ee_target_proximity, hand_occupancy, target_contact,
#        articulation_progress, orientation_alignment
# 这些 slot 描述执行动作的 "前提条件" 和 "过渡中间状态"
J_E: FrozenSet[SlotName] = frozenset(
    name for name, meta in SLOT_REGISTRY.items() if meta.family == SlotFamily.ENABLING
)

# J_R: 结果/锁存 slot 集合 (4 个)
# 包含: target_goal_proximity, support_relation, containment_relation, completion_latch
# 这些 slot 描述动作的 "目标后果"，具有锁存语义
J_R: FrozenSet[SlotName] = frozenset(
    name for name, meta in SLOT_REGISTRY.items() if meta.family == SlotFamily.RESULT
)

# J_C: 置信度 slot 集合 (1 个)
# 包含: task_visible_confidence
# 不参与 certificate 判定，仅用于观测质量评估
J_C: FrozenSet[SlotName] = frozenset(
    name for name, meta in SLOT_REGISTRY.items() if meta.family == SlotFamily.CONFIDENCE
)

# J_CERT: Certificate 参与 slot 的并集 = J_E ∪ J_R (9 个)
# certificate 验证时只检查这 9 个 slot，置信度 slot 被排除在外
# Certificate-participating slots
J_CERT: FrozenSet[SlotName] = J_E | J_R


def get_family(slot_name: SlotName) -> SlotFamily:
    """Return the family of a slot.

    给定 slot 名称，返回其所属的族 (J_E / J_R / J_C)。
    可用于在 certificate 验证循环中快速判断 slot 角色。
    """
    return SLOT_REGISTRY[slot_name].family


# ===========================================================================
# v1 完整性校验
# ===========================================================================
# 以下 assert 确保族集合的成员与设计文档完全一致。
# 如果 schema.py 中的定义被意外修改，这些断言会在 import 时立即失败，
# 帮助开发者快速定位问题。

# --- J_E 应有且仅有 5 个使能/过渡 slot ---
assert SlotName.EE_TARGET_PROXIMITY in J_E
assert SlotName.HAND_OCCUPANCY in J_E
assert SlotName.TARGET_CONTACT in J_E
assert SlotName.ARTICULATION_PROGRESS in J_E
assert SlotName.ORIENTATION_ALIGNMENT in J_E
assert len(J_E) == 5, f"Expected 5 enabling slots, got {len(J_E)}"

# --- J_R 应有且仅有 4 个结果/锁存 slot ---
assert SlotName.TARGET_GOAL_PROXIMITY in J_R
assert SlotName.SUPPORT_RELATION in J_R
assert SlotName.CONTAINMENT_RELATION in J_R
assert SlotName.COMPLETION_LATCH in J_R
assert len(J_R) == 4, f"Expected 4 result slots, got {len(J_R)}"

# --- J_C 应有且仅有 1 个置信度 slot ---
assert SlotName.TASK_VISIBLE_CONFIDENCE in J_C
assert len(J_C) == 1, f"Expected 1 confidence slot, got {len(J_C)}"

# --- J_CERT = J_E ∪ J_R = 5 + 4 = 9 ---
assert len(J_CERT) == 9, f"Expected 9 cert slots (5+4), got {len(J_CERT)}"
