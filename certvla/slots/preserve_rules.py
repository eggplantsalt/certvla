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

# Type alias for slot state dict
SlotStateDict = Dict[SlotName, Union[int, float, str]]


def latch_preserve(state: SlotStateDict, advance_set: Set[SlotName]) -> Set[SlotName]:
    """Latch-preserve: result slots where completion_latch=1 and not advancing.

    Once a result slot's associated task sub-goal is completed (indicated by
    completion_latch=1), it should be preserved unless it is actively advancing
    in the current chunk.

    Per section 8.5:
      P_t^latch = { j in J_R | completion_latch = 1 and j not in A_t }
    """
    result = set()
    latch_val = state.get(SlotName.COMPLETION_LATCH, 0)
    if latch_val in (1, 1.0, True):
        for j in J_R:
            if j not in advance_set:
                result.add(j)
    return result


# === Support-preserve rule table ===
# These capture structural invariants: "to successfully advance X, you must preserve Y."
# Rules are (trigger_condition, preserved_slot) pairs.

# Rule format: if any slot in `trigger_slots` is in advance_set AND the state condition holds,
# then `preserved_slot` must be preserved.

_SUPPORT_RULES = [
    # During transport (target_goal_proximity advancing), hand must hold target
    {
        "trigger_slots": {SlotName.TARGET_GOAL_PROXIMITY},
        "preserved_slot": SlotName.HAND_OCCUPANCY,
        "state_condition": lambda s: s.get(SlotName.HAND_OCCUPANCY) == "target",
    },
    # During placement into container (containment_relation advancing), keep contact
    {
        "trigger_slots": {SlotName.CONTAINMENT_RELATION},
        "preserved_slot": SlotName.TARGET_CONTACT,
        "state_condition": lambda s: s.get(SlotName.TARGET_CONTACT) in (1, 1.0, True),
    },
    # During placement onto support (support_relation advancing), keep contact
    {
        "trigger_slots": {SlotName.SUPPORT_RELATION},
        "preserved_slot": SlotName.TARGET_CONTACT,
        "state_condition": lambda s: s.get(SlotName.TARGET_CONTACT) in (1, 1.0, True),
    },
    # When advancing containment, keep container articulated (open)
    {
        "trigger_slots": {SlotName.CONTAINMENT_RELATION},
        "preserved_slot": SlotName.ARTICULATION_PROGRESS,
        "state_condition": lambda s: s.get(SlotName.ARTICULATION_PROGRESS, 0.0) > 0.5,
    },
    # During transport, maintain target contact
    {
        "trigger_slots": {SlotName.TARGET_GOAL_PROXIMITY},
        "preserved_slot": SlotName.TARGET_CONTACT,
        "state_condition": lambda s: s.get(SlotName.TARGET_CONTACT) in (1, 1.0, True),
    },
]


def support_preserve(state: SlotStateDict, advance_set: Set[SlotName]) -> Set[SlotName]:
    """Support-preserve: enabling conditions required by current advance set.

    Uses a rule table of structural invariants. Each rule says:
    "If advancing slot X and condition Y holds, then preserve slot Z."
    """
    result = set()
    for rule in _SUPPORT_RULES:
        if rule["trigger_slots"] & advance_set:  # any trigger in advance set
            if rule["state_condition"](state):
                preserved = rule["preserved_slot"]
                # A slot cannot be simultaneously advance and preserve
                if preserved not in advance_set:
                    result.add(preserved)
    return result


def compute_preserve_set(state: SlotStateDict, advance_set: Set[SlotName]) -> Set[SlotName]:
    """Compute the full preserve set: latch-preserve | support-preserve.

    The result never overlaps with advance_set.
    """
    preserve = latch_preserve(state, advance_set) | support_preserve(state, advance_set)
    # Safety: remove any slot that is in advance (advance takes priority)
    preserve -= advance_set
    return preserve
