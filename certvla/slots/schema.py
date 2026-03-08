"""
CertVLA v1 slot vocabulary definition.

Frozen v1 schema: 10 slots with fixed names, domains, families.
This file is the single source of truth for the slot vocabulary.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


class SlotName(str, Enum):
    """Names of all slots in the v1 vocabulary."""

    EE_TARGET_PROXIMITY = "ee_target_proximity"
    HAND_OCCUPANCY = "hand_occupancy"
    TARGET_CONTACT = "target_contact"
    TARGET_GOAL_PROXIMITY = "target_goal_proximity"
    SUPPORT_RELATION = "support_relation"
    CONTAINMENT_RELATION = "containment_relation"
    ARTICULATION_PROGRESS = "articulation_progress"
    ORIENTATION_ALIGNMENT = "orientation_alignment"
    COMPLETION_LATCH = "completion_latch"
    TASK_VISIBLE_CONFIDENCE = "task_visible_confidence"


class SlotDomain(str, Enum):
    """Value domain types for slots."""

    BINARY = "binary"                # {0, 1}
    CATEGORICAL = "categorical"      # finite set of string labels
    CONTINUOUS = "continuous"         # [0, 1]
    CONFIDENCE = "confidence"        # [0, 1], semantically distinct from CONTINUOUS


class SlotFamily(str, Enum):
    """Slot family membership (J_E, J_R, J_C)."""

    ENABLING = "enabling"      # J_E: transit / enabling slots
    RESULT = "result"          # J_R: result / latch slots
    CONFIDENCE = "confidence"  # J_C: observability / confidence slots


@dataclass(frozen=True)
class SlotMeta:
    """Metadata for a single slot in the vocabulary."""

    name: SlotName
    domain: SlotDomain
    family: SlotFamily
    categories: Optional[Tuple[str, ...]] = None  # only for CATEGORICAL domain
    valid_range: Tuple[float, float] = (0.0, 1.0)  # only meaningful for CONTINUOUS/CONFIDENCE

    def __post_init__(self):
        if self.domain == SlotDomain.CATEGORICAL and not self.categories:
            raise ValueError(f"Categorical slot {self.name} must define categories")
        if self.domain == SlotDomain.BINARY:
            # Binary range is implicitly {0, 1}; store (0, 1) for consistency
            object.__setattr__(self, "valid_range", (0.0, 1.0))

    def validate_value(self, value: Union[int, float, str]) -> bool:
        """Check whether a value is in this slot's valid domain."""
        if self.domain == SlotDomain.BINARY:
            return value in (0, 1, 0.0, 1.0, True, False)
        elif self.domain == SlotDomain.CATEGORICAL:
            return value in self.categories
        elif self.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
            return isinstance(value, (int, float)) and self.valid_range[0] <= float(value) <= self.valid_range[1]
        return False

    @property
    def num_categories(self) -> int:
        """Number of categories for categorical slots, 2 for binary, 1 for continuous."""
        if self.domain == SlotDomain.CATEGORICAL:
            return len(self.categories)
        elif self.domain == SlotDomain.BINARY:
            return 2
        else:
            return 1  # continuous / confidence: single scalar


# === Frozen v1 slot registry ===

SLOT_REGISTRY: Dict[SlotName, SlotMeta] = {
    SlotName.EE_TARGET_PROXIMITY: SlotMeta(
        name=SlotName.EE_TARGET_PROXIMITY,
        domain=SlotDomain.CONTINUOUS,
        family=SlotFamily.ENABLING,
    ),
    SlotName.HAND_OCCUPANCY: SlotMeta(
        name=SlotName.HAND_OCCUPANCY,
        domain=SlotDomain.CATEGORICAL,
        family=SlotFamily.ENABLING,
        categories=("empty", "target", "other"),
    ),
    SlotName.TARGET_CONTACT: SlotMeta(
        name=SlotName.TARGET_CONTACT,
        domain=SlotDomain.BINARY,
        family=SlotFamily.ENABLING,
    ),
    SlotName.TARGET_GOAL_PROXIMITY: SlotMeta(
        name=SlotName.TARGET_GOAL_PROXIMITY,
        domain=SlotDomain.CONTINUOUS,
        family=SlotFamily.RESULT,
    ),
    SlotName.SUPPORT_RELATION: SlotMeta(
        name=SlotName.SUPPORT_RELATION,
        domain=SlotDomain.CATEGORICAL,
        family=SlotFamily.RESULT,
        categories=("none", "on_goal", "on_other"),
    ),
    SlotName.CONTAINMENT_RELATION: SlotMeta(
        name=SlotName.CONTAINMENT_RELATION,
        domain=SlotDomain.CATEGORICAL,
        family=SlotFamily.RESULT,
        categories=("none", "in_goal", "in_other"),
    ),
    SlotName.ARTICULATION_PROGRESS: SlotMeta(
        name=SlotName.ARTICULATION_PROGRESS,
        domain=SlotDomain.CONTINUOUS,
        family=SlotFamily.ENABLING,
    ),
    SlotName.ORIENTATION_ALIGNMENT: SlotMeta(
        name=SlotName.ORIENTATION_ALIGNMENT,
        domain=SlotDomain.CONTINUOUS,
        family=SlotFamily.ENABLING,
    ),
    SlotName.COMPLETION_LATCH: SlotMeta(
        name=SlotName.COMPLETION_LATCH,
        domain=SlotDomain.BINARY,
        family=SlotFamily.RESULT,
    ),
    SlotName.TASK_VISIBLE_CONFIDENCE: SlotMeta(
        name=SlotName.TASK_VISIBLE_CONFIDENCE,
        domain=SlotDomain.CONFIDENCE,
        family=SlotFamily.CONFIDENCE,
    ),
}

SLOT_VOCAB_SIZE: int = len(SLOT_REGISTRY)
assert SLOT_VOCAB_SIZE == 10, f"v1 vocabulary must have exactly 10 slots, got {SLOT_VOCAB_SIZE}"


def get_slot_meta(name: SlotName) -> SlotMeta:
    """Look up metadata for a slot by name."""
    return SLOT_REGISTRY[name]
