"""
Training stage configuration and curriculum scheduler.

Stage 1  state         : learn z_t readout               (L_state)
Stage 2  certificate   : add cert head                    (+ L_role, L_goal)
Stage 3  policy        : add cert action head             (+ L_act, L_cons, L_dep)
Stage 4  counterfactual: add augmented invariance/breaking (+ L_cf)
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class TrainingStage(str, Enum):
    STAGE_1_STATE = "stage1_state"
    STAGE_2_CERTIFICATE = "stage2_certificate"
    STAGE_3_POLICY = "stage3_policy"
    STAGE_4_COUNTERFACTUAL = "stage4_counterfactual"


@dataclass
class StageConfig:
    """Configuration for one training stage.

    Attributes:
        stage: Which stage this config describes.
        lambda_s .. lambda_cf: Loss weights (0.0 = disabled).
        lambda_pre: Internal weight for preserve term inside L_cons.
        dep_margin: Margin for L_dep.
        focal_gamma: Focal CE gamma for L_role.
        cf_mu: Margin for L_cf breaking term.
        freeze_backbone: Freeze base VLA backbone (vision + LLM).
        freeze_state: Freeze state token + readout.
        freeze_certificate: Freeze certificate head.
        freeze_action: Freeze cert action head.
    """
    stage: TrainingStage = TrainingStage.STAGE_1_STATE

    # --- loss weights ---
    lambda_s: float = 0.0       # L_state
    lambda_r: float = 0.0       # L_role
    lambda_g: float = 0.0       # L_goal
    lambda_a: float = 0.0       # L_act
    lambda_c: float = 0.0       # L_cons
    lambda_d: float = 0.0       # L_dep
    lambda_cf: float = 0.0      # L_cf

    # --- loss hyper-params ---
    lambda_pre: float = 1.0     # preserve weight inside L_cons
    dep_margin: float = 0.1     # L_dep margin
    focal_gamma: float = 2.0    # L_role focal gamma
    cf_mu: float = 1.0          # L_cf breaking margin

    # --- frozen module groups ---
    freeze_backbone: bool = True
    freeze_state: bool = False
    freeze_certificate: bool = True
    freeze_action: bool = True

    description: str = ""

    def loss_weights(self) -> Dict[str, float]:
        """Return a dict suitable for ``cert_total_loss``."""
        return {
            "lambda_s": self.lambda_s,
            "lambda_r": self.lambda_r,
            "lambda_g": self.lambda_g,
            "lambda_a": self.lambda_a,
            "lambda_c": self.lambda_c,
            "lambda_d": self.lambda_d,
            "lambda_cf": self.lambda_cf,
        }


# ── Default stage configs ────────────────────────────────────

DEFAULT_STAGES: Dict[TrainingStage, StageConfig] = {
    TrainingStage.STAGE_1_STATE: StageConfig(
        stage=TrainingStage.STAGE_1_STATE,
        lambda_s=1.0,
        freeze_backbone=True,
        freeze_state=False,
        freeze_certificate=True,
        freeze_action=True,
        description="Learn state readout from z_t",
    ),
    TrainingStage.STAGE_2_CERTIFICATE: StageConfig(
        stage=TrainingStage.STAGE_2_CERTIFICATE,
        lambda_s=1.0,
        lambda_r=1.0,
        lambda_g=1.0,
        freeze_backbone=True,
        freeze_state=False,
        freeze_certificate=False,
        freeze_action=True,
        description="Learn certificate role + goal prediction",
    ),
    TrainingStage.STAGE_3_POLICY: StageConfig(
        stage=TrainingStage.STAGE_3_POLICY,
        lambda_s=1.0,
        lambda_r=1.0,
        lambda_g=1.0,
        lambda_a=1.0,
        lambda_c=0.5,
        lambda_d=0.5,
        freeze_backbone=True,
        freeze_state=False,
        freeze_certificate=False,
        freeze_action=False,
        description="Train cert-conditioned action head",
    ),
    TrainingStage.STAGE_4_COUNTERFACTUAL: StageConfig(
        stage=TrainingStage.STAGE_4_COUNTERFACTUAL,
        lambda_s=1.0,
        lambda_r=1.0,
        lambda_g=1.0,
        lambda_a=1.0,
        lambda_c=0.5,
        lambda_d=0.5,
        lambda_cf=0.5,
        freeze_backbone=True,
        freeze_state=False,
        freeze_certificate=False,
        freeze_action=False,
        description="Full training with counterfactual augmentation",
    ),
}


class CurriculumScheduler:
    """Step-based stage scheduler.

    Given a global training step, returns the corresponding
    ``TrainingStage`` and ``StageConfig``.

    Stage boundaries are expressed as ``(start_step, end_step)`` tuples.
    Steps past every boundary remain on the final stage.
    """

    def __init__(
        self,
        stages: Optional[Dict[TrainingStage, StageConfig]] = None,
        stage_boundaries: Optional[Dict[TrainingStage, Tuple[int, int]]] = None,
    ):
        """
        Args:
            stages: Stage configs to use (defaults to DEFAULT_STAGES).
            stage_boundaries: {stage -> (start, end)} step ranges.
        """
        self.stages = stages if stages is not None else dict(DEFAULT_STAGES)
        self.stage_boundaries = stage_boundaries if stage_boundaries is not None else {
            TrainingStage.STAGE_1_STATE:          (0,     5_000),
            TrainingStage.STAGE_2_CERTIFICATE:    (5_000, 15_000),
            TrainingStage.STAGE_3_POLICY:         (15_000, 40_000),
            TrainingStage.STAGE_4_COUNTERFACTUAL: (40_000, 60_000),
        }
        # Ordered list for fast lookup
        self._order: List[Tuple[TrainingStage, int, int]] = sorted(
            [(s, lo, hi) for s, (lo, hi) in self.stage_boundaries.items()],
            key=lambda x: x[1],
        )

    def get_stage(self, step: int) -> TrainingStage:
        """Return stage for *step*."""
        for stage, lo, hi in self._order:
            if lo <= step < hi:
                return stage
        # Past all boundaries -> last stage
        return self._order[-1][0]

    def get_config(self, step: int) -> StageConfig:
        """Return ``StageConfig`` for *step*."""
        return self.stages[self.get_stage(step)]

    def get_loss_weights(self, step: int) -> Dict[str, float]:
        """Return loss-weight dict for ``cert_total_loss``."""
        return self.get_config(step).loss_weights()

    def should_compute_dep(self, step: int) -> bool:
        """Whether the current stage needs a negative-cert forward pass."""
        return self.get_config(step).lambda_d > 0.0

    def should_compute_cf(self, step: int) -> bool:
        """Whether the current stage needs augmented pairs."""
        return self.get_config(step).lambda_cf > 0.0
