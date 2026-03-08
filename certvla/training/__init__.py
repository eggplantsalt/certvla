"""
CertVLA training layer: losses, curriculum, scheduled sampling.

Phase 3 scope: loss functions, staged training configuration,
and scheduled sampling framework.
"""

from certvla.training.losses import (
    cert_state_loss,
    cert_role_loss,
    cert_goal_loss,
    cert_action_loss,
    cert_consistency_loss,
    cert_dependence_loss,
    cert_counterfactual_loss,
    cert_total_loss,
)
from certvla.training.curriculum import TrainingStage, StageConfig, CurriculumScheduler
from certvla.training.sched_sampling import ScheduledSampler
