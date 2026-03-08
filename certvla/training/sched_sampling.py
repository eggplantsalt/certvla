"""
Scheduled sampling for the persistent state token.

v1 baseline: every training sample starts from the learnable z_0
(teacher-force probability = 1.0).  This module provides the
framework for future episode-sequential training (v2), where the
teacher-force probability is annealed so the model gradually
transitions from using ground-truth-derived z_{t-1} to its own
predicted z_{t-1}.

Three schedules are implemented:
    constant : always returns a fixed probability (v1 default: 1.0)
    linear   : linearly decay from start_prob to end_prob
    cosine   : cosine-annealed decay from start_prob to end_prob
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SamplingSchedule(str, Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"


@dataclass
class ScheduledSampler:
    """Controls the teacher-forcing probability for the state token.

    Attributes:
        schedule: One of ``constant``, ``linear``, ``cosine``.
        start_prob: Teacher-force probability at warmup_steps.
        end_prob: Teacher-force probability at total_steps.
        warmup_steps: Steps before decay begins (prob stays at start_prob).
        total_steps: Step at which prob reaches end_prob.
    """
    schedule: SamplingSchedule = SamplingSchedule.CONSTANT
    start_prob: float = 1.0
    end_prob: float = 0.0
    warmup_steps: int = 0
    total_steps: int = 10_000

    def get_teacher_force_prob(self, step: int) -> float:
        """Return the teacher-forcing probability for this training step.

        Returns a float in [end_prob, start_prob].
        """
        if self.schedule == SamplingSchedule.CONSTANT:
            return self.start_prob

        if step < self.warmup_steps:
            return self.start_prob

        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = min((step - self.warmup_steps) / decay_steps, 1.0)

        if self.schedule == SamplingSchedule.LINEAR:
            return self.start_prob + (self.end_prob - self.start_prob) * progress

        if self.schedule == SamplingSchedule.COSINE:
            return self.end_prob + (self.start_prob - self.end_prob) * (
                1.0 + math.cos(math.pi * progress)
            ) / 2.0

        raise ValueError(f"Unknown schedule: {self.schedule}")

    def should_use_teacher(self, step: int) -> bool:
        """Stochastically decide whether to teacher-force at *step*.

        True  -> use ground-truth (or z_0) as z_{t-1}.
        False -> use the model's own previous prediction as z_{t-1}.
        """
        return random.random() < self.get_teacher_force_prob(step)
