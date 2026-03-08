"""
Inference-time logging and debug traces.

Records per-step data for analysis and visualization:
- CertVLAOutput snapshots (z_t, state readout, role probs, gap)
- Repair attempt counts and acceptance decisions
- Aggregated episode statistics
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

import torch

from certvla.model.outputs import CertVLAOutput
from certvla.slots.schema import SlotName

logger = logging.getLogger("certvla.inference")


@dataclass
class StepRecord:
    """One inference step record.

    Attributes:
        output: CertVLAOutput (detached tensors for logging).
        gap: GapResult from gap computation.
        repair_attempt: Which attempt this record is for (0 = initial).
        accepted: Whether this attempt was accepted as final.
    """
    output: Any = None         # CertVLAOutput
    gap: Any = None            # GapResult
    repair_attempt: int = 0
    accepted: bool = False


@dataclass
class EpisodeTrace:
    """Full trace for one inference episode.

    Attributes:
        steps: List of accepted StepRecords (one per timestep).
        all_attempts: List of ALL StepRecords including repair rejects.
        warnings: List of warning messages.
        metadata: Free-form metadata dict.
    """
    steps: List[StepRecord] = field(default_factory=list)
    all_attempts: List[StepRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_repairs(self) -> int:
        return sum(1 for s in self.steps if s.repair_attempt > 0)

    @property
    def gap_history(self) -> List[float]:
        """Return list of aggregated gap values for accepted steps."""
        out = []
        for s in self.steps:
            if s.gap is not None:
                out.append(s.gap.aggregated.mean().item())
        return out

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics for the episode."""
        gaps = self.gap_history
        return {
            "num_steps": self.num_steps,
            "num_repairs": self.num_repairs,
            "total_attempts": len(self.all_attempts),
            "mean_gap": sum(gaps) / len(gaps) if gaps else 0.0,
            "max_gap": max(gaps) if gaps else 0.0,
            "num_warnings": len(self.warnings),
        }


class InferenceLogger:
    """Logger for inference-time debugging.

    Collects StepRecords into EpisodeTraces, and optionally writes
    to the Python logging system.

    Usage::

        logger = InferenceLogger(verbose=True)
        logger.begin_episode(metadata={"task": "pick_and_place"})

        # ... during inference ...
        logger.log_step(record)

        logger.end_episode()
        trace = logger.get_last_trace()
    """

    def __init__(self, verbose: bool = False, max_episodes: int = 100):
        """
        Args:
            verbose: If True, emit Python log messages for each step.
            max_episodes: Maximum number of episode traces to keep in memory.
        """
        self.verbose = verbose
        self.max_episodes = max_episodes
        self._traces: List[EpisodeTrace] = []
        self._current: Optional[EpisodeTrace] = None

    def begin_episode(self, metadata: Optional[Dict[str, Any]] = None):
        """Start recording a new episode."""
        self._current = EpisodeTrace(metadata=metadata or {})
        if self.verbose:
            logger.info("Episode started: %s", metadata)

    def log_step(self, record: StepRecord):
        """Record one inference step."""
        if self._current is None:
            self._current = EpisodeTrace()

        self._current.all_attempts.append(record)
        if record.accepted:
            self._current.steps.append(record)

        if self.verbose and record.accepted:
            gap_val = "N/A"
            if record.gap is not None:
                gap_val = f"{record.gap.aggregated.mean().item():.4f}"
            logger.info(
                "Step %d | attempt=%d | gap=%s | accepted=%s",
                len(self._current.steps) - 1,
                record.repair_attempt,
                gap_val,
                record.accepted,
            )

    def log_warning(self, message: str):
        """Record a warning during inference."""
        if self._current is not None:
            self._current.warnings.append(message)
        if self.verbose:
            logger.warning(message)

    def end_episode(self):
        """Finalise the current episode trace."""
        if self._current is not None:
            self._traces.append(self._current)
            if self.verbose:
                logger.info("Episode ended: %s", self._current.summary())
            # Prune oldest
            if len(self._traces) > self.max_episodes:
                self._traces = self._traces[-self.max_episodes:]
            self._current = None

    def get_last_trace(self) -> Optional[EpisodeTrace]:
        """Return the most recent completed episode trace."""
        return self._traces[-1] if self._traces else None

    def get_all_traces(self) -> List[EpisodeTrace]:
        """Return all stored episode traces."""
        return list(self._traces)

    def clear(self):
        """Clear all stored traces."""
        self._traces.clear()
        self._current = None
