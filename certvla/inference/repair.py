"""
Short-horizon local repair controller.

When the aggregated certificate gap Gamma_t exceeds a threshold,
the repair controller triggers re-prediction: the model is called
again (possibly multiple times) with the same observation to attempt
a lower-gap action chunk.

v1 repair strategy:
    1. Normal forward -> compute gap
    2. If gap > threshold -> re-forward up to max_repair_steps times
    3. Accept the action chunk with the lowest gap
    4. If all attempts exceed threshold -> accept lowest-gap attempt
       and log a warning

This is NOT a full replanner.  It is a local, stateless retry loop
that relies on stochasticity in the model (dropout, temperature)
or on the gated state update providing slightly different z_t.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from certvla.model.outputs import CertVLAOutput
from certvla.inference.gap import GapResult, slot_gap, aggregate_certificate_gap
from certvla.inference.logging import InferenceLogger, StepRecord


@dataclass
class RepairConfig:
    """Configuration for the repair controller.

    Attributes:
        gap_threshold: Gamma_t above which repair is triggered.
        max_repair_steps: Maximum number of re-forward attempts.
        slot_weights: Static per-slot importance for gap aggregation.
        use_best_of_n: If True, always pick the lowest-gap attempt
            even if it still exceeds threshold.
    """
    gap_threshold: float = 0.3
    max_repair_steps: int = 3
    slot_weights: Optional[Dict] = None
    use_best_of_n: bool = True


class RepairController:
    """Short-horizon local repair loop.

    Usage::

        controller = RepairController(config, model_fn, logger)
        actions, gap_result, n_repairs = controller.step(
            last_hidden_states, state_token_pos, action_start_pos,
            z_prev, confidence_weights,
        )

    The ``model_fn`` callable must have signature::

        model_fn(last_hidden_states, state_token_pos, action_start_pos,
                 z_prev) -> CertVLAOutput

    This decouples the repair controller from the concrete model class.
    At integration time, ``model_fn`` wraps the CertVLAWrapper.forward call.
    """

    def __init__(
        self,
        config: RepairConfig,
        model_fn: Callable[..., CertVLAOutput],
        logger: Optional["InferenceLogger"] = None,
    ):
        self.config = config
        self.model_fn = model_fn
        self.logger = logger

    def _compute_gap(
        self,
        output: CertVLAOutput,
        state_readout_tH: Optional[Dict] = None,
        confidence_weights: Optional[Dict] = None,
    ) -> GapResult:
        """Compute certificate gap from a CertVLAOutput.

        For gap computation we need state_readout at t+H.  In the
        closed-loop setting the model has only processed o_t, so
        state_readout_tH is not available from this forward pass.

        v1 fallback: use the goal predictions as a proxy for the
        expected chunk-end state (advance slots) and the current
        state readout for preserve slots.  This gives a self-consistency
        gap rather than a ground-truth gap.
        """
        if state_readout_tH is None:
            # Proxy: goal_preds for advance, state_readout for preserve.
            # This yields gamma_j = p_adv * d(goal, goal) + p_pre * d(s_t, s_t)
            #                      = 0 always, which is not useful.
            # Better proxy: use goal_preds as the "expected t+H" for all slots.
            state_readout_tH = output.goal_preds

        per_slot = slot_gap(
            output.role_logits,
            output.state_readout,
            output.goal_preds,
            state_readout_tH,
        )
        return aggregate_certificate_gap(
            per_slot,
            output.role_logits,
            slot_weights=self.config.slot_weights,
            confidence_weights=confidence_weights,
        )

    @torch.no_grad()
    def step(
        self,
        last_hidden_states: torch.Tensor,
        state_token_pos: int,
        action_start_pos: int,
        z_prev: Optional[torch.Tensor] = None,
        confidence_weights: Optional[Dict] = None,
        state_readout_tH: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, GapResult, int]:
        """Run one inference step with optional repair loop.

        Args:
            last_hidden_states: LLM output. Shape (B, seq, llm_dim).
            state_token_pos: Position of the state token in the sequence.
            action_start_pos: Start position of action tokens.
            z_prev: Previous state (None -> use z_0).
            confidence_weights: Per-slot confidence for gap aggregation.
            state_readout_tH: Optional ground-truth or model-predicted
                state at t+H for gap computation.

        Returns:
            actions: Best action chunk. Shape (B, H, action_dim).
            gap_result: GapResult for the selected attempt.
            n_repairs: Number of repair re-forwards performed (0 = no repair).
        """
        # Initial forward
        output = self.model_fn(
            last_hidden_states, state_token_pos, action_start_pos, z_prev,
        )
        gap = self._compute_gap(output, state_readout_tH, confidence_weights)

        best_output = output
        best_gap = gap
        best_gap_val = gap.aggregated.mean().item()
        n_repairs = 0

        # Check if repair is needed (mean gap across batch)
        if best_gap_val <= self.config.gap_threshold:
            if self.logger:
                self.logger.log_step(StepRecord(
                    output=output, gap=gap, repair_attempt=0, accepted=True,
                ))
            return output.actions, gap, 0

        # Repair loop
        for attempt in range(1, self.config.max_repair_steps + 1):
            output_r = self.model_fn(
                last_hidden_states, state_token_pos, action_start_pos, z_prev,
            )
            gap_r = self._compute_gap(output_r, state_readout_tH, confidence_weights)
            gap_val = gap_r.aggregated.mean().item()

            if self.logger:
                self.logger.log_step(StepRecord(
                    output=output_r, gap=gap_r,
                    repair_attempt=attempt, accepted=False,
                ))

            if gap_val < best_gap_val:
                best_output = output_r
                best_gap = gap_r
                best_gap_val = gap_val

            if best_gap_val <= self.config.gap_threshold:
                break

            n_repairs = attempt

        # Accept best attempt
        if self.logger:
            self.logger.log_step(StepRecord(
                output=best_output, gap=best_gap,
                repair_attempt=n_repairs, accepted=True,
            ))
            if best_gap_val > self.config.gap_threshold:
                self.logger.log_warning(
                    f"Repair exhausted ({n_repairs} attempts), "
                    f"accepting gap={best_gap_val:.4f} > threshold={self.config.gap_threshold}"
                )

        return best_output.actions, best_gap, n_repairs
