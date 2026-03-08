"""
CertVLA inference layer: certificate gap, repair loop, logging.

Phase 4 scope: closed-loop inference with gap monitoring and
short-horizon local repair.
"""

from certvla.inference.gap import (
    slot_gap,
    aggregate_certificate_gap,
    GapResult,
)
from certvla.inference.repair import RepairController
from certvla.inference.logging import InferenceLogger
