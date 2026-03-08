"""
Frozen v1 slot vocabulary configuration.

This module re-exports the v1 schema as a single configuration object.
Used by configs/ layer; content is identical to certvla.slots.schema.SLOT_REGISTRY.
"""

from certvla.slots.schema import SLOT_REGISTRY, SLOT_VOCAB_SIZE, SlotName, SlotDomain, SlotFamily, SlotMeta
from certvla.data.certificate_mining import MiningThresholds

# v1 default thresholds (conservative starting values)
DEFAULT_MINING_THRESHOLDS = MiningThresholds(
    tau_delta=0.1,
    tau_rho=0.6,
    tau_upsilon=0.05,
    tau_R=0.1,
    L_future=5,
    epsilon_j=0.05,
)

# v1 goal signature aggregation window
DEFAULT_GOAL_K = 5

# v1 chunk size (matches LIBERO NUM_ACTIONS_CHUNK)
DEFAULT_CHUNK_SIZE = 8
