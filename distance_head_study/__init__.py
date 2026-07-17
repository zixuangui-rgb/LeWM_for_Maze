"""Staged, fail-closed DistanceHead study for pooled Vector-JEPA Maze planning."""

from __future__ import annotations

PROTOCOL_ID = "procgen-maze-distance-head-staged-v1"
EXPERIMENT_FAMILY = "distance_head_study"
FORMAT_VERSION = 1
ACTION_IDS = (1, 2, 3, 4)
MODEL_ACTION_VOCAB_SIZE = 5
DEVELOPMENT_BACKBONE_SEEDS = (42, 43, 44)
CONFIRMATION_SEED_START = 1001

__all__ = [
    "ACTION_IDS",
    "CONFIRMATION_SEED_START",
    "DEVELOPMENT_BACKBONE_SEEDS",
    "EXPERIMENT_FAMILY",
    "FORMAT_VERSION",
    "MODEL_ACTION_VOCAB_SIZE",
    "PROTOCOL_ID",
]
