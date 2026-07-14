"""Planner-frontier experiments for pooled-vector JEPA on Procgen Maze."""

EXPERIMENT_FAMILY = "vector_jepa_planner_frontier"
FORMAT_VERSION = 1
PROTOCOL_ID = "vector-jepa-planner-frontier-v1"

ACTION_IDS = (1, 2, 3, 4)
INVERSE_ACTION = {1: 2, 2: 1, 3: 4, 4: 3}
REACHABILITY_BINS = (1, 2, 4, 8, 16, 32, 64, 128)

__all__ = [
    "ACTION_IDS",
    "EXPERIMENT_FAMILY",
    "FORMAT_VERSION",
    "INVERSE_ACTION",
    "PROTOCOL_ID",
    "REACHABILITY_BINS",
]
