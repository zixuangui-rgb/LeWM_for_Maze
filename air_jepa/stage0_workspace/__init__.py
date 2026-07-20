"""AIR-JEPA Stage 0: frozen Spatial-JEPA iterative workspace experiment."""

EXPERIMENT_ID = "procgen-maze-air0-workspace-v1"
FORMAT_VERSION = 1
METHODS = (
    "j0_static",
    "j1_static",
    "j1_receding",
    "air0_direct",
    "air0_jepa",
)
ALL_METHODS = METHODS
AIR_METHODS = ("air0_direct", "air0_jepa")
RECURRENT_METHODS = ("j1_receding", "air0_direct", "air0_jepa")
SYSTEM_SEEDS = (42, 43, 44)
PAIRING_AUDIT_BATCHES = 128

__all__ = [
    "AIR_METHODS",
    "ALL_METHODS",
    "EXPERIMENT_ID",
    "FORMAT_VERSION",
    "METHODS",
    "PAIRING_AUDIT_BATCHES",
    "RECURRENT_METHODS",
    "SYSTEM_SEEDS",
]
