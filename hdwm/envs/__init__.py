"""Environment implementations for HDWM experiments."""

from hdwm.config import (
    EnvConfig,
    FourRoomsConfig,
    GridWorld2DConfig,
    IceWorld2DConfig,
    ProcgenMazeConfig,
    RingWorldConfig,
)
from hdwm.envs.four_rooms import FourRoomsEnv
from hdwm.envs.grid_world_2d import GridWorld2DEnv
from hdwm.envs.ice_world_2d import IceWorld2DEnv
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.envs.ring_world import RingWorldEnv, SequenceBatch


def make_env(
    config: EnvConfig,
    seed: int | None = None,
) -> RingWorldEnv | GridWorld2DEnv | IceWorld2DEnv | ProcgenMazeEnv | FourRoomsEnv:
    """Build the environment implementation for a validated env config."""

    if isinstance(config, RingWorldConfig):
        return RingWorldEnv(config, seed=seed)
    if isinstance(config, IceWorld2DConfig):
        return IceWorld2DEnv(config, seed=seed)
    if isinstance(config, GridWorld2DConfig):
        return GridWorld2DEnv(config, seed=seed)
    if isinstance(config, ProcgenMazeConfig):
        return ProcgenMazeEnv(config, seed=seed)
    if isinstance(config, FourRoomsConfig):
        return FourRoomsEnv(config, seed=seed)
    raise TypeError(f"unsupported env config: {type(config).__name__}")


__all__ = [
    "FourRoomsConfig",
    "FourRoomsEnv",
    "GridWorld2DConfig",
    "GridWorld2DEnv",
    "IceWorld2DConfig",
    "IceWorld2DEnv",
    "ProcgenMazeConfig",
    "ProcgenMazeEnv",
    "RingWorldConfig",
    "RingWorldEnv",
    "SequenceBatch",
    "make_env",
]
