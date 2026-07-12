"""Deterministic data construction for the fixed BC and LeWM baselines."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from spatial_jepa_planning.common import (
    ACTION_IDS,
    bfs_distances_from,
    create_env,
    next_state,
    validate_manifest_entry,
)


@dataclass
class BCSizePool:
    size: int
    wall_masks: np.ndarray
    goals: np.ndarray
    map_indices: np.ndarray
    states: np.ndarray
    labels: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.states.shape[0])


@dataclass
class BCDatasetIndex:
    pools: dict[int, BCSizePool]
    sample_sizes: np.ndarray
    sample_local_indices: np.ndarray
    sample_labels: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.sample_sizes.shape[0])


def _optimal_label(env: Any, state: int, distances: np.ndarray) -> int:
    current = int(distances[state])
    if current <= 0:
        raise ValueError("BC samples must exclude the goal and unreachable states")
    optimal_slots = []
    for slot, action in enumerate(ACTION_IDS):
        candidate = next_state(env, state, action)
        if candidate != state and int(distances[candidate]) == current - 1:
            optimal_slots.append(slot)
    if len(optimal_slots) != 1:
        raise ValueError(
            "Procgen perfect-maze BC target must have one shortest action; "
            f"state={state}, targets={optimal_slots}"
        )
    return optimal_slots[0]


def build_bc_dataset(entries: list[dict[str, Any]]) -> BCDatasetIndex:
    """Index every non-goal free state without materializing image tensors."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(int(entry["maze_size"]), []).append(entry)
    pools: dict[int, BCSizePool] = {}
    global_sizes: list[np.ndarray] = []
    global_indices: list[np.ndarray] = []
    global_labels: list[np.ndarray] = []
    for size, size_entries in sorted(grouped.items()):
        wall_masks: list[np.ndarray] = []
        goals: list[int] = []
        sample_maps: list[int] = []
        sample_states: list[int] = []
        sample_labels: list[int] = []
        for map_index, entry in enumerate(size_entries):
            env = validate_manifest_entry(entry)
            wall_masks.append(np.asarray(env._maze_mask, dtype=bool))
            goal = int(env._goal_position)
            goals.append(goal)
            distances = bfs_distances_from(env._maze_mask, goal, size)
            free = np.flatnonzero((~env._maze_mask).reshape(-1))
            for state_value in free:
                state = int(state_value)
                if state == goal:
                    continue
                sample_maps.append(map_index)
                sample_states.append(state)
                sample_labels.append(_optimal_label(env, state, distances))
        pool = BCSizePool(
            size=size,
            wall_masks=np.stack(wall_masks),
            goals=np.asarray(goals, dtype=np.int64),
            map_indices=np.asarray(sample_maps, dtype=np.int32),
            states=np.asarray(sample_states, dtype=np.int32),
            labels=np.asarray(sample_labels, dtype=np.int64),
        )
        pools[size] = pool
        global_sizes.append(np.full(pool.sample_count, size, dtype=np.int16))
        global_indices.append(np.arange(pool.sample_count, dtype=np.int32))
        global_labels.append(pool.labels)
    if not pools:
        raise ValueError("BC training needs at least one manifest entry")
    return BCDatasetIndex(
        pools=pools,
        sample_sizes=np.concatenate(global_sizes),
        sample_local_indices=np.concatenate(global_indices),
        sample_labels=np.concatenate(global_labels),
    )


def render_bc_batch(
    dataset: BCDatasetIndex,
    global_indices: np.ndarray,
    *,
    canvas_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a mixed-size batch exactly in the historical bottom/right padding."""

    batch_size = int(len(global_indices))
    observations = np.zeros((batch_size, canvas_size, canvas_size, 5), dtype=np.float32)
    labels = np.empty(batch_size, dtype=np.int64)
    selected_sizes = dataset.sample_sizes[global_indices]
    selected_local = dataset.sample_local_indices[global_indices]
    for size_value in np.unique(selected_sizes):
        size = int(size_value)
        if size > canvas_size:
            raise ValueError(f"training size {size} exceeds BC canvas {canvas_size}")
        batch_positions = np.flatnonzero(selected_sizes == size_value)
        local = selected_local[batch_positions]
        pool = dataset.pools[size]
        map_indices = pool.map_indices[local]
        states = pool.states[local]
        goals = pool.goals[map_indices]
        walls = pool.wall_masks[map_indices]
        rendered = np.zeros((len(local), size, size, 5), dtype=np.float32)
        rendered[..., 0] = 1.0
        rendered[..., 0][walls] = 0.0
        rendered[..., 1][walls] = 1.0
        row_index = np.arange(len(local))
        goal_y, goal_x = np.divmod(goals, size)
        state_y, state_x = np.divmod(states, size)
        rendered[row_index, goal_y, goal_x, 0] = 0.0
        rendered[row_index, goal_y, goal_x, 3] = 1.0
        rendered[row_index, state_y, state_x, 0] = 0.0
        rendered[row_index, state_y, state_x, 2] = 1.0
        observations[batch_positions, :size, :size] = rendered
        labels[batch_positions] = pool.labels[local]
    inputs = torch.from_numpy(observations).permute(0, 3, 1, 2).contiguous()
    return inputs, torch.from_numpy(labels)


def epoch_batches(
    sample_count: int,
    batch_size: int,
    *,
    seed: int,
    epoch: int,
    namespace: int = 1701,
) -> Iterator[np.ndarray]:
    """Yield a reproducible global permutation with complete epoch coverage."""

    if sample_count <= 0 or batch_size <= 0 or epoch <= 0:
        raise ValueError("sample_count, batch_size, and epoch must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, namespace]))
    permutation = rng.permutation(sample_count)
    for start in range(0, sample_count, batch_size):
        yield permutation[start : start + batch_size]


def materialize_bc_dataset(
    dataset: BCDatasetIndex,
    *,
    canvas_size: int,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache exact one-hot observations as uint8 for repeated full epochs."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    observations = torch.empty(
        (dataset.sample_count, 5, canvas_size, canvas_size), dtype=torch.uint8
    )
    for start in range(0, dataset.sample_count, chunk_size):
        end = min(start + chunk_size, dataset.sample_count)
        rendered, labels = render_bc_batch(
            dataset,
            np.arange(start, end, dtype=np.int64),
            canvas_size=canvas_size,
        )
        observations[start:end] = rendered.to(dtype=torch.uint8)
        expected = torch.from_numpy(dataset.sample_labels[start:end])
        if not torch.equal(labels, expected):
            raise RuntimeError("BC materialization changed target order")
    return observations, torch.from_numpy(dataset.sample_labels.copy())


def sample_lewm_sequence(
    entry: dict[str, Any],
    *,
    rng: np.random.Generator,
    batch_size: int,
    sequence_length: int,
) -> Any:
    runtime_entry = dict(entry)
    runtime_entry["env_seed"] = int(rng.integers(2**31))
    env = create_env(runtime_entry)
    if int(env._goal_position) != int(entry["goal_cell"]):
        raise ValueError("runtime seed changed the locked topology goal")
    return env.sample_sequence(
        batch_size=batch_size,
        sequence_length=sequence_length,
    )


__all__ = [
    "BCDatasetIndex",
    "BCSizePool",
    "build_bc_dataset",
    "epoch_batches",
    "materialize_bc_dataset",
    "render_bc_batch",
    "sample_lewm_sequence",
]
