"""Deterministic map-state sampling and exact AIR training labels."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from diagnostics.common import (
    ACTION_IDS,
    bfs_distances_from,
    next_state,
    observe_state,
)
from spatial_jepa_planning.common import ManifestSampler, validate_manifest_entry

LOCKED_TRAIN_COUNTS = {size: 400 for size in range(9, 22, 2)}


@dataclass(frozen=True)
class AIRBatch:
    """One current observation and all four exact one-step successors per task."""

    current_observation: torch.Tensor
    successor_observations: torch.Tensor
    candidate_distances: torch.Tensor
    optimal_action_mask: torch.Tensor
    moving_action_mask: torch.Tensor
    current_distance: torch.Tensor
    current_states: torch.Tensor
    successor_states: torch.Tensor
    task_ids: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.current_observation.shape[0])

    @property
    def maze_size(self) -> int:
        return int(self.current_observation.shape[1])

    def validate(self) -> None:
        batch, height, width, channels = self.current_observation.shape
        expected_successors = (batch, 4, height, width, channels)
        if self.successor_observations.shape != expected_successors:
            raise ValueError("successor observations must be [B,4,H,W,C]")
        if self.candidate_distances.shape != (batch, 4):
            raise ValueError("candidate distances must be [B,4]")
        if self.optimal_action_mask.shape != (batch, 4):
            raise ValueError("optimal action mask must be [B,4]")
        if self.moving_action_mask.shape != (batch, 4):
            raise ValueError("moving action mask must be [B,4]")
        if self.current_distance.shape != (batch,):
            raise ValueError("current distance must be [B]")
        if self.current_states.shape != (batch,):
            raise ValueError("current states must be [B]")
        if self.successor_states.shape != (batch, 4):
            raise ValueError("successor states must be [B,4]")
        if len(self.task_ids) != batch:
            raise ValueError("task_ids length must match the batch")
        if channels != 5 or height != width:
            raise ValueError("AIR0 requires square five-channel Procgen observations")
        if not bool((self.current_distance > 0).all()):
            raise ValueError("training states must be reachable and non-goal")
        if not bool((self.candidate_distances >= 0).all()):
            raise ValueError("candidate BFS distances must be finite and non-negative")
        if not bool((self.optimal_action_mask.sum(dim=1) >= 1).all()):
            raise ValueError("every sample must have at least one optimal action")
        minima = self.candidate_distances.min(dim=1, keepdim=True).values
        expected = self.candidate_distances == minima
        if not torch.equal(self.optimal_action_mask.bool(), expected):
            raise ValueError("optimal action mask is inconsistent with BFS distances")
        expected_moving = self.successor_states != self.current_states[:, None]
        if not torch.equal(self.moving_action_mask.bool(), expected_moving):
            raise ValueError("moving action mask is inconsistent with successors")


@dataclass(frozen=True)
class AIRRNGStreams:
    """Independent streams prevent one method branch from shifting another."""

    entries: np.random.Generator
    states: np.random.Generator
    iterations: np.random.Generator
    diagnostics: np.random.Generator
    stream_seeds: dict[str, int]


def make_rng_streams(seed: int) -> AIRRNGStreams:
    sequence = np.random.SeedSequence(int(seed))
    names = ("entries", "states", "iterations", "diagnostics")
    children = sequence.spawn(len(names))
    integer_seeds = {
        name: int(child.generate_state(1, dtype=np.uint64)[0])
        for name, child in zip(names, children, strict=True)
    }
    generators = [np.random.default_rng(child) for child in children]
    return AIRRNGStreams(*generators, stream_seeds=integer_seeds)


def paired_stream_record(batch: AIRBatch, *, iterations: int) -> dict[str, Any]:
    """Return the canonical per-batch record shared by L0 and formal training."""

    if iterations <= 0:
        raise ValueError("paired stream iterations must be positive")
    batch.validate()
    return {
        "task_ids": list(batch.task_ids),
        "states": batch.current_states.detach().cpu().tolist(),
        "candidate_distances": batch.candidate_distances.detach().cpu().tolist(),
        "optimal_action_mask": batch.optimal_action_mask.detach().cpu().tolist(),
        "iterations": int(iterations),
    }


def require_balanced_training_manifest(entries: list[dict[str, Any]]) -> None:
    """Same-size batches are map-uniform only when every size has equal mass."""

    counts = Counter(int(entry["maze_size"]) for entry in entries)
    if not counts:
        raise ValueError("training manifest is empty")
    if dict(sorted(counts.items())) != LOCKED_TRAIN_COUNTS:
        raise ValueError(
            "AIR0 requires exactly 400 train topologies at sizes 9..21; "
            f"got {dict(sorted(counts.items()))}"
        )
    if len({str(entry.get("task_hash")) for entry in entries}) != len(entries):
        raise ValueError("training manifest contains duplicate task hashes")


def task_identifier(entry: dict[str, Any]) -> str:
    value = entry.get("task_hash")
    if value:
        return str(value)
    return (
        f"sz{int(entry['maze_size'])}_topo{int(entry['topology_seed'])}_"
        f"start{int(entry['start_cell'])}"
    )


def sample_training_batch(
    sampler: ManifestSampler,
    *,
    entry_rng: np.random.Generator,
    state_rng: np.random.Generator,
    batch_size: int,
    device: torch.device,
) -> AIRBatch:
    """Sample a paired batch without consulting any evaluation manifest."""

    entries = sampler.sample(entry_rng, batch_size)
    observations: list[np.ndarray] = []
    successor_observations: list[np.ndarray] = []
    candidate_distances: list[list[int]] = []
    optimal_masks: list[list[bool]] = []
    moving_masks: list[list[bool]] = []
    current_distances: list[int] = []
    current_states: list[int] = []
    successor_states: list[list[int]] = []
    task_ids: list[str] = []

    for entry in entries:
        env = validate_manifest_entry(entry, check_bfs=False)
        goal = int(env._goal_position)
        distances = bfs_distances_from(env._maze_mask, goal, int(env.config.width))
        free = np.flatnonzero((~env._maze_mask).reshape(-1))
        candidates = free[(free != goal) & (distances[free] > 0)]
        if not len(candidates):
            raise ValueError(f"manifest task has no reachable non-goal state: {entry}")
        state = int(state_rng.choice(candidates))
        next_states = [next_state(env, state, action) for action in ACTION_IDS]
        next_distances = [int(distances[candidate]) for candidate in next_states]
        if any(distance < 0 for distance in next_distances):
            raise ValueError("Procgen successor unexpectedly became unreachable")
        best = min(next_distances)

        observations.append(observe_state(env, state))
        successor_observations.append(
            np.stack([observe_state(env, candidate) for candidate in next_states])
        )
        candidate_distances.append(next_distances)
        optimal_masks.append([distance == best for distance in next_distances])
        moving_masks.append([candidate != state for candidate in next_states])
        current_distances.append(int(distances[state]))
        current_states.append(state)
        successor_states.append(next_states)
        task_ids.append(task_identifier(entry))

    batch = AIRBatch(
        current_observation=torch.as_tensor(
            np.stack(observations), dtype=torch.float32, device=device
        ),
        successor_observations=torch.as_tensor(
            np.stack(successor_observations), dtype=torch.float32, device=device
        ),
        candidate_distances=torch.as_tensor(
            candidate_distances, dtype=torch.long, device=device
        ),
        optimal_action_mask=torch.as_tensor(
            optimal_masks, dtype=torch.bool, device=device
        ),
        moving_action_mask=torch.as_tensor(
            moving_masks, dtype=torch.bool, device=device
        ),
        current_distance=torch.as_tensor(
            current_distances, dtype=torch.long, device=device
        ),
        current_states=torch.as_tensor(current_states, dtype=torch.long, device=device),
        successor_states=torch.as_tensor(
            successor_states, dtype=torch.long, device=device
        ),
        task_ids=tuple(task_ids),
    )
    batch.validate()
    return batch


def select_progressive_iterations(
    *,
    step: int,
    phase_steps: int,
    k_train: tuple[int, ...],
    rng: np.random.Generator,
) -> int:
    """Uniformly sample from the unlocked K prefix for the current phase."""

    if not 1 <= step <= phase_steps * len(k_train):
        raise ValueError("step is outside the locked progressive schedule")
    phase = min((step - 1) // phase_steps, len(k_train) - 1)
    return int(rng.choice(k_train[: phase + 1]))


def progressive_iteration_signature(
    *,
    seed: int,
    steps: int,
    phase_steps: int,
    k_train: tuple[int, ...],
) -> dict[str, Any]:
    if steps != phase_steps * len(k_train):
        raise ValueError("formal progressive schedule must fill every K phase")
    rng = make_rng_streams(seed).iterations
    digest = hashlib.sha256()
    counts: Counter[int] = Counter()
    for step in range(1, steps + 1):
        iterations = select_progressive_iterations(
            step=step,
            phase_steps=phase_steps,
            k_train=k_train,
            rng=rng,
        )
        counts[iterations] += 1
        digest.update(iterations.to_bytes(4, "big", signed=False))
    return {
        "sha256": digest.hexdigest(),
        "counts": {str(key): value for key, value in sorted(counts.items())},
    }


__all__ = [
    "AIRBatch",
    "AIRRNGStreams",
    "LOCKED_TRAIN_COUNTS",
    "make_rng_streams",
    "paired_stream_record",
    "progressive_iteration_signature",
    "require_balanced_training_manifest",
    "sample_training_batch",
    "select_progressive_iterations",
    "task_identifier",
]
