"""Leakage-safe BFS-labelled batches built from training topologies only."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from final_closure.common import (
    bfs_distances_from,
    next_state,
    observe_state,
)
from final_closure.data import sample_lewm_sequence
from spatial_jepa_planning.common import validate_manifest_entry
from vector_jepa_planner_frontier import ACTION_IDS


@dataclass(frozen=True)
class MaterializedTopology:
    entry: dict[str, Any]
    env: Any
    distances: np.ndarray
    eligible_states: np.ndarray
    free_states: np.ndarray


@dataclass(frozen=True)
class PlannerRawBatch:
    source_observations: torch.Tensor
    successor_observations: torch.Tensor
    goal_observations: torch.Tensor
    comparison_observations: torch.Tensor
    source_states: torch.Tensor
    successor_states: torch.Tensor
    goal_states: torch.Tensor
    action_ids: torch.Tensor
    bfs_distances: torch.Tensor
    join_labels: torch.Tensor
    optimal_action_chunks: torch.Tensor
    optimal_chunks_are_full: bool
    maze_size: int
    task_hashes: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.action_ids.shape[0])

    def to(self, device: torch.device) -> PlannerRawBatch:
        return PlannerRawBatch(
            source_observations=self.source_observations.to(device),
            successor_observations=self.successor_observations.to(device),
            goal_observations=self.goal_observations.to(device),
            comparison_observations=self.comparison_observations.to(device),
            source_states=self.source_states.to(device),
            successor_states=self.successor_states.to(device),
            goal_states=self.goal_states.to(device),
            action_ids=self.action_ids.to(device),
            bfs_distances=self.bfs_distances.to(device),
            join_labels=self.join_labels.to(device),
            optimal_action_chunks=self.optimal_action_chunks.to(device),
            optimal_chunks_are_full=self.optimal_chunks_are_full,
            maze_size=self.maze_size,
            task_hashes=self.task_hashes,
        )


@dataclass(frozen=True)
class CounterexampleRawBatch:
    source_observations: torch.Tensor
    good_actions: torch.Tensor
    bad_actions: torch.Tensor
    maze_size: int
    task_hashes: tuple[str, ...]


@dataclass(frozen=True)
class MaterializedCounterexample:
    task_hash: str
    maze_size: int
    env: Any
    source_state: int
    good_actions: tuple[int, ...]
    bad_actions: tuple[int, ...]


@dataclass(frozen=True)
class JEPATrajectoryBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor
    maze_size: int
    task_hash: str


class JEPATrajectorySampler:
    """Balanced-size, train-manifest trajectory sampler for Track J losses."""

    def __init__(self, entries: list[dict[str, Any]], *, sequence_length: int) -> None:
        if not entries or sequence_length < 2:
            raise ValueError("JEPA trajectory sampler requires entries and T >= 2")
        by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            by_size[int(entry["maze_size"])].append(entry)
        self.by_size = dict(by_size)
        self.sizes = tuple(sorted(self.by_size))
        self.sequence_length = int(sequence_length)

    def sample(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        size_slot: int,
        device: torch.device,
    ) -> JEPATrajectoryBatch:
        if batch_size <= 0 or size_slot < 0:
            raise ValueError("JEPA trajectory batch/size slot is invalid")
        maze_size = self.sizes[size_slot % len(self.sizes)]
        entries = self.by_size[maze_size]
        entry = entries[int(rng.integers(len(entries)))]
        batch = sample_lewm_sequence(
            entry,
            rng=rng,
            batch_size=batch_size,
            sequence_length=self.sequence_length,
        )
        observations = batch.observations.to(device=device, dtype=torch.float32)
        actions = batch.actions.to(device=device, dtype=torch.long)
        states = batch.states.to(device=device, dtype=torch.long)
        expected_observations = (batch_size, self.sequence_length)
        if observations.shape[:2] != expected_observations:
            raise ValueError("JEPA trajectory observations have an invalid shape")
        if actions.shape != (batch_size, self.sequence_length - 1):
            raise ValueError("JEPA trajectory actions have an invalid shape")
        if states.shape != expected_observations:
            raise ValueError("JEPA trajectory states have an invalid shape")
        if not torch.isfinite(observations).all():
            raise FloatingPointError("JEPA trajectory contains non-finite observations")
        return JEPATrajectoryBatch(
            observations=observations,
            actions=actions,
            states=states,
            maze_size=maze_size,
            task_hash=str(entry["task_hash"]),
        )


class CounterexampleBatchSampler:
    """Uniform-over-size sampler for frozen, provenance-checked hard negatives."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        entries: list[dict[str, Any]],
        *,
        horizon: int,
        expected_negative_source: str,
    ) -> None:
        if not records:
            raise ValueError("joint ranker training requires mined counterexamples")
        entry_by_hash = {str(entry["task_hash"]): entry for entry in entries}
        if len(entry_by_hash) != len(entries):
            raise ValueError("training manifest contains duplicate task hashes")
        materialized: list[MaterializedCounterexample] = []
        seen: set[str] = set()
        for record in records:
            task_hash = str(record["task_hash"])
            if task_hash in seen:
                raise ValueError(f"duplicate counterexample task: {task_hash}")
            seen.add(task_hash)
            if task_hash not in entry_by_hash:
                raise ValueError(
                    f"counterexample is outside the training split: {task_hash}"
                )
            entry = entry_by_hash[task_hash]
            env = validate_manifest_entry(entry)
            maze_size = int(record["maze_size"])
            if (
                maze_size != int(entry["maze_size"])
                or int(record["topology_seed"]) != int(entry["topology_seed"])
                or int(record["goal_state"]) != int(entry["goal_cell"])
            ):
                raise ValueError(
                    f"counterexample manifest labels disagree: {task_hash}"
                )
            source_state = int(record["source_state"])
            cell_count = maze_size * maze_size
            if not 0 <= source_state < cell_count:
                raise ValueError(f"counterexample source is out of bounds: {task_hash}")
            if bool(env._maze_mask.reshape(-1)[source_state]):
                raise ValueError(f"counterexample source lies on a wall: {task_hash}")
            good_actions = tuple(int(action) for action in record["good_actions"])
            bad_actions = tuple(
                int(action) for action in record["false_optimistic_actions"]
            )
            if len(good_actions) != horizon or len(bad_actions) != horizon:
                raise ValueError(f"counterexample horizon mismatch: {task_hash}")
            if any(action not in ACTION_IDS for action in good_actions + bad_actions):
                raise ValueError(
                    f"counterexample contains an invalid action: {task_hash}"
                )
            if good_actions == bad_actions:
                raise ValueError(
                    f"counterexample positive equals negative: {task_hash}"
                )
            if record.get("negative_source") != expected_negative_source:
                raise ValueError(f"counterexample negative type mismatch: {task_hash}")
            outcome = record.get("outcome", {})
            if outcome.get("false_optimistic") is not True:
                raise ValueError(
                    f"counterexample was not false optimistic: {task_hash}"
                )
            materialized.append(
                MaterializedCounterexample(
                    task_hash=task_hash,
                    maze_size=maze_size,
                    env=env,
                    source_state=source_state,
                    good_actions=good_actions,
                    bad_actions=bad_actions,
                )
            )
        by_size: dict[int, list[MaterializedCounterexample]] = defaultdict(list)
        for record in materialized:
            by_size[record.maze_size].append(record)
        self.by_size = dict(by_size)
        self.sizes = tuple(sorted(self.by_size))

    def sample(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        device: torch.device,
    ) -> CounterexampleRawBatch:
        if batch_size <= 0:
            raise ValueError("counterexample batch_size must be positive")
        maze_size = int(rng.choice(self.sizes))
        pool = self.by_size[maze_size]
        indices = rng.integers(len(pool), size=batch_size)
        selected = [pool[int(index)] for index in indices]
        observations = torch.as_tensor(
            np.stack(
                [observe_state(record.env, record.source_state) for record in selected]
            ),
            dtype=torch.float32,
            device=device,
        )
        return CounterexampleRawBatch(
            source_observations=observations,
            good_actions=torch.tensor(
                [record.good_actions for record in selected],
                dtype=torch.long,
                device=device,
            ),
            bad_actions=torch.tensor(
                [record.bad_actions for record in selected],
                dtype=torch.long,
                device=device,
            ),
            maze_size=maze_size,
            task_hashes=tuple(record.task_hash for record in selected),
        )


class PlannerBatchSampler:
    """Uniform-over-size sampler with topology-labelled real transitions."""

    def __init__(
        self,
        entries: list[dict[str, Any]],
        *,
        horizon: int,
        require_full_chunk: bool = True,
    ) -> None:
        if not entries:
            raise ValueError("planner sampler requires manifest entries")
        if horizon <= 0:
            raise ValueError("planner horizon must be positive")
        self.entries = entries
        self.horizon = int(horizon)
        self.require_full_chunk = bool(require_full_chunk)
        by_size: dict[int, list[int]] = defaultdict(list)
        for index, entry in enumerate(entries):
            by_size[int(entry["maze_size"])].append(index)
        self.by_size = dict(by_size)
        self.sizes = tuple(sorted(self.by_size))
        self._topology_cache: dict[int, MaterializedTopology] = {}
        self._eligible_indices_by_size: dict[int, tuple[int, ...]] = {}

    def _materialize(self, index: int) -> MaterializedTopology:
        if index in self._topology_cache:
            return self._topology_cache[index]
        entry = self.entries[index]
        env = validate_manifest_entry(entry)
        goal = int(entry["goal_cell"])
        size = int(entry["maze_size"])
        distances = bfs_distances_from(env._maze_mask, goal, size)
        free = np.flatnonzero((~env._maze_mask).reshape(-1)).astype(np.int64)
        eligible = (
            free[distances[free] >= self.horizon]
            if self.require_full_chunk
            else free[distances[free] >= 0]
        )
        topology = MaterializedTopology(
            entry=entry,
            env=env,
            distances=distances,
            eligible_states=eligible,
            free_states=free,
        )
        if len(self._topology_cache) >= 512:
            self._topology_cache.pop(next(iter(self._topology_cache)))
        self._topology_cache[index] = topology
        return topology

    def _eligible_topology(
        self,
        index: int,
        *,
        size: int,
        rng: np.random.Generator,
    ) -> MaterializedTopology:
        topology = self._materialize(index)
        if topology.eligible_states.size > 0:
            return topology
        eligible_indices = self._eligible_indices_by_size.get(size)
        if eligible_indices is None:
            eligible_indices = tuple(
                candidate
                for candidate in self.by_size[size]
                if self._materialize(candidate).eligible_states.size > 0
            )
            self._eligible_indices_by_size[size] = eligible_indices
        if not eligible_indices:
            raise ValueError(
                f"size {size} has no topology supporting horizon {self.horizon}"
            )
        return self._materialize(int(rng.choice(eligible_indices)))

    def sample(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        device: torch.device | None = None,
    ) -> PlannerRawBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        size = int(rng.choice(self.sizes))
        indices = rng.choice(self.by_size[size], size=batch_size, replace=True)
        source_observations: list[np.ndarray] = []
        successor_observations: list[np.ndarray] = []
        goal_observations: list[np.ndarray] = []
        comparison_observations: list[np.ndarray] = []
        source_states: list[int] = []
        successor_states: list[int] = []
        goal_states: list[int] = []
        action_ids: list[int] = []
        bfs_distances: list[int] = []
        join_labels: list[float] = []
        chunks: list[list[int]] = []
        task_hashes: list[str] = []
        for raw_index in indices.tolist():
            topology = self._eligible_topology(int(raw_index), size=size, rng=rng)
            env = topology.env
            source = int(rng.choice(topology.eligible_states))
            goal = int(topology.entry["goal_cell"])
            moving = [
                action
                for action in ACTION_IDS
                if next_state(env, source, action) != source
            ]
            if not moving:
                raise RuntimeError("free maze state has no moving action")
            action = int(rng.choice(moving))
            successor = next_state(env, source, action)
            chunk = (
                self._optimal_chunk(topology, source, rng)
                if self.require_full_chunk
                else [ACTION_IDS[0]] * self.horizon
            )
            same_pair = bool(rng.integers(2))
            if same_pair:
                comparison = successor
            else:
                if bool(rng.integers(2)):
                    comparison = source
                else:
                    alternatives = topology.free_states[
                        topology.free_states != successor
                    ]
                    comparison = int(rng.choice(alternatives))
            source_observations.append(observe_state(env, source))
            successor_observations.append(observe_state(env, successor))
            goal_observations.append(observe_state(env, goal))
            comparison_observations.append(observe_state(env, comparison))
            source_states.append(source)
            successor_states.append(successor)
            goal_states.append(goal)
            action_ids.append(action)
            bfs_distances.append(int(topology.distances[source]))
            join_labels.append(float(same_pair))
            chunks.append(chunk)
            task_hashes.append(str(topology.entry["task_hash"]))
        target_device = device or torch.device("cpu")

        def observations(values: list[np.ndarray]) -> torch.Tensor:
            return torch.as_tensor(
                np.stack(values), dtype=torch.float32, device=target_device
            )

        return PlannerRawBatch(
            source_observations=observations(source_observations),
            successor_observations=observations(successor_observations),
            goal_observations=observations(goal_observations),
            comparison_observations=observations(comparison_observations),
            source_states=torch.tensor(
                source_states, dtype=torch.long, device=target_device
            ),
            successor_states=torch.tensor(
                successor_states, dtype=torch.long, device=target_device
            ),
            goal_states=torch.tensor(
                goal_states, dtype=torch.long, device=target_device
            ),
            action_ids=torch.tensor(action_ids, dtype=torch.long, device=target_device),
            bfs_distances=torch.tensor(
                bfs_distances, dtype=torch.long, device=target_device
            ),
            join_labels=torch.tensor(
                join_labels, dtype=torch.float32, device=target_device
            ),
            optimal_action_chunks=torch.tensor(
                chunks, dtype=torch.long, device=target_device
            ),
            optimal_chunks_are_full=self.require_full_chunk,
            maze_size=size,
            task_hashes=tuple(task_hashes),
        )

    def _optimal_chunk(
        self,
        topology: MaterializedTopology,
        source: int,
        rng: np.random.Generator,
    ) -> list[int]:
        state = int(source)
        actions: list[int] = []
        for _ in range(self.horizon):
            current_distance = int(topology.distances[state])
            optimal = [
                action
                for action in ACTION_IDS
                if int(topology.distances[next_state(topology.env, state, action)])
                == current_distance - 1
            ]
            if not optimal:
                raise RuntimeError("BFS-labelled state has no distance-reducing action")
            action = int(rng.choice(optimal))
            actions.append(action)
            state = next_state(topology.env, state, action)
        return actions


def planner_chunk_eligibility(
    entries: list[dict[str, Any]], *, horizon: int
) -> dict[str, Any]:
    eligible_by_size: defaultdict[int, int] = defaultdict(int)
    ineligible_by_size: defaultdict[int, int] = defaultdict(int)
    ineligible_tasks: list[str] = []
    for entry in entries:
        env = validate_manifest_entry(entry)
        distances = bfs_distances_from(
            env._maze_mask, int(entry["goal_cell"]), int(entry["maze_size"])
        )
        if int(distances.max()) >= horizon:
            eligible_by_size[int(entry["maze_size"])] += 1
        else:
            ineligible_by_size[int(entry["maze_size"])] += 1
            ineligible_tasks.append(str(entry["task_hash"]))
    return {
        "horizon": int(horizon),
        "eligible_count": int(sum(eligible_by_size.values())),
        "ineligible_count": int(sum(ineligible_by_size.values())),
        "eligible_by_size": {
            str(size): count for size, count in sorted(eligible_by_size.items())
        },
        "ineligible_by_size": {
            str(size): count for size, count in sorted(ineligible_by_size.items())
        },
        "ineligible_task_hashes": sorted(ineligible_tasks),
    }


def encode_planner_batch(
    model: torch.nn.Module,
    batch: PlannerRawBatch,
    *,
    gradients: bool,
) -> dict[str, torch.Tensor]:
    observations = torch.stack(
        [
            batch.source_observations,
            batch.successor_observations,
            batch.goal_observations,
            batch.comparison_observations,
        ],
        dim=1,
    )
    manager = torch.enable_grad() if gradients else torch.no_grad()
    with manager:
        encoded = model.encoder(observations, batch.maze_size)
        embeddings, _ = model.embedding_projector(encoded)
    if embeddings.shape[:2] != (batch.batch_size, 4):
        raise ValueError("unexpected planner-batch embedding shape")
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("planner-batch encoder produced non-finite vectors")
    return {
        "source": embeddings[:, 0],
        "successor": embeddings[:, 1],
        "goal": embeddings[:, 2],
        "comparison": embeddings[:, 3],
    }


__all__ = [
    "CounterexampleBatchSampler",
    "CounterexampleRawBatch",
    "JEPATrajectoryBatch",
    "JEPATrajectorySampler",
    "MaterializedTopology",
    "PlannerBatchSampler",
    "PlannerRawBatch",
    "encode_planner_batch",
    "planner_chunk_eligibility",
]
