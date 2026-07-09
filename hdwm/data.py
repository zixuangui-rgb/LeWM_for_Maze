"""Training data utilities."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from hdwm.config import (
    BatchSampleStrategy,
    EnvConfig,
    ProcgenMazeConfig,
    SequenceDataConfig,
)
from hdwm.envs import GridWorld2DEnv, SequenceBatch, make_env
from hdwm.envs.procgen_maze import ProcgenMazeEnv


class RingWorldSequenceDataset(IterableDataset[SequenceBatch]):
    """Infinite stream of environment sequence batches."""

    def __init__(
        self,
        env_config: EnvConfig,
        data_config: SequenceDataConfig,
        seed: int = 0,
        max_batches: int | None = None,
        split: Literal["train", "validation"] = "train",
    ) -> None:
        self.env_config = env_config
        self.data_config = data_config
        self.seed = seed
        self.max_batches = max_batches
        self.split = split

    def __iter__(self) -> Iterator[SequenceBatch]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        # Offset worker seeds so multi-worker loading remains deterministic without
        # sampling identical trajectories in every worker.
        env = make_env(self.env_config, seed=self.seed + worker_id)

        batch_idx = 0
        while self.max_batches is None or batch_idx < self.max_batches:
            if isinstance(env, GridWorld2DEnv):
                yield sample_sequence_batch(
                    env=env,
                    batch_size=self.data_config.batch_size,
                    sequence_length=self.data_config.sequence_length,
                    context_length=self.data_config.context_length,
                    batch_sample_strategy=self.data_config.batch_sample_strategy,
                    virtual_border=env.virtual_border_for_split(self.split),
                )
            else:
                yield sample_sequence_batch(
                    env=env,
                    batch_size=self.data_config.batch_size,
                    sequence_length=self.data_config.sequence_length,
                    context_length=self.data_config.context_length,
                    batch_sample_strategy=self.data_config.batch_sample_strategy,
                )
            batch_idx += 1


def sample_sequence_batch(
    env: object,
    batch_size: int,
    sequence_length: int,
    context_length: int | None = None,
    batch_sample_strategy: BatchSampleStrategy = BatchSampleStrategy.SAME_WITHIN_BATCH,
    virtual_border: tuple[int, int, int, int] | None = None,
    context_sample_strategy: BatchSampleStrategy | None = None,
) -> SequenceBatch:
    """Sample either normal trajectories or ICWM context-packed trajectories."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    if context_sample_strategy is not None:
        batch_sample_strategy = context_sample_strategy

    if batch_sample_strategy == BatchSampleStrategy.SAME_WITHIN_BATCH:
        if context_length is not None:
            if context_length <= 0:
                raise ValueError("context_length must be positive")
            flat_batch = _sample_single_sequence_batch(
                env=env,
                batch_size=batch_size * context_length,
                sequence_length=sequence_length,
                virtual_border=virtual_border,
            )
            return _reshape_context_batch(
                flat_batch,
                batch_size=batch_size,
                context_length=context_length,
            )
        return _sample_single_sequence_batch(
            env=env,
            batch_size=batch_size,
            sequence_length=sequence_length,
            virtual_border=virtual_border,
        )
    if batch_sample_strategy != BatchSampleStrategy.DIFFERENT_WITHIN_BATCH:
        raise ValueError(f"unsupported batch_sample_strategy: {batch_sample_strategy}")

    if context_length is None:
        sequence_batches = [
            _sample_single_sequence_batch(
                env=env,
                batch_size=1,
                sequence_length=sequence_length,
                virtual_border=virtual_border,
            )
            for _ in range(batch_size)
        ]
        return _concat_sequence_batches(sequence_batches)
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    context_batches = [
        _sample_single_sequence_batch(
            env=env,
            batch_size=context_length,
            sequence_length=sequence_length,
            virtual_border=virtual_border,
        )
        for _ in range(batch_size)
    ]
    return _stack_context_batches(context_batches)


def _sample_single_sequence_batch(
    env: object,
    batch_size: int,
    sequence_length: int,
    virtual_border: tuple[int, int, int, int] | None,
) -> SequenceBatch:
    if isinstance(env, GridWorld2DEnv):
        return env.sample_sequence(
            batch_size=batch_size,
            sequence_length=sequence_length,
            virtual_border=virtual_border,
        )
    sample_sequence = getattr(env, "sample_sequence", None)
    if sample_sequence is None:
        raise TypeError(f"unsupported sequence environment: {type(env).__name__}")
    return sample_sequence(
        batch_size=batch_size,
        sequence_length=sequence_length,
    )


def _concat_sequence_batches(sequence_batches: list[SequenceBatch]) -> SequenceBatch:
    if not sequence_batches:
        raise ValueError("sequence_batches must be non-empty")
    obstacle_masks = [
        batch.obstacle_masks
        for batch in sequence_batches
        if batch.obstacle_masks is not None
    ]
    if obstacle_masks and len(obstacle_masks) != len(sequence_batches):
        raise ValueError(
            "either all sequence batches or none must include obstacle_masks"
        )
    return SequenceBatch(
        observations=torch.cat(
            [batch.observations for batch in sequence_batches],
            dim=0,
        ),
        states=torch.cat([batch.states for batch in sequence_batches], dim=0),
        noise_masks=torch.cat([batch.noise_masks for batch in sequence_batches], dim=0),
        actions=torch.cat([batch.actions for batch in sequence_batches], dim=0),
        actual_deltas=torch.cat(
            [batch.actual_deltas for batch in sequence_batches],
            dim=0,
        ),
        noop_masks=torch.cat([batch.noop_masks for batch in sequence_batches], dim=0),
        obstacle_masks=torch.cat(obstacle_masks, dim=0) if obstacle_masks else None,
    )


def _stack_context_batches(context_batches: list[SequenceBatch]) -> SequenceBatch:
    if not context_batches:
        raise ValueError("context_batches must be non-empty")
    obstacle_masks = [
        batch.obstacle_masks
        for batch in context_batches
        if batch.obstacle_masks is not None
    ]
    if obstacle_masks and len(obstacle_masks) != len(context_batches):
        raise ValueError(
            "either all context batches or none must include obstacle_masks"
        )
    return SequenceBatch(
        observations=torch.stack(
            [batch.observations for batch in context_batches],
            dim=0,
        ),
        states=torch.stack([batch.states for batch in context_batches], dim=0),
        noise_masks=torch.stack(
            [batch.noise_masks for batch in context_batches], dim=0
        ),
        actions=torch.stack([batch.actions for batch in context_batches], dim=0),
        actual_deltas=torch.stack(
            [batch.actual_deltas for batch in context_batches],
            dim=0,
        ),
        noop_masks=torch.stack([batch.noop_masks for batch in context_batches], dim=0),
        obstacle_masks=torch.stack(obstacle_masks, dim=0) if obstacle_masks else None,
    )


def _reshape_context_batch(
    batch: SequenceBatch,
    batch_size: int,
    context_length: int,
) -> SequenceBatch:
    expected_flat_batch_size = batch_size * context_length
    if batch.observations.shape[0] != expected_flat_batch_size:
        raise ValueError(
            f"expected flat batch size {expected_flat_batch_size}, "
            f"got {batch.observations.shape[0]}"
        )

    def reshape_first_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch_size, context_length, *tensor.shape[1:])

    return SequenceBatch(
        observations=reshape_first_dim(batch.observations),
        states=reshape_first_dim(batch.states),
        noise_masks=reshape_first_dim(batch.noise_masks),
        actions=reshape_first_dim(batch.actions),
        actual_deltas=reshape_first_dim(batch.actual_deltas),
        noop_masks=reshape_first_dim(batch.noop_masks),
        obstacle_masks=reshape_first_dim(batch.obstacle_masks)
        if batch.obstacle_masks is not None
        else None,
    )


def sequence_batch_collate(batch: list[SequenceBatch]) -> SequenceBatch:
    """Return the single pre-batched item produced by the iterable dataset."""

    if len(batch) != 1:
        raise ValueError("RingWorldSequenceDataset already returns batches")
    return batch[0]


def move_sequence_batch(batch: SequenceBatch, device: torch.device) -> SequenceBatch:
    """Move a sequence batch to the requested device."""

    return SequenceBatch(
        observations=batch.observations.to(device=device),
        states=batch.states.to(device=device),
        noise_masks=batch.noise_masks.to(device=device),
        actions=batch.actions.to(device=device),
        actual_deltas=batch.actual_deltas.to(device=device),
        noop_masks=batch.noop_masks.to(device=device),
        obstacle_masks=batch.obstacle_masks.to(device=device)
        if batch.obstacle_masks is not None
        else None,
    )


class ManifestSequenceDataset(IterableDataset[SequenceBatch]):
    """Dataset that samples sequences from a JSONL manifest of maze configurations.

    Each manifest line is a JSON dict with keys:
        maze_size, topology_seed, level_seed, topology_hash, layout_hash, task_hash
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_config: SequenceDataConfig,
        seed: int = 0,
        max_batches: int | None = None,
        shuffle: bool = True,
    ) -> None:
        self.manifest_path = str(manifest_path)
        self.data_config = data_config
        self.seed = seed
        self.max_batches = max_batches
        self.shuffle = shuffle

        with open(self.manifest_path) as f:
            self._entries = [json.loads(line) for line in f if line.strip()]
        if not self._entries:
            raise ValueError(f"manifest is empty: {self.manifest_path}")

    def __iter__(self) -> Iterator[SequenceBatch]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.seed + worker_id)

        entries = list(self._entries)
        if self.shuffle:
            rng.shuffle(entries)

        batch_idx = 0
        while self.max_batches is None or batch_idx < self.max_batches:
            entry = entries[batch_idx % len(entries)]
            sz = entry["maze_size"]
            env_config = ProcgenMazeConfig(
                height=sz, width=sz, observation_channels=5,
                p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
                resample_maze_per_sequence=False,
                topology_seed=entry["topology_seed"],
            )
            env = ProcgenMazeEnv(env_config, seed=int(rng.integers(0, 2**31)))
            batch = env.sample_sequence(
                batch_size=self.data_config.batch_size,
                sequence_length=self.data_config.sequence_length,
            )
            yield batch
            batch_idx += 1
