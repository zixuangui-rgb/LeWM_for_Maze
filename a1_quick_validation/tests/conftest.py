from __future__ import annotations

import pytest
import torch

from distance_head_study.data import TrainingBatch


@pytest.fixture
def synthetic_batch() -> TrainingBatch:
    generator = torch.Generator().manual_seed(701)
    batch_size = 8
    latent_dim = 256
    source = torch.randn(batch_size, latent_dim, generator=generator)
    goal = torch.randn(batch_size, latent_dim, generator=generator)
    raw = torch.arange(1, batch_size + 1, dtype=torch.float32)
    next_distances = torch.stack(
        [raw, (raw - 1).clamp_min(0), raw, raw + 1, raw + 2], dim=1
    )
    valid = torch.ones(batch_size, 5, dtype=torch.bool)
    valid[:, 0] = False
    optimal = torch.zeros_like(valid)
    optimal[:, 1] = True
    path_steps = torch.arange(13, dtype=torch.float32)
    path_distances = (raw[:, None] - path_steps[None, :]).clamp_min(0)
    batch = TrainingBatch(
        source=source,
        goal=goal,
        raw_distance=raw,
        max_distance=torch.full((batch_size,), 128.0),
        next_latents=torch.randn(batch_size, 5, latent_dim, generator=generator),
        next_distances=next_distances,
        valid_actions=valid,
        optimal_actions=optimal,
        history_latents=torch.randn(batch_size, 3, latent_dim, generator=generator),
        history_actions=torch.tensor([[4, 1, 2]]).repeat(batch_size, 1),
        path_latents=torch.randn(batch_size, 13, latent_dim, generator=generator),
        path_distances=path_distances,
        triangle_latent=torch.randn(batch_size, latent_dim, generator=generator),
        triangle_source_distance=torch.arange(1, batch_size + 1, dtype=torch.float32),
        maze_size=11,
        topology_positions=torch.zeros(batch_size, dtype=torch.long),
        source_indices=torch.arange(batch_size),
        next_indices=torch.arange(batch_size * 5).reshape(batch_size, 5),
        history_indices=torch.arange(batch_size * 3).reshape(batch_size, 3),
        path_indices=torch.arange(batch_size * 13).reshape(batch_size, 13),
        triangle_index=torch.arange(batch_size),
    )
    batch.validate()
    return batch
