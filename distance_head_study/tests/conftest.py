from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from distance_head_study import PROTOCOL_ID
from distance_head_study.common import (
    canonical_json_sha256,
    load_method_catalog,
    sha256_file,
)
from distance_head_study.data import TrainingBatch


@pytest.fixture
def decision_root(tmp_path: Path) -> Path:
    evidence = tmp_path / "immutable_evidence.txt"
    evidence.write_text("locked\n", encoding="utf-8")
    evidence_hashes = {evidence.as_posix(): sha256_file(evidence)}
    choices = {
        "a_target_parent": (
            "b_dh_cem",
            ["b_dh_cem", "a1_log"],
        ),
        "a_sampling_parent": (
            "a2_distance_balanced",
            ["a2_distance_balanced", "a3_full_horizon"],
        ),
        "b_structural_winner": (
            "b2_bellman",
            ["b2_bellman", "b3_multistep"],
        ),
        "b_parent": (
            "b1_listwise",
            ["b1_listwise", "b2_bellman", "b3_multistep", "b5_local_structural"],
        ),
        "c_parent": (
            "c1_predicted_listwise",
            ["c1_predicted_listwise", "c2_dual_calibration"],
        ),
        "finalist_lock": (
            "d2_trm_full",
            ["d2_trm_full", "d4_reachability"],
        ),
    }
    for name, (selected, eligible) in choices.items():
        payload = {
            "protocol_id": PROTOCOL_ID,
            "decision_name": name,
            "selected_method": selected,
            "eligible_methods": eligible,
            "input_hashes": evidence_hashes,
        }
        payload["decision_sha256"] = canonical_json_sha256(payload)
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


@pytest.fixture
def method_catalog():
    return load_method_catalog("distance_head_study/configs/methods.json")


@pytest.fixture
def synthetic_batch() -> TrainingBatch:
    generator = torch.Generator().manual_seed(7)
    batch_size = 8
    latent_dim = 256
    source = torch.randn(batch_size, latent_dim, generator=generator)
    goal = torch.randn(batch_size, latent_dim, generator=generator)
    raw = torch.arange(1, batch_size + 1, dtype=torch.float32)
    next_distances = torch.stack(
        [
            raw,
            (raw - 1).clamp_min(0),
            raw,
            raw + 1,
            raw + 2,
        ],
        dim=1,
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
