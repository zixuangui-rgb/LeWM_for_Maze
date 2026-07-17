from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from distance_head_study.common import read_jsonl, sha256_file
from distance_head_study.data import (
    CACHE_SCHEMA,
    ShardedGoalDataset,
    _history_for_source,
    _sample_sources,
    build_topology_shard,
    evenly_spaced_indices,
    load_backbone_checkpoint,
    sample_training_batch,
    true_candidate_distances,
)
from distance_head_study.losses import compute_objective_terms, weighted_total
from distance_head_study.methods import resolve_method
from distance_head_study.models import build_distance_head
from distance_head_study.schemas import SamplerKind
from distance_head_study.train_head import _predict_all_actions, _trajectory_batch
from final_closure.models import serialize_lewm_config
from hdwm.config import LEWMCNNConfig, ProcgenMazeConfig
from scripts.train.train_dim256 import Unisize256


def test_sampled_history_actions_align_with_state_transitions() -> None:
    transitions = np.tile(np.arange(3)[:, None], (1, 5))
    transitions[0, 1] = 1
    transitions[1, 1] = 2
    transitions[1, 2] = 0
    transitions[2, 2] = 1
    indices, actions = _history_for_source(
        transitions, source=2, rng=np.random.default_rng(11)
    )
    assert int(transitions[indices[0], actions[1]]) == int(indices[1])
    assert int(transitions[indices[1], actions[2]]) == int(indices[2])
    assert int(indices[2]) == 2


class _ShapeCheckingPredictor(nn.Module):
    def forward(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        assert embeddings.shape[1] == 3
        assert actions.shape == (embeddings.shape[0], 2)
        output = torch.zeros_like(embeddings[:, :2])
        output[:, -1] = embeddings[:, 1] + actions[:, -1:].to(embeddings)
        return output


class _MockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = _ShapeCheckingPredictor()


def test_all_action_prediction_uses_current_state_and_candidate_action(
    synthetic_batch,
) -> None:
    history = torch.zeros_like(synthetic_batch.history_latents)
    history[:, 0] = 10.0
    history[:, 1] = 20.0
    history[:, 2] = 30.0
    batch = replace(synthetic_batch, history_latents=history)
    predicted = _predict_all_actions(_MockModel(), batch, gradients=False)
    for action in range(5):
        assert torch.equal(
            predicted[:, action],
            torch.full_like(predicted[:, action], 30.0 + action),
        )


def test_tiny_lewm_checkpoint_cache_and_predicted_objective_step(
    tmp_path: Path,
    method_catalog,
    decision_root,
) -> None:
    entry = read_jsonl("distance_head_study/manifests/d_cal.jsonl")[0]
    maze_size = int(entry["maze_size"])
    environment = ProcgenMazeConfig(
        height=maze_size,
        width=maze_size,
        observation_channels=5,
        p_noise=0.0,
        p_noop=0.0,
        p_action_turn=0.0,
        p_action_stay=0.0,
        resample_maze_per_sequence=False,
    )
    model_config = LEWMCNNConfig(
        env_config=environment,
        latent_dim=256,
        model_dim=16,
        cnn_channels=(4,),
        predictor_layers=1,
        predictor_heads=4,
        encoder_heads=4,
        latent_batch_norm=True,
        embedding_stage="post_bn",
        sigreg_stage="post_bn",
    )
    source_model = Unisize256(model_config, max_size=31)
    checkpoint = tmp_path / "tiny_lewm.pt"
    torch.save(
        {
            "model_config": serialize_lewm_config(model_config),
            "model_state_dict": source_model.state_dict(),
        },
        checkpoint,
    )
    model, _ = load_backbone_checkpoint(checkpoint, torch.device("cpu"), freeze=True)
    shard_payload = build_topology_shard(
        entry,
        model,
        backbone_path=checkpoint,
        device=torch.device("cpu"),
        encode_batch_size=32,
    )
    shard = tmp_path / "shard.pt"
    torch.save(shard_payload, shard)
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "records": [
                    {
                        "position": 0,
                        "task_hash": entry["task_hash"],
                        "maze_size": maze_size,
                        "path": shard.as_posix(),
                        "sha256": sha256_file(shard),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    batch = sample_training_batch(
        ShardedGoalDataset(index),
        sampler=SamplerKind.UNIFORM,
        effective_batch_size=8,
        pairs_per_topology=4,
        schedule_seed=17,
        backbone_seed=42,
        step=0,
    )
    predicted = _predict_all_actions(model, batch, gradients=False)
    method, _ = resolve_method(
        method_catalog, "c1_predicted_listwise", decision_root=decision_root
    )
    assert method.head is not None and method.objectives is not None
    head = build_distance_head(method.head)
    terms = compute_objective_terms(
        head,
        method,
        batch,
        predicted_next=predicted,
    )
    assert "predicted_listwise" in terms
    weights = {name: float(getattr(method.objectives, name)) for name in terms}
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    total = weighted_total(terms, weights)
    total.backward()
    optimizer.step()
    assert torch.isfinite(total)


def test_legacy_rollout_labels_use_effective_horizon_minus_one(
    synthetic_batch,
) -> None:
    transitions = torch.tensor(
        [
            [0, 1, 0, 0, 0],
            [1, 2, 0, 1, 1],
            [2, 2, 1, 2, 2],
        ]
    )
    shard = {
        "next_indices": transitions,
        "goal_distances": torch.tensor([2, 1, 0]),
    }

    class Dataset:
        @staticmethod
        def get(position: int):
            assert position == 0
            return shard

    batch = replace(
        synthetic_batch,
        topology_positions=torch.zeros(8, dtype=torch.long),
        source_indices=torch.zeros(8, dtype=torch.long),
    )
    candidates = torch.tensor([[[1, 1, 2], [1, 2, 1]]])
    distances = true_candidate_distances(
        Dataset(),
        batch,
        candidates,
        context_indices=torch.tensor([0]),
        executed_action_count=2,
    )
    assert distances.tolist() == [[0.0, 2.0]]


def test_context_indices_are_deterministic_and_spread_across_the_batch() -> None:
    assert evenly_spaced_indices(8, 2).tolist() == [2, 6]
    assert evenly_spaced_indices(8, 8).tolist() == list(range(8))
    with pytest.raises(ValueError, match="context count"):
        evenly_spaced_indices(8, 9)


def test_trm_horizon_counts_executed_actions_not_warmup_slots(
    monkeypatch, synthetic_batch
) -> None:
    observed: dict[str, object] = {}

    class FakeWorldModel:
        def __init__(self, model, *, device, history_size):
            del model, device
            assert history_size == 3

        def rollout(self, context, sequences, *, semantics, gradients):
            del context, semantics, gradients
            observed["rollout_slots"] = int(sequences.shape[1])
            return SimpleNamespace(terminal=torch.zeros(sequences.shape[0], 256))

    def fake_distances(
        dataset,
        batch,
        sequences,
        *,
        context_indices,
        executed_action_count,
    ):
        del dataset, batch
        observed["executed_actions"] = int(executed_action_count)
        observed["context_indices"] = tuple(context_indices.tolist())
        return torch.zeros(sequences.shape[:2])

    monkeypatch.setattr(
        "distance_head_study.train_head.VectorWorldModel", FakeWorldModel
    )
    monkeypatch.setattr(
        "distance_head_study.train_head.true_candidate_distances", fake_distances
    )
    trajectory = _trajectory_batch(
        nn.Identity(),
        object(),
        synthetic_batch,
        torch.ones(64, 12, dtype=torch.long),
        contexts=2,
        horizon=1,
        device=torch.device("cpu"),
        gradients=False,
    )
    assert observed == {
        "rollout_slots": 2,
        "executed_actions": 1,
        "context_indices": (2, 6),
    }
    assert trajectory.horizon == 2
    assert torch.equal(
        trajectory.max_distance,
        synthetic_batch.max_distance.index_select(0, torch.tensor([2, 6])),
    )


def _write_synthetic_cache(tmp_path: Path) -> Path:
    latent_dim = 256
    state_count = 6
    latents = torch.arange(state_count * latent_dim, dtype=torch.float32).reshape(
        state_count, latent_dim
    )
    next_indices = torch.arange(state_count)[:, None].repeat(1, 5)
    next_indices[:, 1] = torch.clamp(torch.arange(state_count) + 1, max=5)
    next_indices[:, 2] = torch.clamp(torch.arange(state_count) - 1, min=0)
    valid = next_indices != torch.arange(state_count)[:, None]
    valid[:, 0] = False
    optimal = torch.zeros_like(valid)
    optimal[:-1, 1] = True
    # The environment permits leaving the goal; supervised shortest paths must
    # nevertheless treat goal arrival as absorbing.
    optimal[-1, 2] = True
    payload = {
        "metadata": {
            "task_hash": "task-0",
            "goal_index": 5,
            "max_goal_distance": 5,
        },
        "latents": latents,
        "goal_distances": torch.tensor([5, 4, 3, 2, 1, 0]),
        "all_pairs_bfs": torch.tensor(
            [[abs(left - right) for right in range(6)] for left in range(6)]
        ),
        "next_indices": next_indices,
        "valid_actions": valid,
        "optimal_actions": optimal,
    }
    shard = tmp_path / "shard.pt"
    torch.save(payload, shard)
    index = {
        "schema": CACHE_SCHEMA,
        "records": [
            {
                "position": 0,
                "task_hash": "task-0",
                "maze_size": 11,
                "path": shard.as_posix(),
                "sha256": sha256_file(shard),
            }
        ],
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "sampler",
    [
        SamplerKind.UNIFORM,
        SamplerKind.DISTANCE_BALANCED,
        SamplerKind.DECISION_BALANCED,
        SamplerKind.FULL_HORIZON,
    ],
)
def test_training_sampler_is_deterministic(
    tmp_path: Path, sampler: SamplerKind
) -> None:
    dataset = ShardedGoalDataset(_write_synthetic_cache(tmp_path))
    kwargs = {
        "sampler": sampler,
        "effective_batch_size": 8,
        "pairs_per_topology": 4,
        "schedule_seed": 123,
        "backbone_seed": 42,
        "step": 9,
    }
    first = sample_training_batch(dataset, **kwargs)
    second = sample_training_batch(dataset, **kwargs)
    for name, value in first.__dict__.items():
        other = getattr(second, name)
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), name
        else:
            assert value == other


def test_hard_crossfit_cannot_run_without_signed_mining_artifact(
    tmp_path: Path,
) -> None:
    dataset = ShardedGoalDataset(_write_synthetic_cache(tmp_path))
    with pytest.raises(ValueError, match="signed mining artifact"):
        sample_training_batch(
            dataset,
            sampler=SamplerKind.HARD_CROSSFIT,
            effective_batch_size=8,
            pairs_per_topology=4,
            schedule_seed=1,
            backbone_seed=42,
            step=0,
        )


def test_full_horizon_sampler_is_not_the_distance_bin_arm() -> None:
    shard = {"goal_distances": torch.arange(25, 0, -1)}
    distance_balanced = _sample_sources(
        shard,
        120,
        SamplerKind.DISTANCE_BALANCED,
        np.random.default_rng(17),
    )
    full_horizon = _sample_sources(
        shard,
        120,
        SamplerKind.FULL_HORIZON,
        np.random.default_rng(17),
    )
    assert not np.array_equal(distance_balanced, full_horizon)
    full_distances = shard["goal_distances"].numpy()[full_horizon]
    horizon_strata = np.digitize(full_distances, (1, 3, 5, 8, 12), right=True)
    counts = np.bincount(horizon_strata, minlength=6)
    assert counts.tolist() == [20] * 6


def test_shortest_path_prefix_stays_at_goal_after_arrival(tmp_path: Path) -> None:
    dataset = ShardedGoalDataset(_write_synthetic_cache(tmp_path))
    batch = sample_training_batch(
        dataset,
        sampler=SamplerKind.UNIFORM,
        effective_batch_size=8,
        pairs_per_topology=4,
        schedule_seed=9,
        backbone_seed=42,
        step=0,
    )
    for distances in batch.path_distances.tolist():
        first_goal = distances.index(0.0)
        assert distances[first_goal:] == [0.0] * (len(distances) - first_goal)
