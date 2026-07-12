#!/usr/bin/env python3
"""Fast CPU integration test across data, training loss, and both controllers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from final_closure.audit_protocol import audit_config
from final_closure.common import (
    corrected_actions,
    load_config,
    read_jsonl,
    set_seed,
    validate_manifest_entry,
)
from final_closure.data import build_bc_dataset, render_bc_batch, sample_lewm_sequence
from final_closure.evaluate import BCController, LeWMCEMController, run_episode
from final_closure.models import BCPolicyConfig, DeepCNNPolicy, build_lewm
from final_closure.train import lewm_loss
from hdwm.losses import SIGReg


def main() -> None:
    set_seed(123, deterministic=True)
    config, _ = load_config("final_closure/configs/default.json")
    audit_config(config)
    entries = read_jsonl(config["paths"]["train_manifest"])
    small_entries = entries[:2]
    dataset = build_bc_dataset(small_entries)
    observations, labels = render_bc_batch(
        dataset,
        np.asarray([0, dataset.sample_count - 1]),
        canvas_size=21,
    )
    assert observations.shape == (2, 5, 21, 21)
    assert labels.shape == (2,)
    bc = DeepCNNPolicy(BCPolicyConfig(dropout=0.3, action_count=4))
    logits = bc(observations)
    assert logits.shape == (2, 4)
    bc_loss = F.cross_entropy(logits, labels)
    bc_loss.backward()
    assert torch.isfinite(bc_loss)
    assert any(parameter.grad is not None for parameter in bc.parameters())

    lewm_baseline = config["baselines"][1]
    lewm, _ = build_lewm(lewm_baseline["train"])
    sequence = sample_lewm_sequence(
        small_entries[0],
        rng=np.random.default_rng(123),
        batch_size=2,
        sequence_length=2,
    )
    total, metrics = lewm_loss(
        lewm,
        SIGReg(knots=5, num_proj=8),
        sequence,
        maze_size=int(small_entries[0]["maze_size"]),
        device=torch.device("cpu"),
        weights={
            "prediction": 1.0,
            "sigreg": 0.09,
            "absolute": 0.1,
            "relative": 1.0,
            "goal": 0.5,
        },
    )
    total.backward()
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in metrics.values())

    entry = entries[0]
    env = validate_manifest_entry(entry)
    start = int(entry["start_cell"])
    previous = None
    assert corrected_actions(env, start, previous)
    bc.eval()
    controller = BCController(
        bc,
        device=torch.device("cpu"),
        canvas_size=21,
        action_selection="unmasked",
    )
    row = run_episode(entry, controller, task_index=0, max_steps=3)
    assert row["path_length"] <= 3
    assert 0.0 <= row["spl"] <= 1.0
    assert row["task_id"] == entry["task_hash"]

    smoke_planner = {
        **lewm_baseline["planner"],
        "horizon": 2,
        "num_candidates": 8,
        "num_elites": 2,
    }
    lewm.eval()
    cem = LeWMCEMController(
        lewm,
        smoke_planner,
        device=torch.device("cpu"),
        evaluation_seed=42,
        action_selection="unmasked",
    )
    observation = env.reset(options={"start_state": start})[0]
    cem.reset(env, observation, task_index=0)
    action, action_metrics = cem.choose(env, observation, start, None)
    assert action in (1, 2, 3, 4)
    assert action_metrics["cem_calls"] == 1.0
    print("final_closure smoke test passed")


if __name__ == "__main__":
    main()
