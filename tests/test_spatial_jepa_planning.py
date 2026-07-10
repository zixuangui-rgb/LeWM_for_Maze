from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from spatial_jepa_planning import EXPERIMENT_FAMILY, FORMAT_VERSION
from spatial_jepa_planning.audit_protocol import audit_config
from spatial_jepa_planning.common import (
    ACTION_IDS,
    ACTION_TO_SLOT,
    build_map_targets,
    next_state,
    read_jsonl,
    resolve_device,
    save_checkpoint,
    sha256_file,
    validate_manifest_entry,
)
from spatial_jepa_planning.losses import PlannerLossWeights, planner_loss
from spatial_jepa_planning.models import (
    OracleValueIteration,
    PlannerConfig,
    PlannerOutput,
    SpatialRepresentation,
    SpatialRepresentationConfig,
    build_planner,
    make_ema_target,
    neighbor_stack,
    update_ema_target,
)
from spatial_jepa_planning.summarize import validate_metadata, validate_task_rows

ROOT = Path(__file__).resolve().parents[1]


def test_neighbor_stack_action_order() -> None:
    values = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    neighbors = neighbor_stack(values, fill=99.0)
    assert neighbors[0, :, 1, 1].tolist() == [1.0, 7.0, 3.0, 5.0]


def test_auto_device_resolves_to_available_backend() -> None:
    device = resolve_device("auto")
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_summary_rejects_sampled_or_duplicate_task_results() -> None:
    metadata = {
        "eval_manifest_sha256": "locked",
        "max_steps": 128,
        "seed": 42,
        "task_count": 2,
        "max_per_size": 0,
        "limit": 0,
        "action_selection": "corrected",
        "mode": "learned",
        "recompute_every_step": False,
        "comparable_to_full900": True,
    }
    validate_metadata(
        {"metadata": metadata},
        expected_eval_hash="locked",
        max_steps=128,
        expected_task_count=2,
        evaluation_seed=42,
    )
    validate_task_rows(
        {"task_rows": [{"task_id": "a"}, {"task_id": "b"}]},
        expected_task_count=2,
    )
    with pytest.raises(ValueError, match="evaluation seed"):
        validate_metadata(
            {"metadata": {**metadata, "seed": 43}},
            expected_eval_hash="locked",
            max_steps=128,
            expected_task_count=2,
            evaluation_seed=42,
        )
    with pytest.raises(ValueError, match="unique task IDs"):
        validate_task_rows(
            {"task_rows": [{"task_id": "a"}, {"task_id": "a"}]},
            expected_task_count=2,
        )


def test_map_targets_match_environment_transitions() -> None:
    entry = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")[0]
    env = validate_manifest_entry(entry)
    targets = build_map_targets(env, torch.device("cpu"))
    distances = targets["distance"].numpy()
    free = np.flatnonzero((~env._maze_mask).reshape(-1))
    for state in free.tolist():
        row, col = divmod(state, env.config.width)
        for action in ACTION_IDS:
            slot = ACTION_TO_SLOT[action]
            candidate = next_state(env, state, action)
            expected_valid = candidate != state
            assert bool(targets["valid_action_mask"][slot, row, col]) == expected_valid
            if bool(targets["optimal_action_mask"][slot, row, col]):
                assert (
                    distances.reshape(-1)[candidate] == distances.reshape(-1)[state] - 1
                )


def test_spatial_representation_shapes_and_ema() -> None:
    config = SpatialRepresentationConfig(
        spatial_dim=8,
        planning_dim=12,
        encoder_blocks=1,
        predictor_blocks=1,
    )
    online = SpatialRepresentation(config)
    target = make_ema_target(online)
    observations = torch.randn(2, 3, 9, 9, 5)
    dynamics = online.dynamics_latent(observations)
    planning = online.planning_latent(observations)
    assert dynamics.shape == (2, 3, 8, 9, 9)
    assert planning.shape == (2, 3, 12, 9, 9)
    with torch.no_grad():
        next(online.parameters()).add_(1.0)
    before = next(target.parameters()).clone()
    update_ema_target(online, target, momentum=0.5)
    assert not torch.equal(before, next(target.parameters()))
    assert all(not parameter.requires_grad for parameter in target.parameters())


def test_planner_loss_backpropagates_for_both_architectures() -> None:
    entry = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")[1]
    env = validate_manifest_entry(entry)
    observation = torch.as_tensor(
        __import__("diagnostics.common", fromlist=["observe_state"]).observe_state(
            env, int(entry["start_cell"])
        ),
        dtype=torch.float32,
    ).unsqueeze(0)
    targets = {
        name: value.unsqueeze(0)
        for name, value in build_map_targets(env, torch.device("cpu")).items()
    }
    features = observation.permute(0, 3, 1, 2)
    for planner_type in ("feedforward", "feedforward_dilated", "iterative"):
        planner = build_planner(
            PlannerConfig(
                input_channels=5,
                hidden_dim=8,
                planner_type=planner_type,
                depth=2,
            )
        )
        outputs = planner(features, iterations=3)
        loss, metrics = planner_loss(
            outputs,
            targets,
            PlannerLossWeights(),
            distance_scale=128.0,
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(metrics["action"])
        loss.backward()
        assert any(parameter.grad is not None for parameter in planner.parameters())


def test_iteration_budget_masks_unreachable_far_cells() -> None:
    entry = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")[3]
    env = validate_manifest_entry(entry)
    targets = {
        name: value.unsqueeze(0)
        for name, value in build_map_targets(env, torch.device("cpu")).items()
    }
    height = width = int(entry["maze_size"])
    base_value = torch.zeros((1, height, width))
    far = torch.nonzero(targets["distance"][0] > 1, as_tuple=False)[0]
    changed_value = base_value.clone()
    changed_value[0, far[0], far[1]] = 100.0
    logits = torch.zeros((1, 4, height, width))
    first = PlannerOutput(base_value, logits, logits, logits[:, :1], 1)
    second = PlannerOutput(changed_value, logits, logits, logits[:, :1], 1)
    weights = PlannerLossWeights(
        value=1.0,
        action=0.0,
        valid=0.0,
        bellman=0.0,
        gap=0.0,
    )
    first_loss, _ = planner_loss(
        [first], targets, weights, distance_scale=128.0, iteration_budgeted=True
    )
    second_loss, _ = planner_loss(
        [second], targets, weights, distance_scale=128.0, iteration_budgeted=True
    )
    assert torch.equal(first_loss, second_loss)


def test_budgeted_bellman_ignores_unresolved_frontier() -> None:
    entry = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")[3]
    env = validate_manifest_entry(entry)
    targets = {
        name: value.unsqueeze(0)
        for name, value in build_map_targets(env, torch.device("cpu")).items()
    }
    distance = targets["distance"]
    first_value = distance.clone()
    second_value = distance.clone()
    second_value[distance > 2] = 0.0
    height = width = int(entry["maze_size"])
    logits = torch.zeros((1, 4, height, width))
    first = PlannerOutput(first_value, logits, logits, logits[:, :1], 2)
    second = PlannerOutput(second_value, logits, logits, logits[:, :1], 2)
    weights = PlannerLossWeights(
        value=0.0,
        action=0.0,
        valid=0.0,
        bellman=1.0,
        gap=0.0,
    )
    first_loss, _ = planner_loss(
        [first], targets, weights, distance_scale=128.0, iteration_budgeted=True
    )
    second_loss, _ = planner_loss(
        [second], targets, weights, distance_scale=128.0, iteration_budgeted=True
    )
    assert torch.equal(first_loss, second_loss)


def test_oracle_vi_recovers_bfs_distance() -> None:
    entry = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")[2]
    env = validate_manifest_entry(entry)
    targets = build_map_targets(env, torch.device("cpu"))
    output = OracleValueIteration()(
        ~targets["free_mask"].unsqueeze(0),
        targets["goal_mask"].bool().unsqueeze(0),
        iterations=entry["maze_size"] ** 2,
    )
    free = targets["free_mask"]
    assert torch.equal(output.value[0][free], targets["distance"][free])


def test_default_config_passes_alignment_audit() -> None:
    with open(ROOT / "spatial_jepa_planning/configs/default.json") as stream:
        config = json.load(stream)
    result = audit_config(config)
    assert len(result["seeds"]) >= 3
    assert "r4_raw_iterative_progressive" in result["planners"]


def test_runner_dry_run_builds_commands() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "spatial_jepa_planning/run_plan.py",
            "--stages",
            "full",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [
        shlex.split(line[2:])
        for line in completed.stdout.splitlines()
        if line.startswith("$ ")
    ]
    outputs = [
        command[command.index("--output") + 1]
        for command in commands
        if "--output" in command
    ]
    assert len(commands) == 70
    assert len(outputs) == len(set(outputs)) == 70
    labelled_evaluations = [
        command
        for command in commands
        if command[1].endswith("evaluate.py") and "--training-seed" in command
    ]
    assert len(labelled_evaluations) == 33
    assert all(
        command[command.index("--seed") + 1] == "42" for command in labelled_evaluations
    )
    spatial_train_commands = [
        command
        for command in commands
        if command[1].endswith("train.py") and "--representation-ckpt" in command
    ]
    assert len(spatial_train_commands) == 12
    for command in spatial_train_commands:
        seed = command[command.index("--seed") + 1]
        checkpoint = command[command.index("--representation-ckpt") + 1]
        assert checkpoint.endswith(f"spatial_info_sigreg_seed{seed}.pt")


def test_runner_rejects_unknown_variant_and_duplicate_seed() -> None:
    for extra_args, message in (
        (["--variants", "typo"], "unknown or disabled variants"),
        (["--seeds", "42,42"], "contains duplicates"),
    ):
        completed = subprocess.run(
            [
                sys.executable,
                "spatial_jepa_planning/run_plan.py",
                "--stages",
                "anchors",
                "--dry-run",
                *extra_args,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert message in completed.stderr


def test_oracle_bfs_cli_preserves_fixed_task_count(tmp_path: Path) -> None:
    output = tmp_path / "oracle.json"
    subprocess.run(
        [
            sys.executable,
            "spatial_jepa_planning/evaluate.py",
            "--mode",
            "oracle_bfs",
            "--output",
            str(output),
            "--limit",
            "3",
            "--progress-every",
            "0",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(output.read_text())
    assert data["metadata"]["task_count"] == 3
    assert len(data["results"]["task_rows"]) == 3
    assert data["results"]["navigation"]["overall"]["sr"] == 1.0


def test_learned_evaluator_cli_loads_new_checkpoint(tmp_path: Path) -> None:
    config = PlannerConfig(
        input_channels=5,
        hidden_dim=8,
        planner_type="feedforward_dilated",
        depth=2,
    )
    planner = build_planner(config)
    checkpoint = tmp_path / "planner.pt"
    eval_manifest = ROOT / "data/splits/unisize_eval_manifest.jsonl"
    save_checkpoint(
        checkpoint,
        {
            "experiment_family": EXPERIMENT_FAMILY,
            "format_version": FORMAT_VERSION,
            "stage": "planner",
            "input_mode": "raw",
            "planner_config": config.to_dict(),
            "planner_state_dict": planner.state_dict(),
            "protocol": {
                "seed": 42,
                "eval_manifest_sha256": sha256_file(eval_manifest),
            },
        },
    )
    output = tmp_path / "learned.json"
    subprocess.run(
        [
            sys.executable,
            "spatial_jepa_planning/evaluate.py",
            "--mode",
            "learned",
            "--planner-ckpt",
            str(checkpoint),
            "--output",
            str(output),
            "--iterations",
            "2",
            "--limit",
            "2",
            "--max-steps",
            "4",
            "--progress-every",
            "0",
            "--training-seed",
            "42",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(output.read_text())
    assert data["metadata"]["training_seed"] == 42
    assert "2" in data["results"]
    assert data["results"]["2"]["field"]["overall"]["n_states"] > 0
