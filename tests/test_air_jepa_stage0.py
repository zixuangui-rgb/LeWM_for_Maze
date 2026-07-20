from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from air_jepa.stage0_workspace import EXPERIMENT_ID, FORMAT_VERSION
from air_jepa.stage0_workspace.audit_protocol import (
    FORMAL_PAIRING_BATCHES,
    pairing_audit,
)
from air_jepa.stage0_workspace.checkpoints import (
    load_air_checkpoint,
    save_air_checkpoint,
    validate_source_planner_payload,
    validate_source_representation_payload,
)
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    load_config,
    package_files,
    read_jsonl,
    relative_path,
    require_h800_device,
    runtime_metadata,
    sha256_file,
    signed_payload,
    state_dict_sha256,
)
from air_jepa.stage0_workspace.data import (
    LOCKED_TRAIN_COUNTS,
    make_rng_streams,
    progressive_iteration_signature,
    require_balanced_training_manifest,
    sample_training_batch,
    select_progressive_iterations,
)
from air_jepa.stage0_workspace.diagnose import (
    deterministic_states,
    summarize_state_rows,
)
from air_jepa.stage0_workspace.evaluate import (
    _validate_request,
    candidate_actions,
    classify_failure,
    run_navigation_with_diagnostics,
)
from air_jepa.stage0_workspace.evaluate_oracle import build_oracle_action_fn
from air_jepa.stage0_workspace.generate_manifests import (
    SIZES,
    generate_all,
    validate_generated,
)
from air_jepa.stage0_workspace.losses import (
    air_loss,
    deep_supervision_weights,
    distributional_cost_loss,
    future_prediction_loss,
    tie_aware_action_loss,
)
from air_jepa.stage0_workspace.models import (
    AIRWorkspaceModel,
    LocalNeighborhoodAttention,
)
from air_jepa.stage0_workspace.plan_jobs import (
    build_jobs,
    scientific_matrix_from_jobs,
    validate_jobs,
)
from air_jepa.stage0_workspace.protocol import (
    build_protocol_payload,
    expected_matrix,
    require_role_allowed,
)
from air_jepa.stage0_workspace.run_jobs import validate_job_plan_payload
from air_jepa.stage0_workspace.schemas import Stage0Config
from air_jepa.stage0_workspace.summarize import (
    ArtifactLoader,
    _spearman,
    _validate_diagnostic_ranking,
    compute_accounting,
    crossed_bootstrap_difference,
    diagnostic_summary,
    paired_checkpoint_audit,
    require_benchmark,
    require_bridge_audit,
    require_protocol_audit,
)
from air_jepa.stage0_workspace.train import ChannelMoments
from diagnostics.common import (
    ACTION_IDS,
    bfs_distances_from,
    next_state,
    observe_state,
)
from spatial_jepa_planning.common import (
    ManifestSampler,
    summarize_rows,
    validate_manifest_entry,
)
from spatial_jepa_planning.models import (
    SpatialRepresentation,
    SpatialRepresentationConfig,
)


@pytest.fixture(scope="module")
def config() -> Stage0Config:
    return load_config()


def _formal_runtime() -> dict[str, object]:
    runtime = runtime_metadata()
    runtime.update(
        {
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
            "python_hash_seed": "0",
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_device_name": "NVIDIA H800",
            "cuda_device_capability": [9, 0],
        }
    )
    return runtime


def test_config_rejects_scientific_budget_changes(config: Stage0Config) -> None:
    payload = config.model_dump(mode="json", by_alias=True)
    changed = copy.deepcopy(payload)
    changed["training"]["batch_size"] = 16
    with pytest.raises(ValidationError, match="batch_size"):
        Stage0Config.model_validate(changed)
    changed = copy.deepcopy(payload)
    changed["model"]["hidden_dim"] = 128
    with pytest.raises(ValidationError, match="hidden_dim"):
        Stage0Config.model_validate(changed)
    changed = copy.deepcopy(payload)
    changed["gates"]["green_overall_mean"] = 0.85
    with pytest.raises(ValidationError, match="green_overall_mean"):
        Stage0Config.model_validate(changed)
    changed = copy.deepcopy(payload)
    changed["statistics"]["familywise_alpha"] = 0.10
    with pytest.raises(ValidationError, match="familywise_alpha"):
        Stage0Config.model_validate(changed)


def test_package_fingerprint_covers_transitive_runtime_dependencies() -> None:
    covered = {relative_path(path) for path in package_files()}
    required = {
        "uv.lock",
        "pyproject.toml",
        "tests/test_air_jepa_stage0.py",
        "hdwm/config.py",
        "hdwm/envs/action_utils.py",
        "hdwm/models/lewm.py",
        "scripts/train/train_dim256.py",
    }
    assert required <= covered


def test_formal_device_gate_rejects_cpu() -> None:
    with pytest.raises(RuntimeError, match="requires an NVIDIA H800"):
        require_h800_device(torch.device("cpu"))


def test_model_has_locked_readouts_and_shared_parameters(config: Stage0Config) -> None:
    torch.manual_seed(1)
    model = AIRWorkspaceModel(config.model)
    latent = torch.randn(1, 64, 9, 9)
    outputs = model(latent, iterations=32, deep_supervision_every=16)
    assert [output.iterations for output in outputs] == [16, 32]
    assert outputs[-1].energy.shape == (1, 4)
    assert outputs[-1].cost_logits.shape == (1, 4, 129)
    assert outputs[-1].predicted_future is not None
    parameter_ids_before = {id(parameter) for parameter in model.reasoner.parameters()}
    model(latent, iterations=4)
    parameter_ids_after = {id(parameter) for parameter in model.reasoner.parameters()}
    assert parameter_ids_before == parameter_ids_after
    assert model.analytical_macs(25, 128) > model.analytical_macs(25, 16)
    one = model.analytical_mac_breakdown(25, 1)
    four = model.analytical_mac_breakdown(25, 4)
    assert model.analytical_macs(25, 4) == sum(four.values())
    assert four["shared_reasoner"] == 4 * one["shared_reasoner"]
    for component in ("adapter_and_goal", "future_decoder", "energy_head"):
        assert four[component] == one[component]


def test_energy_head_has_no_token_only_classifier_bypass(
    config: Stage0Config,
) -> None:
    torch.manual_seed(9)
    model = AIRWorkspaceModel(config.model).eval()
    fields = torch.zeros(2, 4, 64, 9, 9)
    mask = torch.ones(2, 9, 9, dtype=torch.bool)
    goal = torch.randn(2, 1, 64)
    actions = torch.randn(2, 4, 64)
    with torch.no_grad():
        first, _ = model.energy_head(fields, goal, actions, mask)
        second, _ = model.energy_head(
            fields,
            goal * 7.0 + 3.0,
            actions * -5.0,
            mask,
        )
    assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)


def test_local_attention_masks_padding_and_stays_finite() -> None:
    module = LocalNeighborhoodAttention(hidden_dim=8, heads=2)
    inputs = torch.randn(2, 8, 5, 5)
    mask = torch.ones(2, 5, 5, dtype=torch.bool)
    mask[:, -1] = False
    output = module(inputs, mask)
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, :, -1], torch.zeros_like(output[:, :, -1]))


def test_tie_action_and_distributional_cost_losses() -> None:
    energy = torch.tensor([[1.0, 1.0, 3.0, 4.0]], requires_grad=True)
    optimal = torch.tensor([[True, True, False, False]])
    tied = tie_aware_action_loss(energy, optimal)
    swapped = tie_aware_action_loss(energy[:, [1, 0, 2, 3]], optimal)
    assert torch.allclose(tied, swapped)
    logits = torch.zeros(1, 4, 129)
    distance = torch.tensor([[0, 12, 128, 999]])
    loss = distributional_cost_loss(logits, distance, max_distance=128)
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_future_loss_explicitly_compares_copy_current() -> None:
    source = torch.zeros(2, 3, 4, 4)
    target = torch.randn(2, 4, 3, 4, 4)
    mask = torch.ones(2, 4, 4, dtype=torch.bool)
    perfect = future_prediction_loss(
        target,
        target,
        source,
        valid_mask=mask,
        epsilon=1e-4,
    )
    copy = future_prediction_loss(
        source[:, None].expand_as(target),
        target,
        source,
        valid_mask=mask,
        epsilon=1e-4,
    )
    assert float(perfect.total) == pytest.approx(0.0)
    assert float(copy.normalized_delta) == pytest.approx(
        float(copy.copy_delta_normalized)
    )


def test_channel_moments_match_exact_population_statistics() -> None:
    values = torch.tensor(
        [
            [
                [[[1.0, 3.0]], [[2.0, 6.0]]],
                [[[5.0, 7.0]], [[4.0, 8.0]]],
                [[[9.0, 11.0]], [[6.0, 10.0]]],
                [[[13.0, 15.0]], [[8.0, 12.0]]],
            ]
        ]
    )
    moments = ChannelMoments(channels=2)
    moments.update(values)
    summary = moments.summary()
    flattened = values.permute(2, 0, 1, 3, 4).reshape(2, -1).double()
    assert summary["count_per_channel"] == 8
    assert summary["mean"] == pytest.approx(flattened.mean(dim=1).tolist())
    assert summary["variance"] == pytest.approx(
        flattened.var(dim=1, unbiased=False).tolist()
    )


def test_air_loss_backpropagates_for_both_matched_methods(config: Stage0Config) -> None:
    latent = torch.randn(2, 64, 9, 9)
    target = torch.randn(2, 4, 64, 9, 9)
    distances = torch.tensor([[2, 1, 3, 2], [4, 5, 3, 3]])
    optimal = distances == distances.min(dim=1, keepdim=True).values
    mask = torch.ones(2, 9, 9, dtype=torch.bool)
    initial_states = []
    for method in ("air0_direct", "air0_jepa"):
        torch.manual_seed(91)
        model = AIRWorkspaceModel(config.model)
        initial_states.append(state_dict_sha256(model.state_dict()))
        outputs = model(latent, iterations=4, valid_mask=mask)
        result = air_loss(
            outputs,
            successor_latent=target,
            source_latent=latent,
            candidate_distances=distances,
            optimal_action_mask=optimal,
            valid_mask=mask,
            weights=config.training.methods[method],
            max_distance=128,
            target_variance_epsilon=1e-4,
        )
        result.total.backward()
        assert torch.isfinite(result.total)
        assert any(parameter.grad is not None for parameter in model.parameters())
    assert initial_states[0] == initial_states[1]


def test_deep_supervision_weights_are_k_over_total(config: Stage0Config) -> None:
    model = AIRWorkspaceModel(config.model)
    outputs = model(
        torch.randn(1, 64, 9, 9),
        iterations=48,
        deep_supervision_every=16,
    )
    weights = deep_supervision_weights(outputs)
    assert [output.iterations for output in outputs] == [16, 32, 48]
    assert torch.allclose(weights, torch.tensor([1 / 6, 2 / 6, 3 / 6]))


def test_exact_successor_data_and_paired_rng(config: Stage0Config) -> None:
    entries = read_jsonl(config.paths.train_manifest)
    require_balanced_training_manifest(entries)
    first_streams = make_rng_streams(42)
    second_streams = make_rng_streams(42)
    sampler = ManifestSampler(entries)
    first = sample_training_batch(
        sampler,
        entry_rng=first_streams.entries,
        state_rng=first_streams.states,
        batch_size=8,
        device=torch.device("cpu"),
    )
    second = sample_training_batch(
        sampler,
        entry_rng=second_streams.entries,
        state_rng=second_streams.states,
        batch_size=8,
        device=torch.device("cpu"),
    )
    assert first.task_ids == second.task_ids
    assert torch.equal(first.current_states, second.current_states)
    assert torch.equal(first.successor_states, second.successor_states)
    for index, task_id in enumerate(first.task_ids):
        entry = next(entry for entry in entries if entry["task_hash"] == task_id)
        env = validate_manifest_entry(entry, check_bfs=False)
        for slot, action in enumerate(ACTION_IDS):
            assert int(first.successor_states[index, slot]) == next_state(
                env, int(first.current_states[index]), action
            )


def test_progressive_schedule_unlocks_only_prefix(config: Stage0Config) -> None:
    rng = np.random.default_rng(3)
    assert {
        select_progressive_iterations(
            step=step,
            phase_steps=config.training.phase_steps,
            k_train=config.training.k_train,
            rng=rng,
        )
        for step in range(1, 100)
    } == {4}
    rng = np.random.default_rng(4)
    values = {
        select_progressive_iterations(
            step=5001,
            phase_steps=config.training.phase_steps,
            k_train=config.training.k_train,
            rng=rng,
        )
        for _ in range(100)
    }
    assert values == {4, 8}


def test_generated_roles_have_locked_counts_and_are_reproducible(
    config: Stage0Config,
) -> None:
    generated = generate_all(config)
    validate_generated(generated)
    train_counts = Counter(
        entry["maze_size"] for entry in read_jsonl(config.paths.train_manifest)
    )
    assert train_counts == Counter(LOCKED_TRAIN_COUNTS)
    assert len(generated["air_early"]) == 210
    for role in ("air_dev", "air_select", "air_final"):
        assert Counter(entry["maze_size"] for entry in generated[role]) == Counter(
            {size: 100 for size in SIZES}
        )
        on_disk = read_jsonl(getattr(config.paths, f"{role}_manifest"))
        assert generated[role] == on_disk


def test_protocol_rebuild_and_sealed_roles(config: Stage0Config) -> None:
    protocol = build_protocol_payload(config)
    assert protocol["manifests"]["air_dev"]["rows"] == 900
    assert protocol["matrix"] == expected_matrix(config)
    counts = {key: len(value) for key, value in protocol["matrix"].items()}
    assert counts["historical_bridges"] == 6
    assert counts["air_early_context"] == 4
    assert counts["air_early_interventions"] == 30
    assert counts["air_early_diagnostics"] == 1
    assert sum(counts.values()) == 135
    require_role_allowed("air_dev")
    for role in ("air_select", "air_final"):
        with pytest.raises(PermissionError, match="sealed"):
            require_role_allowed(role)


def test_job_dag_contains_complete_locked_matrix(config: Stage0Config) -> None:
    jobs = build_jobs(config, "config.json")
    validate_jobs(jobs, config)
    executable = scientific_matrix_from_jobs(jobs, config)
    for key, rows in expected_matrix(config).items():
        assert Counter(json.dumps(row, sort_keys=True) for row in executable[key]) == (
            Counter(json.dumps(row, sort_keys=True) for row in rows)
        )
    ids = {job.job_id for job in jobs}
    assert len(jobs) == len(ids)
    assert sum(job.job_id.startswith("train_air0_") for job in jobs) == 6
    assert sum(job.job_id.startswith("l2_primary_") for job in jobs) == 9
    assert sum(job.job_id.startswith("l3_curve_") for job in jobs) == 54
    assert sum(job.job_id == "l3_oracle_bfs" for job in jobs) == 1
    assert len(jobs) == 143
    assert jobs[-1].job_id == "l3_release"
    priorities = {job.job_id: job.priority for job in jobs}
    assert priorities["l0_protocol_audit"] < priorities["train_air0_direct_s42"]
    assert priorities["train_air0_direct_s42"] < priorities["l1_release"]
    assert priorities["l1_release"] < priorities["train_air0_direct_s44"]
    assert priorities["train_air0_direct_s44"] < priorities["l2_release"]
    assert priorities["l2_release"] < priorities["l3_release"]

    changed = list(jobs)
    index = next(
        i
        for i, job in enumerate(changed)
        if job.job_id == "l2_primary_air0_jepa_s42_k128"
    )
    command = list(changed[index].command)
    method_index = command.index("--method") + 1
    command[method_index] = "air0_direct"
    changed[index] = replace(changed[index], command=tuple(command))
    with pytest.raises(ValueError, match="scientific cells differ"):
        validate_jobs(changed, config)


def test_runner_rebuilds_and_rejects_tampered_job_plan(config: Stage0Config) -> None:
    config_path = str(DEFAULT_CONFIG)
    jobs = build_jobs(config, config_path)
    payload = signed_payload(
        {
            "schema": "air-jepa-stage0-job-plan-v1",
            "experiment_id": config.experiment_id,
            "score_independent": True,
            "automatic_continue_after_quicklooks": True,
            "job_count": len(jobs),
            "protocol_sha256": "protocol",
            "package_sha256": "package",
            "source_lock_sha256": "source",
            "git_commit": "test-commit",
            "code_fingerprint": "code",
            "runtime": runtime_metadata(),
            "jobs": [job.as_dict() for job in jobs],
        },
        "job_plan_sha256",
    )
    loaded = validate_job_plan_payload(
        payload,
        config=config,
        config_path=config_path,
        protocol_sha256="protocol",
        package_sha256="package",
        package_code_fingerprint="code",
        source_lock_sha256="source",
    )
    assert len(loaded) == 143

    tampered = copy.deepcopy(payload)
    tampered.pop("job_plan_sha256")
    tampered["jobs"][0]["priority"] = 999
    with pytest.raises(ValueError, match="executable locked DAG"):
        validate_job_plan_payload(
            signed_payload(tampered, "job_plan_sha256"),
            config=config,
            config_path=config_path,
            protocol_sha256="protocol",
            package_sha256="package",
            package_code_fingerprint="code",
            source_lock_sha256="source",
        )

    wrong_identity = copy.deepcopy(payload)
    wrong_identity.pop("job_plan_sha256")
    wrong_identity["experiment_id"] = "wrong-experiment"
    with pytest.raises(ValueError, match="identity/provenance"):
        validate_job_plan_payload(
            signed_payload(wrong_identity, "job_plan_sha256"),
            config=config,
            config_path=config_path,
            protocol_sha256="protocol",
            package_sha256="package",
            package_code_fingerprint="code",
            source_lock_sha256="source",
        )


def test_formal_evaluator_rejects_cells_outside_locked_matrix(
    config: Stage0Config,
) -> None:
    valid = argparse.Namespace(
        seed=42,
        k=16,
        method="air0_direct",
        intervention="normal",
        split_role="air_early",
        action_protocol="unmasked",
    )
    _validate_request(valid, config, formal=True)
    invalid = copy.copy(valid)
    invalid.seed = 43
    with pytest.raises(ValueError, match="absent from the protocol matrix"):
        _validate_request(invalid, config, formal=True)
    preflight = copy.copy(valid)
    preflight.split_role = "preflight"
    with pytest.raises(ValueError, match="not part of the locked matrix"):
        _validate_request(preflight, config, formal=True)


def _fake_result(seed: int, offset: int) -> dict[str, object]:
    rows = []
    for size in (9, 23):
        for index in range(6):
            rows.append(
                {
                    "task_id": f"size{size}-task{index}",
                    "maze_size": size,
                    "success": float((index + seed + offset) % 3 != 0),
                }
            )
    return {"task_rows": rows}


def test_crossed_bootstrap_is_deterministic_and_paired() -> None:
    first = [_fake_result(seed, 1) for seed in range(3)]
    second = [_fake_result(seed, 0) for seed in range(3)]
    kwargs = {
        "predicate": lambda row: True,
        "samples": 500,
        "seed": 123,
        "family_size": 4,
        "alpha": 0.05,
    }
    left = crossed_bootstrap_difference(first, second, **kwargs)
    right = crossed_bootstrap_difference(first, second, **kwargs)
    assert left == right
    assert left["task_count"] == 12
    assert left["seed_count"] == 3


def test_spearman_handles_monotonicity_and_constant_curves() -> None:
    assert _spearman([0.0, 1.0, 2.0], [0.2, 0.4, 0.9]) == pytest.approx(1.0)
    assert _spearman([0.0, 1.0, 2.0], [0.9, 0.4, 0.2]) == pytest.approx(-1.0)
    assert _spearman([0.0, 1.0, 2.0], [0.5, 0.5, 0.5]) is None


def test_copy_relative_improvement_is_computed_per_eligible_state(
    config: Stage0Config,
) -> None:
    diagnostic = {
        "summary": {
            "future": {
                "predicted_variance": 1.0,
                "target_variance": 1.0,
                "predicted_candidate_pairwise": 1.0,
                "target_candidate_pairwise": 1.0,
                "normalized_delta_error": 2.0,
                "copy_delta_normalized": 3.0,
            },
            "predicted": {"local_top1": 0.75},
            "true_future": {"local_top1": 0.8},
            "permuted": {"local_top1": 0.5},
            "predicted_true_choice_agreement": 0.7,
            "prediction_flip_rate": 0.1,
            "energy_wrong_with_true_future_rate": 0.2,
            "distance": {
                "target_clipped_rate": 0.0,
                "predicted": {
                    "expected_mae": 1.0,
                    "expected_rmse": 2.0,
                    "expected_spearman": 0.5,
                    "categorical_accuracy": 0.4,
                    "top_class_ece_15": 0.2,
                },
                "true_future": {
                    "expected_mae": 0.5,
                    "expected_rmse": 1.0,
                    "expected_spearman": 0.8,
                    "categorical_accuracy": 0.6,
                    "top_class_ece_15": 0.1,
                },
            },
        },
        "metadata": {"seed": 42},
        "state_rows": [
            {"copy_delta_normalized": 2.0, "normalized_delta_error": 1.0},
            {"copy_delta_normalized": 0.0, "normalized_delta_error": 0.0},
            {"copy_delta_normalized": 4.0, "normalized_delta_error": 3.0},
        ],
    }
    summary = diagnostic_summary([diagnostic], config)
    assert summary["copy_relative_improvement"] == pytest.approx(0.375)
    assert summary["copy_relative_eligible_states"] == 2
    assert summary["copy_relative_excluded_states"] == 1
    assert summary["distance"]["predicted"]["expected_mae"] == 1.0


def test_distance_diagnostics_report_clipped_regression_rank_and_calibration() -> None:
    predicted = {
        "top1": True,
        "regret": 0,
        "margin": 1.0,
        "chosen_slot": 0,
        "energy": [1.0, 2.0, 4.0, 128.0],
    }
    true_future = {
        **predicted,
        "energy": [1.0, 2.0, 3.0, 128.0],
    }
    row = {
        "candidate_distances": [1, 2, 3, 200],
        "predicted_ranking": predicted,
        "true_ranking": true_future,
        "copy_ranking": predicted,
        "permuted_ranking": predicted,
        "zero_ranking": predicted,
        "predicted_cost_class": [1, 2, 4, 128],
        "predicted_cost_confidence": [0.8, 0.7, 0.6, 0.9],
        "true_future_cost_class": [1, 2, 3, 128],
        "true_future_cost_confidence": [0.8, 0.7, 0.6, 0.9],
        "normalized_field_error": 1.0,
        "normalized_delta_error": 1.0,
        "copy_delta_normalized": 2.0,
        "predicted_candidate_pairwise": 1.0,
        "target_candidate_pairwise": 1.0,
        "predicted_variance": 1.0,
        "target_variance": 1.0,
        "local_error_type": "correct",
    }
    summary = summarize_state_rows([row])
    assert summary["distance"]["target_clipped_rate"] == pytest.approx(0.25)
    assert summary["distance"]["predicted"]["expected_mae"] == pytest.approx(0.25)
    assert summary["distance"]["predicted"]["expected_rmse"] == pytest.approx(0.5)
    assert summary["distance"]["predicted"]["categorical_accuracy"] == pytest.approx(
        0.75
    )
    assert summary["distance"]["true_future"]["expected_mae"] == 0.0


def test_source_payload_validation_requires_exact_representation(
    config: Stage0Config,
) -> None:
    representation = SpatialRepresentation(SpatialRepresentationConfig())
    state = representation.state_dict()
    source_protocol = {
        "seed": 42,
        "train_manifest_sha256": sha256_file(config.paths.train_manifest),
        "development_manifest_sha256": sha256_file(
            config.paths.historical_development_manifest
        ),
        "eval_manifest_sha256": sha256_file(
            config.paths.historical_confirmatory_manifest
        ),
        "git_dirty": False,
        "git_commit": "test-commit",
        "code_fingerprint": "test-fingerprint",
    }
    representation_payload = {
        "experiment_family": "spatial_jepa_planning",
        "format_version": 2,
        "stage": "representation",
        "variant_name": "spatial_info_sigreg",
        "protocol": source_protocol,
        "training_accounting": {"optimizer_steps": 30_000},
        "representation_state_dict": state,
    }
    state_hash = validate_source_representation_payload(
        representation_payload, seed=42, config=config
    )
    planner_payload = {
        "experiment_family": "spatial_jepa_planning",
        "format_version": 2,
        "stage": "planner",
        "input_mode": "spatial_jepa",
        "variant_name": "j1_spatial_iterative_frozen",
        "protocol": source_protocol,
        "training_accounting": {"optimizer_steps": 30_000},
        "training_args": {"encoder_mode": "frozen"},
        "planner_config": {"planner_type": "iterative"},
        "representation_state_dict": state,
        "planner_parameter_count": 100,
        "representation_planning_parameter_count": 50,
        "total_inference_parameter_count": 150,
        "representation_inference_conv_macs": {"21": 50, "25": 60},
        "planner_inference_conv_macs": {
            "21": {"4": 100, "128": 3200},
            "25": {"4": 120, "128": 3840},
        },
    }
    validate_source_planner_payload(
        planner_payload,
        seed=42,
        variant="j1_spatial_iterative_frozen",
        representation_state_sha256=state_hash,
        config=config,
    )
    changed = copy.deepcopy(planner_payload)
    first_key = next(iter(changed["representation_state_dict"]))
    changed["representation_state_dict"][first_key] = (
        changed["representation_state_dict"][first_key] + 1.0
    )
    with pytest.raises(ValueError, match="exact seed-matched"):
        validate_source_planner_payload(
            changed,
            seed=42,
            variant="j1_spatial_iterative_frozen",
            representation_state_sha256=state_hash,
            config=config,
        )


def test_air_checkpoint_rejects_nonfinal_and_loads_exact_state(
    config: Stage0Config, tmp_path: Path
) -> None:
    model = AIRWorkspaceModel(config.model)
    state = model.state_dict()
    payload = {
        "experiment_id": config.experiment_id,
        "format_version": 1,
        "method": "air0_jepa",
        "seed": 42,
        "optimizer_steps": 30_000,
        "config": config.model_dump(mode="json", by_alias=True),
        "model_state_dict": state,
        "model_state_sha256": state_dict_sha256(state),
    }
    path = tmp_path / "air.pt"
    torch.save(payload, path)
    loaded, _ = load_air_checkpoint(
        path,
        config=config,
        method="air0_jepa",
        seed=42,
        device=torch.device("cpu"),
    )
    assert state_dict_sha256(loaded.state_dict()) == payload["model_state_sha256"]
    with pytest.raises(ValueError, match="signed final checkpoint"):
        load_air_checkpoint(
            path,
            config=config,
            method="air0_jepa",
            seed=42,
            device=torch.device("cpu"),
            require_formal=True,
        )
    payload["formal"] = True
    payload["checkpoint_role"] = "final_step"
    torch.save(payload, path)
    load_air_checkpoint(
        path,
        config=config,
        method="air0_jepa",
        seed=42,
        device=torch.device("cpu"),
        require_formal=True,
    )
    payload["optimizer_steps"] = 29_999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="final 30k"):
        load_air_checkpoint(
            path,
            config=config,
            method="air0_jepa",
            seed=42,
            device=torch.device("cpu"),
        )


def test_air_checkpoint_save_is_atomic_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    save_air_checkpoint(path, {"tensor": torch.ones(2)})
    assert path.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_air_checkpoint(path, {"tensor": torch.zeros(2)})

    failed_path = tmp_path / "failed.pt"

    def fail_after_partial_write(payload: object, stream: object) -> None:
        del payload
        stream.write(b"partial")
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected serialization failure"):
        save_air_checkpoint(failed_path, {"tensor": torch.ones(2)})
    assert not failed_path.exists()
    assert not failed_path.with_suffix(".pt.tmp").exists()


def test_diagnostic_state_selection_and_action_protocol(config: Stage0Config) -> None:
    entry = read_jsonl(config.paths.air_dev_manifest)[0]
    states = deterministic_states(entry, count=24)
    assert states == deterministic_states(entry, count=24)
    assert 0 < len(states) <= 24
    env = validate_manifest_entry(entry, check_bfs=False)
    state = states[0]
    assert candidate_actions(env, state, None, "unmasked") == list(ACTION_IDS)
    corrected = candidate_actions(env, state, None, "corrected")
    assert all(next_state(env, state, action) != state for action in corrected)


def test_diagnostic_ranking_audit_recomputes_choice_regret_and_margin(
    tmp_path: Path,
) -> None:
    ranking = {
        "top1": True,
        "regret": 0,
        "margin": 2.0,
        "chosen_slot": 1,
        "energy": [3.0, 1.0, 2.0, 4.0],
    }
    _validate_diagnostic_ranking(
        ranking,
        candidate_distances=[4, 2, 2, 5],
        optimal_action_mask=[False, True, True, False],
        path=tmp_path / "diagnostic.json",
    )
    corrupted = {**ranking, "chosen_slot": 2}
    with pytest.raises(ValueError, match="chosen slot"):
        _validate_diagnostic_ranking(
            corrupted,
            candidate_distances=[4, 2, 2, 5],
            optimal_action_mask=[False, True, True, False],
            path=tmp_path / "diagnostic.json",
        )


def test_bfs_oracle_strictly_decreases_distance(config: Stage0Config) -> None:
    entry = read_jsonl(config.paths.air_dev_manifest)[0]
    env = validate_manifest_entry(entry, check_bfs=False)
    state = int(entry["start_cell"])
    distances = bfs_distances_from(
        env._maze_mask,
        int(entry["goal_cell"]),
        int(entry["maze_size"]),
    )
    action, _ = build_oracle_action_fn()(env, observe_state(env, state), state, None)
    assert int(distances[next_state(env, state, action)]) == int(distances[state]) - 1
    row = run_navigation_with_diagnostics(
        entry,
        action_fn=build_oracle_action_fn(),
        max_steps=config.evaluation.max_steps,
    )
    assert row["distance_decrease_actions"] == row["path_length"]
    assert row["distance_flat_actions"] == 0
    assert row["distance_increase_actions"] == 0


def test_pairing_audit_reports_identical_method_streams(config: Stage0Config) -> None:
    report = pairing_audit(config, batches=2)
    assert set(report) == {"42", "43", "44"}
    assert all(item["checked_batches"] == 2 for item in report.values())


def test_release_loader_recomputes_rows_and_aggregates(
    config: Stage0Config, tmp_path: Path
) -> None:
    entry = read_jsonl(config.paths.air_early_manifest)[0]
    manifest = tmp_path / "mini_manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    config_payload = config.model_dump(mode="json", by_alias=True)
    config_payload["paths"]["air_dev_manifest"] = str(manifest)
    mini_config = Stage0Config.model_validate(config_payload)
    row = run_navigation_with_diagnostics(
        entry,
        action_fn=build_oracle_action_fn(),
        max_steps=config.evaluation.max_steps,
    )
    row["elapsed_seconds"] = 0.01
    row["failure_reason"] = classify_failure(row)
    payload = {
        "schema": "air-jepa-stage0-evaluation-v1",
        "metadata": {
            "experiment_id": config.experiment_id,
            "method": "oracle_bfs",
            "seed": 0,
            "k": 0,
            "split_role": "air_dev",
            "evidence_role": "EVALUATOR_ORACLE",
            "action_protocol": "unmasked",
            "intervention": "normal",
            "task_count": 1,
            "max_steps": config.evaluation.max_steps,
            "manifest": relative_path(manifest),
            "manifest_sha256": sha256_file(manifest),
            "checkpoint_sha256": None,
            "protocol_sha256": "protocol",
            "package_sha256": "package",
            "source_lock_sha256": "source",
            "git_commit": "test-commit",
            "git_dirty": False,
            "code_fingerprint": "code",
            "runtime": _formal_runtime(),
            "formal": True,
        },
        "navigation": summarize_rows(
            [row],
            seen_max_size=config.evaluation.seen_max_size,
            max_steps=config.evaluation.max_steps,
        ),
        "task_rows": [row],
    }
    path = tmp_path / "valid.json"
    atomic_json_dump(path, payload)
    loader = ArtifactLoader(
        mini_config,
        protocol_sha256="protocol",
        package_sha256="package",
        package_code_fingerprint="code",
        source_lock_sha256="source",
        expected_checkpoint_hashes={},
    )
    loaded = loader.evaluation(
        path,
        role="air_dev",
        method="oracle_bfs",
        seed=0,
        k=0,
    )
    assert loaded["navigation"]["overall"]["sr"] == 1.0
    corrupted = copy.deepcopy(payload)
    corrupted["navigation"]["overall"]["sr"] = 0.0
    corrupted_path = tmp_path / "corrupted.json"
    atomic_json_dump(corrupted_path, corrupted)
    with pytest.raises(ValueError, match="not reproducible"):
        loader.evaluation(
            corrupted_path,
            role="air_dev",
            method="oracle_bfs",
            seed=0,
            k=0,
        )

    wrong_calls = copy.deepcopy(payload)
    wrong_calls["task_rows"][0]["auxiliary"]["inference_calls"] = 999.0
    wrong_calls_path = tmp_path / "wrong_calls.json"
    atomic_json_dump(wrong_calls_path, wrong_calls)
    with pytest.raises(ValueError, match="inference-call semantics"):
        loader.evaluation(
            wrong_calls_path,
            role="air_dev",
            method="oracle_bfs",
            seed=0,
            k=0,
        )


def test_paired_checkpoint_audit_rejects_unmatched_initialization(
    config: Stage0Config, tmp_path: Path
) -> None:
    config_payload = config.model_dump(mode="json", by_alias=True)
    config_payload["paths"]["air_checkpoint_template"] = str(
        tmp_path / "{method}_seed{seed}.pt"
    )
    temp_config = Stage0Config.model_validate(config_payload)
    model = AIRWorkspaceModel(config.model)
    state = model.state_dict()
    count = sum(parameter.numel() for parameter in model.parameters())
    workspace_macs = {
        str(size): {
            str(k): model.analytical_macs(size, k) for k in config.evaluation.k_values
        }
        for size in (21, 25)
    }
    workspace_components = {
        str(size): {
            str(k): model.analytical_mac_breakdown(size, k)
            for k in config.evaluation.k_values
        }
        for size in (21, 25)
    }
    representation_macs = {"21": 10, "25": 12}
    iteration_rng = make_rng_streams(42).iterations
    iteration_schedule = [
        select_progressive_iterations(
            step=step,
            phase_steps=config.training.phase_steps,
            k_train=config.training.k_train,
            rng=iteration_rng,
        )
        for step in range(1, config.training.steps + 1)
    ]
    cumulative_k: Counter[int] = Counter()
    training_log = []
    for step in range(config.training.log_every, config.training.steps + 1, 500):
        window = Counter(
            iteration_schedule[step - config.training.log_every : step]
        )
        cumulative_k.update(window)
        training_log.append(
            {
                "step": step,
                "total": 1.0,
                "action": 1.0,
                "future": 1.0,
                "cost": 1.0,
                "future_field_normalized": 1.0,
                "future_delta_normalized": 1.0,
                "future_field_raw_mse": 1.0,
                "future_delta_raw_mse": 1.0,
                "copy_delta_normalized": 1.0,
                "gradient_norm": 1.0,
                "iterations": float(np.mean(iteration_schedule[step - 500 : step])),
                "learning_rate": 1e-3,
                "window_steps": 500,
                "window_elapsed_seconds": 1.0,
                "steps_per_second": 500.0,
                "window_k_counts": {
                    str(key): value for key, value in sorted(window.items())
                },
                "cumulative_k_counts": {
                    str(key): value for key, value in sorted(cumulative_k.items())
                },
                "peak_cuda_memory_bytes": 1,
            }
        )
    iteration_audit = progressive_iteration_signature(
        seed=42,
        steps=config.training.steps,
        phase_steps=config.training.phase_steps,
        k_train=config.training.k_train,
    )
    common = {
        "experiment_id": EXPERIMENT_ID,
        "format_version": FORMAT_VERSION,
        "seed": 42,
        "formal": True,
        "checkpoint_role": "final_step",
        "optimizer_steps": config.training.steps,
        "config": temp_config.model_dump(mode="json", by_alias=True),
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "protocol_sha256": "protocol",
        "package_sha256": "package",
        "source_lock_sha256": "source",
        "git_commit": "test-commit",
        "git_dirty": False,
        "code_fingerprint": "code",
        "model_state_dict": state,
        "model_state_sha256": state_dict_sha256(state),
        "initial_model_state_sha256": "same-initialization",
        "paired_sample_stream_sha256": "same-stream",
        "progressive_iteration_stream_sha256": iteration_audit["sha256"],
        "paired_sample_stream_prefix_batches": FORMAL_PAIRING_BATCHES,
        "paired_sample_stream_prefix_sha256": "l0-stream",
        "model_seed": 70_042,
        "rng_stream_seeds": make_rng_streams(42).stream_seeds,
        "k_counts": iteration_audit["counts"],
        "source_representation": {"state_sha256": "representation"},
        "model_parameter_count": count,
        "model_trainable_parameter_count": count,
        "component_parameter_counts": {"all_air_components": count},
        "representation_planning_parameter_count": 2,
        "representation_trainable_parameter_count": 0,
        "representation_component_parameter_counts": {
            "encoder": 1,
            "planning_projector": 1,
        },
        "total_inference_parameter_count": count + 2,
        "workspace_analytical_macs": workspace_macs,
        "workspace_analytical_macs_by_component": workspace_components,
        "representation_planning_conv_macs": representation_macs,
        "total_inference_macs": {
            str(size): {
                str(k): representation_macs[str(size)]
                + workspace_macs[str(size)][str(k)]
                for k in config.evaluation.k_values
            }
            for size in (21, 25)
        },
        "training_accounting": {
            "elapsed_seconds": 1.0,
            "map_state_examples": 240_000,
            "successor_examples": 960_000,
            "peak_cuda_memory_bytes": 1,
        },
        "training_log": training_log,
        "gradient_history": [
            {
                "step": step,
                "iterations": iteration_schedule[step - 1],
                "action_norm": 1.0,
            }
            for step in [1, *range(500, 30_001, 500)]
        ],
        "future_target_channel_moments": {
            "count_per_channel": 1,
            "mean": [0.0] * 64,
            "variance": [0.0] * 64,
        },
        "runtime": _formal_runtime(),
    }
    source_lock = {
        "records": {
            "42": {"representation": {"state_sha256": "representation"}}
        }
    }
    l0_pairing = {
        "42": {
            "checked_batches": FORMAL_PAIRING_BATCHES,
            "initial_model_state_sha256": "same-initialization",
            "sample_stream_sha256": "l0-stream",
        }
    }
    for method in ("air0_direct", "air0_jepa"):
        torch.save(
            {**common, "method": method},
            tmp_path / f"{method}_seed42.pt",
        )
    report = paired_checkpoint_audit(
        temp_config,
        seeds=(42,),
        source_lock=source_lock,
        l0_pairing=l0_pairing,
        protocol_sha256="protocol",
        package_sha256="package",
        package_code_fingerprint="code",
        source_lock_sha256="source",
    )
    assert report["passed"] is True
    changed = {**common, "method": "air0_jepa"}
    changed["initial_model_state_sha256"] = "different-initialization"
    torch.save(changed, tmp_path / "air0_jepa_seed42.pt")
    with pytest.raises(ValueError, match="signed L0 pairing audit"):
        paired_checkpoint_audit(
            temp_config,
            seeds=(42,),
            source_lock=source_lock,
            l0_pairing=l0_pairing,
            protocol_sha256="protocol",
            package_sha256="package",
            package_code_fingerprint="code",
            source_lock_sha256="source",
        )

    wrong_source = copy.deepcopy(common)
    wrong_source["method"] = "air0_jepa"
    wrong_source["source_representation"] = {"state_sha256": "wrong"}
    torch.save(wrong_source, tmp_path / "air0_jepa_seed42.pt")
    with pytest.raises(ValueError, match="representation lineage"):
        paired_checkpoint_audit(
            temp_config,
            seeds=(42,),
            source_lock=source_lock,
            l0_pairing=l0_pairing,
            protocol_sha256="protocol",
            package_sha256="package",
            package_code_fingerprint="code",
            source_lock_sha256="source",
        )

    wrong_iterations = copy.deepcopy(common)
    wrong_iterations["method"] = "air0_jepa"
    wrong_iterations["progressive_iteration_stream_sha256"] = "wrong-k-stream"
    torch.save(wrong_iterations, tmp_path / "air0_jepa_seed42.pt")
    with pytest.raises(ValueError, match="cumulative K accounting"):
        paired_checkpoint_audit(
            temp_config,
            seeds=(42,),
            source_lock=source_lock,
            l0_pairing=l0_pairing,
            protocol_sha256="protocol",
            package_sha256="package",
            package_code_fingerprint="code",
            source_lock_sha256="source",
        )

    restored = {**common, "method": "air0_jepa"}
    restored["paired_sample_stream_sha256"] = "different-full-stream"
    torch.save(restored, tmp_path / "air0_jepa_seed42.pt")
    with pytest.raises(ValueError, match="pairing mismatch"):
        paired_checkpoint_audit(
            temp_config,
            seeds=(42,),
            source_lock=source_lock,
            l0_pairing=l0_pairing,
            protocol_sha256="protocol",
            package_sha256="package",
            package_code_fingerprint="code",
            source_lock_sha256="source",
        )

    wrong_schedule = copy.deepcopy(common)
    wrong_schedule["method"] = "air0_jepa"
    wrong_schedule["training_log"][0]["iterations"] = 128.0
    wrong_schedule["training_log"][0]["window_k_counts"] = {"128": 500}
    wrong_schedule["training_log"][0]["cumulative_k_counts"] = {"128": 500}
    torch.save(wrong_schedule, tmp_path / "air0_jepa_seed42.pt")
    with pytest.raises(ValueError, match="invalid training log window"):
        paired_checkpoint_audit(
            temp_config,
            seeds=(42,),
            source_lock=source_lock,
            l0_pairing=l0_pairing,
            protocol_sha256="protocol",
            package_sha256="package",
            package_code_fingerprint="code",
            source_lock_sha256="source",
        )


def test_compute_accounting_builds_locked_k_and_mac_curves(
    config: Stage0Config, tmp_path: Path
) -> None:
    source_lock: dict[str, object] = {"records": {}}
    checkpoint_records = []
    loaded = {}
    for seed in config.seeds:
        source_payload = {
            "planner_parameter_count": 100,
            "representation_planning_parameter_count": 50,
            "total_inference_parameter_count": 150,
            "planner_inference_conv_macs": {
                str(size): {str(k): 100 + 10 * k for k in config.training.k_train}
                for size in (21, 25)
            },
            "representation_inference_conv_macs": {"21": 50, "25": 50},
        }
        source_path = tmp_path / f"j1_seed{seed}.pt"
        torch.save(source_payload, source_path)
        source_lock["records"][str(seed)] = {
            "j1": {"path": str(source_path), "file_sha256": sha256_file(source_path)}
        }
        for method in ("air0_direct", "air0_jepa"):
            checkpoint_records.append(
                {
                    "method": method,
                    "seed": seed,
                    "component_parameter_counts": {"reasoner": 80},
                    "representation_component_parameter_counts": {
                        "encoder": 30,
                        "planning_projector": 20,
                    },
                    "total_inference_parameter_count": 130,
                    "total_inference_macs": {
                        str(size): {
                            str(k): 90 + 5 * k for k in config.evaluation.k_values
                        }
                        for size in (21, 25)
                    },
                    "training_accounting": {
                        "elapsed_seconds": 1.0,
                        "map_state_examples": 240_000,
                        "successor_examples": 960_000,
                        "peak_cuda_memory_bytes": 1,
                    },
                }
            )
        for method in ("j1_receding", "air0_direct", "air0_jepa"):
            for k in config.evaluation.k_values:
                loaded[(method, seed, k)] = {
                    "metadata": {"elapsed_seconds": 1.0},
                    "navigation": {
                        "overall": {"sr": 1.0, "spl": 1.0},
                        "ood": {"sr": 1.0, "spl": 1.0},
                        "by_size": {
                            "21": {"sr": 1.0},
                            "25": {"sr": 1.0},
                        },
                    },
                    "task_rows": [{"success": True, "elapsed_seconds": 0.01}],
                }
    report = compute_accounting(
        config,
        checkpoint_audit={"records": checkpoint_records},
        loaded=loaded,
        source_lock=source_lock,
        locked_compute_match={
            "k_by_size": {"21": 128, "25": 128},
            "joint_k": 128,
            "performance_used": False,
        },
    )
    assert len(report["parameter_rows"]) == 9
    assert len(report["quality_vs_k_and_macs"]) == 126
    assert len(report["runtime_rows"]) == 63
    assert report["compute_match_joint_k"] == 128
    assert report["compute_match_k_by_size"] == {"21": 128, "25": 128}


def test_protocol_audit_gate_requires_formal_hardware_and_pairing(
    config: Stage0Config,
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "protocol_audit.json"
    config_payload = config.model_dump(mode="json", by_alias=True)
    config_payload["paths"]["audit_output"] = str(audit_path)
    temp_config = Stage0Config.model_validate(config_payload)
    payload = signed_payload(
        {
            "schema": "air-jepa-stage0-protocol-audit-v1",
            "experiment_id": config.experiment_id,
            "passed": True,
            "protocol_sha256": "protocol",
            "package_sha256": "package",
            "source_lock_sha256": "source",
            "git_commit": "test-commit",
            "git_dirty": False,
            "code_fingerprint": "code",
            "runtime": _formal_runtime(),
            "hardware": {
                "skipped": False,
                "formal_eligible": True,
                "devices": [
                    {"index": index, "name": "NVIDIA H800", "total_memory": 1}
                    for index in range(4)
                ],
            },
            "pairing": {
                str(seed): {
                    "checked_batches": FORMAL_PAIRING_BATCHES,
                    "initial_model_state_sha256": f"init-{seed}",
                    "sample_stream_sha256": f"stream-{seed}",
                }
                for seed in config.seeds
            },
            "sealed_roles": {},
            "matrix_counts": {
                key: len(value) for key, value in expected_matrix(config).items()
            },
        },
        "protocol_audit_sha256",
    )
    atomic_json_dump(audit_path, payload)
    report = require_protocol_audit(
        temp_config,
        protocol_sha256="protocol",
        package_sha256="package",
        package_code_fingerprint="code",
        source_lock_sha256="source",
    )
    assert report["pairing_batches"] == FORMAL_PAIRING_BATCHES
    assert report["pairing"] == payload["pairing"]

    invalid = copy.deepcopy(payload)
    invalid.pop("protocol_audit_sha256")
    invalid["hardware"]["devices"][0]["name"] = "NVIDIA A100"
    atomic_json_dump(
        audit_path,
        signed_payload(invalid, "protocol_audit_sha256"),
    )
    with pytest.raises(ValueError, match="four verified H800"):
        require_protocol_audit(
            temp_config,
            protocol_sha256="protocol",
            package_sha256="package",
            package_code_fingerprint="code",
            source_lock_sha256="source",
        )


def test_bridge_audit_gate_requires_one_formal_runtime(
    config: Stage0Config,
    tmp_path: Path,
) -> None:
    config_payload = config.model_dump(mode="json", by_alias=True)
    config_payload["paths"]["run_root"] = str(tmp_path)
    temp_config = Stage0Config.model_validate(config_payload)
    path = tmp_path / "audits" / "historical_bridge_parity.json"
    payload = signed_payload(
        {
            "schema": "air-jepa-stage0-bridge-audit-v1",
            "experiment_id": config.experiment_id,
            "protocol_sha256": "protocol",
            "package_sha256": "package",
            "source_lock_sha256": "source",
            "runtime": _formal_runtime(),
            "comparisons": {
                f"cell-{index}": {"exact_parity": True} for index in range(6)
            },
            "passed": True,
            "failures": [],
        },
        "bridge_audit_sha256",
    )
    atomic_json_dump(path, payload)
    report = require_bridge_audit(
        temp_config,
        protocol_sha256="protocol",
        package_sha256="package",
        source_lock_sha256="source",
    )
    assert report["cells"] == 6

    invalid = copy.deepcopy(payload)
    invalid.pop("bridge_audit_sha256")
    invalid["runtime"]["cuda_device_name"] = "CPU"
    atomic_json_dump(path, signed_payload(invalid, "bridge_audit_sha256"))
    with pytest.raises(ValueError, match="runtime is not deterministic H800"):
        require_bridge_audit(
            temp_config,
            protocol_sha256="protocol",
            package_sha256="package",
            source_lock_sha256="source",
        )


def test_benchmark_gate_recomputes_locked_compute_match(
    config: Stage0Config,
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    config_payload = config.model_dump(mode="json", by_alias=True)
    config_payload["paths"]["benchmark_output"] = str(benchmark_path)
    temp_config = Stage0Config.model_validate(config_payload)
    workspace = {
        str(size): {str(k): 100 * k for k in config.evaluation.k_values}
        for size in (21, 25)
    }
    components = {
        size: {k: {"shared_reasoner": value} for k, value in curve.items()}
        for size, curve in workspace.items()
    }
    representation = {"21": 50, "25": 50}
    air_total = {
        size: {k: representation[size] + value for k, value in curve.items()}
        for size, curve in workspace.items()
    }
    payload = signed_payload(
        {
            "schema": "air-jepa-stage0-benchmark-v1",
            "experiment_id": config.experiment_id,
            "performance_blind": True,
            "protocol_sha256": "protocol",
            "package_sha256": "package",
            "source_lock_sha256": "source",
            "git_commit": "test-commit",
            "git_dirty": False,
            "code_fingerprint": "code",
            "runtime": _formal_runtime(),
            "k128_forward": {
                "tasks": 50,
                "seconds_total": 1.0,
                "seconds_mean": 0.02,
                "tasks_per_second": 50.0,
            },
            "k128_forward_backward": {
                "iterations": 128,
                "repeats": 5,
                "seconds_mean": 0.1,
            },
            "peak_cuda_memory_bytes": 1,
            "parameter_counts": {
                "frozen_representation": 50,
                "air_workspace": 80,
                "air_total_inference": 130,
                "j1_total_inference": 150,
            },
            "workspace_analytical_macs": workspace,
            "workspace_analytical_macs_by_component": components,
            "representation_planning_conv_macs": representation,
            "air_total_inference_macs": air_total,
            "j1_k128_total_inference_macs": {"21": 20_000, "25": 20_000},
            "compute_match": {
                "rule": "test",
                "k_by_size": {"21": 128, "25": 128},
                "joint_k": 128,
                "performance_used": False,
            },
        },
        "benchmark_sha256",
    )
    atomic_json_dump(benchmark_path, payload)
    report = require_benchmark(
        temp_config,
        protocol_sha256="protocol",
        package_sha256="package",
        package_code_fingerprint="code",
        source_lock_sha256="source",
    )
    assert report["compute_match"]["joint_k"] == 128

    invalid = copy.deepcopy(payload)
    invalid.pop("benchmark_sha256")
    invalid["compute_match"]["joint_k"] = 64
    atomic_json_dump(benchmark_path, signed_payload(invalid, "benchmark_sha256"))
    with pytest.raises(ValueError, match="compute-match lock"):
        require_benchmark(
            temp_config,
            protocol_sha256="protocol",
            package_sha256="package",
            package_code_fingerprint="code",
            source_lock_sha256="source",
        )

    invalid_repeats = copy.deepcopy(payload)
    invalid_repeats.pop("benchmark_sha256")
    invalid_repeats["k128_forward_backward"]["repeats"] = 1
    atomic_json_dump(
        benchmark_path,
        signed_payload(invalid_repeats, "benchmark_sha256"),
    )
    with pytest.raises(ValueError, match="timing/memory evidence"):
        require_benchmark(
            temp_config,
            protocol_sha256="protocol",
            package_sha256="package",
            package_code_fingerprint="code",
            source_lock_sha256="source",
        )
