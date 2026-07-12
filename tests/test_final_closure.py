from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from final_closure import FIGURE_FILENAMES, TABLE_FILENAMES
from final_closure.audit_protocol import audit_config
from final_closure.common import (
    ACTION_IDS,
    RERUN_REASONS,
    analysis_spec_sha256,
    corrected_actions,
    crossed_paired_bootstrap,
    experiment_code_fingerprint,
    git_commit,
    load_config,
    next_state,
    prepare_rerun,
    read_jsonl,
    require_study_open,
    sha256_file,
    task_seed,
    validate_task_rows,
)
from final_closure.data import (
    build_bc_dataset,
    epoch_batches,
    materialize_bc_dataset,
    render_bc_batch,
)
from final_closure.evaluate import BCController, run_episode
from final_closure.models import (
    BCPolicyConfig,
    DeepCNNPolicy,
    build_lewm,
    deserialize_lewm_config,
    serialize_lewm_config,
)
from final_closure.run_plan import main as run_plan_main
from final_closure.run_plan import randomized_jobs
from final_closure.summarize import (
    checkpoint_hash_is_valid,
    interval_status,
    source_spatial_variant,
    spatial_k_curves,
    validate_action_protocol_consistency,
    validate_baseline_compute,
    validate_protocol_audit,
    validate_records_against_manifest,
    validate_spatial_checkpoint,
)
from final_closure.verify_closure import verify_closure_gate
from spatial_jepa_planning.common import (
    bfs_distances_from,
    observe_state,
    validate_manifest_entry,
)
from spatial_jepa_planning.common import (
    experiment_code_fingerprint as spatial_code_fingerprint,
)
from spatial_jepa_planning.run_plan import training_spec_sha256 as spatial_training_spec

ROOT = Path(__file__).resolve().parents[1]


def load_protocol() -> tuple[dict, dict]:
    return load_config(ROOT / "final_closure/configs/default.json")


def test_protocol_matrix_is_exact() -> None:
    config, lock = load_protocol()
    report = audit_config(config)
    assert lock["analysis_spec_sha256"] == analysis_spec_sha256(config, lock)
    assert report["seeds"] == list(range(42, 52))
    assert report["comparison_count"] == 4
    assert report["simultaneous_alpha"] == pytest.approx(0.0125)
    assert lock["confirmatory_manifest"]["expected_exact_oracle_sr"] == pytest.approx(
        881 / 900
    )


def test_load_config_rejects_a_mutated_scientific_setting(tmp_path: Path) -> None:
    config, _ = load_protocol()
    changed = json.loads(json.dumps(config))
    changed["paths"]["protocol_lock"] = str(
        ROOT / "final_closure/configs/protocol_lock.json"
    )
    changed["baselines"][1]["planner"]["horizon"] = 13
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="independently locked analysis spec"):
        load_config(path)


def test_closure_gate_makes_the_study_immutable(tmp_path: Path) -> None:
    config = {"paths": {"closure_gate": str(tmp_path / "gate.json")}}
    require_study_open(config)
    Path(config["paths"]["closure_gate"]).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="study is immutable"):
        require_study_open(config)


def test_summary_rejects_a_nonformal_protocol_audit() -> None:
    config, lock = load_protocol()
    audit = {
        "protocol_id": config["protocol_id"],
        "status": "passed",
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "metadata": {"formal_audit": False},
    }
    with pytest.raises(ValueError, match="formal audit flag"):
        validate_protocol_audit(audit, config=config, lock=lock)


def test_rerun_requires_reason_and_preserves_superseded_hash(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    output.write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="allowed rerun reason"):
        prepare_rerun([output], overwrite=True, reason="")
    record = prepare_rerun(
        [output],
        overwrite=True,
        reason="manifest_checkpoint_or_code_hash_mismatch",
    )
    assert record is not None
    assert record["superseded_outputs"][str(output)] == sha256_file(output)
    with pytest.raises(ValueError, match="no selected output exists"):
        prepare_rerun(
            [tmp_path / "missing.json"],
            overwrite=True,
            reason="interrupted_execution",
        )


def test_run_plan_rejects_broad_score_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_plan",
            "--stages",
            "full",
            "--rerun-execution-failures",
            "--rerun-reason",
            "non_finite_output",
            "--dry-run",
        ],
    )
    with pytest.raises(ValueError, match="exactly one explicit stage"):
        run_plan_main()


def test_artifact_schema_is_complete() -> None:
    assert len(TABLE_FILENAMES) == 9
    assert "per_seed_results.csv" in TABLE_FILENAMES
    assert "size_generalization.csv" in TABLE_FILENAMES
    assert "assistance_effects.csv" in TABLE_FILENAMES
    assert len(FIGURE_FILENAMES) == 5


def test_locked_run_order_is_complete_reproducible_and_interleaved() -> None:
    config, _ = load_protocol()
    first = randomized_jobs(
        config["baselines"],
        config["seeds"],
        run_order_seed=config["protocol"]["run_order_seed"],
    )
    second = randomized_jobs(
        config["baselines"],
        config["seeds"],
        run_order_seed=config["protocol"]["run_order_seed"],
    )
    labels = [(item["name"], seed) for item, seed in first]
    assert labels == [(item["name"], seed) for item, seed in second]
    assert len(labels) == 20
    assert len(set(labels)) == 20
    assert {name for name, _ in labels[:5]} == {
        "bc_deepcnn_fixed",
        "lewm_l2_cem_seqlen2",
    }


def test_locked_manifest_hashes_and_source_fingerprint() -> None:
    config, lock = load_protocol()
    for role in (
        "train_manifest",
        "development_manifest",
        "confirmatory_manifest",
    ):
        assert sha256_file(ROOT / config["paths"][role]) == lock[role]["sha256"]
    assert (
        spatial_code_fingerprint()
        == lock["source_spatial_experiment"]["code_fingerprint"]
    )


def test_analysis_spec_changes_when_a_scientific_setting_changes() -> None:
    config, lock = load_protocol()
    original = analysis_spec_sha256(config, lock)
    changed = json.loads(json.dumps(config))
    changed["baselines"][1]["planner"]["horizon"] = 13
    assert analysis_spec_sha256(changed, lock) != original


def test_bc_renderer_matches_environment_observation_and_historical_padding() -> None:
    config, _ = load_protocol()
    entry = read_jsonl(ROOT / config["paths"]["train_manifest"])[0]
    dataset = build_bc_dataset([entry])
    observations, labels = render_bc_batch(
        dataset, np.asarray([0], dtype=np.int64), canvas_size=21
    )
    pool = dataset.pools[int(entry["maze_size"])]
    state = int(pool.states[0])
    env = validate_manifest_entry(entry)
    expected = observe_state(env, state)
    actual = observations[0].permute(1, 2, 0).numpy()
    size = int(entry["maze_size"])
    assert np.array_equal(actual[:size, :size], expected)
    assert np.count_nonzero(actual[size:, :]) == 0
    assert np.count_nonzero(actual[:, size:]) == 0
    assert int(labels[0]) in range(4)


def test_every_bc_target_reduces_oracle_distance_by_one() -> None:
    config, _ = load_protocol()
    entry = read_jsonl(ROOT / config["paths"]["train_manifest"])[0]
    dataset = build_bc_dataset([entry])
    pool = dataset.pools[int(entry["maze_size"])]
    env = validate_manifest_entry(entry)
    distances = bfs_distances_from(
        env._maze_mask, int(env._goal_position), int(env.config.width)
    )
    for state, slot in zip(pool.states.tolist(), pool.labels.tolist(), strict=True):
        candidate = next_state(env, int(state), ACTION_IDS[int(slot)])
        assert distances[candidate] == distances[int(state)] - 1


def test_bc_uint8_cache_preserves_inputs_and_target_order() -> None:
    config, _ = load_protocol()
    entry = read_jsonl(ROOT / config["paths"]["train_manifest"])[0]
    dataset = build_bc_dataset([entry])
    cached, labels = materialize_bc_dataset(dataset, canvas_size=21, chunk_size=7)
    rendered, expected_labels = render_bc_batch(
        dataset,
        np.arange(dataset.sample_count, dtype=np.int64),
        canvas_size=21,
    )
    assert cached.dtype == torch.uint8
    assert torch.equal(cached.float(), rendered)
    assert torch.equal(labels, expected_labels)


def test_epoch_batching_is_complete_reproducible_and_epoch_specific() -> None:
    first = list(epoch_batches(103, 16, seed=42, epoch=1))
    repeat = list(epoch_batches(103, 16, seed=42, epoch=1))
    second_epoch = list(epoch_batches(103, 16, seed=42, epoch=2))
    first_flat = np.concatenate(first)
    assert np.array_equal(first_flat, np.concatenate(repeat))
    assert sorted(first_flat.tolist()) == list(range(103))
    assert not np.array_equal(first_flat, np.concatenate(second_epoch))


def test_deepcnn_accepts_seen_canvas_and_larger_ood_size() -> None:
    model = DeepCNNPolicy(BCPolicyConfig())
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 5, 21, 21)).shape == (2, 4)
        assert model(torch.zeros(2, 5, 25, 25)).shape == (2, 4)


def test_lewm_config_round_trip_uses_repository_unisize_model() -> None:
    config, _ = load_protocol()
    baseline = config["baselines"][1]
    model, model_config = build_lewm(baseline["train"])
    restored = deserialize_lewm_config(serialize_lewm_config(model_config))
    assert restored.latent_dim == 256
    assert tuple(restored.cnn_channels) == (64, 128, 256)
    assert model.encoder.size_embed.num_embeddings == 32


def _state_with_invalid_and_moving_actions(entry: dict) -> tuple[object, int, int]:
    env = validate_manifest_entry(entry)
    free = np.flatnonzero((~env._maze_mask).reshape(-1))
    for state_value in free:
        state = int(state_value)
        invalid = [
            action for action in ACTION_IDS if next_state(env, state, action) == state
        ]
        if invalid and corrected_actions(env, state, None):
            return env, state, invalid[0]
    raise AssertionError("expected a boundary or wall-adjacent free state")


def test_corrected_bc_replaces_invalid_proposal_but_unmasked_does_not() -> None:
    config, _ = load_protocol()
    entry = read_jsonl(ROOT / config["paths"]["train_manifest"])[0]
    env, state, invalid_action = _state_with_invalid_and_moving_actions(entry)
    invalid_slot = ACTION_IDS.index(invalid_action)

    class FixedPolicy(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(inputs.shape[0], 4)
            logits[:, invalid_slot] = 10.0
            return logits

    observation = observe_state(env, state)
    unmasked = BCController(
        FixedPolicy(),
        device=torch.device("cpu"),
        canvas_size=21,
        action_selection="unmasked",
    )
    proposed, raw_metrics = unmasked.choose(env, observation, state, None)
    assert proposed == invalid_action
    assert raw_metrics["proposed_invalid"] == 1.0
    corrected = BCController(
        FixedPolicy(),
        device=torch.device("cpu"),
        canvas_size=21,
        action_selection="corrected",
    )
    selected, corrected_metrics = corrected.choose(env, observation, state, None)
    assert selected != invalid_action
    assert next_state(env, state, selected) != state
    assert corrected_metrics["assisted_action"] == 1.0


def test_episode_schema_counts_an_unmasked_wall_collision() -> None:
    config, _ = load_protocol()
    entry = read_jsonl(ROOT / config["paths"]["train_manifest"])[0]
    env, state, invalid_action = _state_with_invalid_and_moving_actions(entry)
    modified = dict(entry)
    modified["start_cell"] = state
    modified["bfs_path_length"] = int(
        bfs_distances_from(env._maze_mask, int(env._goal_position), env.config.width)[
            state
        ]
    )
    modified.pop("task_hash", None)

    class InvalidController:
        def reset(self, env, observation, task_index):
            del env, observation, task_index

        def choose(self, env, observation, state, previous):
            del env, observation, state, previous
            return invalid_action, {}

    row = run_episode(modified, InvalidController(), task_index=0, max_steps=1)
    assert row["path_length"] == 1
    assert row["invalid_actions"] == 1
    assert row["success"] is False
    assert row["spl"] == 0.0


def test_cem_seed_matches_historical_per_task_schedule() -> None:
    assert task_seed(42, 0, 0) == 4_200_000_000
    assert task_seed(42, 1, 0) != task_seed(42, 0, 0)
    assert task_seed(42, 0, 1) == task_seed(42, 0, 0) + 1


def test_crossed_bootstrap_pairs_both_seed_and_task() -> None:
    baseline = [
        [{"task_id": "a", "success": 0}, {"task_id": "b", "success": 1}],
        [{"task_id": "a", "success": 0}, {"task_id": "b", "success": 0}],
    ]
    candidate = [
        [{"task_id": "a", "success": 1}, {"task_id": "b", "success": 1}],
        [{"task_id": "a", "success": 1}, {"task_id": "b", "success": 0}],
    ]
    effect = crossed_paired_bootstrap(
        candidate,
        baseline,
        metric="success",
        samples=1000,
        alpha=0.05,
        seed=7,
    )
    assert effect["delta"] == 0.5
    assert effect["seed_resampling"] == "paired_across_methods"
    independent = crossed_paired_bootstrap(
        candidate,
        baseline,
        metric="success",
        samples=1000,
        alpha=0.05,
        seed=7,
        pair_seeds=False,
    )
    assert independent["delta"] == 0.5
    assert independent["seed_resampling"] == "independent_across_methods"
    stratified_candidate = [
        [{**row, "maze_size": 9 if row["task_id"] == "a" else 11} for row in seed_rows]
        for seed_rows in candidate
    ]
    stratified_baseline = [
        [{**row, "maze_size": 9 if row["task_id"] == "a" else 11} for row in seed_rows]
        for seed_rows in baseline
    ]
    stratified = crossed_paired_bootstrap(
        stratified_candidate,
        stratified_baseline,
        metric="success",
        samples=100,
        alpha=0.05,
        seed=7,
        pair_seeds=False,
        task_strata_key="maze_size",
    )
    assert stratified["task_resampling"] == "paired_by_task_id_within_maze_size"
    with pytest.raises(ValueError, match="identical task IDs"):
        crossed_paired_bootstrap(
            candidate,
            [[{"task_id": "c", "success": 0}], baseline[1]],
            metric="success",
            samples=10,
            alpha=0.05,
            seed=7,
        )


def test_result_validation_rejects_duplicate_rows_and_bad_spl() -> None:
    with pytest.raises(ValueError, match="unique task IDs"):
        validate_task_rows(
            [
                {
                    "task_id": "a",
                    "success": True,
                    "spl": 1.0,
                    "maze_size": 9,
                    "optimal_length": 1,
                    "path_length": 1,
                },
                {
                    "task_id": "a",
                    "success": False,
                    "spl": 0.0,
                    "maze_size": 9,
                    "optimal_length": 2,
                    "path_length": 3,
                },
            ],
            2,
        )
    with pytest.raises(ValueError, match="SPL"):
        validate_task_rows(
            [
                {
                    "task_id": "a",
                    "success": True,
                    "spl": 1.2,
                    "maze_size": 9,
                    "optimal_length": 1,
                    "path_length": 1,
                }
            ],
            1,
        )
    with pytest.raises(ValueError, match="path length"):
        validate_task_rows(
            [
                {
                    "task_id": "a",
                    "success": True,
                    "spl": 0.5,
                    "maze_size": 9,
                    "optimal_length": 64,
                    "path_length": 129,
                }
            ],
            1,
        )


def test_result_validation_rejects_inconsistent_failure_diagnostics() -> None:
    row = {
        "task_id": "a",
        "success": False,
        "spl": 0.0,
        "maze_size": 9,
        "optimal_length": 3,
        "path_length": 2,
        "invalid_actions": 0,
        "repeat_states": 1,
        "max_state_visits": 2,
        "loop_or_cycle": True,
        "episode_seconds": 0.1,
        "auxiliary": {"policy_forward_calls": 2.0},
    }
    with pytest.raises(ValueError, match="loop/cycle flag"):
        validate_task_rows([row], 1)
    row["loop_or_cycle"] = False
    row["auxiliary"]["policy_forward_calls"] = float("nan")
    with pytest.raises(ValueError, match="named and finite"):
        validate_task_rows([row], 1)


def test_baseline_compute_is_recomputed_from_task_rows() -> None:
    config, _ = load_protocol()
    baseline = config["baselines"][0]
    rows = [
        {
            "path_length": 2,
            "maze_size": 9,
            "episode_seconds": 0.25,
            "auxiliary": {
                "policy_forward_calls": 2.0,
                "proposed_invalid": 1.0,
                "proposed_backtrack": 0.0,
                "assisted_action": 0.0,
            },
        }
    ]
    compute = {
        "task_count": 1,
        "decision_count": 2,
        "wallclock_seconds": 0.25,
        "auxiliary_totals": dict(rows[0]["auxiliary"]),
        "forward_macs_by_maze_size": {"9": 123},
    }
    validate_baseline_compute(
        compute, rows, baseline=baseline, action_selection="unmasked"
    )
    compute["decision_count"] = 1
    with pytest.raises(ValueError, match="compute decision count"):
        validate_baseline_compute(
            compute, rows, baseline=baseline, action_selection="unmasked"
        )


def test_action_protocol_change_requires_recorded_assistance() -> None:
    raw_row = {
        "task_id": "task",
        "success": False,
        "path_length": 2,
        "spl": 0.0,
        "invalid_actions": 1,
        "repeat_states": 1,
        "max_state_visits": 2,
        "loop_or_cycle": False,
        "final_bfs_distance": 3,
        "auxiliary": {"assisted_action": 0.0},
    }
    corrected_row = {**raw_row, "success": True, "final_bfs_distance": 0}
    primary = {
        "bc": [
            {
                "metadata": {"training_seed": 42},
                "task_rows": [raw_row],
            }
        ]
    }
    corrected = {
        "bc": [
            {
                "metadata": {"training_seed": 42},
                "task_rows": [corrected_row],
            }
        ]
    }
    with pytest.raises(ValueError, match="without a recorded assisted action"):
        validate_action_protocol_consistency(primary, corrected)
    corrected_row["auxiliary"] = {"assisted_action": 1.0}
    validate_action_protocol_consistency(primary, corrected)


def test_spatial_iteration_curves_cannot_silently_drop_a_seed_key() -> None:
    records = {
        "j1": [
            {"all_iterations": {"4": {}}, "metadata": {}},
            {"all_iterations": {"4": {}, "8": {}}, "metadata": {}},
        ]
    }
    with pytest.raises(ValueError, match="differ across seeds"):
        spatial_k_curves(records)


def test_interval_status_never_claims_equivalence() -> None:
    assert interval_status({"ci_low": 0.01, "ci_high": 0.02}) == "positive_interval"
    assert interval_status({"ci_low": -0.02, "ci_high": -0.01}) == "negative_interval"
    assert (
        interval_status({"ci_low": -0.01, "ci_high": 0.01}) == "interval_overlaps_zero"
    )


def test_spatial_checkpoint_metadata_uses_planner_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "planner.pt"
    checkpoint.write_bytes(b"locked")
    checkpoint_hash_is_valid(
        {
            "planner_ckpt": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
    )
    with pytest.raises(ValueError, match="SHA256"):
        checkpoint_hash_is_valid(
            {"planner_ckpt": str(checkpoint), "checkpoint_sha256": "wrong"}
        )


def test_imported_spatial_checkpoint_identity_is_not_only_a_file_hash(
    tmp_path: Path,
) -> None:
    config, lock = load_protocol()
    source = lock["source_spatial_experiment"]
    source_config = json.loads((ROOT / source["config_path"]).read_text())
    method = config["spatial_methods"][0]
    variant, train_config, representation_name = source_spatial_variant(
        source_config, method["name"]
    )
    expected_spec = spatial_training_spec(
        source_config,
        train_config,
        variant_name=method["name"],
        seed=42,
        representation_name=representation_name,
    )
    planner_macs = {
        "25": {str(value): value for value in source["evaluation_iterations"]}
    }
    representation_macs = {"25": 1}
    metadata = {
        "source_representation_sha256": None,
        "planner_parameter_count": 1,
        "representation_planning_parameter_count": 0,
        "total_inference_parameter_count": 1,
        "planner_inference_conv_macs": planner_macs,
        "representation_inference_conv_macs": representation_macs,
    }
    checkpoint_data = {
        "experiment_family": source["experiment_family"],
        "format_version": source["format_version"],
        "stage": "planner",
        "variant_name": method["name"],
        "input_mode": variant["input_mode"],
        "analysis_spec_sha256": source["analysis_spec_sha256"],
        "experiment_spec_sha256": expected_spec,
        "protocol": {
            "seed": 42,
            "git_commit": source["git_commit"],
            "git_dirty": False,
            "code_fingerprint": source["code_fingerprint"],
        },
        **metadata,
        "planner_state_dict": {"weight": torch.ones(1)},
    }
    path = tmp_path / "spatial.pt"
    torch.save(checkpoint_data, path)
    validate_spatial_checkpoint(
        path,
        source=source,
        method=method,
        source_config=source_config,
        seed=42,
        expected_training_spec=expected_spec,
        metadata=metadata,
    )
    checkpoint_data["variant_name"] = "wrong_variant"
    torch.save(checkpoint_data, path)
    with pytest.raises(ValueError, match="variant_name"):
        validate_spatial_checkpoint(
            path,
            source=source,
            method=method,
            source_config=source_config,
            seed=42,
            expected_training_spec=expected_spec,
            metadata=metadata,
        )


def test_summary_recomputes_navigation_from_manifest_rows() -> None:
    config, _ = load_protocol()
    entry = read_jsonl(ROOT / config["paths"]["confirmatory_manifest"])[0]
    row = {
        "task_id": entry["task_hash"],
        "maze_size": entry["maze_size"],
        "topology_seed": entry["topology_seed"],
        "start_cell": entry["start_cell"],
        "goal_cell": entry["goal_cell"],
        "optimal_length": entry["bfs_path_length"],
        "success": True,
        "path_length": entry["bfs_path_length"],
        "spl": 1.0,
        "invalid_actions": 0,
        "loop_or_cycle": False,
    }
    from spatial_jepa_planning.common import summarize_rows

    navigation = summarize_rows([row], 21, 128)
    validate_records_against_manifest(
        {"method": [{"task_rows": [row], "navigation": navigation}]},
        [entry],
    )
    stale = json.loads(json.dumps(navigation))
    stale["overall"]["sr"] = 0.0
    with pytest.raises(ValueError, match="stale"):
        validate_records_against_manifest(
            {"method": [{"task_rows": [row], "navigation": stale}]},
            [entry],
        )


def test_closure_verifier_detects_artifact_tampering(tmp_path: Path) -> None:
    config, lock = load_protocol()
    configured = json.loads(json.dumps(config))
    configured["paths"]["protocol_lock"] = str(
        ROOT / "final_closure/configs/protocol_lock.json"
    )
    configured["paths"]["train_manifest"] = str(
        ROOT / config["paths"]["train_manifest"]
    )
    configured["paths"]["development_manifest"] = str(
        ROOT / config["paths"]["development_manifest"]
    )
    configured["paths"]["confirmatory_manifest"] = str(
        ROOT / config["paths"]["confirmatory_manifest"]
    )
    configured["paths"]["audit_output"] = str(tmp_path / "audit.json")
    configured["paths"]["summary_json"] = str(tmp_path / "summary.json")
    configured["paths"]["paper_report"] = str(tmp_path / "PAPER_RESULTS.md")
    configured["paths"]["closure_gate"] = str(tmp_path / "gate.json")
    configured["paths"]["table_dir"] = str(tmp_path / "tables")
    configured["paths"]["figure_dir"] = str(tmp_path / "figures")
    configured["paths"]["checkpoint_template"] = str(
        tmp_path / "checkpoints/{name}_seed{seed}.pt"
    )
    configured["paths"]["development_result_template"] = str(
        tmp_path / "runs/{name}/seed{seed}/development_{action_selection}.json"
    )
    configured["paths"]["confirmatory_result_template"] = str(
        tmp_path / "runs/{name}/seed{seed}/confirmatory_{action_selection}.json"
    )
    configured["paths"]["spatial_result_template"] = str(
        tmp_path / "spatial/{name}/seed{seed}/confirmatory_unmasked.json"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(configured), encoding="utf-8")
    audit_path = Path(configured["paths"]["audit_output"])
    audit_path.write_text("{}", encoding="utf-8")
    report_path = Path(configured["paths"]["paper_report"])
    report_path.write_text("paper", encoding="utf-8")
    artifact_paths = [report_path]
    for directory_key, names in (
        ("table_dir", TABLE_FILENAMES),
        ("figure_dir", FIGURE_FILENAMES),
    ):
        directory = Path(configured["paths"][directory_key])
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            path = directory / name
            path.write_text(name, encoding="utf-8")
            artifact_paths.append(path)
    artifacts = {str(path): sha256_file(path) for path in artifact_paths}
    sources = [
        config_path,
        Path(configured["paths"]["protocol_lock"]),
        Path(configured["paths"]["train_manifest"]),
        Path(configured["paths"]["development_manifest"]),
        Path(configured["paths"]["confirmatory_manifest"]),
        audit_path,
        ROOT / lock["source_spatial_experiment"]["config_path"],
        ROOT / lock["source_spatial_experiment"]["protocol_lock_path"],
    ]
    for baseline in config["baselines"]:
        for seed in config["seeds"]:
            checkpoint = Path(
                configured["paths"]["checkpoint_template"].format(
                    name=baseline["name"], seed=seed
                )
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("checkpoint", encoding="utf-8")
            sources.append(checkpoint)
            for split_role in ("development", "confirmatory"):
                template = configured["paths"][f"{split_role}_result_template"]
                for action_selection in ("unmasked", "corrected"):
                    result = Path(
                        template.format(
                            name=baseline["name"],
                            seed=seed,
                            action_selection=action_selection,
                        )
                    )
                    result.parent.mkdir(parents=True, exist_ok=True)
                    result.write_text("{}", encoding="utf-8")
                    sources.append(result)
    for method in config["spatial_methods"]:
        for seed in config["seeds"]:
            checkpoint = (
                tmp_path / "spatial_checkpoints" / (f"{method['name']}_seed{seed}.pt")
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("spatial checkpoint", encoding="utf-8")
            result = Path(
                configured["paths"]["spatial_result_template"].format(
                    name=method["name"], seed=seed
                )
            )
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(
                json.dumps({"metadata": {"planner_ckpt": str(checkpoint)}}),
                encoding="utf-8",
            )
            sources.extend([result, checkpoint])
    summary = {
        "protocol": {
            "analysis_spec_sha256": analysis_spec_sha256(configured, lock),
            "git_commit": git_commit(),
            "git_dirty": False,
            "code_fingerprint": experiment_code_fingerprint(),
        },
        "artifacts": artifacts,
        "source_file_count": len(sources),
        "rerun_records": [],
    }
    summary_path = Path(configured["paths"]["summary_json"])
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    gate = {
        "format_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "analysis_spec_sha256": analysis_spec_sha256(configured, lock),
        "git_commit": git_commit(),
        "code_fingerprint": experiment_code_fingerprint(),
        "completion_is_score_independent": True,
        "required_training_seeds": config["seeds"],
        "required_tasks_per_seed": 900,
        "required_primary_methods": [
            *(item["name"] for item in config["spatial_methods"]),
            *(item["name"] for item in config["baselines"]),
        ],
        "source_files": {str(path): sha256_file(path) for path in sources},
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "artifacts": artifacts,
        "rerun_records": [],
        "rerun_allowed_only_for": list(RERUN_REASONS),
        "rerun_for_low_or_surprising_score": False,
        "next_architecture_search_authorized": False,
    }
    Path(configured["paths"]["closure_gate"]).write_text(
        json.dumps(gate), encoding="utf-8"
    )
    assert verify_closure_gate(config_path)["status"] == "verified"
    report_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_closure_gate(config_path)
