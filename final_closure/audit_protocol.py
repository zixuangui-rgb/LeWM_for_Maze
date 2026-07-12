#!/usr/bin/env python3
"""Fail-fast audit of every fixed final-closure design and data invariant."""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from final_closure import PROTOCOL_ID
from final_closure.common import (
    ACTION_IDS,
    RERUN_REASONS,
    analysis_spec_sha256,
    atomic_json_dump,
    count_by_size,
    environment_summary,
    experiment_code_fingerprint,
    git_commit,
    git_worktree_dirty,
    load_config,
    load_json,
    prepare_rerun,
    read_jsonl,
    require_clean_worktree,
    require_new_output,
    require_study_open,
    sha256_file,
    validate_manifest_entry,
    verify_holdout,
)
from spatial_jepa_planning.common import (
    experiment_code_fingerprint as spatial_code_fingerprint,
)
from spatial_jepa_planning.run_plan import analysis_spec_sha256 as spatial_analysis_spec

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="final_closure/configs/default.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--skip-entry-regeneration", action="store_true")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-reason", choices=RERUN_REASONS, default="")
    return parser.parse_args()


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"protocol mismatch for {label}: {actual!r} != {expected!r}")


def audit_config(config: dict[str, Any]) -> dict[str, Any]:
    assert_equal(config["protocol_id"], PROTOCOL_ID, "protocol_id")
    assert_equal(
        config["study_role"],
        "post_confirmatory_fixed_baseline_addendum",
        "study_role",
    )
    assert_equal(config["seeds"], list(range(42, 52)), "training seeds")
    protocol = config["protocol"]
    expected_protocol = {
        "max_steps": 128,
        "seen_max_size": 21,
        "evaluation_seed": 42,
        "run_order_seed": 20260712,
        "full_eval_count": 900,
        "action_ids": [1, 2, 3, 4],
        "primary_action_selection": "unmasked",
        "diagnostic_action_selections": ["corrected"],
        "allow_confirmatory_model_selection": False,
        "allow_score_triggered_reruns": False,
    }
    assert_equal(protocol, expected_protocol, "evaluation protocol")
    assert_equal(tuple(protocol["action_ids"]), ACTION_IDS, "environment action order")
    baselines = config["baselines"]
    assert_equal(
        [item["name"] for item in baselines],
        ["bc_deepcnn_fixed", "lewm_l2_cem_seqlen2"],
        "baseline matrix",
    )
    bc = baselines[0]
    assert_equal(bc["kind"], "bc", "BC kind")
    expected_bc_train = {
        "epochs": 200,
        "batch_size": 128,
        "architecture": "historical_deepcnn_res2_down_res1_pool_mlp512_256",
        "target_population": "all_non_goal_free_states",
        "epoch_order": "seeded_global_permutation",
        "epoch_permutation_namespace": 1701,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "scheduler": "cosine_epoch",
        "grad_clip": 1.0,
        "dropout": 0.3,
        "train_canvas_size": 21,
        "observation_cache": "uint8_cpu_full_state",
        "class_count": 4,
        "checkpoint_selection": "final_epoch",
        "log_every_epochs": 10,
    }
    assert_equal(bc["train"], expected_bc_train, "BC frozen training config")
    lewm = baselines[1]
    assert_equal(lewm["kind"], "lewm_l2_cem", "LeWM kind")
    expected_lewm_train = {
        "steps": 30000,
        "batch_size": 256,
        "sequence_length": 2,
        "architecture": "unisize256_sizecond_cnn_projector_transformer",
        "entry_schedule": "historical_step_mod_manifest_length_starting_at_one",
        "environment_seed_stream": "numpy_default_rng_training_seed",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "latent_dim": 256,
        "max_size_embedding": 31,
        "cnn_channels": [64, 128, 256],
        "latent_batch_norm": True,
        "embedding_stage": "post_bn",
        "sigreg_stage": "post_bn",
        "predictor_heads": 16,
        "sigreg_knots": 17,
        "sigreg_num_proj": 1024,
        "lambda_prediction": 1.0,
        "lambda_sigreg": 0.09,
        "lambda_abs_position": 0.1,
        "lambda_relative_position": 1.0,
        "lambda_goal_position": 0.5,
        "checkpoint_selection": "final_step",
        "log_every_steps": 500,
    }
    assert_equal(lewm["train"], expected_lewm_train, "LeWM frozen training config")
    expected_planner = {
        "history_size": 3,
        "context_action_initialization": "repeat_action_id_4",
        "horizon": 12,
        "num_candidates": 64,
        "num_elites": 8,
        "cem_iters": 1,
        "momentum": 0.1,
        "replan_every_step": True,
        "allowed_actions": [1, 2, 3, 4],
        "cem_seed_schedule": "historical_eval_seed_task_step",
        "score": "squared_latent_l2",
    }
    assert_equal(lewm["planner"], expected_planner, "LeWM frozen CEM config")
    assert_equal(
        [
            (item["name"], item["primary_iterations"])
            for item in config["spatial_methods"]
        ],
        [
            ("r4_raw_iterative_progressive", 128),
            ("j1_spatial_iterative_frozen", 128),
        ],
        "imported spatial primary methods",
    )
    expected_pairs = [
        ("j1_spatial_iterative_frozen", "bc_deepcnn_fixed"),
        ("j1_spatial_iterative_frozen", "lewm_l2_cem_seqlen2"),
        ("r4_raw_iterative_progressive", "bc_deepcnn_fixed"),
        ("r4_raw_iterative_progressive", "lewm_l2_cem_seqlen2"),
    ]
    actual_pairs = [
        (item["candidate"], item["baseline"])
        for item in config["analysis"]["comparisons"]
    ]
    assert_equal(actual_pairs, expected_pairs, "secondary comparison family")
    assert_equal(config["analysis"]["familywise_alpha"], 0.05, "FWER alpha")
    assert_equal(config["analysis"]["bootstrap_samples"], 20000, "bootstrap draws")
    assert_equal(config["analysis"]["bootstrap_seed"], 20260712, "bootstrap seed")
    assert_equal(
        config["analysis"]["multiplicity_method"],
        "bonferroni_simultaneous_percentile_ci",
        "multiplicity method",
    )
    assert_equal(config["analysis"]["primary_metric"], "success", "primary metric")
    assert_equal(config["analysis"]["secondary_metrics"], ["spl"], "secondary metrics")
    assert_equal(
        config["analysis"]["cross_method_seed_resampling"],
        "independent",
        "cross-method seed resampling",
    )
    assert_equal(
        config["analysis"]["cross_method_task_resampling"],
        "paired_by_task_hash_within_maze_size",
        "cross-method task resampling",
    )
    assert_equal(
        config["analysis"]["action_protocol_seed_resampling"],
        "paired_same_checkpoint",
        "action-protocol seed resampling",
    )
    assert_equal(
        config["analysis"]["claim_status"],
        "secondary_fixed_addendum_not_new_confirmatory_hypotheses",
        "claim status",
    )
    job_order = [
        {"baseline": baseline["name"], "seed": seed}
        for baseline in baselines
        for seed in config["seeds"]
    ]
    random.Random(int(protocol["run_order_seed"])).shuffle(job_order)
    if len(job_order) != 20 or len(
        {(item["baseline"], item["seed"]) for item in job_order}
    ) != len(job_order):
        raise ValueError("locked run order is incomplete or duplicated")
    return {
        "seeds": config["seeds"],
        "baselines": [item["name"] for item in baselines],
        "spatial_methods": [item["name"] for item in config["spatial_methods"]],
        "comparison_count": len(expected_pairs),
        "simultaneous_alpha": 0.05 / len(expected_pairs),
        "training_job_order": job_order,
    }


def audit_manifests(
    config: dict[str, Any],
    lock: dict[str, Any],
    *,
    regenerate_entries: bool,
) -> dict[str, Any]:
    entries_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in (
        "train_manifest",
        "development_manifest",
        "confirmatory_manifest",
    ):
        path = config["paths"][role]
        assert_equal(str(path), str(lock[role]["path"]), f"{role}.path")
        assert_equal(sha256_file(path), lock[role]["sha256"], f"{role}.sha256")
        entries = read_jsonl(path)
        entries_by_role[role] = entries
        assert_equal(len(entries), int(lock[role]["count"]), f"{role}.count")
        assert_equal(
            count_by_size(entries), lock[role]["counts_by_size"], f"{role}.sizes"
        )
        task_ids = [entry.get("task_hash") for entry in entries]
        if len(set(task_ids)) != len(entries) or any(
            value is None for value in task_ids
        ):
            raise ValueError(f"{role} must contain unique explicit task hashes")
        layouts = [entry.get("layout_hash") for entry in entries]
        if len(set(layouts)) != len(entries) or any(value is None for value in layouts):
            raise ValueError(f"{role} must contain unique explicit layout hashes")
        if regenerate_entries:
            for entry in entries:
                validate_manifest_entry(entry)
    overlaps = {
        "train_vs_development": verify_holdout(
            entries_by_role["train_manifest"],
            entries_by_role["development_manifest"],
        ),
        "train_vs_confirmatory": verify_holdout(
            entries_by_role["train_manifest"],
            entries_by_role["confirmatory_manifest"],
        ),
        "development_vs_confirmatory": verify_holdout(
            entries_by_role["development_manifest"],
            entries_by_role["confirmatory_manifest"],
        ),
    }
    confirmatory = entries_by_role["confirmatory_manifest"]
    failures = sum(int(entry["bfs_path_length"]) > 128 for entry in confirmatory)
    assert_equal(
        failures,
        int(lock["confirmatory_manifest"]["step_cap_failures"]),
        "confirmatory step-cap failures",
    )
    ceiling = 1.0 - failures / len(confirmatory)
    if not np.isclose(
        ceiling,
        float(lock["confirmatory_manifest"]["expected_exact_oracle_sr"]),
    ):
        raise ValueError("confirmatory oracle ceiling differs from protocol lock")
    return {
        "hashes": {role: lock[role]["sha256"] for role in entries_by_role},
        "counts": {role: len(entries) for role, entries in entries_by_role.items()},
        "overlaps": overlaps,
        "confirmatory_step_cap_failures": failures,
        "confirmatory_oracle_ceiling": ceiling,
        "entries_regenerated": regenerate_entries,
    }


def audit_source(lock: dict[str, Any]) -> dict[str, Any]:
    source = lock["source_spatial_experiment"]
    commit = str(source["git_commit"])
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"locked spatial source commit is unavailable: {commit}")
    current_fingerprint = spatial_code_fingerprint()
    assert_equal(
        current_fingerprint,
        source["code_fingerprint"],
        "source spatial code fingerprint",
    )
    assert_equal(
        source["required_methods"],
        {
            "r4_raw_iterative_progressive": 128,
            "j1_spatial_iterative_frozen": 128,
        },
        "source spatial result matrix",
    )
    assert_equal(
        source["evaluation_iterations"],
        [4, 8, 16, 32, 64, 128, 256],
        "source spatial evaluation iterations",
    )
    source_config_path = Path(source["config_path"])
    source_lock_path = Path(source["protocol_lock_path"])
    assert_equal(
        sha256_file(source_config_path),
        source["config_sha256"],
        "source spatial config hash",
    )
    assert_equal(
        sha256_file(source_lock_path),
        source["protocol_lock_sha256"],
        "source spatial protocol-lock hash",
    )
    assert_equal(
        spatial_analysis_spec(load_json(source_config_path)),
        source["analysis_spec_sha256"],
        "source spatial analysis spec",
    )
    return {
        "git_commit": commit,
        "code_fingerprint": current_fingerprint,
        "required_methods": source["required_methods"],
        "evaluation_iterations": source["evaluation_iterations"],
        "config_sha256": source["config_sha256"],
        "protocol_lock_sha256": source["protocol_lock_sha256"],
        "analysis_spec_sha256": source["analysis_spec_sha256"],
    }


def main() -> None:
    args = parse_args()
    config, lock = load_config(args.config)
    require_study_open(config)
    require_clean_worktree(args.allow_dirty_worktree)
    output = args.output or config["paths"]["audit_output"]
    rerun = prepare_rerun([output], overwrite=args.overwrite, reason=args.rerun_reason)
    require_new_output(output, args.overwrite)
    report = {
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "metadata": {
            "formal_audit": not args.allow_dirty_worktree,
            "git_commit": git_commit(),
            "git_dirty": git_worktree_dirty(),
            "code_fingerprint": experiment_code_fingerprint(),
            "runtime": environment_summary(),
            "rerun": rerun,
        },
        "config": audit_config(config),
        "manifests": audit_manifests(
            config,
            lock,
            regenerate_entries=not args.skip_entry_regeneration,
        ),
        "source_spatial_experiment": audit_source(lock),
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
    }
    atomic_json_dump(output, report)
    print(f"final-closure protocol audit passed: {output}")


if __name__ == "__main__":
    main()
