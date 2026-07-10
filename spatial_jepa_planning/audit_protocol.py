#!/usr/bin/env python3
"""Fail-fast audit of manifests, task regeneration, and experiment alignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_jepa_planning.common import (
    canonical_layout_hash,
    canonical_task_hash,
    count_by_size,
    sha256_file,
    strict_json_dump,
    validate_manifest_entry,
    validate_manifest_pair,
)
from spatial_jepa_planning.generate_confirmatory_manifest import generate_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="spatial_jepa_planning/configs/default.json"
    )
    parser.add_argument(
        "--protocol-lock",
        default="spatial_jepa_planning/configs/protocol_lock.json",
    )
    parser.add_argument(
        "--output", default="spatial_jepa_planning_runs/protocol_audit.json"
    )
    parser.add_argument("--skip-entry-regeneration", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"protocol lock mismatch for {label}: {actual!r} != {expected!r}"
        )


def merged(defaults: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    result = dict(defaults)
    result.update(variant.get("train", {}))
    return result


def audit_config(config: dict[str, Any]) -> dict[str, Any]:
    seeds = [int(seed) for seed in config.get("seeds", [])]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError(
            "confirmatory matrix requires exactly ten distinct training seeds"
        )
    assert_equal(seeds, list(range(42, 52)), "training seeds")
    representation_names = [
        str(item["name"])
        for item in config.get("representations", [])
        if item.get("enabled", True)
    ]
    planner_variants = [
        item for item in config.get("planners", []) if item.get("enabled", True)
    ]
    planner_names = [str(item["name"]) for item in planner_variants]
    assert_equal(
        representation_names,
        ["spatial_info_sigreg"],
        "enabled representation matrix",
    )
    assert_equal(
        planner_names,
        [
            "r0_raw_value_only",
            "r1_raw_action_ce",
            "r2_raw_bellman_gap",
            "r2d_raw_dilated_bellman_gap",
            "r3_raw_iterative_fixed",
            "r4_raw_iterative_progressive",
            "j0_spatial_feedforward",
            "j1_spatial_iterative_frozen",
            "j2_spatial_iterative_lastblock",
            "j3_spatial_iterative_joint",
        ],
        "enabled planner matrix",
    )
    if len(set(representation_names)) != len(representation_names):
        raise ValueError("representation names must be unique")
    if len(set(planner_names)) != len(planner_names):
        raise ValueError("planner names must be unique")
    if set(representation_names) & set(planner_names):
        raise ValueError("representation and planner names must not share output names")
    if not planner_variants:
        raise ValueError("at least one planner variant must be enabled")
    representation_defaults = config["representation_defaults"]
    enabled_representations = [
        item for item in config.get("representations", []) if item.get("enabled", True)
    ]
    assert_equal(
        enabled_representations[0].get("train", {}),
        {},
        "spatial_info_sigreg.train",
    )
    expected_representation_defaults = {
        "stage": "representation",
        "input_mode": "raw",
        "steps": 30000,
        "map_batch_size": 8,
        "trajectories_per_map": 2,
        "seq_len": 4,
        "lr": 0.001,
        "encoder_lr_multiplier": 0.1,
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "grad_clip": 1.0,
        "log_every": 500,
        "spatial_dim": 64,
        "planning_dim": 64,
        "encoder_blocks": 3,
        "predictor_blocks": 2,
        "ema_momentum": 0.99,
        "sigreg_num_proj": 1024,
        "sigreg_max_tokens": 64,
        "lambda_prediction": 1.0,
        "lambda_sigreg": 0.09,
        "lambda_variance": 0.0,
        "lambda_covariance": 0.0,
        "lambda_map_wall": 0.5,
        "lambda_map_agent": 0.25,
        "lambda_map_goal": 0.25,
        "lambda_map_valid": 0.5,
    }
    for key, expected_value in expected_representation_defaults.items():
        assert_equal(
            representation_defaults.get(key),
            expected_value,
            f"representation_defaults.{key}",
        )
    if int(representation_defaults["sigreg_max_tokens"]) > 128:
        raise ValueError("spatial SIGReg token cap is unsafe; expected <= 128")

    defaults = config["planner_defaults"]
    expected_planner_defaults = {
        "stage": "planner",
        "steps": 30000,
        "map_batch_size": 8,
        "trajectories_per_map": 2,
        "seq_len": 4,
        "lr": 0.001,
        "encoder_lr_multiplier": 0.1,
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "grad_clip": 1.0,
        "log_every": 500,
        "encoder_mode": "frozen",
        "planner_hidden_dim": 64,
        "planner_depth": 4,
        "recall": True,
        "train_iterations": "4,8,16,32,64,128",
        "iteration_schedule": "random",
        "deep_supervision_every": 0,
        "distance_scale": 128.0,
        "lambda_value": 1.0,
        "lambda_action": 1.0,
        "lambda_valid": 0.25,
        "lambda_bellman": 0.5,
        "lambda_gap": 0.5,
        "lambda_convergence": 0.0,
        "gap_margin": 1.0,
        "lambda_joint_representation": 1.0,
        "lambda_planner_map": 0.0,
        "gradient_audit_every": 500,
    }
    for key, expected_value in expected_planner_defaults.items():
        assert_equal(
            defaults.get(key),
            expected_value,
            f"planner_defaults.{key}",
        )
    controlled_keys = (
        "steps",
        "map_batch_size",
        "planner_hidden_dim",
        "recall",
        "lr",
        "encoder_lr_multiplier",
        "weight_decay",
        "scheduler",
        "grad_clip",
        "distance_scale",
    )
    expected = {
        "steps": 30000,
        "map_batch_size": 8,
        "planner_hidden_dim": 64,
        "recall": True,
        "lr": 0.001,
        "encoder_lr_multiplier": 0.1,
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "grad_clip": 1.0,
        "distance_scale": 128.0,
    }
    assert_equal(
        {key: defaults[key] for key in controlled_keys},
        expected,
        "preregistered shared planner budget",
    )
    dependencies: dict[str, str] = {}
    planner_name_set = set(planner_names)
    evaluation_iterations = {
        int(item.strip())
        for item in str(config["evaluation"]["iterations"]).split(",")
        if item.strip()
    }
    confirmatory_hypotheses: list[str] = []
    for variant in planner_variants:
        train = merged(defaults, variant)
        actual = {key: train[key] for key in controlled_keys}
        assert_equal(actual, expected, f"shared planner budget for {variant['name']}")
        input_mode = str(variant.get("input_mode", ""))
        if input_mode not in {"raw", "spatial_jepa"}:
            raise ValueError(f"invalid input_mode for {variant['name']}: {input_mode}")
        if input_mode == "spatial_jepa":
            dependency = str(variant.get("representation", ""))
            if dependency not in representation_names:
                raise ValueError(
                    f"planner {variant['name']} references disabled/missing "
                    f"representation {dependency}"
                )
            dependencies[str(variant["name"])] = dependency
        if str(variant.get("decision_source")) not in {"policy", "value"}:
            raise ValueError(f"planner {variant['name']} needs a valid decision_source")
        baseline = variant.get("comparison_baseline")
        if baseline is not None and baseline not in planner_name_set:
            raise ValueError(
                f"planner {variant['name']} references missing baseline {baseline}"
            )
        if baseline == variant["name"]:
            raise ValueError(f"planner {variant['name']} cannot compare against itself")
        primary_iterations = int(variant["primary_iterations"])
        if primary_iterations not in evaluation_iterations:
            raise ValueError(
                f"primary K={primary_iterations} missing from evaluation curve for "
                f"{variant['name']}"
            )
        planner_type = str(train["planner_type"])
        if planner_type.startswith("feedforward"):
            assert_equal(
                primary_iterations,
                int(train["planner_depth"]),
                f"feedforward primary depth for {variant['name']}",
            )
        else:
            train_iterations = {
                int(item.strip())
                for item in str(train["train_iterations"]).split(",")
                if item.strip()
            }
            if primary_iterations not in train_iterations:
                raise ValueError(
                    f"primary K={primary_iterations} was not trained for "
                    f"{variant['name']}"
                )
            if str(train["iteration_schedule"]) == "fixed":
                assert_equal(
                    train_iterations,
                    {primary_iterations},
                    f"fixed-K schedule for {variant['name']}",
                )
        hypothesis = variant.get("hypothesis")
        if hypothesis and hypothesis.get("role") == "confirmatory":
            if baseline is None:
                raise ValueError("confirmatory hypotheses require a paired baseline")
            if hypothesis.get("test") not in {"superiority", "noninferiority"}:
                raise ValueError(f"invalid confirmatory test for {variant['name']}")
            confirmatory_hypotheses.append(str(variant["name"]))

    protocol = config["protocol"]
    assert_equal(int(protocol["max_steps"]), 128, "max_steps")
    assert_equal(int(protocol["eval_limit"]), 0, "eval_limit")
    assert_equal(int(protocol["eval_max_per_size"]), 0, "eval_max_per_size")
    assert_equal(int(protocol["seen_max_size"]), 21, "seen_max_size")
    assert_equal(str(protocol["action_selection"]), "unmasked", "action_selection")
    assert_equal(
        list(protocol["diagnostic_action_selections"]),
        ["model_valid", "corrected"],
        "diagnostic_action_selections",
    )
    evaluation = config["evaluation"]
    assert_equal(
        str(evaluation["iterations"]),
        "4,8,16,32,64,128,256",
        "evaluation.iterations",
    )
    assert_equal(
        str(evaluation["oracle_vi_iterations"]),
        "32,64,128,256",
        "evaluation.oracle_vi_iterations",
    )
    assert_equal(int(evaluation["seed"]), 42, "evaluation.seed")
    assert_equal(
        int(evaluation["field_states_per_maze"]),
        24,
        "evaluation.field_states_per_maze",
    )
    assert_equal(
        int(evaluation["field_pairs_per_maze"]),
        128,
        "evaluation.field_pairs_per_maze",
    )
    assert_equal(
        str(evaluation["decoded_action_selection"]),
        "predicted",
        "evaluation.decoded_action_selection",
    )
    assert_equal(
        confirmatory_hypotheses,
        [
            "r4_raw_iterative_progressive",
            "j1_spatial_iterative_frozen",
            "j2_spatial_iterative_lastblock",
        ],
        "preregistered confirmatory hypotheses",
    )
    by_name = {str(variant["name"]): variant for variant in planner_variants}
    expected_variant_specs = {
        "r0_raw_value_only": (
            "raw",
            "value",
            4,
            None,
            {
                "planner_type": "feedforward",
                "lambda_action": 0.0,
                "lambda_valid": 0.0,
                "lambda_bellman": 0.0,
                "lambda_gap": 0.0,
            },
        ),
        "r1_raw_action_ce": (
            "raw",
            "policy",
            4,
            "r0_raw_value_only",
            {
                "planner_type": "feedforward",
                "lambda_value": 0.0,
                "lambda_action": 1.0,
                "lambda_valid": 0.25,
                "lambda_bellman": 0.0,
                "lambda_gap": 0.0,
            },
        ),
        "r2_raw_bellman_gap": (
            "raw",
            "policy",
            4,
            "r1_raw_action_ce",
            {"planner_type": "feedforward"},
        ),
        "r2d_raw_dilated_bellman_gap": (
            "raw",
            "policy",
            4,
            "r2_raw_bellman_gap",
            {"planner_type": "feedforward_dilated"},
        ),
        "r3_raw_iterative_fixed": (
            "raw",
            "policy",
            64,
            "r2d_raw_dilated_bellman_gap",
            {
                "planner_type": "iterative",
                "train_iterations": "64",
                "iteration_schedule": "fixed",
            },
        ),
        "r4_raw_iterative_progressive": (
            "raw",
            "policy",
            128,
            "r2d_raw_dilated_bellman_gap",
            {
                "planner_type": "iterative",
                "iteration_schedule": "progressive",
                "deep_supervision_every": 16,
            },
        ),
        "j0_spatial_feedforward": (
            "spatial_jepa",
            "policy",
            4,
            "r2d_raw_dilated_bellman_gap",
            {"planner_type": "feedforward_dilated", "encoder_mode": "frozen"},
        ),
        "j1_spatial_iterative_frozen": (
            "spatial_jepa",
            "policy",
            128,
            "r4_raw_iterative_progressive",
            {
                "planner_type": "iterative",
                "encoder_mode": "frozen",
                "iteration_schedule": "progressive",
                "deep_supervision_every": 16,
            },
        ),
        "j2_spatial_iterative_lastblock": (
            "spatial_jepa",
            "policy",
            128,
            "j1_spatial_iterative_frozen",
            {
                "planner_type": "iterative",
                "encoder_mode": "last_block",
                "lambda_planner_map": 1.0,
                "iteration_schedule": "progressive",
                "deep_supervision_every": 16,
            },
        ),
        "j3_spatial_iterative_joint": (
            "spatial_jepa",
            "policy",
            128,
            "j2_spatial_iterative_lastblock",
            {
                "stage": "joint",
                "planner_type": "iterative",
                "encoder_mode": "last_block",
                "iteration_schedule": "progressive",
                "deep_supervision_every": 16,
            },
        ),
    }
    for name, (
        input_mode,
        decision_source,
        primary_iterations,
        baseline,
        train_overrides,
    ) in expected_variant_specs.items():
        variant = by_name[name]
        assert_equal(variant.get("input_mode"), input_mode, f"{name}.input_mode")
        assert_equal(
            variant.get("decision_source"),
            decision_source,
            f"{name}.decision_source",
        )
        assert_equal(
            int(variant.get("primary_iterations", -1)),
            primary_iterations,
            f"{name}.primary_iterations",
        )
        assert_equal(
            variant.get("comparison_baseline"),
            baseline,
            f"{name}.comparison_baseline",
        )
        assert_equal(variant.get("train", {}), train_overrides, f"{name}.train")
        if input_mode == "spatial_jepa":
            assert_equal(
                variant.get("representation"),
                "spatial_info_sigreg",
                f"{name}.representation",
            )
    for name, baseline, hypothesis in (
        (
            "r4_raw_iterative_progressive",
            "r2d_raw_dilated_bellman_gap",
            {
                "role": "confirmatory",
                "test": "superiority",
                "minimum_effect_sr": 0.03,
            },
        ),
        (
            "j1_spatial_iterative_frozen",
            "r4_raw_iterative_progressive",
            {
                "role": "confirmatory",
                "test": "noninferiority",
                "margin_sr": 0.03,
            },
        ),
        (
            "j2_spatial_iterative_lastblock",
            "j1_spatial_iterative_frozen",
            {
                "role": "confirmatory",
                "test": "superiority",
                "minimum_effect_sr": 0.03,
            },
        ),
    ):
        assert_equal(
            by_name[name].get("comparison_baseline"),
            baseline,
            f"{name}.comparison_baseline",
        )
        assert_equal(by_name[name].get("hypothesis"), hypothesis, f"{name}.hypothesis")
    assert_equal(int(evaluation["bootstrap_samples"]), 20000, "bootstrap_samples")
    assert_equal(
        float(config["inference"]["familywise_alpha"]),
        0.05,
        "inference.familywise_alpha",
    )
    assert_equal(
        str(config["inference"]["primary_metric"]),
        "success",
        "inference.primary_metric",
    )
    assert_equal(
        str(config["inference"]["multiplicity_control"]),
        "bonferroni_simultaneous_ci",
        "inference.multiplicity_control",
    )
    return {
        "seeds": seeds,
        "representations": representation_names,
        "planners": planner_names,
        "representation_dependencies": dependencies,
        "shared_planner_budget": expected,
        "confirmatory_hypotheses": confirmatory_hypotheses,
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    lock = load_json(args.protocol_lock)
    assert_equal(int(lock.get("schema_version", -1)), 2, "schema_version")
    assert_equal(
        str(lock.get("protocol_id")),
        "spatial-jepa-confirm-v2",
        "protocol_id",
    )
    locked_evaluation = lock["evaluation"]
    assert_equal(int(locked_evaluation["max_steps"]), 128, "locked max_steps")
    assert_equal(
        int(locked_evaluation["seen_max_size"]),
        21,
        "locked seen_max_size",
    )
    assert_equal(
        list(locked_evaluation["action_ids"]),
        [1, 2, 3, 4],
        "locked action_ids",
    )
    assert_equal(
        str(locked_evaluation["primary_action_selection"]),
        "unmasked",
        "locked primary_action_selection",
    )
    assert_equal(
        list(locked_evaluation["diagnostic_action_selections"]),
        ["model_valid", "corrected"],
        "locked diagnostic_action_selections",
    )
    assert_equal(
        str(locked_evaluation["learned_field_recompute"]),
        "once_per_task",
        "locked learned_field_recompute",
    )
    assert_equal(
        str(locked_evaluation["decoded_bfs_action_selection"]),
        "predicted_map_only",
        "locked decoded_bfs_action_selection",
    )
    assert_equal(int(locked_evaluation["full_eval_count"]), 900, "full_eval_count")
    paths = config["paths"]
    train_path = paths["train_manifest"]
    development_path = paths["development_manifest"]
    eval_path = paths["eval_manifest"]
    train_entries, eval_entries, train_eval_overlap = validate_manifest_pair(
        train_path, eval_path
    )
    _, development_entries, train_development_overlap = validate_manifest_pair(
        train_path, development_path
    )
    _, _, development_eval_overlap = validate_manifest_pair(development_path, eval_path)

    for label, path, entries in (
        ("train_manifest", train_path, train_entries),
        ("development_manifest", development_path, development_entries),
        ("eval_manifest", eval_path, eval_entries),
    ):
        expected = lock[label]
        assert_equal(sha256_file(path), expected["sha256"], f"{label}.sha256")
        assert_equal(len(entries), int(expected["count"]), f"{label}.count")
        expected_counts = {
            int(key): int(value) for key, value in expected["counts_by_size"].items()
        }
        assert_equal(count_by_size(entries), expected_counts, f"{label}.counts_by_size")

    canonical_layouts: dict[str, set[str]] = {}
    for label, entries in (
        ("train", train_entries),
        ("development", development_entries),
        ("confirmatory", eval_entries),
    ):
        task_hashes = [entry.get("task_hash") for entry in entries]
        if None in task_hashes or len(set(task_hashes)) != len(task_hashes):
            raise ValueError(f"{label} task_hash values must be present and unique")
        layouts: set[str] = set()
        for entry in entries:
            env = validate_manifest_entry(entry)
            layout_hash = canonical_layout_hash(env._maze_mask)
            if layout_hash in layouts:
                raise ValueError(f"duplicate canonical layout within {label}")
            layouts.add(layout_hash)
            if label == "confirmatory":
                assert_equal(
                    entry.get("layout_hash"),
                    layout_hash,
                    "confirmatory canonical layout hash",
                )
                expected_task_hash = canonical_task_hash(
                    maze_size=int(entry["maze_size"]),
                    layout_hash=layout_hash,
                    start_cell=int(entry["start_cell"]),
                    goal_cell=int(entry["goal_cell"]),
                )
                assert_equal(
                    entry.get("task_hash"),
                    expected_task_hash,
                    "confirmatory canonical task hash",
                )
        canonical_layouts[label] = layouts
    for left, right in (
        ("train", "development"),
        ("train", "confirmatory"),
        ("development", "confirmatory"),
    ):
        assert_equal(
            len(canonical_layouts[left] & canonical_layouts[right]),
            0,
            f"canonical layout overlap {left}/{right}",
        )
    assert_equal(
        eval_entries,
        generate_entries(),
        "confirmatory generator reproduction",
    )
    if not args.skip_entry_regeneration:
        all_entries = [*train_entries, *development_entries, *eval_entries]
        for index, entry in enumerate(all_entries, start=1):
            validate_manifest_entry(entry)
            if index % 500 == 0:
                print(f"validated {index}/{len(all_entries)} tasks")

    step_cap_failures = sum(
        int(entry["bfs_path_length"]) > int(config["protocol"]["max_steps"])
        for entry in eval_entries
    )
    assert_equal(
        step_cap_failures,
        int(lock["evaluation"]["step_cap_failures"]),
        "evaluation.step_cap_failures",
    )
    theoretical_max = (len(eval_entries) - step_cap_failures) / len(eval_entries)
    assert_equal(
        theoretical_max,
        float(lock["evaluation"]["expected_exact_oracle_sr"]),
        "evaluation.expected_exact_oracle_sr",
    )
    anchor_audit: list[dict[str, Any]] = []
    for anchor in lock.get("reference_anchors", []):
        source = anchor.get("source")
        expected_hash = anchor.get("source_sha256")
        exists = bool(source and Path(source).exists())
        actual_hash = sha256_file(source) if exists and expected_hash else None
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"reference anchor hash mismatch: {source}")
        anchor_audit.append(
            {
                "name": anchor["name"],
                "source": source,
                "source_exists": exists,
                "hash_verified": bool(expected_hash and actual_hash == expected_hash),
                "scope": anchor["scope"],
                "sr": anchor["sr"],
                "spl": anchor.get("spl"),
            }
        )

    config_audit = audit_config(config)
    report = {
        "status": "passed",
        "config": args.config,
        "protocol_lock": args.protocol_lock,
        "manifests": {
            "train_count": len(train_entries),
            "development_count": len(development_entries),
            "eval_count": len(eval_entries),
            "holdout_overlap": {
                "train_vs_development": train_development_overlap,
                "train_vs_confirmatory": train_eval_overlap,
                "development_vs_confirmatory": development_eval_overlap,
                "canonical_layout": {
                    "train_vs_development": 0,
                    "train_vs_confirmatory": 0,
                    "development_vs_confirmatory": 0,
                },
            },
            "eval_tasks_over_step_cap": step_cap_failures,
            "step_cap_theoretical_max_sr": theoretical_max,
            "entry_regeneration_checked": not args.skip_entry_regeneration,
        },
        "config_audit": config_audit,
        "anchors": anchor_audit,
    }
    strict_json_dump(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
