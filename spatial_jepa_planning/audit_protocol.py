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
    count_by_size,
    sha256_file,
    strict_json_dump,
    validate_manifest_entry,
    validate_manifest_pair,
)


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
    if len(set(seeds)) < 3:
        raise ValueError(
            "scientific matrix requires at least three distinct training seeds"
        )
    representation_names = [
        str(item["name"])
        for item in config.get("representations", [])
        if item.get("enabled", True)
    ]
    planner_variants = [
        item for item in config.get("planners", []) if item.get("enabled", True)
    ]
    planner_names = [str(item["name"]) for item in planner_variants]
    if len(set(representation_names)) != len(representation_names):
        raise ValueError("representation names must be unique")
    if len(set(planner_names)) != len(planner_names):
        raise ValueError("planner names must be unique")
    if set(representation_names) & set(planner_names):
        raise ValueError("representation and planner names must not share output names")
    if not planner_variants:
        raise ValueError("at least one planner variant must be enabled")
    representation_defaults = config["representation_defaults"]
    if int(representation_defaults["sigreg_max_tokens"]) > 128:
        raise ValueError("spatial SIGReg token cap is unsafe; expected <= 128")

    defaults = config["planner_defaults"]
    controlled_keys = (
        "steps",
        "map_batch_size",
        "lr",
        "encoder_lr_multiplier",
        "weight_decay",
        "scheduler",
        "grad_clip",
        "distance_scale",
    )
    expected = {key: defaults[key] for key in controlled_keys}
    dependencies: dict[str, str] = {}
    planner_name_set = set(planner_names)
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

    protocol = config["protocol"]
    assert_equal(int(protocol["max_steps"]), 128, "max_steps")
    assert_equal(str(protocol["action_selection"]), "corrected", "action_selection")
    evaluation = config["evaluation"]
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
    return {
        "seeds": seeds,
        "representations": representation_names,
        "planners": planner_names,
        "representation_dependencies": dependencies,
        "shared_planner_budget": expected,
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    lock = load_json(args.protocol_lock)
    paths = config["paths"]
    train_path = paths["train_manifest"]
    eval_path = paths["eval_manifest"]
    train_entries, eval_entries, overlap = validate_manifest_pair(train_path, eval_path)

    for label, path, entries in (
        ("train_manifest", train_path, train_entries),
        ("eval_manifest", eval_path, eval_entries),
    ):
        expected = lock[label]
        assert_equal(sha256_file(path), expected["sha256"], f"{label}.sha256")
        assert_equal(len(entries), int(expected["count"]), f"{label}.count")
        expected_counts = {
            int(key): int(value) for key, value in expected["counts_by_size"].items()
        }
        assert_equal(count_by_size(entries), expected_counts, f"{label}.counts_by_size")

    task_hashes = [entry.get("task_hash") for entry in eval_entries]
    if None in task_hashes or len(set(task_hashes)) != len(task_hashes):
        raise ValueError("eval task_hash values must be present and unique")
    if not args.skip_entry_regeneration:
        for index, entry in enumerate([*train_entries, *eval_entries], start=1):
            validate_manifest_entry(entry)
            if index % 500 == 0:
                print(
                    f"validated {index}/{len(train_entries) + len(eval_entries)} tasks"
                )

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
            "eval_count": len(eval_entries),
            "holdout_overlap": overlap,
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
