"""Fail-closed structural audit for the full-900 paired screen."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from final_closure.common import read_jsonl, sha256_file
from vector_jepa_planner_frontier.common import (
    atomic_json_dump,
    validate_manifest_isolation,
)
from vector_jepa_planner_frontier.compat import (
    checkpoint_path,
    validate_source_contract,
)
from vector_jepa_planner_frontier.heads import required_head_names
from vector_jepa_planner_full900_screen.common import (
    load_config,
    load_json,
    resolve_path,
    role_by_name,
    validate_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="vector_jepa_planner_full900_screen/configs/default.json",
    )
    parser.add_argument("--output")
    parser.add_argument("--require-checkpoints", action="store_true")
    return parser.parse_args()


def _leaf_differences(left: Any, right: Any) -> set[str]:
    def visit(left_value: Any, right_value: Any, prefix: str) -> set[str]:
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            output: set[str] = set()
            for key in sorted(set(left_value) | set(right_value)):
                path = f"{prefix}.{key}" if prefix else key
                if key not in left_value or key not in right_value:
                    output.add(path)
                else:
                    output.update(visit(left_value[key], right_value[key], path))
            return output
        return set() if left_value == right_value else {prefix}

    return visit(left.model_dump(mode="json"), right.model_dump(mode="json"), "")


def _manifest_summary(
    rows: list[dict[str, Any]], *, expected_counts: dict[int, int], role: str
) -> dict[str, Any]:
    counts = Counter(int(row["maze_size"]) for row in rows)
    if dict(counts) != expected_counts:
        raise ValueError(f"{role} maze-size counts drifted: {dict(counts)}")
    topology = [(int(row["maze_size"]), int(row["topology_seed"])) for row in rows]
    identities: dict[str, list[Any]] = {
        "topology": topology,
        "layout": [row.get("layout_hash") for row in rows],
        "task": [row.get("task_hash") for row in rows],
    }
    for name, values in identities.items():
        missing = name != "topology" and any(
            not isinstance(value, str) or not value for value in values
        )
        if missing or len(set(values)) != len(values):
            raise ValueError(f"{role} contains missing or duplicate {name} identities")
    return {
        "count": len(rows),
        "size_counts": {str(key): value for key, value in sorted(counts.items())},
        "unique_topologies": len(set(identities["topology"])),
        "unique_layouts": len(set(identities["layout"])),
        "unique_tasks": len(set(identities["task"])),
    }


def audit(
    config: Any, lock: dict[str, Any], *, require_checkpoints: bool
) -> dict[str, Any]:
    validate_lock(config, lock)
    validate_source_contract(config, lock)
    overlaps = validate_manifest_isolation(config)
    expected_seen = {size: 100 for size in range(9, 22, 2)}
    expected_full = {size: 100 for size in range(9, 26, 2)}
    expected_train = {size: 400 for size in range(9, 22, 2)}
    manifest_summaries = {
        "train": _manifest_summary(
            read_jsonl(resolve_path(config.paths.train_manifest)),
            expected_counts=expected_train,
            role="train",
        ),
        "development": _manifest_summary(
            read_jsonl(resolve_path(config.paths.development_manifest)),
            expected_counts=expected_full,
            role="development",
        ),
        "validation": _manifest_summary(
            read_jsonl(resolve_path(config.paths.validation_manifest)),
            expected_counts=expected_seen,
            role="validation",
        ),
        "confirmatory": _manifest_summary(
            read_jsonl(resolve_path(config.paths.confirmatory_manifest)),
            expected_counts=expected_full,
            role="confirmatory",
        ),
    }

    methods = {method.name: method for method in config.methods}
    q1 = [
        method
        for method in config.methods
        if role_by_name(config, method.name).phase == "Q1"
    ]
    for method in q1:
        if method.name == "q1_control_categorical_cem_1x":
            continue
        differences = _leaf_differences(
            methods["q1_control_categorical_cem_1x"], method
        )
        if differences != {"name", "planner.kind"}:
            raise ValueError(
                f"Q1 changed more than search for {method.name}: {differences}"
            )

    matched_pairs = {
        "q2b_vector_dts": (
            "q2b_control_dts_uniform_expansion",
            {"name", "control.dts_expansion"},
        ),
        "q2b_bidirectional": (
            "q2b_control_bidirectional_forward",
            {"name", "planner.kind"},
        ),
        "q2b_denoising_icem": (
            "q2b_control_denoising_uniform",
            {
                "name",
                "proposal.kind",
                "proposal.learned_weight",
                "proposal.uniform_weight",
                "component_checkpoint_required",
            },
        ),
        "q2c_hard_negative_ranker": (
            "q2c_control_random_negative_ranker",
            {"name", "control.ranker_negatives"},
        ),
    }
    pair_audit: dict[str, Any] = {}
    for candidate, (control, allowed) in matched_pairs.items():
        differences = _leaf_differences(methods[candidate], methods[control])
        if differences != allowed:
            raise ValueError(
                f"matched control drift for {candidate}: "
                f"expected={allowed} actual={differences}"
            )
        pair_audit[candidate] = {"control": control, "differences": sorted(differences)}

    dts_direct_differences = _leaf_differences(
        methods["q2b_vector_dts"], methods["q2b_control_dts_direct"]
    )
    if dts_direct_differences != {"name", "control.dts_expansion"}:
        raise ValueError("DTS direct diagnostic drifted from the learned DTS system")

    if any(method.planner.budget.transition_limit != 768 for method in config.methods):
        raise ValueError("a method escaped the 768-transition 1x budget")
    if any(method.track != "F" for method in config.methods):
        raise ValueError("joint backbone updates are forbidden in this screen")
    for method in config.methods:
        requires_heads = bool(required_head_names(method))
        if method.component_checkpoint_required != requires_heads:
            raise ValueError(
                f"component-checkpoint flag disagrees with active heads: {method.name}"
            )

    checkpoints = {
        str(seed): str(checkpoint_path(config, seed=seed))
        for seed in config.replication.final_backbone_seeds
    }
    missing = [path for path in checkpoints.values() if not Path(path).exists()]
    if require_checkpoints and missing:
        raise FileNotFoundError(f"missing historical checkpoints: {missing}")
    checkpoint_sha256s = (
        {seed: sha256_file(path) for seed, path in checkpoints.items()}
        if require_checkpoints
        else {}
    )
    if require_checkpoints and len(set(checkpoint_sha256s.values())) != len(
        checkpoint_sha256s
    ):
        raise ValueError("historical backbone seeds contain duplicate checkpoints")
    return {
        "schema": "vector-jepa-full900-screen-audit-v1",
        "status": "pass",
        "protocol_id": config.protocol_id,
        "manifest_summaries": manifest_summaries,
        "manifest_overlaps": overlaps,
        "matched_control_audit": pair_audit,
        "secondary_control_audit": {
            "q2b_vector_dts": {
                "control": "q2b_control_dts_direct",
                "differences": sorted(dts_direct_differences),
                "selection_role": "descriptive_only_not_advancement_control",
            }
        },
        "method_count": len(config.methods),
        "advancement_candidate_count": sum(
            role.advancement_eligible for role in config.method_roles
        ),
        "all_methods_budget_transitions": 768,
        "source_checkpoints": checkpoints,
        "source_checkpoint_sha256s": checkpoint_sha256s,
        "missing_source_checkpoints": missing,
        "source_checkpoints_required": require_checkpoints,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    result = audit(config, lock, require_checkpoints=args.require_checkpoints)
    if args.output:
        atomic_json_dump(args.output, result)
    print(
        f"audit pass: methods={result['method_count']} "
        f"candidates={result['advancement_candidate_count']} tasks=900"
    )


if __name__ == "__main__":
    main()
