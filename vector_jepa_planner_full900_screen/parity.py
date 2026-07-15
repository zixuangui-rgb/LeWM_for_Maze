"""Compare the quick B0 against the untouched final_closure evaluator task by task."""

from __future__ import annotations

import argparse
from typing import Any

from final_closure.common import (
    read_jsonl,
    sha256_file,
    summarize_rows,
    validate_task_rows,
)
from spatial_jepa_planning.common import task_id
from vector_jepa_planner_frontier.common import method_by_name, validate_finite_tree
from vector_jepa_planner_frontier.compat import checkpoint_path
from vector_jepa_planner_full900_screen.analysis import load_result
from vector_jepa_planner_full900_screen.common import (
    atomic_json_dump,
    load_config,
    load_json,
    resolve_path,
    result_path,
    validate_lock,
)

PARITY_FIELDS = (
    "task_id",
    "maze_size",
    "topology_seed",
    "start_cell",
    "goal_cell",
    "optimal_length",
    "success",
    "path_length",
    "spl",
    "invalid_actions",
    "repeat_states",
    "max_state_visits",
    "loop_or_cycle",
    "final_bfs_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="vector_jepa_planner_full900_screen/configs/default.json",
    )
    parser.add_argument(
        "--action-selection", choices=("corrected_v1", "unmasked"), required=True
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_rows = reference["results"]["task_rows"]
    candidate_rows = candidate["tasks"]
    if len(reference_rows) != 900 or len(candidate_rows) != 900:
        raise ValueError("B0 parity requires both complete full-900 outputs")
    validate_task_rows(reference_rows, 900)
    validate_task_rows(candidate_rows, 900)
    validate_finite_tree(reference)
    validate_finite_tree(candidate)
    mismatches: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(
        zip(reference_rows, candidate_rows, strict=True)
    ):
        changed = {
            key: [left.get(key), right.get(key)]
            for key in PARITY_FIELDS
            if left.get(key) != right.get(key)
        }
        if changed:
            mismatches.append({"task_index": index, "differences": changed})
            continue
        reference_actions = left.get("executed_actions")
        candidate_actions = [
            int(trace["executed_action"]) for trace in right.get("decision_traces", [])
        ]
        if (
            not isinstance(reference_actions, list)
            or [int(action) for action in reference_actions] != candidate_actions
        ):
            mismatches.append(
                {
                    "task_index": index,
                    "differences": {
                        "executed_actions": [reference_actions, candidate_actions]
                    },
                }
            )
    if mismatches:
        raise ValueError(f"B0 parity failed for {len(mismatches)} tasks")
    return {
        "schema": "vector-jepa-full900-b0-parity-v1",
        "status": "pass",
        "task_count": 900,
        "compared_fields": list(PARITY_FIELDS),
        "executed_actions_compared": True,
        "mismatch_count": 0,
    }


def validate_q0_gate(config: Any, lock: dict[str, Any]) -> dict[str, str]:
    """Validate both parity artifacts and the exact result files they bind."""

    digests: dict[str, str] = {}
    for action in config.replication.action_selections:
        parity_path = (
            resolve_path(config.paths.run_root) / "parity" / f"parity_{action}.json"
        )
        value = load_json(parity_path)
        reference_action = "corrected" if action == "corrected_v1" else "unmasked"
        reference_path = (
            resolve_path(config.paths.run_root)
            / "parity"
            / f"reference_seed42_{reference_action}.json"
        )
        candidate_path = result_path(
            config,
            method="b0_legacy_l2_cem",
            backbone_seed=42,
            planner_seed=0,
            action_selection=action,
        )
        if (
            value.get("schema") != "vector-jepa-full900-b0-parity-v1"
            or value.get("status") != "pass"
            or value.get("task_count") != 900
            or value.get("compared_fields") != list(PARITY_FIELDS)
            or value.get("protocol_id") != config.protocol_id
            or value.get("quick_spec_sha256") != lock["quick_spec_sha256"]
            or value.get("action_selection") != action
            or value.get("executed_actions_compared") is not True
            or value.get("mismatch_count") != 0
            or not reference_path.exists()
            or not candidate_path.exists()
            or value.get("reference_sha256") != sha256_file(reference_path)
            or value.get("candidate_sha256") != sha256_file(candidate_path)
        ):
            raise RuntimeError(f"Q0 parity gate failed: {parity_path}")
        digests[action] = sha256_file(parity_path)
    return digests


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_lock(config, lock)
    reference_path = resolve_path(args.reference)
    candidate_path = resolve_path(args.candidate)
    output_path = resolve_path(args.output)
    if output_path.exists():
        raise FileExistsError("Q0 parity artifact is immutable")
    expected_reference_action = (
        "corrected" if args.action_selection == "corrected_v1" else "unmasked"
    )
    expected_reference_path = (
        resolve_path(config.paths.run_root)
        / "parity"
        / f"reference_seed42_{expected_reference_action}.json"
    )
    baseline = method_by_name(config, "b0_legacy_l2_cem")
    expected_candidate_path = result_path(
        config,
        method=baseline.name,
        backbone_seed=42,
        planner_seed=0,
        action_selection=args.action_selection,
    )
    if (
        reference_path.resolve() != expected_reference_path.resolve()
        or candidate_path.resolve() != expected_candidate_path.resolve()
    ):
        raise ValueError("Q0 parity inputs are not the canonical frozen paths")
    reference = load_json(reference_path)
    candidate = load_result(
        config,
        lock,
        method=baseline,
        backbone_seed=42,
        planner_seed=0,
        action_selection=args.action_selection,
    )
    reference_metadata = reference.get("metadata", {})
    candidate_metadata = candidate.get("metadata", {})
    source_path = checkpoint_path(config, seed=42)
    if (
        reference_metadata.get("protocol_id") != config.protocol_id
        or reference_metadata.get("quick_spec_sha256") != lock["quick_spec_sha256"]
        or reference_metadata.get("code_fingerprint") != lock["code_fingerprint"]
        or reference_metadata.get("role") != "q0_untouched_final_closure_reference"
        or reference_metadata.get("source_config_sha256")
        != lock["source_baseline"]["config_sha256"]
        or reference_metadata.get("source_lock_sha256")
        != lock["source_baseline"]["lock_sha256"]
        or resolve_path(str(reference_metadata.get("checkpoint", ""))).resolve()
        != source_path.resolve()
        or reference_metadata.get("checkpoint_sha256") != sha256_file(source_path)
        or reference_metadata.get("training_seed") != 42
        or reference_metadata.get("evaluation_seed") != config.protocol.evaluation_seed
        or reference_metadata.get("action_selection") != expected_reference_action
    ):
        raise ValueError("Q0 reference provenance mismatch")
    if (
        candidate_metadata.get("protocol_id") != config.protocol_id
        or candidate_metadata.get("quick_spec_sha256") != lock["quick_spec_sha256"]
        or candidate.get("action_selection") != args.action_selection
    ):
        raise ValueError("Q0 candidate provenance mismatch")
    manifest_rows = read_jsonl(resolve_path(config.paths.development_manifest))
    expected_task_ids = [task_id(row) for row in manifest_rows]
    reference_rows = reference.get("results", {}).get("task_rows", [])
    if [str(row.get("task_id")) for row in reference_rows] != expected_task_ids:
        raise ValueError("Q0 reference task order/hash mismatch")
    expected_reference_summary = summarize_rows(
        reference_rows,
        seen_max_size=config.protocol.seen_max_size,
        max_steps=config.protocol.max_steps,
    )
    if reference.get("results", {}).get("navigation") != expected_reference_summary:
        raise ValueError("Q0 reference summary no longer matches its task rows")
    result = compare(reference, candidate)
    result.update(
        {
            "protocol_id": config.protocol_id,
            "quick_spec_sha256": lock["quick_spec_sha256"],
            "action_selection": args.action_selection,
            "reference_sha256": sha256_file(reference_path),
            "candidate_sha256": sha256_file(candidate_path),
        }
    )
    atomic_json_dump(output_path, result)
    print(f"B0 parity pass: {result['task_count']} tasks")


__all__ = ["compare", "validate_q0_gate"]


if __name__ == "__main__":
    main()
