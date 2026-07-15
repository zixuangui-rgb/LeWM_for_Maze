"""Freeze one scorer-compatible Q1 search parent after the seed-42 full-900 run."""

from __future__ import annotations

import argparse

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import method_by_name
from vector_jepa_planner_full900_screen.analysis import (
    compute_per_decision,
    load_result,
    sr,
)
from vector_jepa_planner_full900_screen.common import (
    atomic_json_dump,
    load_config,
    load_json,
    resolve_path,
    result_path,
    role_by_name,
    validate_lock,
)
from vector_jepa_planner_full900_screen.methods import validate_q1_selection
from vector_jepa_planner_full900_screen.parity import validate_q0_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="vector_jepa_planner_full900_screen/configs/default.json",
    )
    return parser.parse_args()


def _assert_bridge_parity(left: dict, right: dict) -> None:
    fields = (
        "task_id",
        "success",
        "path_length",
        "invalid_actions",
        "loop_or_cycle",
        "final_bfs_distance",
    )
    for left_row, right_row in zip(left["tasks"], right["tasks"], strict=True):
        if any(left_row.get(field) != right_row.get(field) for field in fields):
            raise ValueError("instrumented categorical-CEM bridge diverged from B0")
        left_actions = [
            int(trace["executed_action"]) for trace in left_row["decision_traces"]
        ]
        right_actions = [
            int(trace["executed_action"]) for trace in right_row["decision_traces"]
        ]
        if left_actions != right_actions:
            raise ValueError(
                "instrumented categorical-CEM bridge changed executed actions"
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_lock(config, lock)
    q0_parity_sha256s = validate_q0_gate(config, lock)
    output = resolve_path(config.paths.p2_selection)
    if output.exists():
        raise FileExistsError("Q1 parent decision is immutable")
    baseline = method_by_name(config, "b0_legacy_l2_cem")
    bridge = method_by_name(config, "q1_control_categorical_cem_1x")
    inputs: dict[str, str] = {}
    for action in config.replication.action_selections:
        b0_result = load_result(
            config,
            lock,
            method=baseline,
            backbone_seed=42,
            planner_seed=0,
            action_selection=action,
        )
        bridge_result = load_result(
            config,
            lock,
            method=bridge,
            backbone_seed=42,
            planner_seed=0,
            action_selection=action,
        )
        _assert_bridge_parity(b0_result, bridge_result)
        inputs[f"{baseline.name}:{action}"] = sha256_file(
            result_path(
                config,
                method=baseline.name,
                backbone_seed=42,
                planner_seed=0,
                action_selection=action,
            )
        )
        inputs[f"{bridge.name}:{action}"] = sha256_file(
            result_path(
                config,
                method=bridge.name,
                backbone_seed=42,
                planner_seed=0,
                action_selection=action,
            )
        )
    parent_names = [
        role.name
        for role in config.method_roles
        if role.phase == "Q1" and role.role in {"candidate", "bridge_control"}
    ]
    rows = []
    for name in parent_names:
        method = method_by_name(config, name)
        corrected = load_result(
            config,
            lock,
            method=method,
            backbone_seed=42,
            planner_seed=0,
            action_selection="corrected_v1",
        )
        unmasked = load_result(
            config,
            lock,
            method=method,
            backbone_seed=42,
            planner_seed=0,
            action_selection="unmasked",
        )
        for action in config.replication.action_selections:
            inputs[f"{name}:{action}"] = sha256_file(
                result_path(
                    config,
                    method=name,
                    backbone_seed=42,
                    planner_seed=0,
                    action_selection=action,
                )
            )
        compute = compute_per_decision(corrected)
        rows.append(
            {
                "method": name,
                "corrected_sr": sr(corrected),
                "corrected_ood_sr": sr(corrected, "ood"),
                "unmasked_sr": sr(unmasked),
                "planner_forward_calls_per_decision": compute["planner_forward_calls"],
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["corrected_sr"],
            -row["corrected_ood_sr"],
            -row["unmasked_sr"],
            row["planner_forward_calls_per_decision"],
            row["method"],
        ),
    )
    selected = ranked[0]["method"]
    if role_by_name(config, selected).phase != "Q1":
        raise AssertionError("Q1 selection escaped its candidate family")
    payload = {
        "schema": "vector-jepa-full900-q1-parent-v1",
        "protocol_id": config.protocol_id,
        "quick_spec_sha256": lock["quick_spec_sha256"],
        "q0_parity_sha256s": q0_parity_sha256s,
        "selection_metric": "seed42_full900_corrected_sr",
        "tie_breaks": [
            "corrected_ood_sr",
            "unmasked_sr",
            "planner_forward_calls_per_decision",
            "method_name",
        ],
        "categorical_bridge_exact_task_parity": True,
        "selected_parent": selected,
        "ranked_candidates": ranked,
        "input_sha256s": dict(sorted(inputs.items())),
    }
    atomic_json_dump(output, payload)
    validate_q1_selection(config, lock)
    print(f"frozen Q1 parent: {selected}")


if __name__ == "__main__":
    main()
