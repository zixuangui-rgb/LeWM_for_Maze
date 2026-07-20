#!/usr/bin/env python3
"""Require exact behavioral parity with historical J0/J1 static evaluations."""

from __future__ import annotations

import argparse
import math
from typing import Any

from air_jepa.stage0_workspace.checkpoints import verify_source_lock
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    formal_runtime_valid,
    format_template,
    load_config,
    prepare_new_output,
    read_json,
    relative_path,
    require_clean_worktree,
    runtime_signature,
    sha256_file,
    signed_payload,
)
from air_jepa.stage0_workspace.protocol import (
    verify_package_lock,
    verify_protocol_lock,
)

CORE_FIELDS = (
    "task_id",
    "maze_size",
    "topology_seed",
    "start_cell",
    "goal_cell",
    "optimal_length",
    "success",
    "path_length",
    "invalid_actions",
    "repeat_states",
    "max_state_visits",
    "loop_or_cycle",
    "final_bfs_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _historical_result(payload: dict[str, Any], k: int) -> dict[str, Any]:
    results = payload.get("results")
    if isinstance(results, dict) and str(k) in results:
        result = results[str(k)]
    elif "task_rows" in payload and "navigation" in payload:
        result = payload
    else:
        raise ValueError(f"cannot locate K={k} historical result payload")
    if not isinstance(result, dict):
        raise ValueError("historical result must be an object")
    return result


def _rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(row["task_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate task IDs in bridge result")
    return indexed


def compare_result(
    historical: dict[str, Any],
    current: dict[str, Any],
    *,
    k: int,
) -> dict[str, Any]:
    old = _historical_result(historical, k)
    old_rows = _rows_by_id(old["task_rows"])
    new_rows = _rows_by_id(current["task_rows"])
    if set(old_rows) != set(new_rows):
        raise ValueError("historical/current bridge task sets differ")
    mismatches: list[dict[str, Any]] = []
    for task_id in sorted(old_rows):
        left = old_rows[task_id]
        right = new_rows[task_id]
        for field in CORE_FIELDS:
            if left.get(field) != right.get(field):
                mismatches.append(
                    {
                        "task_id": task_id,
                        "field": field,
                        "historical": left.get(field),
                        "current": right.get(field),
                    }
                )
        if not math.isclose(
            float(left.get("spl", 0.0)),
            float(right.get("spl", 0.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            mismatches.append(
                {
                    "task_id": task_id,
                    "field": "spl",
                    "historical": left.get("spl"),
                    "current": right.get("spl"),
                }
            )
    old_navigation = old["navigation"]
    new_navigation = current["navigation"]
    for partition in ("overall", "seen", "ood"):
        for metric in ("n", "sr", "spl", "invalid_rate", "loop_or_cycle_rate"):
            left = old_navigation[partition][metric]
            right = new_navigation[partition][metric]
            if isinstance(left, float) or isinstance(right, float):
                equal = math.isclose(
                    float(left), float(right), rel_tol=0.0, abs_tol=1e-12
                )
            else:
                equal = left == right
            if not equal:
                mismatches.append(
                    {
                        "task_id": "aggregate",
                        "field": f"{partition}.{metric}",
                        "historical": left,
                        "current": right,
                    }
                )
    return {
        "task_count": len(old_rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "exact_parity": not mismatches,
    }


def main() -> None:
    args = parse_args()
    require_clean_worktree(allow_dirty=False)
    config = load_config(args.config)
    protocol = verify_protocol_lock(config)
    package = verify_package_lock(config)
    source = verify_source_lock(config)
    comparisons: dict[str, Any] = {}
    bridge_runtime: dict[str, Any] | None = None
    bridge_runtime_signatures: set[tuple[Any, ...]] = set()
    for seed in config.seeds:
        for method, historical_template, k in (
            ("j0_static", config.paths.historical_j0_result_template, 4),
            ("j1_static", config.paths.historical_j1_result_template, 128),
        ):
            historical_path = format_template(historical_template, seed=seed)
            current_path = format_template(
                config.paths.result_template,
                split_role="historical",
                method=method,
                seed=seed,
                action_protocol="unmasked",
                k=k,
            )
            if not historical_path.is_file() or not current_path.is_file():
                raise FileNotFoundError(
                    f"bridge comparison needs both {historical_path} and {current_path}"
                )
            source_key = (
                "historical_j0_result"
                if method == "j0_static"
                else "historical_j1_result"
            )
            source_result = source["records"][str(seed)][source_key]
            if (
                relative_path(historical_path) != source_result["path"]
                or sha256_file(historical_path) != source_result["file_sha256"]
            ):
                raise ValueError(
                    "historical source result differs from source lock: "
                    f"{historical_path}"
                )
            current = read_json(current_path)
            if current.get("schema") != "air-jepa-stage0-evaluation-v1":
                raise ValueError(f"bridge result schema mismatch: {current_path}")
            metadata = current.get("metadata", {})
            expected_metadata = {
                "experiment_id": config.experiment_id,
                "method": method,
                "seed": seed,
                "k": k,
                "split_role": "historical",
                "evidence_role": "HISTORICAL_BRIDGE",
                "action_protocol": "unmasked",
                "intervention": "normal",
                "task_count": 900,
                "max_steps": config.evaluation.max_steps,
                "manifest": relative_path(
                    config.paths.historical_confirmatory_manifest
                ),
                "manifest_sha256": sha256_file(
                    config.paths.historical_confirmatory_manifest
                ),
                "formal": True,
                "protocol_sha256": protocol["protocol_sha256"],
                "package_sha256": package["package_sha256"],
                "code_fingerprint": package["code_fingerprint"],
                "source_lock_sha256": source["source_lock_sha256"],
                "git_dirty": False,
            }
            for field, expected in expected_metadata.items():
                if metadata.get(field) != expected:
                    raise ValueError(
                        f"bridge result {field} mismatch in {current_path}"
                    )
            checkpoint_key = "j0" if method == "j0_static" else "j1"
            if metadata.get("checkpoint_sha256") != source["records"][str(seed)][
                checkpoint_key
            ]["file_sha256"]:
                raise ValueError(
                    f"bridge result checkpoint lineage mismatch in {current_path}"
                )
            runtime = metadata.get("runtime", {})
            if not formal_runtime_valid(runtime):
                raise ValueError(
                    f"bridge result runtime is not deterministic H800: {current_path}"
                )
            bridge_runtime_signatures.add(runtime_signature(runtime))
            bridge_runtime = bridge_runtime or runtime
            key = f"{method}_seed{seed}"
            comparisons[key] = {
                "historical_path": relative_path(historical_path),
                "historical_sha256": sha256_file(historical_path),
                "current_path": relative_path(current_path),
                "current_sha256": sha256_file(current_path),
                **compare_result(read_json(historical_path), current, k=k),
            }
    failures = [key for key, value in comparisons.items() if not value["exact_parity"]]
    if len(bridge_runtime_signatures) != 1 or bridge_runtime is None:
        raise ValueError("historical bridge results mix incompatible runtimes")
    payload = signed_payload(
        {
            "schema": "air-jepa-stage0-bridge-audit-v1",
            "experiment_id": config.experiment_id,
            "protocol_sha256": protocol["protocol_sha256"],
            "package_sha256": package["package_sha256"],
            "source_lock_sha256": source["source_lock_sha256"],
            "runtime": bridge_runtime,
            "comparisons": comparisons,
            "passed": not failures,
            "failures": failures,
        },
        "bridge_audit_sha256",
    )
    output = args.output or (
        str(config.paths.run_root) + "/audits/historical_bridge_parity.json"
    )
    prepare_new_output(output)
    atomic_json_dump(output, payload)
    if failures:
        raise RuntimeError(f"historical bridge parity failed: {failures}")
    print(f"saved={relative_path(output)} exact_parity=true")


if __name__ == "__main__":
    main()
