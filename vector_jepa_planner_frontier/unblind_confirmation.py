"""Validate every opaque run, then materialize the named confirmatory results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_bytes_dump,
    atomic_json_dump,
    load_json,
    load_study_config,
    require_clean_worktree,
    resolve_path,
    validate_finite_tree,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.confirmation import load_confirmation_artifacts
from vector_jepa_planner_frontier.effective_methods import (
    effective_method_sha256,
    resolve_effective_method,
)
from vector_jepa_planner_frontier.evaluate import candidate_trace_path
from vector_jepa_planner_frontier.summarize import (
    candidate_mechanism_summary,
    load_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    return parser.parse_args()


def _validated_opaque_result(
    row: dict[str, Any],
    *,
    config: Any,
    lock: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = Path(str(row["opaque_output"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    method = resolve_effective_method(config, lock, str(row["method"]))
    value = load_result(
        path,
        analysis_hash=analysis_spec_sha256(config, lock),
        method=str(row["method"]),
        backbone_seed=int(row["backbone_seed"]),
        planner_seed=int(row["planner_seed"]),
        search_seed=int(row["search_seed"]),
        split_role="confirmatory",
        action_selection=str(row["action_selection"]),
        expected_count=int(lock["confirmatory_manifest"]["count"]),
        expected_manifest_sha256=lock["confirmatory_manifest"]["sha256"],
        expected_code_fingerprint=lock["code_fingerprint"],
        expected_method_sha256=effective_method_sha256(method),
    )
    if value.get("opaque_run_id") != row["run_id"]:
        raise ValueError(f"opaque result run-id mismatch: {path}")
    candidate_mechanism_summary(
        value,
        method=str(row["method"]),
        backbone_seed=int(row["backbone_seed"]),
        planner_seed=int(row["planner_seed"]),
        search_seed=int(row["search_seed"]),
        action_selection=str(row["action_selection"]),
    )
    return path, value


def _publish_named_result(
    row: dict[str, Any],
    *,
    opaque_path: Path,
    value: dict[str, Any],
) -> dict[str, str]:
    formal_path = Path(str(row["formal_output"]))
    formal_candidate = candidate_trace_path(formal_path)
    opaque_candidate = Path(str(value["candidate_traces"]["path"]))
    opaque_result_sha = sha256_file(opaque_path)
    opaque_candidate_sha = sha256_file(opaque_candidate)
    if opaque_candidate_sha != value["candidate_traces"]["sha256"]:
        raise ValueError(
            f"candidate hash changed during unblinding: {opaque_candidate}"
        )
    if formal_candidate.exists():
        if sha256_file(formal_candidate) != opaque_candidate_sha:
            raise ValueError(f"partial unblinding conflict: {formal_candidate}")
    else:
        atomic_bytes_dump(formal_candidate, opaque_candidate.read_bytes())
    named = dict(value)
    named["candidate_traces"] = {
        **dict(value["candidate_traces"]),
        "path": str(formal_candidate),
        "sha256": opaque_candidate_sha,
    }
    named["unblinding"] = {
        "schema": "vector-jepa-unblinded-result-v1",
        "run_id": row["run_id"],
        "opaque_result": str(opaque_path),
        "opaque_result_sha256": opaque_result_sha,
        "opaque_candidate_sha256": opaque_candidate_sha,
    }
    validate_finite_tree(named)
    if formal_path.exists():
        existing = load_json(formal_path)
        if existing != named:
            raise ValueError(f"partial unblinding conflict: {formal_path}")
    else:
        atomic_json_dump(formal_path, named)
    return {
        "run_id": str(row["run_id"]),
        "opaque_result_sha256": opaque_result_sha,
        "formal_result": str(formal_path),
        "formal_result_sha256": sha256_file(formal_path),
        "formal_candidate_sha256": sha256_file(formal_candidate),
    }


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    require_clean_worktree(allow_dirty=False)
    marker_path = resolve_path(config.paths.confirmation_unblinded)
    if marker_path.exists():
        raise FileExistsError("the confirmatory family is already unblinded")
    confirmation, mapping, _ = load_confirmation_artifacts(
        config, lock, require_opened=True
    )
    rows = list(mapping["runs"])
    formal_paths = [str(row["formal_output"]) for row in rows]
    if len(formal_paths) != len(set(formal_paths)):
        raise ValueError("confirmation mapping contains duplicate formal outputs")

    # First validate the entire blinded family. No named result is published until
    # this pass succeeds for every run and every candidate-trace artifact.
    validated = [
        (row, *_validated_opaque_result(row, config=config, lock=lock)) for row in rows
    ]
    published = [
        _publish_named_result(row, opaque_path=path, value=value)
        for row, path, value in validated
    ]
    marker = {
        "schema": "vector-jepa-confirmation-unblinded-v1",
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "confirmation_lock_sha256": sha256_file(
            resolve_path(config.paths.confirmation_lock)
        ),
        "mapping_sha256": confirmation["mapping_sha256"],
        "schedule_sha256": confirmation["schedule_sha256"],
        "run_count": len(rows),
        "unblinded_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": published,
    }
    atomic_json_dump(marker_path, marker)


if __name__ == "__main__":
    main()
