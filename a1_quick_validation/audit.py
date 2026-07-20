"""Fail-closed scientific and compatibility audit for the quick package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from a1_quick_validation import ALL_METHODS, NEW_METHODS
from a1_quick_validation.common import (
    DEFAULT_PROFILE,
    atomic_json_dump,
    canonical_json_sha256,
    resolve_path,
)
from a1_quick_validation.profile import verify_package_lock
from distance_head_study.common import (
    load_study_config,
    read_jsonl,
    sha256_file,
)
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.protocol import verify_protocol_lock


def _task_ids(path: str | Path) -> set[str]:
    return {str(row["task_hash"]) for row in read_jsonl(path)}


def run_audit(profile_path: str | Path = DEFAULT_PROFILE) -> dict[str, Any]:
    profile, package_lock, quick_lock = verify_package_lock(profile_path)
    quick_config = load_study_config(profile.paths.quick_config)
    source_config = load_study_config(profile.paths.source_config)
    source_lock = verify_protocol_lock(source_config, regenerate=True)
    verify_protocol_lock(quick_config, regenerate=True)
    locked_sections = ("splits", "seeds", "planner", "training", "analysis")
    parity = {}
    for section in locked_sections:
        source_value = getattr(source_config, section).model_dump(mode="json")
        quick_value = getattr(quick_config, section).model_dump(mode="json")
        if source_value != quick_value:
            raise ValueError(f"quick protocol changed locked {section}")
        parity[section] = canonical_json_sha256(source_value)
    manifest_roles = ("train", "legacy", "cal", "screen", "select", "confirm", "stress")
    manifests = {}
    for role in manifest_roles:
        source_path = resolve_path(getattr(source_config.paths, f"{role}_manifest"))
        quick_path = resolve_path(getattr(quick_config.paths, f"{role}_manifest"))
        if source_path != quick_path or sha256_file(source_path) != sha256_file(
            quick_path
        ):
            raise ValueError(f"quick protocol changed the {role} manifest")
        manifests[role] = {
            "path": source_path.as_posix(),
            "sha256": sha256_file(source_path),
            "count": len(read_jsonl(source_path)),
        }
    train_tasks = _task_ids(quick_config.paths.train_manifest)
    heldout_overlap = {}
    for role in ("screen", "select", "confirm"):
        overlap = train_tasks & _task_ids(
            getattr(quick_config.paths, f"{role}_manifest")
        )
        heldout_overlap[role] = len(overlap)
        if overlap:
            raise ValueError(f"train/{role} task leakage")
    resolved = {}
    for method_name in ALL_METHODS:
        method, method_hash, decisions = load_and_resolve_method(
            quick_config.paths.method_catalog,
            method_name,
            decision_root=quick_config.paths.decision_root,
            protocol_lock=quick_lock,
        )
        if decisions:
            raise ValueError("quick methods must not have dynamic parents")
        if method.uses_test_bfs:
            raise ValueError("quick ranking method uses test-time BFS")
        if method_name in NEW_METHODS and method.training_scope.value != "frozen":
            raise ValueError("quick method unexpectedly updates JEPA parameters")
        resolved[method_name] = {
            "sha256": method_hash,
            "role": method.role,
            "head": method.head.model_dump(mode="json") if method.head else None,
            "objectives": (
                method.objectives.model_dump(mode="json") if method.objectives else None
            ),
            "planner": method.planner.model_dump(mode="json"),
        }
    for method_name in ("b_l2_cem", "b_dh_cem", "a1_log"):
        source_method, source_hash, source_decisions = load_and_resolve_method(
            source_config.paths.method_catalog,
            method_name,
            decision_root=source_config.paths.decision_root,
            protocol_lock=source_lock,
        )
        quick_method, quick_hash, _ = load_and_resolve_method(
            quick_config.paths.method_catalog,
            method_name,
            decision_root=quick_config.paths.decision_root,
            protocol_lock=quick_lock,
        )
        if (
            source_decisions
            or source_method != quick_method
            or source_hash != quick_hash
        ):
            raise ValueError(f"reference method parity failed: {method_name}")
    if set(profile.q1.methods) != set(ALL_METHODS):
        raise ValueError("Q1 does not cover the complete locked method set")
    if profile.q2.split_role != "select" or profile.q3.split_role != "legacy":
        raise ValueError("quick stage split roles changed")
    output = {
        "schema": "a1-quick-validation-audit-v1",
        "profile_id": profile.profile_id,
        "package_lock_sha256": package_lock["package_lock_sha256"],
        "quick_analysis_spec_sha256": quick_lock["analysis_spec_sha256"],
        "quick_protocol_lock_sha256": quick_lock["protocol_lock_sha256"],
        "source_analysis_spec_sha256": source_lock["analysis_spec_sha256"],
        "source_protocol_lock_sha256": source_lock["protocol_lock_sha256"],
        "locked_section_parity": parity,
        "manifests": manifests,
        "heldout_task_overlap": heldout_overlap,
        "methods": resolved,
        "planner_parity": True,
        "training_parity": True,
        "topology_holdout_preserved": True,
        "test_bfs_enters_action_selection": False,
        "confirmatory_claim_allowed": False,
        "claim_boundary": profile.claim_boundary,
    }
    output["audit_sha256"] = canonical_json_sha256(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = run_audit(args.profile)
    if args.output:
        path = atomic_json_dump(args.output, result)
        print(path)
    else:
        print(result["audit_sha256"])


if __name__ == "__main__":
    main()
