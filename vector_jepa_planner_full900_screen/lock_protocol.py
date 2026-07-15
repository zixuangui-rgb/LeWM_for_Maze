"""Create or verify the immutable full-900 screening protocol lock."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from final_closure.common import read_jsonl, sha256_file
from vector_jepa_planner_frontier.common import analysis_spec_sha256
from vector_jepa_planner_full900_screen import PROTOCOL_ID
from vector_jepa_planner_full900_screen.common import (
    atomic_json_dump,
    code_fingerprint,
    load_config,
    load_json,
    quick_spec_sha256,
    resolve_path,
    validate_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="vector_jepa_planner_full900_screen/configs/default.json",
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def artifact_record(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"cannot lock missing artifact: {resolved}")
    return {
        "path": str(resolved.relative_to(resolve_path("."))),
        "sha256": sha256_file(resolved),
    }


def manifest_record(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    rows = read_jsonl(resolved)
    return {
        **artifact_record(resolved),
        "count": len(rows),
        "size_counts": {
            str(size): count
            for size, count in sorted(
                Counter(int(row["maze_size"]) for row in rows).items()
            )
        },
        "unique_task_hashes": len({str(row["task_hash"]) for row in rows}),
        "unique_topology_hashes": len(
            {str(row.get("topology_hash", row["task_hash"])) for row in rows}
        ),
    }


def build_lock(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    package = "vector_jepa_planner_full900_screen"
    document_paths = {
        "protocol_document": f"{package}/EXPERIMENT_PROTOCOL.md",
        "methods_document": f"{package}/METHODS.md",
        "compatibility_document": f"{package}/COMPATIBILITY.md",
        "runbook_document": f"{package}/ENGINEER_RUNBOOK.md",
        "claims_document": f"{package}/CLAIMS_AND_STOP_RULES.md",
        "result_schema_document": f"{package}/RESULT_SCHEMA.md",
        "readme_document": f"{package}/README.md",
        "implementation_audit_document": f"{package}/IMPLEMENTATION_AUDIT.md",
        "handoff_document": f"{package}/HANDOFF_CHECKLIST.md",
        "validation_document": f"{package}/VALIDATION.md",
        "test_suite": "tests/test_vector_jepa_planner_full900_screen.py",
    }
    lock: dict[str, Any] = {
        "schema": "vector-jepa-full900-screen-lock-v1",
        "status": "locked",
        "protocol_id": PROTOCOL_ID,
        "amendments": artifact_record(config.paths.amendments),
        "amendment_document": artifact_record(config.paths.amendment_document),
        "amendment_before": artifact_record(config.paths.amendment_before),
        "amendment_after": artifact_record(config.paths.amendment_after),
        **{name: artifact_record(path) for name, path in document_paths.items()},
        "method_config": artifact_record(config_path),
        "environment_lock": artifact_record("pyproject.toml"),
        "train_manifest": manifest_record(config.paths.train_manifest),
        "development_manifest": manifest_record(config.paths.development_manifest),
        "validation_manifest": manifest_record(config.paths.validation_manifest),
        "confirmatory_manifest": manifest_record(config.paths.confirmatory_manifest),
        "source_baseline": {
            "name": "lewm_l2_cem_seqlen2",
            "config_path": str(config.paths.source_config),
            "config_sha256": sha256_file(resolve_path(config.paths.source_config)),
            "lock_path": str(config.paths.source_lock),
            "lock_sha256": sha256_file(resolve_path(config.paths.source_lock)),
        },
        "code_fingerprint": code_fingerprint(),
    }
    lock["analysis_spec_sha256"] = analysis_spec_sha256(config, lock)
    lock["quick_spec_sha256"] = quick_spec_sha256(config, lock)
    return lock


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock_path = resolve_path(config.paths.protocol_lock)
    if args.check:
        validate_lock(config, load_json(lock_path))
        print(f"protocol lock verified: {lock_path}")
        return
    value = build_lock(args.config)
    atomic_json_dump(lock_path, value)
    validate_lock(config, load_json(lock_path))
    print(f"protocol locked: {lock_path}")


if __name__ == "__main__":
    main()
