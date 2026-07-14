"""Create or verify the immutable analysis lock before any formal run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import (
    ROOT,
    analysis_spec_sha256,
    atomic_json_dump,
    experiment_code_fingerprint,
    load_study_config,
    manifest_record,
    resolve_path,
    validate_manifest_isolation,
)
from vector_jepa_planner_frontier.compat import SOURCE_BASELINE_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def build_lock(config_path: str | Path) -> dict[str, Any]:
    config = load_study_config(config_path)
    resolved_config_path = resolve_path(config_path)
    config_label = resolved_config_path.relative_to(ROOT).as_posix()
    validate_manifest_isolation(config)
    lock: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": "1.0",
        "protocol_id": config.protocol_id,
        "status": "locked",
        "study_role": config.study_role,
        "confirmation_opened": False,
        "amendments": {
            "path": str(config.paths.amendments),
            "sha256": sha256_file(resolve_path(config.paths.amendments)),
        },
        "amendment_document": {
            "path": str(config.paths.amendment_document),
            "sha256": sha256_file(resolve_path(config.paths.amendment_document)),
        },
        "amendment_before": {
            "path": str(config.paths.amendment_before),
            "sha256": sha256_file(resolve_path(config.paths.amendment_before)),
        },
        "amendment_after": {
            "path": str(config.paths.amendment_after),
            "sha256": sha256_file(resolve_path(config.paths.amendment_after)),
        },
        "protocol_document": {
            "path": "vector_jepa_planner_frontier/EXPERIMENT_PROTOCOL.md",
            "sha256": sha256_file(
                resolve_path("vector_jepa_planner_frontier/EXPERIMENT_PROTOCOL.md")
            ),
        },
        "method_config": {
            "path": config_label,
            "sha256": sha256_file(resolved_config_path),
        },
        "environment_lock": {
            "path": "uv.lock",
            "sha256": sha256_file(resolve_path("uv.lock")),
        },
        "code_fingerprint": experiment_code_fingerprint(),
        "primary_hypotheses": ["H1", "H2"],
        "training_seeds": list(config.protocol.training_seeds),
        "planner_seeds": list(config.protocol.planner_seeds),
        "search_seeds": list(config.protocol.search_seeds),
        "train_manifest": manifest_record(config.paths.train_manifest),
        "development_manifest": manifest_record(config.paths.development_manifest),
        "validation_manifest": manifest_record(config.paths.validation_manifest),
        "confirmatory_manifest": manifest_record(config.paths.confirmatory_manifest),
        "source_baseline": {
            "name": SOURCE_BASELINE_NAME,
            "config_path": str(config.paths.source_config),
            "config_sha256": sha256_file(resolve_path(config.paths.source_config)),
            "lock_path": str(config.paths.source_lock),
            "lock_sha256": sha256_file(resolve_path(config.paths.source_lock)),
        },
    }
    lock["analysis_spec_sha256"] = analysis_spec_sha256(config, lock)
    return lock


def canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    output = resolve_path(config.paths.protocol_lock)
    expected = build_lock(args.config)
    if args.check:
        if not output.exists():
            raise FileNotFoundError(output)
        with open(output, encoding="utf-8") as stream:
            actual = json.load(stream)
        if canonical(actual) != canonical(expected):
            raise ValueError("protocol lock does not reproduce from current artifacts")
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite protocol lock: {output}")
    atomic_json_dump(output, expected)


if __name__ == "__main__":
    main()
