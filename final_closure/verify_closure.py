#!/usr/bin/env python3
"""Re-hash and validate a completed final-closure artifact tree."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from final_closure import FIGURE_FILENAMES, FORMAT_VERSION, TABLE_FILENAMES
from final_closure.common import (
    RERUN_REASONS,
    analysis_spec_sha256,
    experiment_code_fingerprint,
    git_commit,
    load_config,
    load_json,
    sha256_file,
    validate_rerun_record,
)


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"closure mismatch for {label}: {actual!r} != {expected!r}")


def verify_hash_map(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"closure {label} hash map is empty")
    output: dict[str, str] = {}
    for name, expected in value.items():
        path = Path(str(name))
        if not path.is_file():
            raise FileNotFoundError(f"closure {label} file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"closure {label} SHA256 mismatch for {path}: {actual} != {expected}"
            )
        output[str(path)] = str(expected)
    return output


def resolved_paths(value: dict[str, str]) -> set[Path]:
    return {Path(path).resolve() for path in value}


def verify_closure_gate(config_path: str | Path) -> dict[str, Any]:
    config, lock = load_config(config_path)
    gate_path = Path(config["paths"]["closure_gate"])
    if not gate_path.is_file():
        raise FileNotFoundError(f"closure gate is missing: {gate_path}")
    gate = load_json(gate_path)
    assert_equal(gate.get("format_version"), FORMAT_VERSION, "format version")
    assert_equal(gate.get("protocol_id"), config["protocol_id"], "protocol ID")
    assert_equal(gate.get("status"), "complete", "status")
    assert_equal(
        gate.get("analysis_spec_sha256"),
        analysis_spec_sha256(config, lock),
        "analysis spec",
    )
    assert_equal(gate.get("git_commit"), git_commit(), "Git commit")
    assert_equal(
        gate.get("code_fingerprint"),
        experiment_code_fingerprint(),
        "code fingerprint",
    )
    assert_equal(gate.get("completion_is_score_independent"), True, "completion rule")
    seeds = [int(value) for value in config["seeds"]]
    assert_equal(gate.get("required_training_seeds"), seeds, "training seeds")
    assert_equal(
        int(gate.get("required_tasks_per_seed", -1)),
        int(config["protocol"]["full_eval_count"]),
        "tasks per seed",
    )
    expected_methods = [
        *(item["name"] for item in config["spatial_methods"]),
        *(item["name"] for item in config["baselines"]),
    ]
    assert_equal(
        gate.get("required_primary_methods"), expected_methods, "primary methods"
    )
    assert_equal(
        gate.get("rerun_allowed_only_for"), list(RERUN_REASONS), "rerun reasons"
    )
    assert_equal(gate.get("rerun_for_low_or_surprising_score"), False, "score reruns")
    assert_equal(
        gate.get("next_architecture_search_authorized"), False, "architecture stop"
    )

    source_files = verify_hash_map(gate.get("source_files"), "source")
    expected_sources = {
        Path(config_path).resolve(),
        Path(config["paths"]["protocol_lock"]).resolve(),
        Path(config["paths"]["train_manifest"]).resolve(),
        Path(config["paths"]["development_manifest"]).resolve(),
        Path(config["paths"]["confirmatory_manifest"]).resolve(),
        Path(config["paths"]["audit_output"]).resolve(),
        Path(lock["source_spatial_experiment"]["config_path"]).resolve(),
        Path(lock["source_spatial_experiment"]["protocol_lock_path"]).resolve(),
    }
    for baseline in config["baselines"]:
        for seed in seeds:
            expected_sources.add(
                Path(
                    config["paths"]["checkpoint_template"].format(
                        name=baseline["name"], seed=seed
                    )
                ).resolve()
            )
            for split_role in ("development", "confirmatory"):
                template = config["paths"][f"{split_role}_result_template"]
                for action_selection in (
                    config["protocol"]["primary_action_selection"],
                    *config["protocol"]["diagnostic_action_selections"],
                ):
                    expected_sources.add(
                        Path(
                            template.format(
                                name=baseline["name"],
                                seed=seed,
                                action_selection=action_selection,
                            )
                        ).resolve()
                    )
    for method in config["spatial_methods"]:
        for seed in seeds:
            result_path = Path(
                config["paths"]["spatial_result_template"].format(
                    name=method["name"], seed=seed
                )
            )
            expected_sources.add(result_path.resolve())
            metadata = load_json(result_path).get("metadata", {})
            checkpoint = (
                metadata.get("checkpoint")
                or metadata.get("planner_ckpt")
                or metadata.get("representation_ckpt")
            )
            if not checkpoint:
                raise ValueError(
                    f"spatial result has no checkpoint path: {result_path}"
                )
            expected_sources.add(Path(str(checkpoint)).resolve())
    assert_equal(resolved_paths(source_files), expected_sources, "source file set")
    assert_equal(len(source_files), len(expected_sources), "unique source paths")

    artifacts = verify_hash_map(gate.get("artifacts"), "artifact")
    expected_artifacts = {
        Path(config["paths"]["paper_report"]).resolve(),
        *(
            Path(config["paths"]["table_dir"]).joinpath(name).resolve()
            for name in TABLE_FILENAMES
        ),
        *(
            Path(config["paths"]["figure_dir"]).joinpath(name).resolve()
            for name in FIGURE_FILENAMES
        ),
    }
    assert_equal(resolved_paths(artifacts), expected_artifacts, "artifact set")

    summary_path = Path(str(gate.get("summary_path", "")))
    assert_equal(
        summary_path.resolve(),
        Path(config["paths"]["summary_json"]).resolve(),
        "summary path",
    )
    if not summary_path.is_file():
        raise FileNotFoundError(f"closure summary is missing: {summary_path}")
    assert_equal(sha256_file(summary_path), gate.get("summary_sha256"), "summary hash")
    summary = load_json(summary_path)
    protocol = summary.get("protocol", {})
    assert_equal(
        protocol.get("analysis_spec_sha256"),
        gate["analysis_spec_sha256"],
        "summary analysis spec",
    )
    assert_equal(protocol.get("git_commit"), gate["git_commit"], "summary commit")
    assert_equal(
        protocol.get("code_fingerprint"),
        gate["code_fingerprint"],
        "summary code fingerprint",
    )
    assert_equal(protocol.get("git_dirty"), False, "summary dirty flag")
    assert_equal(summary.get("artifacts"), gate["artifacts"], "summary artifacts")
    assert_equal(
        int(summary.get("source_file_count", -1)),
        len(source_files),
        "summary source count",
    )
    rerun_records = gate.get("rerun_records")
    if not isinstance(rerun_records, list):
        raise ValueError("closure rerun records must be a list")
    for index, record in enumerate(rerun_records):
        if not isinstance(record, dict) or not record.get("source"):
            raise ValueError(f"closure rerun record {index} lacks a source")
        validate_rerun_record(
            {
                "reason": record.get("reason"),
                "superseded_outputs": record.get("superseded_outputs"),
            },
            f"closure rerun record {index}",
        )
    assert_equal(summary.get("rerun_records"), rerun_records, "summary rerun records")
    return {
        "protocol_id": config["protocol_id"],
        "status": "verified",
        "source_file_count": len(source_files),
        "artifact_count": len(artifacts),
        "rerun_record_count": len(rerun_records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="final_closure/configs/default.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_closure_gate(args.config)
    print(
        "final-closure gate verified: "
        f"sources={report['source_file_count']} "
        f"artifacts={report['artifact_count']} "
        f"reruns={report['rerun_record_count']}"
    )


if __name__ == "__main__":
    main()
