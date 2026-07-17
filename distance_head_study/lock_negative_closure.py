"""Verify every required reserve family before enabling a broad negative claim."""

from __future__ import annotations

import argparse
from pathlib import Path

from distance_head_study.common import (
    atomic_json_dump,
    canonical_json_sha256,
    load_study_config,
    merge_hash_bindings,
    require_clean_worktree,
    resolve_path,
    sha256_file,
)
from distance_head_study.evidence import diagnostic_evidence_hashes
from distance_head_study.gates import load_signed_artifact
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.results import (
    load_complete_rows,
    result_directory,
    result_evidence_hashes,
)
from distance_head_study.taxonomy import (
    NEGATIVE_CLOSURE_CANDIDATES,
    NEGATIVE_CLOSURE_REQUIRED_RUNS,
)


def _complete(path: Path) -> Path:
    if (path / "rows.jsonl").exists():
        return path
    if (path / "merged" / "rows.jsonl").exists():
        return path / "merged"
    raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--screen-decision", required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    protocol_lock = verify_protocol_lock(config)
    decision = load_signed_artifact(
        args.screen_decision,
        signature_field="decision_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    if (
        decision.get("analysis_spec_sha256") != protocol_lock["analysis_spec_sha256"]
        or decision.get("protocol_lock_sha256") != protocol_lock["protocol_lock_sha256"]
    ):
        raise ValueError("closure decision uses another protocol lock")
    if (
        decision.get("decision_name") != "closure_selection"
        or tuple(decision.get("eligible_methods", ())) != NEGATIVE_CLOSURE_CANDIDATES
    ):
        raise ValueError("negative closure requires the complete closure selection")
    input_hashes: dict[str, str] = merge_hash_bindings(
        {
            resolve_path(args.screen_decision).as_posix(): sha256_file(
                args.screen_decision
            )
        },
        decision["input_hashes"],
    )
    for method in NEGATIVE_CLOSURE_REQUIRED_RUNS:
        for head_seed in config.seeds.screen_head_seeds:
            diagnostic = resolve_path(
                f"distance_head_study_runs/diagnostics/screen/{method}/"
                f"backbone42_head{head_seed}.json"
            )
            diagnostic_payload = load_signed_artifact(
                diagnostic,
                signature_field="diagnostic_sha256",
                expected_protocol_id=config.protocol_id,
            )
            expected_diagnostic = (
                diagnostic_payload.get("analysis_spec_sha256")
                == protocol_lock["analysis_spec_sha256"]
                and diagnostic_payload.get("protocol_lock_sha256")
                == protocol_lock["protocol_lock_sha256"]
                and diagnostic_payload.get("split_role") == "screen"
                and diagnostic_payload.get("method") == method
                and int(diagnostic_payload.get("backbone_seed", -1)) == 42
                and int(diagnostic_payload.get("head_seed", -1)) == head_seed
                and int(diagnostic_payload.get("sample_count", -1))
                == config.analysis.diagnostic_batches
                * config.training.effective_batch_size
                and int(
                    diagnostic_payload.get("cache_binding", {}).get(
                        "diagnostic_limit", -1
                    )
                )
                == 0
            )
            if not expected_diagnostic:
                raise ValueError(
                    f"negative-closure diagnostic differs from protocol: {diagnostic}"
                )
            input_hashes = merge_hash_bindings(
                input_hashes,
                diagnostic_evidence_hashes(
                    diagnostic,
                    diagnostic_payload,
                    split_role="screen",
                    backbone_seed=42,
                    protocol_lock=protocol_lock,
                ),
            )
            for protocol in config.planner.action_protocols:
                directory = _complete(
                    result_directory(
                        config,
                        split_role="screen",
                        method=method,
                        backbone_seed=42,
                        head_seed=head_seed,
                        action_protocol=protocol,
                    )
                )
                metadata, _ = load_complete_rows(directory)
                expected_result = (
                    metadata.get("analysis_spec_sha256")
                    == protocol_lock["analysis_spec_sha256"]
                    and metadata.get("protocol_lock_sha256")
                    == protocol_lock["protocol_lock_sha256"]
                    and metadata.get("split_role") == "screen"
                    and metadata.get("method", {}).get("name") == method
                    and int(metadata.get("backbone_seed", -1)) == 42
                    and int(metadata.get("head_seed", -1)) == head_seed
                    and metadata.get("action_protocol") == protocol
                    and int(metadata.get("diagnostic_limit", -1)) == 0
                )
                if not expected_result:
                    raise ValueError(
                        f"negative-closure result differs from protocol: {directory}"
                    )
                input_hashes = merge_hash_bindings(
                    input_hashes, result_evidence_hashes(directory, metadata)
                )
    payload = {
        "schema": "distance-head-negative-closure-v1",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": protocol_lock["analysis_spec_sha256"],
        "protocol_lock_sha256": protocol_lock["protocol_lock_sha256"],
        "complete": True,
        "required_methods": list(NEGATIVE_CLOSURE_REQUIRED_RUNS),
        "backbone_seed": 42,
        "head_seeds": list(config.seeds.screen_head_seeds),
        "action_protocols": list(config.planner.action_protocols),
        "screen_decision_sha256": decision["decision_sha256"],
        "input_hashes": input_hashes,
        "claim_boundary": "only the preregistered method families listed here",
    }
    payload["negative_closure_sha256"] = canonical_json_sha256(payload)
    output = resolve_path(config.paths.negative_closure_lock)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite negative closure lock: {output}")
    atomic_json_dump(output, payload)
    print(output)


if __name__ == "__main__":
    main()
