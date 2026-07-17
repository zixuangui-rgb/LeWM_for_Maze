"""Result-path, shard merge, and paired task-table helpers."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from distance_head_study import MODEL_ACTION_VOCAB_SIZE, PROTOCOL_ID
from distance_head_study.common import (
    atomic_json_dump,
    atomic_text_dump,
    canonical_json_sha256,
    merge_hash_bindings,
    read_jsonl,
    resolve_path,
    sha256_file,
)
from distance_head_study.schemas import StudyConfig
from spatial_jepa_planning.common import summarize_rows, task_id


def result_directory(
    config: StudyConfig,
    *,
    split_role: str,
    method: str,
    backbone_seed: int,
    head_seed: int,
    action_protocol: str,
) -> Path:
    return resolve_path(
        config.paths.result_template.format(
            split_role=split_role,
            method=method,
            backbone_seed=int(backbone_seed),
            head_seed=int(head_seed),
            action_protocol=action_protocol,
        )
    )


def load_complete_rows(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = resolve_path(path)
    metadata_path = directory / "metadata.json"
    rows_path = directory / "rows.jsonl"
    summary_path = directory / "summary.json"
    if (
        not metadata_path.exists()
        or not rows_path.exists()
        or not summary_path.exists()
    ):
        raise FileNotFoundError(f"incomplete result directory: {directory}")
    with open(metadata_path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema") != "distance-head-task-results-v1":
        raise ValueError(f"result metadata schema mismatch: {metadata_path}")
    if metadata.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"result protocol ID mismatch: {metadata_path}")
    protocol_lock_sha256 = metadata.get("protocol_lock_sha256")
    if not isinstance(protocol_lock_sha256, str) or len(protocol_lock_sha256) != 64:
        raise ValueError(f"result omits its protocol-lock binding: {metadata_path}")
    signature = metadata.get("run_spec_sha256")
    unsigned = {
        key: value for key, value in metadata.items() if key != "run_spec_sha256"
    }
    if signature != canonical_json_sha256(unsigned):
        raise ValueError(f"result metadata signature mismatch: {metadata_path}")
    checkpoint = metadata.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"result metadata omits checkpoint provenance: {metadata_path}"
        )
    method = metadata.get("method")
    planner = method.get("planner") if isinstance(method, dict) else None
    if not isinstance(method, dict) or not isinstance(planner, dict):
        raise ValueError(f"result metadata omits method/planner spec: {metadata_path}")
    method_role = method.get("role")
    planner_kind = planner.get("kind")
    if method_role not in {"baseline", "candidate", "control", "oracle"} or (
        planner_kind
        not in {
            "model_free_greedy",
            "predictor_greedy",
            "categorical_cem",
            "icem",
            "beam",
            "best_first",
        }
    ):
        raise ValueError(f"result method/planner spec is invalid: {metadata_path}")
    action_protocol = metadata.get("action_protocol")
    if action_protocol not in {"corrected_v1", "unmasked"}:
        raise ValueError(f"result action protocol is invalid: {metadata_path}")
    uses_test_bfs = method.get("uses_test_bfs") is True
    for path_key, hash_key in (
        ("backbone_path", "backbone_sha256"),
        ("head_checkpoint_path", "head_checkpoint_sha256"),
    ):
        checkpoint_path = checkpoint.get(path_key)
        if checkpoint_path is None:
            continue
        checkpoint_hash = checkpoint.get(hash_key)
        if (
            not isinstance(checkpoint_path, str)
            or not isinstance(checkpoint_hash, str)
            or not resolve_path(checkpoint_path).exists()
            or sha256_file(checkpoint_path) != checkpoint_hash
        ):
            raise ValueError(
                f"result checkpoint changed after evaluation: {checkpoint_path}"
            )
    input_hashes = metadata.get("input_row_hashes", {})
    if input_hashes:
        for input_path, expected_hash in input_hashes.items():
            if sha256_file(resolve_path(input_path)) != expected_hash:
                raise ValueError(f"merged input rows changed: {input_path}")
    rows = read_jsonl(rows_path)
    identifiers = [str(row["task_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate task IDs in {rows_path}")
    manifest_path_value = metadata.get("manifest_path")
    manifest_hash = metadata.get("manifest_sha256")
    if not isinstance(manifest_path_value, str) or not isinstance(manifest_hash, str):
        raise ValueError(f"result metadata omits its manifest binding: {metadata_path}")
    manifest_path = resolve_path(manifest_path_value)
    if sha256_file(manifest_path) != manifest_hash:
        raise ValueError(f"result manifest changed after evaluation: {manifest_path}")
    manifest_rows = read_jsonl(manifest_path)
    num_shards = int(metadata.get("num_shards", -1))
    shard_index = metadata.get("shard_index")
    merged = int(metadata.get("merged_from_shards", 0))
    if num_shards < 1:
        raise ValueError(f"result has invalid shard metadata: {metadata_path}")
    if merged:
        if merged != num_shards or shard_index is not None:
            raise ValueError(
                f"merged result shard metadata is inconsistent: {directory}"
            )
        assigned = manifest_rows
    else:
        if not isinstance(shard_index, int) or not 0 <= shard_index < num_shards:
            raise ValueError(f"result has invalid shard index: {metadata_path}")
        assigned = [
            entry
            for index, entry in enumerate(manifest_rows)
            if index % num_shards == shard_index
        ]
    diagnostic_limit = int(metadata.get("diagnostic_limit", 0))
    if diagnostic_limit < 0:
        raise ValueError(f"result has a negative diagnostic limit: {metadata_path}")
    if diagnostic_limit:
        if merged:
            raise ValueError("partial diagnostic shards cannot be merged formally")
        assigned = assigned[:diagnostic_limit]
    expected_tasks = {task_id(entry): entry for entry in assigned}
    if len(expected_tasks) != len(assigned):
        raise ValueError(f"bound manifest has duplicate task IDs: {manifest_path}")
    if set(identifiers) != set(expected_tasks):
        raise ValueError(f"result tasks differ from the bound manifest: {rows_path}")
    for row in rows:
        required = {
            "task_id",
            "maze_size",
            "topology_seed",
            "start_cell",
            "goal_cell",
            "success",
            "path_length",
            "optimal_length",
            "spl",
            "final_bfs_distance",
            "failure_mode",
            "loop_or_cycle",
            "invalid_actions",
            "repeat_states",
            "max_state_visits",
            "proposed_invalid",
            "proposed_backtrack",
            "assistance_count",
            "assistance_rate",
            "plan_transitions",
            "fallback_transitions",
            "mean_best_cost",
            "episode_seconds",
        }
        if not required <= set(row):
            raise ValueError(f"result row is missing required fields: {rows_path}")
        path_length = int(row["path_length"])
        optimal = int(row["optimal_length"])
        spl = float(row["spl"])
        success = bool(row["success"])
        if path_length < 0 or path_length > 128 or optimal <= 0:
            raise ValueError(f"result row has impossible path lengths: {rows_path}")
        if not 0.0 <= spl <= 1.0:
            raise ValueError(f"result row has SPL outside [0,1]: {rows_path}")
        invalid = int(row["invalid_actions"])
        repeats = int(row["repeat_states"])
        max_visits = int(row["max_state_visits"])
        proposed_invalid = int(row["proposed_invalid"])
        proposed_backtrack = int(row["proposed_backtrack"])
        assistance = int(row["assistance_count"])
        assistance_rate = float(row["assistance_rate"])
        plan_transitions = int(row["plan_transitions"])
        fallback_transitions = int(row["fallback_transitions"])
        if (
            invalid < 0
            or repeats < 0
            or max_visits < 1
            or proposed_invalid < 0
            or proposed_backtrack < 0
            or assistance < 0
            or plan_transitions < 0
            or fallback_transitions < 0
            or not 0.0 <= assistance_rate <= 1.0
            or not math.isfinite(float(row["mean_best_cost"]))
            or float(row["episode_seconds"]) < 0.0
        ):
            raise ValueError(f"result row has invalid diagnostic metrics: {rows_path}")
        if (
            invalid > path_length
            or proposed_invalid > path_length
            or proposed_backtrack > path_length
            or assistance > path_length
            or repeats > path_length
            or max_visits > path_length + 1
            or not math.isclose(
                assistance_rate,
                assistance / max(path_length, 1),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or bool(row["loop_or_cycle"]) != (max_visits >= 4)
        ):
            raise ValueError(
                f"result row diagnostics are internally inconsistent: {rows_path}"
            )
        if action_protocol == "corrected_v1":
            if invalid != 0:
                raise ValueError(
                    f"corrected result contains an invalid executed action: {rows_path}"
                )
            expected_fallback = (
                0 if uses_test_bfs else MODEL_ACTION_VOCAB_SIZE * assistance
            )
            if fallback_transitions != expected_fallback:
                raise ValueError(
                    f"corrected fallback compute does not reproduce: {rows_path}"
                )
        elif (
            assistance != 0 or fallback_transitions != 0 or invalid != proposed_invalid
        ):
            raise ValueError(
                f"unmasked action/assistance semantics do not reproduce: {rows_path}"
            )
        if method_role == "oracle" or planner_kind == "model_free_greedy":
            if plan_transitions != 0:
                raise ValueError(
                    f"model-free/oracle planner reports predictor compute: {rows_path}"
                )
        elif planner_kind == "predictor_greedy":
            if plan_transitions != MODEL_ACTION_VOCAB_SIZE * path_length:
                raise ValueError(
                    f"predictor-greedy compute does not reproduce: {rows_path}"
                )
        elif not 0 < plan_transitions <= 768 * path_length:
            raise ValueError(
                f"search planner exceeds or omits its transition budget: {rows_path}"
            )
        expected_spl = optimal / max(optimal, path_length) if success else 0.0
        if not math.isclose(spl, expected_spl, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"result row SPL does not reproduce: {rows_path}")
        final_distance = int(row["final_bfs_distance"])
        if final_distance < 0:
            raise ValueError(f"result row has a negative final distance: {rows_path}")
        if success:
            expected_failure = "success"
        elif optimal > 128:
            expected_failure = "step_cap_ineligible"
        elif invalid > path_length / 4:
            expected_failure = "invalid_action"
        elif bool(row["loop_or_cycle"]):
            expected_failure = "loop_or_cycle"
        elif final_distance >= optimal:
            expected_failure = "insufficient_progress"
        else:
            expected_failure = "timeout_inefficient"
        if row["failure_mode"] != expected_failure:
            raise ValueError(f"result row failure mode does not reproduce: {rows_path}")
        if success and (final_distance != 0 or path_length < optimal):
            raise ValueError(f"successful result row violates invariants: {rows_path}")
        entry = expected_tasks[str(row["task_id"])]
        for row_key, entry_key in (
            ("maze_size", "maze_size"),
            ("topology_seed", "topology_seed"),
            ("start_cell", "start_cell"),
            ("goal_cell", "goal_cell"),
            ("optimal_length", "bfs_path_length"),
        ):
            if int(row.get(row_key, -1)) != int(entry[entry_key]):
                raise ValueError(
                    f"result row {row_key} differs from manifest: {rows_path}"
                )
    summary = summarize_rows(rows, seen_max_size=21, max_steps=128)
    with open(summary_path, encoding="utf-8") as stream:
        stored = json.load(stream)
    for key, value in summary.items():
        if stored.get(key) != value:
            raise ValueError(
                f"stored summary field {key!r} does not reproduce: {directory}"
            )
    expected_failure_modes = dict(
        sorted(Counter(str(row["failure_mode"]) for row in rows).items())
    )
    if stored.get("failure_modes") != expected_failure_modes:
        raise ValueError(f"stored failure modes do not reproduce: {directory}")
    expected_assistance = sum(float(row["assistance_rate"]) for row in rows) / len(rows)
    if not math.isclose(
        float(stored.get("mean_assistance_rate", float("nan"))),
        expected_assistance,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"stored assistance rate does not reproduce: {directory}")
    if stored.get("run_spec_sha256") != signature:
        raise ValueError(f"summary/result metadata signatures differ: {directory}")
    return metadata, rows


def result_evidence_hashes(
    path: str | Path, metadata: dict[str, Any] | None = None
) -> dict[str, str]:
    """Bind every immutable file needed to audit a completed result bundle."""

    directory = resolve_path(path)
    if metadata is None:
        metadata, _ = load_complete_rows(directory)
    hashes = {
        (directory / name).as_posix(): sha256_file(directory / name)
        for name in ("metadata.json", "rows.jsonl", "summary.json")
    }

    def bind_recorded_file(
        path_value: Any,
        hash_value: Any,
        *,
        label: str,
    ) -> None:
        nonlocal hashes
        if not isinstance(path_value, str) or not isinstance(hash_value, str):
            raise ValueError(f"result omits {label} provenance")
        resolved = resolve_path(path_value)
        observed = sha256_file(resolved)
        if observed != hash_value:
            raise ValueError(f"result {label} changed: {resolved}")
        hashes = merge_hash_bindings(hashes, {resolved.as_posix(): observed})

    manifest = resolve_path(str(metadata["manifest_path"]))
    bind_recorded_file(
        manifest.as_posix(),
        metadata.get("manifest_sha256"),
        label="manifest",
    )
    checkpoint = metadata["checkpoint"]
    for path_key, hash_key in (
        ("backbone_path", "backbone_sha256"),
        ("head_checkpoint_path", "head_checkpoint_sha256"),
    ):
        checkpoint_path = checkpoint.get(path_key)
        if checkpoint_path is not None:
            bind_recorded_file(
                checkpoint_path,
                checkpoint.get(hash_key),
                label=path_key,
            )
    head_checkpoint_path = checkpoint.get("head_checkpoint_path")
    if head_checkpoint_path is not None:
        head_payload = torch.load(
            resolve_path(str(head_checkpoint_path)),
            map_location="cpu",
            weights_only=False,
        )

        bank = head_payload.get("candidate_bank")
        if not isinstance(bank, dict):
            raise ValueError("head checkpoint omits candidate-bank provenance")
        bind_recorded_file(bank.get("path"), bank.get("sha256"), label="candidate bank")
        cache_bindings = head_payload.get("cache_bindings")
        if not isinstance(cache_bindings, dict) or set(cache_bindings) != {
            "train",
            "cal",
        }:
            raise ValueError("head checkpoint cache provenance is incomplete")
        for split_role, binding in cache_bindings.items():
            if not isinstance(binding, dict):
                raise ValueError("head checkpoint cache provenance is malformed")
            bind_recorded_file(
                binding.get("index_path"),
                binding.get("index_sha256"),
                label=f"{split_role} cache index",
            )
        initialization = head_payload.get("initialization", {})
        if not isinstance(initialization, dict):
            raise ValueError("head checkpoint initialization provenance is malformed")
        parent_path = initialization.get("parent_checkpoint_path")
        if parent_path is not None:
            bind_recorded_file(
                parent_path,
                initialization.get("parent_checkpoint_sha256"),
                label="initialization parent checkpoint",
            )
    for input_path, expected_hash in metadata.get("input_row_hashes", {}).items():
        resolved = resolve_path(str(input_path))
        observed = sha256_file(resolved)
        if observed != expected_hash:
            raise ValueError(f"merged input rows changed: {resolved}")
        hashes = merge_hash_bindings(hashes, {resolved.as_posix(): observed})
    return hashes


def merge_shards(base_directory: str | Path, *, expected_shards: int) -> Path:
    base = resolve_path(base_directory)
    output = base / "merged"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite merged results: {output}")
    metadata_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for shard_index in range(expected_shards):
        shard = base / f"shard_{shard_index:03d}_of_{expected_shards:03d}"
        metadata, shard_rows = load_complete_rows(shard)
        if (
            metadata.get("shard_index") != shard_index
            or metadata.get("num_shards") != expected_shards
            or int(metadata.get("diagnostic_limit", 0)) != 0
        ):
            raise ValueError(f"shard metadata mismatch: {shard}")
        metadata_rows.append(metadata)
        rows.extend(shard_rows)
        shard_rows_path = shard / "rows.jsonl"
        try:
            recorded_path = shard_rows_path.relative_to(resolve_path(".")).as_posix()
        except ValueError:
            recorded_path = shard_rows_path.as_posix()
        input_hashes[recorded_path] = sha256_file(shard_rows_path)
    reference = dict(metadata_rows[0])
    for metadata in metadata_rows[1:]:
        comparable = dict(metadata)
        for key in ("shard_index", "run_spec_sha256"):
            comparable.pop(key, None)
        expected = dict(reference)
        for key in ("shard_index", "run_spec_sha256"):
            expected.pop(key, None)
        if comparable != expected:
            raise ValueError("result shards differ in scientific metadata")
    identifiers = [str(row["task_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("merged result shards overlap in task IDs")
    rows.sort(
        key=lambda row: (
            int(row["maze_size"]),
            int(row["topology_seed"]),
            str(row["task_id"]),
        )
    )
    output.mkdir(parents=True)
    merged_metadata = {
        **reference,
        "shard_index": None,
        "merged_from_shards": expected_shards,
        "input_row_hashes": input_hashes,
    }
    merged_metadata.pop("run_spec_sha256", None)
    merged_metadata["run_spec_sha256"] = canonical_json_sha256(merged_metadata)
    atomic_json_dump(output / "metadata.json", merged_metadata)
    text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    atomic_text_dump(output / "rows.jsonl", text)
    summary = summarize_rows(rows, seen_max_size=21, max_steps=128)
    summary["failure_modes"] = dict(
        sorted(Counter(str(row["failure_mode"]) for row in rows).items())
    )
    summary["mean_assistance_rate"] = sum(
        float(row["assistance_rate"]) for row in rows
    ) / len(rows)
    summary["run_spec_sha256"] = merged_metadata["run_spec_sha256"]
    atomic_json_dump(output / "summary.json", summary)
    return output


def paired_rows(
    candidate: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left = {str(row["task_id"]): row for row in candidate}
    right = {str(row["task_id"]): row for row in baseline}
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:3]
        missing_right = sorted(set(left) - set(right))[:3]
        raise ValueError(
            f"paired task sets differ: candidate_missing={missing_left}, "
            f"baseline_missing={missing_right}"
        )
    return [(left[identifier], right[identifier]) for identifier in sorted(left)]


__all__ = [
    "load_complete_rows",
    "merge_shards",
    "paired_rows",
    "result_evidence_hashes",
    "result_directory",
]
