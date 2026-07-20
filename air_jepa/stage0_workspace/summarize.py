#!/usr/bin/env python3
"""Create preregistered L1/L2/L3 AIR0 releases without adaptive selection."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from air_jepa.stage0_workspace import EXPERIMENT_ID, FORMAT_VERSION
from air_jepa.stage0_workspace.audit_protocol import FORMAL_PAIRING_BATCHES
from air_jepa.stage0_workspace.benchmark import (
    FORMAL_BACKWARD_K,
    FORMAL_BACKWARD_REPEATS,
    FORMAL_BENCHMARK_TASKS,
)
from air_jepa.stage0_workspace.checkpoints import verify_source_lock
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    atomic_text_dump,
    formal_runtime_valid,
    format_template,
    load_config,
    prepare_new_output,
    read_json,
    read_jsonl,
    relative_path,
    require_clean_worktree,
    resolve_path,
    runtime_signature,
    sha256_file,
    signed_payload,
    state_dict_sha256,
    verify_signature,
)
from air_jepa.stage0_workspace.data import (
    make_rng_streams,
    progressive_iteration_signature,
)
from air_jepa.stage0_workspace.diagnose import (
    DISTANCE_ECE_BINS,
    DISTANCE_MAX,
    deterministic_states,
    summarize_state_rows,
)
from air_jepa.stage0_workspace.models import AIRWorkspaceModel
from air_jepa.stage0_workspace.protocol import (
    expected_matrix,
    verify_package_lock,
    verify_protocol_lock,
)
from diagnostics.common import ACTION_IDS, bfs_distances_from, next_state
from spatial_jepa_planning.common import summarize_rows, validate_manifest_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--level", choices=("l1", "l2", "l3"), required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def result_path(
    config: Any,
    *,
    role: str,
    method: str,
    seed: int,
    k: int,
    protocol: str = "unmasked",
) -> Path:
    return format_template(
        config.paths.result_template,
        split_role=role,
        method=method,
        seed=seed,
        action_protocol=protocol,
        k=k,
    )


def intervention_path(
    config: Any,
    *,
    seed: int,
    k: int,
    intervention: str,
) -> Path:
    if intervention == "normal":
        return result_path(
            config,
            role="air_early",
            method="air0_jepa",
            seed=seed,
            k=k,
        )
    return resolve_path(config.paths.run_root) / (
        f"results/air_early/interventions/air0_jepa/seed{seed}_"
        f"unmasked_k{k}_{intervention}.json"
    )


def diagnostic_path(config: Any, *, seed: int, role: str) -> Path:
    if role == "air_dev":
        return format_template(
            config.paths.diagnostic_template,
            method="air0_jepa",
            seed=seed,
        )
    return resolve_path(config.paths.run_root) / (
        f"diagnostics/air_early/air0_jepa/seed{seed}.json"
    )


def _validate_diagnostic_ranking(
    ranking: Any,
    *,
    candidate_distances: list[int],
    optimal_action_mask: list[bool],
    path: Path,
) -> None:
    if not isinstance(ranking, dict):
        raise ValueError(f"diagnostic ranking is not an object: {path}")
    energy = [float(value) for value in ranking.get("energy", [])]
    if len(energy) != 4 or any(
        not math.isfinite(value) or not -1e-5 <= value <= float(DISTANCE_MAX) + 1e-5
        for value in energy
    ):
        raise ValueError(f"diagnostic ranking energy is invalid: {path}")
    chosen = int(ranking.get("chosen_slot", -1))
    expected_chosen = int(np.argmin(np.asarray(energy, dtype=np.float64)))
    if chosen != expected_chosen or not 0 <= chosen < 4:
        raise ValueError(f"diagnostic ranking chosen slot is invalid: {path}")
    if not isinstance(ranking.get("top1"), bool) or bool(ranking["top1"]) != bool(
        optimal_action_mask[chosen]
    ):
        raise ValueError(f"diagnostic ranking top1 is inconsistent: {path}")
    expected_regret = candidate_distances[chosen] - min(candidate_distances)
    if int(ranking.get("regret", -1)) != expected_regret:
        raise ValueError(f"diagnostic ranking regret is inconsistent: {path}")
    bad = [
        value
        for value, optimal in zip(energy, optimal_action_mask, strict=True)
        if not optimal
    ]
    margin = ranking.get("margin")
    if not bad:
        if margin is not None:
            raise ValueError(f"diagnostic all-optimal margin must be null: {path}")
        return
    expected_margin = min(bad) - min(
        value
        for value, optimal in zip(energy, optimal_action_mask, strict=True)
        if optimal
    )
    if margin is None or not math.isclose(
        float(margin), expected_margin, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"diagnostic ranking margin is inconsistent: {path}")


class ArtifactLoader:
    def __init__(
        self,
        config: Any,
        *,
        protocol_sha256: str,
        package_sha256: str,
        package_code_fingerprint: str,
        source_lock_sha256: str,
        expected_checkpoint_hashes: dict[tuple[str, int], str],
    ) -> None:
        self.config = config
        self.protocol_sha256 = protocol_sha256
        self.package_sha256 = package_sha256
        self.package_code_fingerprint = package_code_fingerprint
        self.source_lock_sha256 = source_lock_sha256
        self.expected_checkpoint_hashes = expected_checkpoint_hashes
        self.runtime_signatures: set[tuple[Any, ...]] = set()
        self.loaded_hashes: dict[str, str] = {}

    def _common(self, path: Path, metadata: dict[str, Any]) -> None:
        if metadata.get("formal") is not True:
            raise ValueError(f"release rejects non-formal result: {path}")
        if metadata.get("experiment_id") != self.config.experiment_id:
            raise ValueError(f"artifact experiment ID mismatch: {path}")
        if metadata.get("code_fingerprint") != self.package_code_fingerprint:
            raise ValueError(f"artifact code fingerprint mismatch: {path}")
        for key, expected in (
            ("protocol_sha256", self.protocol_sha256),
            ("package_sha256", self.package_sha256),
            ("source_lock_sha256", self.source_lock_sha256),
        ):
            if metadata.get(key) != expected:
                raise ValueError(f"artifact {key} mismatch: {path}")
        if metadata.get("git_dirty") is not False:
            raise ValueError(f"release rejects dirty-worktree artifact: {path}")
        if not metadata.get("git_commit"):
            raise ValueError(f"release rejects artifact without git provenance: {path}")
        runtime = metadata.get("runtime", {})
        if not formal_runtime_valid(runtime):
            raise ValueError(
                f"formal artifact runtime is not deterministic H800: {path}"
            )
        signature = runtime_signature(runtime)
        self.runtime_signatures.add(signature)
        if len(self.runtime_signatures) > 1:
            raise ValueError("formal results mix incompatible software/GPU runtimes")
        artifact_path = relative_path(path)
        artifact_sha256 = sha256_file(path)
        previous_sha256 = self.loaded_hashes.get(artifact_path)
        if previous_sha256 is not None and previous_sha256 != artifact_sha256:
            raise ValueError(f"formal artifact changed during release: {path}")
        self.loaded_hashes[artifact_path] = artifact_sha256

    def evaluation(
        self,
        path: Path,
        *,
        role: str,
        method: str,
        seed: int,
        k: int,
        protocol: str = "unmasked",
        intervention: str = "normal",
    ) -> dict[str, Any]:
        payload = read_json(path)
        if payload.get("schema") != "air-jepa-stage0-evaluation-v1":
            raise ValueError(f"invalid evaluation schema: {path}")
        metadata = payload["metadata"]
        self._common(path, metadata)
        expected = {
            "split_role": role,
            "method": method,
            "seed": seed,
            "k": k,
            "action_protocol": protocol,
            "intervention": intervention,
            "max_steps": self.config.evaluation.max_steps,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"evaluation {key} mismatch in {path}")
        expected_evidence_role = (
            "EVALUATOR_ORACLE"
            if method == "oracle_bfs"
            else (
                "ORACLE_INTERVENTION"
                if intervention == "true_future"
                else (
                    "MECHANISM_DIAGNOSTIC"
                    if intervention != "normal" or protocol == "corrected"
                    else (
                        "EARLY_SIGNAL" if role == "air_early" else "PRIMARY_PROVISIONAL"
                    )
                )
            )
        )
        if metadata.get("evidence_role") != expected_evidence_role:
            raise ValueError(f"evaluation evidence role mismatch: {path}")
        if method == "oracle_bfs" and role != "air_dev":
            raise ValueError("BFS evaluator oracle is restricted to AIR_dev")
        manifest_path = {
            "air_early": self.config.paths.air_early_manifest,
            "air_dev": self.config.paths.air_dev_manifest,
        }[role]
        manifest = read_jsonl(manifest_path)
        expected_entries = {str(entry["task_hash"]): entry for entry in manifest}
        expected_ids = set(expected_entries)
        if metadata.get("manifest") != relative_path(manifest_path):
            raise ValueError(f"evaluation manifest path mismatch: {path}")
        if metadata.get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError(f"evaluation manifest hash mismatch: {path}")
        if int(metadata.get("task_count", -1)) != len(expected_ids):
            raise ValueError(f"evaluation metadata task count mismatch: {path}")
        if method == "oracle_bfs":
            if metadata.get("checkpoint_sha256") is not None:
                raise ValueError(
                    f"oracle evaluation unexpectedly has a checkpoint: {path}"
                )
        else:
            checkpoint_hash = self.expected_checkpoint_hashes.get((method, seed))
            if (
                checkpoint_hash is None
                or metadata.get("checkpoint_sha256") != checkpoint_hash
            ):
                raise ValueError(f"evaluation checkpoint lineage mismatch: {path}")
        rows = payload.get("task_rows", [])
        actual_ids = [str(row["task_id"]) for row in rows]
        if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
            raise ValueError(f"evaluation task set mismatch: {path}")
        if len(set(actual_ids)) != len(actual_ids):
            raise ValueError(f"duplicate evaluation task rows: {path}")
        for row in rows:
            entry = expected_entries[str(row["task_id"])]
            expected_fields = {
                "maze_size": int(entry["maze_size"]),
                "topology_seed": int(entry["topology_seed"]),
                "start_cell": int(entry["start_cell"]),
                "goal_cell": int(entry["goal_cell"]),
                "optimal_length": int(entry["bfs_path_length"]),
            }
            for field, expected_value in expected_fields.items():
                if row.get(field) != expected_value:
                    raise ValueError(f"evaluation row {field} mismatch: {path}")
            if not isinstance(row.get("success"), bool):
                raise ValueError(f"evaluation success is not boolean: {path}")
            for field in ("spl", "elapsed_seconds", "final_bfs_distance"):
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {field} in {path}")
            if (
                float(row["elapsed_seconds"]) < 0.0
                or int(row["final_bfs_distance"]) < 0
            ):
                raise ValueError(f"negative evaluation timing/distance in {path}")
            if not 0.0 <= float(row["spl"]) <= 1.0:
                raise ValueError(f"evaluation SPL is outside [0,1]: {path}")
            if not 0 <= int(row["path_length"]) <= self.config.evaluation.max_steps:
                raise ValueError(f"evaluation path length is invalid: {path}")
            if not 0 <= int(row["invalid_actions"]) <= int(row["path_length"]):
                raise ValueError(f"evaluation invalid-action count is invalid: {path}")
            success = bool(row["success"])
            if success != (int(row["final_bfs_distance"]) == 0):
                raise ValueError(f"evaluation success/distance mismatch: {path}")
            if success and int(row["path_length"]) < int(row["optimal_length"]):
                raise ValueError(f"evaluation path beats exact BFS impossibly: {path}")
            if (
                not success
                and int(row["path_length"]) != self.config.evaluation.max_steps
            ):
                raise ValueError(
                    f"failed evaluation did not exhaust the step cap: {path}"
                )
            repeats = int(row.get("repeat_states", -1))
            max_visits = int(row.get("max_state_visits", -1))
            if (
                not 0 <= repeats <= int(row["path_length"])
                or not 1 <= max_visits <= int(row["path_length"]) + 1
                or not isinstance(row.get("loop_or_cycle"), bool)
                or bool(row["loop_or_cycle"]) != (max_visits >= 4)
            ):
                raise ValueError(f"evaluation visit/loop accounting is invalid: {path}")
            movement = {
                field: int(row.get(field, -1))
                for field in (
                    "immediate_backtracks",
                    "distance_decrease_actions",
                    "distance_flat_actions",
                    "distance_increase_actions",
                    "dead_end_recovery_opportunities",
                    "dead_end_recovery_successes",
                    "dead_end_recovery_failures",
                )
            }
            if (
                any(value < 0 for value in movement.values())
                or movement["immediate_backtracks"] > int(row["path_length"])
                or movement["distance_decrease_actions"]
                + movement["distance_flat_actions"]
                + movement["distance_increase_actions"]
                != int(row["path_length"])
                or movement["distance_flat_actions"] != int(row["invalid_actions"])
                or movement["dead_end_recovery_opportunities"]
                != movement["dead_end_recovery_successes"]
                + movement["dead_end_recovery_failures"]
                or movement["dead_end_recovery_opportunities"] > int(row["path_length"])
                or movement["dead_end_recovery_failures"] > int(row["invalid_actions"])
            ):
                raise ValueError(
                    f"evaluation movement diagnostics are inconsistent: {path}"
                )
            expected_spl = (
                int(row["optimal_length"])
                / max(int(row["optimal_length"]), int(row["path_length"]))
                if success
                else 0.0
            )
            if not math.isclose(
                float(row["spl"]), expected_spl, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"evaluation SPL is not reproducible: {path}")
            expected_failure = (
                "success"
                if success
                else (
                    "loop_or_cycle"
                    if bool(row["loop_or_cycle"])
                    else (
                        "invalid_action_stall"
                        if int(row["invalid_actions"])
                        >= max(4, int(row["path_length"]) // 2)
                        else "step_cap_or_unresolved"
                    )
                )
            )
            if row.get("failure_reason") != expected_failure:
                raise ValueError(f"evaluation failure taxonomy mismatch: {path}")
            auxiliary = row.get("auxiliary", {})
            if not isinstance(auxiliary, dict) or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in auxiliary.values()
            ):
                raise ValueError(f"evaluation auxiliary metric is non-finite: {path}")
            expected_calls = (
                1.0
                if method in {"j0_static", "j1_static"}
                else float(row["path_length"])
            )
            if not math.isclose(
                float(auxiliary.get("inference_calls", -1.0)),
                expected_calls,
                rel_tol=0.0,
                abs_tol=0.0,
                ):
                raise ValueError(
                    "evaluation static/receding inference-call semantics changed: "
                    f"{path}"
                )
        recomputed = summarize_rows(
            rows,
            seen_max_size=self.config.evaluation.seen_max_size,
            max_steps=self.config.evaluation.max_steps,
        )
        if payload.get("navigation") != recomputed:
            raise ValueError(
                f"evaluation navigation summary is not reproducible: {path}"
            )
        return payload

    def diagnostic(self, path: Path, *, seed: int, role: str) -> dict[str, Any]:
        payload = read_json(path)
        if payload.get("schema") != "air-jepa-stage0-local-diagnostic-v1":
            raise ValueError(f"invalid diagnostic schema: {path}")
        metadata = payload["metadata"]
        self._common(path, metadata)
        if (
            metadata.get("seed") != seed
            or metadata.get("split_role") != role
            or metadata.get("k") != 128
        ):
            raise ValueError(f"diagnostic identity mismatch: {path}")
        manifest_path = (
            self.config.paths.air_dev_manifest
            if role == "air_dev"
            else self.config.paths.air_early_manifest
        )
        if metadata.get("manifest") != relative_path(manifest_path) or metadata.get(
            "manifest_sha256"
        ) != sha256_file(manifest_path):
            raise ValueError(f"diagnostic manifest lineage mismatch: {path}")
        checkpoint_hash = self.expected_checkpoint_hashes.get(("air0_jepa", seed))
        if (
            checkpoint_hash is None
            or metadata.get("checkpoint_sha256") != checkpoint_hash
        ):
            raise ValueError(f"diagnostic checkpoint lineage mismatch: {path}")
        if int(metadata.get("states_per_maze", -1)) != int(
            self.config.evaluation.local_states_per_maze
        ):
            raise ValueError(f"diagnostic states-per-maze metadata mismatch: {path}")
        if (
            int(metadata.get("distance_max", -1)) != DISTANCE_MAX
            or DISTANCE_MAX != self.config.model.max_distance
            or int(metadata.get("distance_ece_bins", -1)) != DISTANCE_ECE_BINS
            or metadata.get("distance_calibration")
            != "top-class exact-bin ECE over 15 equal-width confidence bins"
        ):
            raise ValueError(f"diagnostic distance protocol mismatch: {path}")
        expected_tasks = 900 if role == "air_dev" else 210
        task_rows = payload.get("task_rows", [])
        state_rows = payload.get("state_rows", [])
        if len(task_rows) != expected_tasks:
            raise ValueError(f"diagnostic task count mismatch: {path}")
        expected_states = expected_tasks * self.config.evaluation.local_states_per_maze
        if len(state_rows) != expected_states:
            raise ValueError(f"diagnostic state count mismatch: {path}")
        if any(
            int(row.get("sampled_states", -1))
            != self.config.evaluation.local_states_per_maze
            for row in task_rows
        ):
            raise ValueError(
                f"diagnostic did not sample exactly 24 states/task: {path}"
            )
        manifest_entries = {
            str(entry["task_hash"]): entry for entry in read_jsonl(manifest_path)
        }
        task_ids = [str(row["task_id"]) for row in task_rows]
        if len(set(task_ids)) != expected_tasks or set(task_ids) != set(
            manifest_entries
        ):
            raise ValueError(f"diagnostic task identity mismatch: {path}")
        states_by_task: defaultdict[str, list[int]] = defaultdict(list)
        environment_cache: dict[str, Any] = {}
        distance_cache: dict[str, np.ndarray] = {}
        for row in state_rows:
            task_id = str(row["task_id"])
            if task_id not in manifest_entries:
                raise ValueError(f"diagnostic state has unknown task: {path}")
            entry = manifest_entries[task_id]
            if int(row["maze_size"]) != int(entry["maze_size"]):
                raise ValueError(f"diagnostic state maze size mismatch: {path}")
            if task_id not in environment_cache:
                env = validate_manifest_entry(entry, check_bfs=False)
                environment_cache[task_id] = env
                distance_cache[task_id] = bfs_distances_from(
                    env._maze_mask,
                    int(env._goal_position),
                    int(env.config.width),
                )
            env = environment_cache[task_id]
            distances = distance_cache[task_id]
            state = int(row["state"])
            if (
                not 0 <= state < int(env.config.width) ** 2
                or bool(env._maze_mask.reshape(-1)[state])
                or state == int(env._goal_position)
                or int(distances[state]) <= 0
                or int(row.get("current_distance", -1)) != int(distances[state])
            ):
                raise ValueError(f"diagnostic sampled state is invalid: {path}")
            candidate_distances = [
                int(distances[next_state(env, state, action)]) for action in ACTION_IDS
            ]
            observed_distances = [
                int(value) for value in row.get("candidate_distances", [])
            ]
            if observed_distances != candidate_distances:
                raise ValueError(f"diagnostic successor BFS labels are invalid: {path}")
            minimum = min(candidate_distances)
            optimal = [distance == minimum for distance in candidate_distances]
            if row.get("optimal_action_mask") != optimal:
                raise ValueError(f"diagnostic optimal action mask is invalid: {path}")
            for ranking_name in (
                "predicted_ranking",
                "true_ranking",
                "copy_ranking",
                "permuted_ranking",
                "zero_ranking",
            ):
                _validate_diagnostic_ranking(
                    row.get(ranking_name),
                    candidate_distances=candidate_distances,
                    optimal_action_mask=optimal,
                    path=path,
                )
            for class_name, confidence_name in (
                ("predicted_cost_class", "predicted_cost_confidence"),
                ("true_future_cost_class", "true_future_cost_confidence"),
            ):
                classes = row.get(class_name, [])
                confidence = row.get(confidence_name, [])
                if (
                    len(classes) != 4
                    or len(confidence) != 4
                    or any(
                        not isinstance(value, int) or not 0 <= value <= DISTANCE_MAX
                        for value in classes
                    )
                    or any(
                        not math.isfinite(float(value))
                        or not 1.0 / (DISTANCE_MAX + 1) - 1e-7
                        <= float(value)
                        <= 1.0 + 1e-7
                        for value in confidence
                    )
                ):
                    raise ValueError(
                        f"diagnostic distance class/confidence is invalid: {path}"
                    )
            nonnegative_fields = (
                "normalized_field_error",
                "normalized_delta_error",
                "copy_delta_normalized",
                "predicted_candidate_pairwise",
                "target_candidate_pairwise",
                "predicted_variance",
                "target_variance",
            )
            if any(
                not math.isfinite(float(row.get(field, math.nan)))
                or float(row[field]) < 0.0
                for field in nonnegative_fields
            ):
                raise ValueError(f"diagnostic future metric is invalid: {path}")
            predicted_correct = bool(row["predicted_ranking"]["top1"])
            true_correct = bool(row["true_ranking"]["top1"])
            expected_error = (
                "correct"
                if predicted_correct
                else (
                    "prediction_flip"
                    if true_correct
                    else "energy_wrong_with_true_future"
                )
            )
            if row.get("local_error_type") != expected_error:
                raise ValueError(f"diagnostic local error type is invalid: {path}")
            states_by_task[task_id].append(state)
        if any(
            len(states) != self.config.evaluation.local_states_per_maze
            or len(set(states)) != len(states)
            for states in states_by_task.values()
        ):
            raise ValueError(
                f"diagnostic state sampling is incomplete/duplicated: {path}"
            )
        for task_id, entry in manifest_entries.items():
            expected = deterministic_states(
                entry,
                count=self.config.evaluation.local_states_per_maze,
            )
            if sorted(states_by_task[task_id]) != sorted(expected):
                raise ValueError(f"diagnostic state selection was changed: {path}")
        if payload.get("summary") != summarize_state_rows(state_rows):
            raise ValueError(f"diagnostic summary is not reproducible: {path}")
        return payload


def require_bridge_audit(
    config: Any,
    *,
    protocol_sha256: str,
    package_sha256: str,
    source_lock_sha256: str,
) -> dict[str, Any]:
    path = resolve_path(config.paths.run_root) / "audits/historical_bridge_parity.json"
    payload = read_json(path)
    if (
        payload.get("schema") != "air-jepa-stage0-bridge-audit-v1"
        or payload.get("experiment_id") != config.experiment_id
        or payload.get("passed") is not True
        or payload.get("failures") != []
    ):
        raise ValueError("historical J0/J1 bridge audit is absent or failed")
    verify_signature(payload, "bridge_audit_sha256")
    for key, expected in (
        ("protocol_sha256", protocol_sha256),
        ("package_sha256", package_sha256),
        ("source_lock_sha256", source_lock_sha256),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"historical bridge audit {key} mismatch")
    comparisons = payload.get("comparisons", {})
    if len(comparisons) != 6 or not all(
        value.get("exact_parity") is True for value in comparisons.values()
    ):
        raise ValueError("historical bridge audit lacks six exact-parity cells")
    runtime = payload.get("runtime", {})
    if not formal_runtime_valid(runtime):
        raise ValueError("historical bridge audit runtime is not deterministic H800")
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "bridge_audit_sha256": payload["bridge_audit_sha256"],
        "cells": len(comparisons),
        "runtime": runtime,
    }


def require_protocol_audit(
    config: Any,
    *,
    protocol_sha256: str,
    package_sha256: str,
    package_code_fingerprint: str,
    source_lock_sha256: str,
) -> dict[str, Any]:
    path = resolve_path(config.paths.audit_output)
    payload = read_json(path)
    if (
        payload.get("schema") != "air-jepa-stage0-protocol-audit-v1"
        or payload.get("experiment_id") != config.experiment_id
        or payload.get("passed") is not True
        or payload.get("git_dirty") is not False
        or not payload.get("git_commit")
        or payload.get("code_fingerprint") != package_code_fingerprint
    ):
        raise ValueError("formal L0 protocol audit is absent or invalid")
    verify_signature(payload, "protocol_audit_sha256")
    for key, expected in (
        ("protocol_sha256", protocol_sha256),
        ("package_sha256", package_sha256),
        ("source_lock_sha256", source_lock_sha256),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"formal L0 protocol audit {key} mismatch")
    hardware = payload.get("hardware", {})
    runtime = payload.get("runtime", {})
    if not formal_runtime_valid(runtime):
        raise ValueError("formal L0 protocol audit runtime is not deterministic H800")
    devices = hardware.get("devices", [])
    if (
        hardware.get("skipped") is not False
        or hardware.get("formal_eligible") is not True
        or len(devices) != config.worker_count
        or any("H800" not in str(device.get("name", "")).upper() for device in devices)
    ):
        raise ValueError("formal L0 protocol audit lacks four verified H800 GPUs")
    pairing = payload.get("pairing", {})
    if set(pairing) != {str(seed) for seed in config.seeds} or any(
        int(record.get("checked_batches", -1)) != FORMAL_PAIRING_BATCHES
        or not record.get("initial_model_state_sha256")
        or not record.get("sample_stream_sha256")
        for record in pairing.values()
    ):
        raise ValueError("formal L0 paired-stream audit is incomplete")
    expected_counts = {
        key: len(value) for key, value in expected_matrix(config).items()
    }
    if payload.get("matrix_counts") != expected_counts:
        raise ValueError(
            "formal L0 audit matrix counts differ from executable protocol"
        )
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "protocol_audit_sha256": payload["protocol_audit_sha256"],
        "hardware": hardware,
        "runtime": runtime,
        "pairing": pairing,
        "pairing_batches": FORMAL_PAIRING_BATCHES,
        "matrix_counts": expected_counts,
    }


def require_benchmark(
    config: Any,
    *,
    protocol_sha256: str,
    package_sha256: str,
    package_code_fingerprint: str,
    source_lock_sha256: str,
) -> dict[str, Any]:
    path = resolve_path(config.paths.benchmark_output)
    payload = read_json(path)
    if (
        payload.get("schema") != "air-jepa-stage0-benchmark-v1"
        or payload.get("experiment_id") != config.experiment_id
        or payload.get("performance_blind") is not True
        or payload.get("git_dirty") is not False
        or not payload.get("git_commit")
        or payload.get("code_fingerprint") != package_code_fingerprint
    ):
        raise ValueError("L0 performance-blind benchmark is absent or invalid")
    verify_signature(payload, "benchmark_sha256")
    for key, expected in (
        ("protocol_sha256", protocol_sha256),
        ("package_sha256", package_sha256),
        ("source_lock_sha256", source_lock_sha256),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"L0 benchmark {key} mismatch")
    compute_match = payload.get("compute_match", {})
    k_by_size = compute_match.get("k_by_size", {})
    if (
        compute_match.get("performance_used") is not False
        or compute_match.get("joint_k") not in config.evaluation.k_values
        or set(k_by_size) != {"21", "25"}
        or any(value not in config.evaluation.k_values for value in k_by_size.values())
        or compute_match.get("joint_k") != min(k_by_size.values())
    ):
        raise ValueError("L0 benchmark compute-match lock is invalid")
    expected_k_keys = {str(k) for k in config.evaluation.k_values}
    for size in (21, 25):
        size_key = str(size)
        workspace = payload.get("workspace_analytical_macs", {}).get(size_key, {})
        components = payload.get("workspace_analytical_macs_by_component", {}).get(
            size_key, {}
        )
        air_total = payload.get("air_total_inference_macs", {}).get(size_key, {})
        if not all(
            set(values) == expected_k_keys
            for values in (workspace, components, air_total)
        ):
            raise ValueError(f"L0 benchmark K/MAC curve is incomplete at size {size}")
        representation = int(
            payload.get("representation_planning_conv_macs", {}).get(size_key, -1)
        )
        j1_total = int(
            payload.get("j1_k128_total_inference_macs", {}).get(size_key, -1)
        )
        if representation <= 0 or j1_total <= representation:
            raise ValueError(
                f"L0 benchmark source MAC accounting failed at size {size}"
            )
        for k in config.evaluation.k_values:
            k_key = str(k)
            if (
                int(workspace[k_key]) <= 0
                or sum(int(value) for value in components[k_key].values())
                != int(workspace[k_key])
                or int(air_total[k_key]) != representation + int(workspace[k_key])
            ):
                raise ValueError(
                    "L0 benchmark AIR component MAC accounting failed at "
                    f"size {size}, K={k}"
                )
        eligible = [
            k
            for k in config.evaluation.k_values
            if int(air_total[str(k)]) <= 1.05 * j1_total
        ]
        if not eligible or max(eligible) != int(k_by_size[size_key]):
            raise ValueError(f"L0 benchmark compute-match K is wrong at size {size}")
    counts = payload.get("parameter_counts", {})
    if (
        int(counts.get("frozen_representation", -1)) <= 0
        or int(counts.get("air_workspace", -1)) <= 0
        or int(counts.get("air_total_inference", -1))
        != int(counts.get("frozen_representation", -1))
        + int(counts.get("air_workspace", -1))
        or int(counts.get("j1_total_inference", -1)) <= 0
    ):
        raise ValueError("L0 benchmark parameter accounting is invalid")
    runtime = payload.get("runtime", {})
    if not formal_runtime_valid(runtime):
        raise ValueError("L0 benchmark runtime is not deterministic H800")
    forward = payload.get("k128_forward", {})
    backward = payload.get("k128_forward_backward", {})
    if (
        int(forward.get("tasks", -1)) != FORMAL_BENCHMARK_TASKS
        or any(
            float(forward.get(field, -1.0)) <= 0.0
            for field in ("seconds_total", "seconds_mean", "tasks_per_second")
        )
        or int(backward.get("iterations", -1)) != FORMAL_BACKWARD_K
        or int(backward.get("repeats", -1)) != FORMAL_BACKWARD_REPEATS
        or float(backward.get("seconds_mean", -1.0)) <= 0.0
        or int(payload.get("peak_cuda_memory_bytes", -1)) <= 0
    ):
        raise ValueError("L0 benchmark timing/memory evidence is invalid")
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "benchmark_sha256": payload["benchmark_sha256"],
        "compute_match": compute_match,
        "parameter_counts": payload["parameter_counts"],
        "peak_cuda_memory_bytes": payload["peak_cuda_memory_bytes"],
        "k128_forward": payload["k128_forward"],
        "runtime": runtime,
    }


def require_sealed_roles_absent(config: Any) -> list[str]:
    roots = [
        resolve_path(config.paths.run_root) / kind / role
        for kind in ("results", "releases")
        for role in ("air_select", "air_final")
    ]
    present = [relative_path(path) for path in roots if path.exists()]
    if present:
        raise PermissionError(f"sealed result roles were opened: {present}")
    return [relative_path(path) for path in roots]


def _validate_training_evidence(
    payload: dict[str, Any],
    *,
    config: Any,
    seed: int,
    path: Path,
) -> None:
    if payload.get("config_sha256") != sha256_file(DEFAULT_CONFIG):
        raise ValueError(f"AIR checkpoint config file hash mismatch: {path}")
    if int(payload.get("model_seed", -1)) != seed + 70_000:
        raise ValueError(f"AIR checkpoint model seed mismatch: {path}")
    if payload.get("rng_stream_seeds") != make_rng_streams(seed).stream_seeds:
        raise ValueError(f"AIR checkpoint RNG stream seeds mismatch: {path}")
    k_counts = {
        int(key): int(value) for key, value in payload.get("k_counts", {}).items()
    }
    expected_iterations = progressive_iteration_signature(
        seed=seed,
        steps=config.training.steps,
        phase_steps=config.training.phase_steps,
        k_train=config.training.k_train,
    )
    if (
        not k_counts
        or set(k_counts) != set(config.training.k_train)
        or sum(k_counts.values()) != config.training.steps
        or any(value <= 0 for value in k_counts.values())
        or {str(key): value for key, value in sorted(k_counts.items())}
        != expected_iterations["counts"]
        or payload.get("progressive_iteration_stream_sha256")
        != expected_iterations["sha256"]
    ):
        raise ValueError(f"AIR checkpoint cumulative K accounting failed: {path}")
    accounting = payload.get("training_accounting", {})
    if (
        int(accounting.get("map_state_examples", -1))
        != config.training.steps * config.training.batch_size
        or int(accounting.get("successor_examples", -1))
        != config.training.steps * config.training.batch_size * 4
        or float(accounting.get("elapsed_seconds", -1.0)) <= 0.0
        or int(accounting.get("peak_cuda_memory_bytes", -1)) <= 0
    ):
        raise ValueError(f"AIR checkpoint training accounting failed: {path}")
    logs = payload.get("training_log", [])
    expected_steps = list(
        range(
            config.training.log_every,
            config.training.steps + 1,
            config.training.log_every,
        )
    )
    if [int(row.get("step", -1)) for row in logs] != expected_steps:
        raise ValueError(f"AIR checkpoint 500-step log schedule is incomplete: {path}")
    window_counts: Counter[int] = Counter()
    required_finite = (
        "total",
        "action",
        "future",
        "cost",
        "future_field_normalized",
        "future_delta_normalized",
        "future_field_raw_mse",
        "future_delta_raw_mse",
        "copy_delta_normalized",
        "gradient_norm",
        "iterations",
        "learning_rate",
        "window_elapsed_seconds",
        "steps_per_second",
    )
    for row in logs:
        step = int(row.get("step", -1))
        current_counts = {
            int(key): int(value)
            for key, value in row.get("window_k_counts", {}).items()
        }
        reported_cumulative = {
            int(key): int(value)
            for key, value in row.get("cumulative_k_counts", {}).items()
        }
        window_counts.update(current_counts)
        phase = min(
            (step - 1) // config.training.phase_steps,
            len(config.training.k_train) - 1,
        )
        allowed_k = set(config.training.k_train[: phase + 1])
        if (
            int(row.get("window_steps", -1)) != config.training.log_every
            or sum(current_counts.values()) != config.training.log_every
            or not set(current_counts) <= set(config.training.k_train)
            or not set(current_counts) <= allowed_k
            or reported_cumulative != dict(window_counts)
            or int(row.get("peak_cuda_memory_bytes", -1)) <= 0
            or any(
                not math.isfinite(float(row.get(key, math.nan)))
                for key in required_finite
            )
            or float(row["window_elapsed_seconds"]) <= 0.0
            or float(row["steps_per_second"]) <= 0.0
        ):
            raise ValueError(
                f"AIR checkpoint has an invalid training log window: {path}"
            )
    if dict(window_counts) != k_counts or max(
        int(row["peak_cuda_memory_bytes"]) for row in logs
    ) != int(accounting["peak_cuda_memory_bytes"]):
        raise ValueError(f"AIR checkpoint window/cumulative accounting differs: {path}")
    gradients = payload.get("gradient_history", [])
    expected_gradient_steps = [
        1,
        *range(
            config.training.gradient_audit_every,
            config.training.steps + 1,
            config.training.gradient_audit_every,
        ),
    ]
    if [int(row.get("step", -1)) for row in gradients] != expected_gradient_steps:
        raise ValueError(
            f"AIR checkpoint gradient audit schedule is incomplete: {path}"
        )
    for row in gradients:
        step = int(row.get("step", -1))
        phase = min(
            (step - 1) // config.training.phase_steps,
            len(config.training.k_train) - 1,
        )
        if int(row.get("iterations", -1)) not in config.training.k_train[
            : phase + 1
        ] or any(
            value is not None and not math.isfinite(float(value))
            for key, value in row.items()
            if key not in {"step", "iterations"}
        ):
            raise ValueError(f"AIR checkpoint gradient audit is invalid: {path}")
    if (
        int(payload.get("paired_sample_stream_prefix_batches", -1))
        != FORMAL_PAIRING_BATCHES
        or not payload.get("paired_sample_stream_prefix_sha256")
        or not payload.get("paired_sample_stream_sha256")
    ):
        raise ValueError(f"AIR checkpoint paired stream evidence is invalid: {path}")
    moments = payload.get("future_target_channel_moments", {})
    means = moments.get("mean", [])
    variances = moments.get("variance", [])
    if (
        int(moments.get("count_per_channel", -1)) <= 0
        or len(means) != config.model.input_dim
        or len(variances) != config.model.input_dim
        or any(not math.isfinite(float(value)) for value in [*means, *variances])
        or any(float(value) < 0.0 for value in variances)
    ):
        raise ValueError(f"AIR checkpoint future target moments are invalid: {path}")


def paired_checkpoint_audit(
    config: Any,
    *,
    seeds: tuple[int, ...],
    source_lock: dict[str, Any],
    l0_pairing: dict[str, Any],
    protocol_sha256: str,
    package_sha256: str,
    package_code_fingerprint: str,
    source_lock_sha256: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    runtime_signatures: set[tuple[Any, ...]] = set()
    reference_model = AIRWorkspaceModel(config.model)
    expected_workspace_macs = {
        str(size): {
            str(k): reference_model.analytical_macs(size, k)
            for k in config.evaluation.k_values
        }
        for size in (21, 25)
    }
    expected_component_macs = {
        str(size): {
            str(k): reference_model.analytical_mac_breakdown(size, k)
            for k in config.evaluation.k_values
        }
        for size in (21, 25)
    }
    for seed in seeds:
        expected_source_representation = source_lock.get("records", {}).get(
            str(seed), {}
        ).get("representation")
        expected_l0_pairing = l0_pairing.get(str(seed))
        if not isinstance(expected_source_representation, dict) or not isinstance(
            expected_l0_pairing, dict
        ):
            raise ValueError(f"AIR checkpoint audit lacks source/L0 seed {seed}")
        by_seed[seed] = {}
        for method in ("air0_direct", "air0_jepa"):
            path = format_template(
                config.paths.air_checkpoint_template,
                method=method,
                seed=seed,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (
                payload.get("experiment_id") != EXPERIMENT_ID
                or int(payload.get("format_version", -1)) != FORMAT_VERSION
                or payload.get("method") != method
                or int(payload.get("seed", -1)) != seed
                or payload.get("config")
                != config.model_dump(mode="json", by_alias=True)
            ):
                raise ValueError(f"AIR checkpoint identity/config mismatch: {path}")
            if (
                payload.get("formal") is not True
                or payload.get("checkpoint_role") != "final_step"
                or int(payload.get("optimizer_steps", -1)) != config.training.steps
            ):
                raise ValueError(f"release rejects non-final AIR checkpoint: {path}")
            for key, expected in (
                ("protocol_sha256", protocol_sha256),
                ("package_sha256", package_sha256),
                ("source_lock_sha256", source_lock_sha256),
            ):
                if payload.get(key) != expected:
                    raise ValueError(f"AIR checkpoint {key} mismatch: {path}")
            if payload.get("git_dirty") is not False:
                raise ValueError(f"AIR checkpoint came from a dirty worktree: {path}")
            if not payload.get("git_commit"):
                raise ValueError(f"AIR checkpoint lacks git provenance: {path}")
            if payload.get("code_fingerprint") != package_code_fingerprint:
                raise ValueError(f"AIR checkpoint code fingerprint mismatch: {path}")
            if payload.get("source_representation") != expected_source_representation:
                raise ValueError(
                    f"AIR checkpoint source representation lineage mismatch: {path}"
                )
            if (
                payload.get("initial_model_state_sha256")
                != expected_l0_pairing.get("initial_model_state_sha256")
                or payload.get("paired_sample_stream_prefix_sha256")
                != expected_l0_pairing.get("sample_stream_sha256")
                or int(payload.get("paired_sample_stream_prefix_batches", -1))
                != FORMAL_PAIRING_BATCHES
            ):
                raise ValueError(
                    f"AIR checkpoint does not match the signed L0 pairing audit: {path}"
                )
            runtime = payload.get("runtime", {})
            if not formal_runtime_valid(runtime):
                raise ValueError(
                    f"AIR checkpoint runtime is not deterministic H800: {path}"
                )
            if state_dict_sha256(payload["model_state_dict"]) != payload.get(
                "model_state_sha256"
            ):
                raise ValueError(f"AIR checkpoint state signature failed: {path}")
            by_seed[seed][method] = payload
            runtime_signatures.add(runtime_signature(payload.get("runtime", {})))
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "path": relative_path(path),
                    "sha256": sha256_file(path),
                    "initial_model_state_sha256": payload["initial_model_state_sha256"],
                    "paired_sample_stream_sha256": payload[
                        "paired_sample_stream_sha256"
                    ],
                    "component_parameter_counts": payload["component_parameter_counts"],
                    "representation_component_parameter_counts": payload[
                        "representation_component_parameter_counts"
                    ],
                    "total_inference_parameter_count": payload[
                        "total_inference_parameter_count"
                    ],
                    "total_inference_macs": payload["total_inference_macs"],
                    "training_accounting": payload["training_accounting"],
                    "runtime": payload["runtime"],
                }
            )
            if payload.get("model_trainable_parameter_count") != payload.get(
                "model_parameter_count"
            ):
                raise ValueError(
                    f"AIR checkpoint did not train the full AIR core: {path}"
                )
            if payload.get("representation_trainable_parameter_count") != 0:
                raise ValueError(
                    f"AIR checkpoint did not freeze representation: {path}"
                )
            if sum(payload["component_parameter_counts"].values()) != payload.get(
                "model_parameter_count"
            ):
                raise ValueError(f"AIR component parameter accounting failed: {path}")
            if sum(
                payload["representation_component_parameter_counts"].values()
            ) != payload.get("representation_planning_parameter_count"):
                raise ValueError(
                    f"representation component parameter accounting failed: {path}"
                )
            if payload.get("workspace_analytical_macs") != expected_workspace_macs:
                raise ValueError(f"AIR workspace MAC curve differs from code: {path}")
            if (
                payload.get("workspace_analytical_macs_by_component")
                != expected_component_macs
            ):
                raise ValueError(f"AIR component MAC curve differs from code: {path}")
            representation_macs = payload.get("representation_planning_conv_macs", {})
            total_macs = payload.get("total_inference_macs", {})
            for size in (21, 25):
                size_key = str(size)
                representation_value = int(representation_macs.get(size_key, -1))
                if representation_value <= 0:
                    raise ValueError(
                        "AIR checkpoint lacks representation MACs "
                        f"at size {size}: {path}"
                    )
                expected_total = {
                    str(k): representation_value
                    + expected_workspace_macs[size_key][str(k)]
                    for k in config.evaluation.k_values
                }
                if total_macs.get(size_key) != expected_total:
                    raise ValueError(
                        "AIR total MAC curve is not reproducible "
                        f"at size {size}: {path}"
                    )
            _validate_training_evidence(
                payload,
                config=config,
                seed=seed,
                path=path,
            )
        direct = by_seed[seed]["air0_direct"]
        treatment = by_seed[seed]["air0_jepa"]
        paired_fields = (
            "initial_model_state_sha256",
            "paired_sample_stream_sha256",
            "progressive_iteration_stream_sha256",
            "paired_sample_stream_prefix_batches",
            "paired_sample_stream_prefix_sha256",
            "rng_stream_seeds",
            "k_counts",
            "source_representation",
            "future_target_channel_moments",
            "model_parameter_count",
            "representation_planning_parameter_count",
            "component_parameter_counts",
            "representation_component_parameter_counts",
            "total_inference_parameter_count",
            "representation_planning_conv_macs",
            "workspace_analytical_macs",
            "workspace_analytical_macs_by_component",
            "total_inference_macs",
        )
        for field in paired_fields:
            if direct.get(field) != treatment.get(field):
                raise ValueError(f"seed {seed} AIR pairing mismatch for {field}")
    if len(runtime_signatures) != 1:
        raise ValueError("AIR checkpoints mix incompatible training runtimes")
    return {
        "passed": True,
        "records": records,
        "runtime_signature": list(next(iter(runtime_signatures))),
    }


def aggregate_breakdowns(
    loaded: dict[tuple[str, int, int], dict[str, Any]],
    config: Any,
) -> dict[str, Any]:
    methods = ("j1_receding", "air0_direct", "air0_jepa")
    by_size = []
    by_path = []
    failures = []
    movement_diagnostics = []
    for method in methods:
        payloads = [loaded[(method, seed, 128)] for seed in config.seeds]
        size_names = sorted(
            payloads[0]["navigation"]["by_size"], key=lambda value: int(value)
        )
        for size in size_names:
            by_size.append(
                {
                    "method": method,
                    "maze_size": int(size),
                    "mean_sr": float(
                        np.mean(
                            [
                                payload["navigation"]["by_size"][size]["sr"]
                                for payload in payloads
                            ]
                        )
                    ),
                    "mean_spl": float(
                        np.mean(
                            [
                                payload["navigation"]["by_size"][size]["spl"]
                                for payload in payloads
                            ]
                        )
                    ),
                }
            )
        for path_bin in payloads[0]["navigation"]["by_shortest_path"]:
            by_path.append(
                {
                    "method": method,
                    "path_bin": path_bin,
                    "mean_sr": float(
                        np.mean(
                            [
                                payload["navigation"]["by_shortest_path"][path_bin][
                                    "sr"
                                ]
                                for payload in payloads
                            ]
                        )
                    ),
                    "mean_n": float(
                        np.mean(
                            [
                                payload["navigation"]["by_shortest_path"][path_bin]["n"]
                                for payload in payloads
                            ]
                        )
                    ),
                }
            )
        counts = Counter(
            str(row["failure_reason"])
            for payload in payloads
            for row in payload["task_rows"]
        )
        failures.append(
            {
                "method": method,
                "counts": dict(sorted(counts.items())),
                "total_seed_task_rows": sum(counts.values()),
            }
        )
        all_rows = [row for payload in payloads for row in payload["task_rows"]]
        actions = sum(int(row["path_length"]) for row in all_rows)
        dead_end_opportunities = sum(
            int(row["dead_end_recovery_opportunities"]) for row in all_rows
        )
        movement_diagnostics.append(
            {
                "method": method,
                "total_seed_task_rows": len(all_rows),
                "total_actions": actions,
                "immediate_backtrack_rate": sum(
                    int(row["immediate_backtracks"]) for row in all_rows
                )
                / max(actions, 1),
                "distance_decrease_rate": sum(
                    int(row["distance_decrease_actions"]) for row in all_rows
                )
                / max(actions, 1),
                "distance_flat_rate": sum(
                    int(row["distance_flat_actions"]) for row in all_rows
                )
                / max(actions, 1),
                "distance_increase_rate": sum(
                    int(row["distance_increase_actions"]) for row in all_rows
                )
                / max(actions, 1),
                "dead_end_recovery_opportunities": dead_end_opportunities,
                "dead_end_recovery_failure_rate": sum(
                    int(row["dead_end_recovery_failures"]) for row in all_rows
                )
                / max(dead_end_opportunities, 1),
            }
        )
    return {
        "by_size": by_size,
        "by_shortest_path": by_path,
        "failures": failures,
        "movement_diagnostics": movement_diagnostics,
    }


def compute_accounting(
    config: Any,
    *,
    checkpoint_audit: dict[str, Any],
    loaded: dict[tuple[str, int, int], dict[str, Any]],
    source_lock: dict[str, Any],
    locked_compute_match: dict[str, Any],
) -> dict[str, Any]:
    """Build descriptive parameter, MAC, runtime, and compute-match tables."""

    air_records = {
        (str(record["method"]), int(record["seed"])): record
        for record in checkpoint_audit["records"]
    }
    source_records: dict[int, dict[str, Any]] = {}
    for seed in config.seeds:
        record = source_lock["records"][str(seed)]["j1"]
        path = resolve_path(record["path"])
        if sha256_file(path) != record["file_sha256"]:
            raise ValueError(f"J1 checkpoint changed during compute audit: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source_records[seed] = payload

    parameter_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        source = source_records[seed]
        planner_parameters = int(source.get("planner_parameter_count", -1))
        representation_parameters = int(
            source.get("representation_planning_parameter_count", -1)
        )
        total_parameters = int(source.get("total_inference_parameter_count", -1))
        if (
            planner_parameters <= 0
            or representation_parameters <= 0
            or total_parameters != planner_parameters + representation_parameters
        ):
            raise ValueError(f"J1 parameter accounting is invalid for seed {seed}")
        parameter_rows.append(
            {
                "method": "j1_receding",
                "seed": seed,
                "components": {
                    "frozen_representation": representation_parameters,
                    "iterative_planner": planner_parameters,
                },
                "total_inference_parameter_count": total_parameters,
            }
        )
        for method in ("air0_direct", "air0_jepa"):
            air = air_records[(method, seed)]
            components = {
                **{
                    f"frozen_representation_{key}": int(value)
                    for key, value in air[
                        "representation_component_parameter_counts"
                    ].items()
                },
                **{
                    f"air_{key}": int(value)
                    for key, value in air["component_parameter_counts"].items()
                },
            }
            if sum(components.values()) != int(air["total_inference_parameter_count"]):
                raise ValueError(
                    "total inference parameter accounting failed for "
                    f"{method} seed {seed}"
                )
            parameter_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "components": components,
                    "total_inference_parameter_count": int(
                        air["total_inference_parameter_count"]
                    ),
                }
            )
            accounting = air["training_accounting"]
            training_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "elapsed_seconds": float(accounting["elapsed_seconds"]),
                    "map_state_examples": int(accounting["map_state_examples"]),
                    "successor_examples": int(accounting["successor_examples"]),
                    "peak_cuda_memory_bytes": int(accounting["peak_cuda_memory_bytes"]),
                }
            )

    def source_total_macs(seed: int, size: int, k: int) -> int:
        payload = source_records[seed]
        curve = {
            int(key): int(value)
            for key, value in payload["planner_inference_conv_macs"][str(size)].items()
        }
        if k in curve:
            planner = curve[k]
        else:
            points = sorted(curve.items())
            if len(points) < 2:
                raise ValueError(f"J1 MAC curve cannot derive K={k} for seed {seed}")
            (first_k, first_value), (second_k, second_value) = points[:2]
            delta_k = second_k - first_k
            delta_value = second_value - first_value
            if delta_k <= 0 or delta_value % delta_k:
                raise ValueError(f"J1 MAC curve is not integral-affine for seed {seed}")
            per_iteration = delta_value // delta_k
            fixed = first_value - first_k * per_iteration
            if any(value != fixed + count * per_iteration for count, value in points):
                raise ValueError(f"J1 MAC curve is not affine for seed {seed}")
            planner = fixed + k * per_iteration
        representation = int(payload["representation_inference_conv_macs"][str(size)])
        if planner <= 0 or representation <= 0:
            raise ValueError(f"J1 MAC accounting is invalid for seed {seed}")
        return planner + representation

    def air_total_macs(method: str, seed: int, size: int, k: int) -> int:
        value = int(
            air_records[(method, seed)]["total_inference_macs"][str(size)][str(k)]
        )
        if value <= 0:
            raise ValueError(f"AIR MAC accounting is invalid for {method} seed {seed}")
        return value

    quality_vs_compute: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for method in ("j1_receding", "air0_direct", "air0_jepa"):
        for seed in config.seeds:
            for k in config.evaluation.k_values:
                result = loaded[(method, seed, k)]
                task_rows = result["task_rows"]
                successful = [row for row in task_rows if bool(row["success"])]
                runtime_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "k": k,
                        "mean_task_wall_seconds": float(
                            np.mean(
                                [float(row["elapsed_seconds"]) for row in task_rows]
                            )
                        ),
                        "mean_success_episode_wall_seconds": (
                            float(
                                np.mean(
                                    [
                                        float(row["elapsed_seconds"])
                                        for row in successful
                                    ]
                                )
                            )
                            if successful
                            else None
                        ),
                        "evaluation_elapsed_seconds": float(
                            result["metadata"]["elapsed_seconds"]
                        ),
                    }
                )
                for size in (21, 25):
                    macs = (
                        source_total_macs(seed, size, k)
                        if method == "j1_receding"
                        else air_total_macs(method, seed, size, k)
                    )
                    sr = float(result["navigation"]["by_size"][str(size)]["sr"])
                    quality_vs_compute.append(
                        {
                            "method": method,
                            "seed": seed,
                            "maze_size": size,
                            "k": k,
                            "total_inference_macs": macs,
                            "sr": sr,
                            "sr_per_gmac": sr * 1_000_000_000.0 / float(macs),
                        }
                    )

    compute_match_by_size: dict[str, int] = {}
    for size in (21, 25):
        eligible = [
            k
            for k in config.evaluation.k_values
            if all(
                air_total_macs(method, seed, size, k)
                <= 1.05 * source_total_macs(seed, size, 128)
                for seed in config.seeds
                for method in ("air0_direct", "air0_jepa")
            )
        ]
        if not eligible:
            raise ValueError(f"no compute-matched AIR K exists at size {size}")
        compute_match_by_size[str(size)] = max(eligible)
    compute_match_k = min(compute_match_by_size.values())
    if (
        compute_match_by_size != locked_compute_match.get("k_by_size")
        or compute_match_k != locked_compute_match.get("joint_k")
        or locked_compute_match.get("performance_used") is not False
    ):
        raise ValueError(
            "L3 compute-match result differs from the performance-blind L0 lock"
        )
    compute_matched_rows = []
    for seed in config.seeds:
        j1 = loaded[("j1_receding", seed, 128)]
        for method in ("air0_direct", "air0_jepa"):
            air = loaded[(method, seed, compute_match_k)]
            compute_matched_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "air_k": compute_match_k,
                    "j1_k": 128,
                    "overall_sr_delta": endpoint(air, "overall")
                    - endpoint(j1, "overall"),
                    "ood_sr_delta": endpoint(air, "ood") - endpoint(j1, "ood"),
                }
            )
    return {
        "parameter_rows": parameter_rows,
        "training_rows": training_rows,
        "runtime_rows": runtime_rows,
        "quality_vs_k_and_macs": quality_vs_compute,
        "compute_match_rule": (
            "largest locked K satisfying AIR total MACs <= 1.05 * "
            "seed-matched J1-receding@K128 at both size 21 and size 25"
        ),
        "compute_match_k_by_size": compute_match_by_size,
        "compute_match_joint_k": compute_match_k,
        "compute_matched_descriptive_rows": compute_matched_rows,
        "compute_matched_inference": "descriptive_secondary_no_additional_CI",
    }


def endpoint(payload: dict[str, Any], partition: str, metric: str = "sr") -> float:
    return float(payload["navigation"][partition][metric])


def _average_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    cursor = 0
    while cursor < len(array):
        end = cursor + 1
        while end < len(array) and array[order[end]] == array[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    return ranks


def _spearman(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman inputs must have equal nontrivial length")
    left = _average_ranks(first)
    right = _average_ranks(second)
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _task_sr(
    payload: dict[str, Any],
    predicate: Callable[[dict[str, Any]], bool],
) -> float:
    selected = [float(row["success"]) for row in payload["task_rows"] if predicate(row)]
    if not selected:
        raise ValueError("K-scaling task stratum is empty")
    return float(np.mean(selected))


def k_scaling_descriptives(
    loaded: dict[tuple[str, int, int], dict[str, Any]],
    config: Any,
) -> list[dict[str, Any]]:
    rows = []
    log_k = [math.log2(k) for k in config.evaluation.k_values]
    for method in ("j1_receding", "air0_direct", "air0_jepa"):
        for seed in config.seeds:
            payloads = [loaded[(method, seed, k)] for k in config.evaluation.k_values]
            overall = [endpoint(payload, "overall") for payload in payloads]
            ood = [endpoint(payload, "ood") for payload in payloads]
            k16 = loaded[(method, seed, 16)]
            k128 = loaded[(method, seed, 128)]
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "spearman_log2k_overall_sr": _spearman(log_k, overall),
                    "spearman_log2k_ood_sr": _spearman(log_k, ood),
                    "k128_minus_k16_overall_sr": endpoint(k128, "overall")
                    - endpoint(k16, "overall"),
                    "k128_minus_k16_ood_sr": endpoint(k128, "ood")
                    - endpoint(k16, "ood"),
                    "k128_minus_k16_path33_64_sr": _task_sr(
                        k128, lambda row: 33 <= int(row["optimal_length"]) <= 64
                    )
                    - _task_sr(k16, lambda row: 33 <= int(row["optimal_length"]) <= 64),
                    "k128_minus_k16_path65_128_sr": _task_sr(
                        k128, lambda row: 65 <= int(row["optimal_length"]) <= 128
                    )
                    - _task_sr(
                        k16, lambda row: 65 <= int(row["optimal_length"]) <= 128
                    ),
                }
            )
    return rows


def seed_table(
    loaded: dict[tuple[str, int, int], dict[str, Any]],
    *,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    k: int,
) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        for seed in seeds:
            result = loaded[(method, seed, k)]
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "k": k,
                    "overall_sr": endpoint(result, "overall"),
                    "seen_sr": endpoint(result, "seen"),
                    "ood_sr": endpoint(result, "ood"),
                    "overall_spl": endpoint(result, "overall", "spl"),
                    "invalid_rate": endpoint(result, "overall", "invalid_rate"),
                    "loop_or_cycle_rate": endpoint(
                        result, "overall", "loop_or_cycle_rate"
                    ),
                }
            )
    return rows


def _matrix(
    payloads: list[dict[str, Any]],
    *,
    predicate: Callable[[dict[str, Any]], bool],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    reference = {
        str(row["task_id"]): row for row in payloads[0]["task_rows"] if predicate(row)
    }
    task_ids = sorted(reference)
    if not task_ids:
        raise ValueError("paired endpoint selected zero tasks")
    values = []
    for payload in payloads:
        indexed = {str(row["task_id"]): row for row in payload["task_rows"]}
        if not set(task_ids) <= set(indexed):
            raise ValueError("paired bootstrap task rows are not aligned")
        values.append([float(indexed[task_id]["success"]) for task_id in task_ids])
    return np.asarray(values, dtype=np.float64), [reference[key] for key in task_ids]


def crossed_bootstrap_difference(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    predicate: Callable[[dict[str, Any]], bool],
    samples: int,
    seed: int,
    family_size: int,
    alpha: float,
) -> dict[str, Any]:
    first_values, task_rows = _matrix(first, predicate=predicate)
    second_values, second_rows = _matrix(second, predicate=predicate)
    if [row["task_id"] for row in task_rows] != [row["task_id"] for row in second_rows]:
        raise ValueError("paired methods use different ordered task identities")
    difference = first_values - second_values
    by_size: defaultdict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(task_rows):
        by_size[int(row["maze_size"])].append(index)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    chunk = 250
    cursor = 0
    while cursor < samples:
        count = min(chunk, samples - cursor)
        seed_draws = rng.integers(
            0,
            difference.shape[0],
            size=(count, difference.shape[0]),
        )
        weighted = np.zeros(count, dtype=np.float64)
        total_tasks = 0
        for size in sorted(by_size):
            indices = np.asarray(by_size[size], dtype=np.int64)
            task_draws = indices[
                rng.integers(0, len(indices), size=(count, len(indices)))
            ]
            selected = difference[seed_draws[:, :, None], task_draws[:, None, :]]
            weighted += selected.mean(axis=(1, 2)) * len(indices)
            total_tasks += len(indices)
        draws[cursor : cursor + count] = weighted / float(total_tasks)
        cursor += count
    tail = alpha / (2.0 * family_size)
    return {
        "estimate": float(difference.mean()),
        "simultaneous_ci": [
            float(np.quantile(draws, tail)),
            float(np.quantile(draws, 1.0 - tail)),
        ],
        "samples": samples,
        "seed": seed,
        "family_size": family_size,
        "alpha": alpha,
        "task_count": len(task_rows),
        "seed_count": difference.shape[0],
    }


def diagnostic_summary(
    diagnostics: list[dict[str, Any]], config: Any
) -> dict[str, Any]:
    summaries = [payload["summary"] for payload in diagnostics]

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for summary in summaries:
            current: Any = summary
            for key in path:
                current = current[key]
            values.append(float(current))
        return float(np.mean(values))

    predicted_variance = mean(("future", "predicted_variance"))
    target_variance = mean(("future", "target_variance"))
    predicted_pairwise = mean(("future", "predicted_candidate_pairwise"))
    target_pairwise = mean(("future", "target_candidate_pairwise"))
    delta = mean(("future", "normalized_delta_error"))
    copy_delta = mean(("future", "copy_delta_normalized"))
    predicted_top1 = mean(("predicted", "local_top1"))
    permuted_top1 = mean(("permuted", "local_top1"))
    copy_relative_values = []
    copy_excluded = 0
    for payload in diagnostics:
        for row in payload["state_rows"]:
            copy_error = float(row["copy_delta_normalized"])
            if copy_error <= config.training.target_variance_epsilon:
                copy_excluded += 1
                continue
            model_error = float(row["normalized_delta_error"])
            copy_relative_values.append((copy_error - model_error) / copy_error)
    if not copy_relative_values:
        raise ValueError("future copy-relative diagnostic has zero eligible states")
    variance_ratio = predicted_variance / max(target_variance, 1e-12)
    candidate_ratio = predicted_pairwise / max(target_pairwise, 1e-12)
    permutation_effect = abs(predicted_top1 - permuted_top1)
    distance_fields = (
        "expected_mae",
        "expected_rmse",
        "expected_spearman",
        "categorical_accuracy",
        "top_class_ece_15",
    )

    def aggregate_distance(mode: str) -> dict[str, float | None]:
        output: dict[str, float | None] = {}
        for field in distance_fields:
            values = [
                payload["summary"]["distance"][mode][field]
                for payload in diagnostics
                if payload["summary"]["distance"][mode][field] is not None
            ]
            output[field] = float(np.mean(values)) if values else None
        return output

    collapse_reasons = []
    if variance_ratio < config.gates.collapse_variance_ratio:
        collapse_reasons.append("prediction_variance")
    if candidate_ratio < config.gates.collapse_candidate_ratio:
        collapse_reasons.append("candidate_diversity")
    if (
        delta >= copy_delta
        and permutation_effect < config.gates.collapse_permutation_effect
    ):
        collapse_reasons.append("copy_not_beaten_and_permutation_insensitive")
    return {
        "predicted_variance_ratio": variance_ratio,
        "candidate_pairwise_ratio": candidate_ratio,
        "normalized_delta_error": delta,
        "copy_delta_normalized": copy_delta,
        "copy_relative_improvement": float(np.mean(copy_relative_values)),
        "copy_relative_eligible_states": len(copy_relative_values),
        "copy_relative_excluded_states": copy_excluded,
        "predicted_local_top1": predicted_top1,
        "true_future_local_top1": mean(("true_future", "local_top1")),
        "permuted_local_top1": permuted_top1,
        "permutation_local_top1_drop": predicted_top1 - permuted_top1,
        "predicted_true_choice_agreement": mean(("predicted_true_choice_agreement",)),
        "prediction_flip_rate": mean(("prediction_flip_rate",)),
        "energy_wrong_with_true_future_rate": mean(
            ("energy_wrong_with_true_future_rate",)
        ),
        "distance": {
            "max_distance": DISTANCE_MAX,
            "ece_bins": DISTANCE_ECE_BINS,
            "target_clipped_rate": float(
                np.mean(
                    [
                        payload["summary"]["distance"]["target_clipped_rate"]
                        for payload in diagnostics
                    ]
                )
            ),
            "predicted": aggregate_distance("predicted"),
            "true_future": aggregate_distance("true_future"),
            "per_seed": [
                {
                    "seed": int(payload["metadata"]["seed"]),
                    **payload["summary"]["distance"],
                }
                for payload in diagnostics
            ],
        },
        "collapse": bool(collapse_reasons),
        "collapse_reasons": collapse_reasons,
    }


def load_primary(
    loader: ArtifactLoader,
    config: Any,
    *,
    role: str,
    seeds: tuple[int, ...],
    k_values: tuple[int, ...],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    loaded = {}
    for method in ("j1_receding", "air0_direct", "air0_jepa"):
        for seed in seeds:
            for k in k_values:
                path = result_path(config, role=role, method=method, seed=seed, k=k)
                loaded[(method, seed, k)] = loader.evaluation(
                    path,
                    role=role,
                    method=method,
                    seed=seed,
                    k=k,
                )
    return loaded


def l1_release(loader: ArtifactLoader, config: Any) -> dict[str, Any]:
    loaded = load_primary(
        loader,
        config,
        role="air_early",
        seeds=(42,),
        k_values=(16, 128),
    )
    interventions = {
        intervention: loader.evaluation(
            intervention_path(config, seed=42, k=128, intervention=intervention),
            role="air_early",
            method="air0_jepa",
            seed=42,
            k=128,
            intervention=intervention,
        )
        for intervention in (
            "copy_current",
            "true_future",
            "future_permutation",
            "future_zero",
        )
    }
    diagnostic = loader.diagnostic(
        diagnostic_path(config, seed=42, role="air_early"),
        seed=42,
        role="air_early",
    )
    mechanism = diagnostic_summary([diagnostic], config)
    jepa = endpoint(loaded[("air0_jepa", 42, 128)], "overall")
    j1 = endpoint(loaded[("j1_receding", 42, 128)], "overall")
    scaling = jepa - endpoint(loaded[("air0_jepa", 42, 16)], "overall")
    if (
        jepa >= config.gates.early_green_sr
        and jepa - j1 >= config.gates.early_green_j1_delta
        and scaling >= config.gates.early_green_k_delta
        and not mechanism["collapse"]
    ):
        signal = "early_green"
    elif (
        jepa < config.gates.early_red_sr
        or mechanism["collapse"]
        or (scaling <= 0.0 and mechanism["permutation_local_top1_drop"] <= 0.0)
    ):
        signal = "early_red"
    else:
        signal = "early_yellow"
    return {
        "evidence_role": "EARLY_SIGNAL",
        "seed_table": seed_table(
            loaded,
            methods=("j1_receding", "air0_direct", "air0_jepa"),
            seeds=(42,),
            k=128,
        ),
        "k128_minus_k16": scaling,
        "air0_jepa_minus_j1": jepa - j1,
        "mechanism": mechanism,
        "intervention_sr": {
            key: endpoint(value, "overall") for key, value in interventions.items()
        },
        "signal": signal,
        "binding_effect": "none; the locked DAG continues automatically",
    }


def l2_release(loader: ArtifactLoader, config: Any) -> dict[str, Any]:
    loaded = load_primary(
        loader,
        config,
        role="air_dev",
        seeds=config.seeds,
        k_values=(128,),
    )
    diagnostics = [
        loader.diagnostic(
            diagnostic_path(config, seed=seed, role="air_dev"),
            seed=seed,
            role="air_dev",
        )
        for seed in config.seeds
    ]
    family_size = 4
    comparisons = {
        "air0_jepa_minus_j1_overall": crossed_bootstrap_difference(
            [loaded[("air0_jepa", seed, 128)] for seed in config.seeds],
            [loaded[("j1_receding", seed, 128)] for seed in config.seeds],
            predicate=lambda row: True,
            samples=config.evaluation.bootstrap_samples,
            seed=config.evaluation.bootstrap_seed,
            family_size=family_size,
            alpha=config.statistics.familywise_alpha,
        ),
        "air0_jepa_minus_direct_overall": crossed_bootstrap_difference(
            [loaded[("air0_jepa", seed, 128)] for seed in config.seeds],
            [loaded[("air0_direct", seed, 128)] for seed in config.seeds],
            predicate=lambda row: True,
            samples=config.evaluation.bootstrap_samples,
            seed=config.evaluation.bootstrap_seed,
            family_size=family_size,
            alpha=config.statistics.familywise_alpha,
        ),
    }
    return {
        "evidence_role": "PRIMARY_PROVISIONAL",
        "seed_table": seed_table(
            loaded,
            methods=("j1_receding", "air0_direct", "air0_jepa"),
            seeds=config.seeds,
            k=128,
        ),
        "paired_comparisons": comparisons,
        "mechanism": diagnostic_summary(diagnostics, config),
        "breakdowns": aggregate_breakdowns(loaded, config),
        "decision": "PROVISIONAL_ONLY",
        "binding_effect": "none; L3 remains mandatory",
    }


def l3_release(loader: ArtifactLoader, config: Any) -> dict[str, Any]:
    loaded = load_primary(
        loader,
        config,
        role="air_dev",
        seeds=config.seeds,
        k_values=config.evaluation.k_values,
    )
    oracle = loader.evaluation(
        resolve_path(config.paths.run_root)
        / "results/air_dev/oracle_bfs/seed0_unmasked_k0.json",
        role="air_dev",
        method="oracle_bfs",
        seed=0,
        k=0,
    )
    static = {}
    for seed in config.seeds:
        for method, k in (("j0_static", 4), ("j1_static", 128)):
            static[(method, seed)] = loader.evaluation(
                result_path(
                    config,
                    role="air_dev",
                    method=method,
                    seed=seed,
                    k=k,
                ),
                role="air_dev",
                method=method,
                seed=seed,
                k=k,
            )
    corrected = {}
    for seed in config.seeds:
        for method in (
            "j0_static",
            "j1_static",
            "j1_receding",
            "air0_direct",
            "air0_jepa",
        ):
            k = 4 if method == "j0_static" else 128
            corrected[(method, seed)] = loader.evaluation(
                result_path(
                    config,
                    role="air_dev",
                    method=method,
                    seed=seed,
                    k=k,
                    protocol="corrected",
                ),
                role="air_dev",
                method=method,
                seed=seed,
                k=k,
                protocol="corrected",
            )
    interventions = {}
    for seed in config.seeds:
        for k in (16, 128):
            for intervention in (
                "normal",
                "copy_current",
                "true_future",
                "future_permutation",
                "future_zero",
            ):
                interventions[(seed, k, intervention)] = loader.evaluation(
                    intervention_path(
                        config, seed=seed, k=k, intervention=intervention
                    ),
                    role="air_early",
                    method="air0_jepa",
                    seed=seed,
                    k=k,
                    intervention=intervention,
                )
    diagnostics = [
        loader.diagnostic(
            diagnostic_path(config, seed=seed, role="air_dev"),
            seed=seed,
            role="air_dev",
        )
        for seed in config.seeds
    ]
    family_size = 4
    common = {
        "samples": config.evaluation.bootstrap_samples,
        "seed": config.evaluation.bootstrap_seed,
        "family_size": family_size,
        "alpha": config.statistics.familywise_alpha,
    }
    comparisons = {
        "air0_jepa_minus_j1_overall": crossed_bootstrap_difference(
            [loaded[("air0_jepa", seed, 128)] for seed in config.seeds],
            [loaded[("j1_receding", seed, 128)] for seed in config.seeds],
            predicate=lambda row: True,
            **common,
        ),
        "air0_jepa_minus_direct_overall": crossed_bootstrap_difference(
            [loaded[("air0_jepa", seed, 128)] for seed in config.seeds],
            [loaded[("air0_direct", seed, 128)] for seed in config.seeds],
            predicate=lambda row: True,
            **common,
        ),
        "air0_jepa_k128_minus_k16_ood": crossed_bootstrap_difference(
            [loaded[("air0_jepa", seed, 128)] for seed in config.seeds],
            [loaded[("air0_jepa", seed, 16)] for seed in config.seeds],
            predicate=lambda row: (
                int(row["maze_size"]) > config.evaluation.seen_max_size
            ),
            **common,
        ),
        "air0_jepa_k128_minus_k16_path33_128": crossed_bootstrap_difference(
            [loaded[("air0_jepa", seed, 128)] for seed in config.seeds],
            [loaded[("air0_jepa", seed, 16)] for seed in config.seeds],
            predicate=lambda row: 33 <= int(row["optimal_length"]) <= 128,
            **common,
        ),
    }
    table = seed_table(
        loaded,
        methods=("j1_receding", "air0_direct", "air0_jepa"),
        seeds=config.seeds,
        k=128,
    )
    jepa_rows = [row for row in table if row["method"] == "air0_jepa"]
    overall_mean = float(np.mean([row["overall_sr"] for row in jepa_rows]))
    ood_mean = float(np.mean([row["ood_sr"] for row in jepa_rows]))
    mechanism = diagnostic_summary(diagnostics, config)
    permutation_sr_drop = float(
        np.mean(
            [
                endpoint(interventions[(seed, 128, "normal")], "overall")
                - endpoint(interventions[(seed, 128, "future_permutation")], "overall")
                for seed in config.seeds
            ]
        )
    )
    green_checks = {
        "overall_mean": overall_mean >= config.gates.green_overall_mean,
        "overall_each_seed": all(
            row["overall_sr"] >= config.gates.green_overall_each_seed
            for row in jepa_rows
        ),
        "ood_mean": ood_mean >= config.gates.green_ood_mean,
        "ood_each_seed": all(
            row["ood_sr"] >= config.gates.green_ood_each_seed for row in jepa_rows
        ),
        "j1_noninferiority": comparisons["air0_jepa_minus_j1_overall"][
            "simultaneous_ci"
        ][0]
        > config.gates.j1_noninferiority_margin,
        "direct_noninferiority": comparisons["air0_jepa_minus_direct_overall"][
            "simultaneous_ci"
        ][0]
        > config.gates.direct_noninferiority_margin,
        "k_scaling": max(
            comparisons["air0_jepa_k128_minus_k16_ood"]["estimate"],
            comparisons["air0_jepa_k128_minus_k16_path33_128"]["estimate"],
        )
        >= config.gates.k_scaling_min_delta,
        "future_copy_improvement": mechanism["copy_relative_improvement"]
        >= config.gates.future_copy_improvement,
        "future_permutation_effect": (
            mechanism["permutation_local_top1_drop"]
            >= config.gates.permutation_local_top1_drop
            or permutation_sr_drop >= config.gates.permutation_sr_drop
        ),
        "no_collapse": not mechanism["collapse"],
    }
    if all(green_checks.values()):
        decision = "GREEN"
        reasons = ["all preregistered Green criteria passed"]
    else:
        clear_red = []
        if overall_mean < config.gates.yellow_overall_floor:
            clear_red.append("mean overall SR below the locked 0.80 floor")
        if mechanism["collapse"]:
            clear_red.append("future collapse criterion triggered")
        if (
            comparisons["air0_jepa_minus_direct_overall"]["simultaneous_ci"][1]
            < config.gates.direct_noninferiority_margin
        ):
            clear_red.append("AIR0-jepa is clearly worse than matched direct control")
        if (
            mechanism["true_future_local_top1"] <= mechanism["predicted_local_top1"]
            and mechanism["predicted_local_top1"] < 0.5
        ):
            clear_red.append("true future does not repair weak energy ranking")
        decision = "RED" if clear_red else "YELLOW"
        reasons = clear_red or [
            "at least one Green criterion failed without a locked clear-Red condition"
        ]
    curves = []
    for method in ("j1_receding", "air0_direct", "air0_jepa"):
        for seed in config.seeds:
            for k in config.evaluation.k_values:
                payload = loaded[(method, seed, k)]
                curves.append(
                    {
                        "method": method,
                        "seed": seed,
                        "k": k,
                        "overall_sr": endpoint(payload, "overall"),
                        "ood_sr": endpoint(payload, "ood"),
                    }
                )
    return {
        "evidence_role": "FINAL_CLOSURE",
        "seed_table": table,
        "oracle_ceiling": {
            "overall_sr": endpoint(oracle, "overall"),
            "seen_sr": endpoint(oracle, "seen"),
            "ood_sr": endpoint(oracle, "ood"),
            "overall_spl": endpoint(oracle, "overall", "spl"),
            "step_cap_ceiling": float(
                oracle["navigation"]["overall"]["step_cap_ceiling"]
            ),
            "evidence_role": "EVALUATOR_ORACLE",
        },
        "k_curves": curves,
        "k_scaling_descriptives": k_scaling_descriptives(loaded, config),
        "static_bridges": [
            {
                "method": method,
                "seed": seed,
                "overall_sr": endpoint(payload, "overall"),
                "ood_sr": endpoint(payload, "ood"),
            }
            for (method, seed), payload in sorted(static.items())
        ],
        "assistance_gap": [
            {
                "method": method,
                "seed": seed,
                "corrected_sr": endpoint(payload, "overall"),
                "unmasked_sr": endpoint(
                    (
                        static[(method, seed)]
                        if method in {"j0_static", "j1_static"}
                        else loaded[(method, seed, 128)]
                    ),
                    "overall",
                ),
            }
            for (method, seed), payload in sorted(corrected.items())
        ],
        "paired_comparisons": comparisons,
        "mechanism": mechanism,
        "intervention_sr": [
            {
                "seed": seed,
                "k": k,
                "intervention": intervention,
                "overall_sr": endpoint(payload, "overall"),
                "ood_sr": endpoint(payload, "ood"),
            }
            for (seed, k, intervention), payload in sorted(interventions.items())
        ],
        "permutation_sr_drop_early210": permutation_sr_drop,
        "breakdowns": aggregate_breakdowns(loaded, config),
        "green_checks": green_checks,
        "decision": decision,
        "decision_reasons": reasons,
        "sealed_roles_remain_unopened": ["air_select", "air_final"],
    }


def markdown_release(payload: dict[str, Any]) -> str:
    body = payload["results"]
    lines = [
        f"# AIR-JEPA Stage 0 {payload['level'].upper()} Release",
        "",
        f"- Evidence role: `{body['evidence_role']}`",
        f"- Decision/signal: `{body.get('decision', body.get('signal'))}`",
        f"- Release hash: `{payload['release_sha256']}`",
        "",
        "## Seed Table",
        "",
        "| Method | Seed | K | Overall SR | Seen SR | OOD SR | SPL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in body["seed_table"]:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['k']} | "
            f"{row['overall_sr']:.4f} | {row['seen_sr']:.4f} | "
            f"{row['ood_sr']:.4f} | {row['overall_spl']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This release is generated by the locked analysis code. Corrected and "
            "oracle-intervention results are diagnostics and are not absolute-ability "
            "scores. AIR_select and AIR_final remain sealed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    require_clean_worktree(allow_dirty=False)
    config = load_config(args.config)
    protocol = verify_protocol_lock(config)
    package = verify_package_lock(config)
    source = verify_source_lock(config)
    protocol_audit = require_protocol_audit(
        config,
        protocol_sha256=protocol["protocol_sha256"],
        package_sha256=package["package_sha256"],
        package_code_fingerprint=package["code_fingerprint"],
        source_lock_sha256=source["source_lock_sha256"],
    )
    bridge_audit = require_bridge_audit(
        config,
        protocol_sha256=protocol["protocol_sha256"],
        package_sha256=package["package_sha256"],
        source_lock_sha256=source["source_lock_sha256"],
    )
    benchmark = require_benchmark(
        config,
        protocol_sha256=protocol["protocol_sha256"],
        package_sha256=package["package_sha256"],
        package_code_fingerprint=package["code_fingerprint"],
        source_lock_sha256=source["source_lock_sha256"],
    )
    sealed_paths = require_sealed_roles_absent(config)
    checkpoint_seeds = (42,) if args.level == "l1" else config.seeds
    checkpoint_audit = paired_checkpoint_audit(
        config,
        seeds=checkpoint_seeds,
        source_lock=source,
        l0_pairing=protocol_audit["pairing"],
        protocol_sha256=protocol["protocol_sha256"],
        package_sha256=package["package_sha256"],
        package_code_fingerprint=package["code_fingerprint"],
        source_lock_sha256=source["source_lock_sha256"],
    )
    checkpoint_runtime = runtime_signature(checkpoint_audit["records"][0]["runtime"])
    if (
        runtime_signature(protocol_audit["runtime"]) != checkpoint_runtime
        or runtime_signature(benchmark["runtime"]) != checkpoint_runtime
        or runtime_signature(bridge_audit["runtime"]) != checkpoint_runtime
    ):
        raise ValueError(
            "L0 audit, bridge, benchmark, and AIR training runtimes are not identical"
        )
    expected_checkpoint_hashes: dict[tuple[str, int], str] = {}
    for seed in config.seeds:
        expected_checkpoint_hashes[("j0_static", seed)] = source["records"][str(seed)][
            "j0"
        ]["file_sha256"]
        for method in ("j1_static", "j1_receding"):
            expected_checkpoint_hashes[(method, seed)] = source["records"][str(seed)][
                "j1"
            ]["file_sha256"]
    for record in checkpoint_audit["records"]:
        expected_checkpoint_hashes[(record["method"], record["seed"])] = record[
            "sha256"
        ]
    loader = ArtifactLoader(
        config,
        protocol_sha256=protocol["protocol_sha256"],
        package_sha256=package["package_sha256"],
        package_code_fingerprint=package["code_fingerprint"],
        source_lock_sha256=source["source_lock_sha256"],
        expected_checkpoint_hashes=expected_checkpoint_hashes,
    )
    loader.loaded_hashes[protocol_audit["path"]] = protocol_audit["sha256"]
    loader.loaded_hashes[bridge_audit["path"]] = bridge_audit["sha256"]
    loader.loaded_hashes[benchmark["path"]] = benchmark["sha256"]
    for record in checkpoint_audit["records"]:
        loader.loaded_hashes[record["path"]] = record["sha256"]
    if args.level == "l1":
        results = l1_release(loader, config)
    elif args.level == "l2":
        results = l2_release(loader, config)
    else:
        results = l3_release(loader, config)
        primary = load_primary(
            loader,
            config,
            role="air_dev",
            seeds=config.seeds,
            k_values=config.evaluation.k_values,
        )
        results["compute_accounting"] = compute_accounting(
            config,
            checkpoint_audit=checkpoint_audit,
            loaded=primary,
            source_lock=source,
            locked_compute_match=benchmark["compute_match"],
        )
    if loader.runtime_signatures != {checkpoint_runtime}:
        raise ValueError(
            "AIR training and formal evaluation runtimes are not identical"
        )
    results["l0_protocol_hardware_audit"] = protocol_audit
    results["historical_bridge_audit"] = bridge_audit
    results["l0_performance_blind_benchmark"] = benchmark
    results["paired_checkpoint_audit"] = checkpoint_audit
    results["sealed_result_paths_absent"] = sealed_paths
    payload = signed_payload(
        {
            "schema": "air-jepa-stage0-release-v1",
            "experiment_id": config.experiment_id,
            "level": args.level,
            "protocol_sha256": protocol["protocol_sha256"],
            "package_sha256": package["package_sha256"],
            "source_lock_sha256": source["source_lock_sha256"],
            "analysis": config.statistics.model_dump(mode="json"),
            "artifact_hashes": loader.loaded_hashes,
            "results": results,
        },
        "release_sha256",
    )
    verify_signature(payload, "release_sha256")
    output = resolve_path(
        args.output or config.paths.release_template.format(level=args.level)
    )
    markdown = output.with_suffix(".md")
    prepare_new_output(output)
    prepare_new_output(markdown)
    atomic_json_dump(output, payload)
    atomic_text_dump(markdown, markdown_release(payload))
    print(
        f"saved={relative_path(output)} role={results['evidence_role']} "
        f"decision={results.get('decision', results.get('signal'))}"
    )


if __name__ == "__main__":
    main()
