#!/usr/bin/env python3
"""Evaluate AIR and source bridges under the locked Stage-0 protocols."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from air_jepa.stage0_workspace import AIR_METHODS, ALL_METHODS
from air_jepa.stage0_workspace.checkpoints import (
    load_air_checkpoint,
    load_frozen_representation,
    load_source_planner,
    verify_source_lock,
)
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    code_fingerprint,
    format_template,
    git_commit,
    git_worktree_dirty,
    load_config,
    prepare_new_output,
    relative_path,
    require_clean_worktree,
    require_h800_device,
    resolve_device,
    resolve_path,
    runtime_metadata,
    set_seed,
    sha256_file,
)
from air_jepa.stage0_workspace.models import AIRWorkspaceModel, require_finite_output
from air_jepa.stage0_workspace.protocol import (
    expected_matrix,
    require_role_allowed,
    verify_package_lock,
    verify_protocol_lock,
)
from diagnostics.common import (
    ACTION_IDS,
    ACTION_TO_SLOT,
    bfs_distances_from,
    next_state,
    observe_state,
)
from spatial_jepa_planning.common import (
    read_jsonl,
    summarize_rows,
    validate_manifest_entry,
)
from spatial_jepa_planning.evaluate import run_navigation
from spatial_jepa_planning.models import PlannerOutput, SpatialRepresentation

INTERVENTIONS = (
    "normal",
    "copy_current",
    "true_future",
    "future_permutation",
    "future_zero",
)
FUTURE_PERMUTATION = (1, 0, 3, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--method", choices=ALL_METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--split-role",
        choices=(
            "preflight",
            "air_early",
            "air_dev",
            "historical",
            "air_select",
            "air_final",
        ),
        required=True,
    )
    parser.add_argument(
        "--action-protocol", choices=("unmasked", "corrected"), default="unmasked"
    )
    parser.add_argument("--intervention", choices=INTERVENTIONS, default="normal")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--smoke-limit", type=int, default=0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


def manifest_for_role(config: Any, role: str) -> Path:
    require_role_allowed(role)
    mapping = {
        "preflight": config.paths.preflight_manifest,
        "air_early": config.paths.air_early_manifest,
        "air_dev": config.paths.air_dev_manifest,
        "historical": config.paths.historical_confirmatory_manifest,
    }
    return resolve_path(mapping[role])


def corrected_actions(env: Any, state: int, previous: int | None) -> list[int]:
    moving: list[int] = []
    non_backtracking: list[int] = []
    for action in ACTION_IDS:
        candidate = next_state(env, state, action)
        if candidate == state:
            continue
        moving.append(int(action))
        if previous is None or candidate != previous:
            non_backtracking.append(int(action))
    return non_backtracking or moving or [int(action) for action in ACTION_IDS]


def candidate_actions(
    env: Any,
    state: int,
    previous: int | None,
    action_protocol: str,
) -> list[int]:
    if action_protocol == "unmasked":
        return [int(action) for action in ACTION_IDS]
    if action_protocol == "corrected":
        return corrected_actions(env, state, previous)
    raise ValueError(f"unsupported action protocol: {action_protocol}")


def classify_failure(row: dict[str, Any]) -> str:
    if bool(row["success"]):
        return "success"
    if bool(row["loop_or_cycle"]):
        return "loop_or_cycle"
    if int(row["invalid_actions"]) >= max(4, int(row["path_length"]) // 2):
        return "invalid_action_stall"
    return "step_cap_or_unresolved"


def run_navigation_with_diagnostics(
    entry: dict[str, Any],
    *,
    action_fn: Callable[
        [Any, np.ndarray, int, int | None], tuple[int, dict[str, float]]
    ],
    max_steps: int,
) -> dict[str, Any]:
    """Run the historical evaluator while adding auditable movement counters."""

    counters = {
        "immediate_backtracks": 0,
        "distance_decrease_actions": 0,
        "distance_flat_actions": 0,
        "distance_increase_actions": 0,
        "dead_end_recovery_opportunities": 0,
        "dead_end_recovery_successes": 0,
        "dead_end_recovery_failures": 0,
    }
    cached_env: Any | None = None
    cached_distances: np.ndarray | None = None

    def traced_action_fn(
        env: Any,
        observation: np.ndarray,
        state: int,
        previous: int | None,
    ) -> tuple[int, dict[str, float]]:
        nonlocal cached_distances, cached_env
        action, metrics = action_fn(env, observation, state, previous)
        candidate = next_state(env, int(state), int(action))
        if env is not cached_env:
            cached_distances = bfs_distances_from(
                env._maze_mask,
                int(env._goal_position),
                int(env.config.width),
            )
            cached_env = env
        if cached_distances is None:
            raise RuntimeError(
                "navigation diagnostic distance cache was not initialized"
            )
        current_distance = int(cached_distances[int(state)])
        candidate_distance = int(cached_distances[int(candidate)])
        if candidate_distance < current_distance:
            counters["distance_decrease_actions"] += 1
        elif candidate_distance > current_distance:
            counters["distance_increase_actions"] += 1
        else:
            counters["distance_flat_actions"] += 1
        if (
            previous is not None
            and int(previous) != int(state)
            and int(candidate) == int(previous)
        ):
            counters["immediate_backtracks"] += 1
        moving_successors = {
            next_state(env, int(state), int(candidate_action))
            for candidate_action in ACTION_IDS
            if next_state(env, int(state), int(candidate_action)) != int(state)
        }
        if int(state) != int(env._goal_position) and len(moving_successors) == 1:
            counters["dead_end_recovery_opportunities"] += 1
            if int(candidate) == int(state):
                counters["dead_end_recovery_failures"] += 1
            else:
                counters["dead_end_recovery_successes"] += 1
        return int(action), metrics

    row = run_navigation(entry, action_fn=traced_action_fn, max_steps=max_steps)
    row.update(counters)
    if counters["distance_decrease_actions"] + counters[
        "distance_flat_actions"
    ] + counters["distance_increase_actions"] != int(row["path_length"]):
        raise RuntimeError("movement diagnostics did not account for every action")
    if counters["dead_end_recovery_opportunities"] != (
        counters["dead_end_recovery_successes"] + counters["dead_end_recovery_failures"]
    ):
        raise RuntimeError("dead-end recovery accounting is inconsistent")
    return row


def choose_source_action(
    output: PlannerOutput,
    *,
    env: Any,
    state: int,
    previous: int | None,
    action_protocol: str,
) -> int:
    row, col = divmod(int(state), int(env.config.width))
    candidates = candidate_actions(env, state, previous, action_protocol)
    return max(
        candidates,
        key=lambda action: float(
            output.policy_logits[0, ACTION_TO_SLOT[action], row, col]
        ),
    )


def _spatial_mask(latent: torch.Tensor) -> torch.Tensor:
    return torch.ones(
        (latent.shape[0], latent.shape[-2], latent.shape[-1]),
        dtype=torch.bool,
        device=latent.device,
    )


def _true_successor_latents(
    representation: SpatialRepresentation,
    *,
    env: Any,
    state: int,
    device: torch.device,
) -> torch.Tensor:
    observations = np.stack(
        [observe_state(env, next_state(env, state, action)) for action in ACTION_IDS]
    )
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    with torch.no_grad():
        return representation.planning_latent(tensor).unsqueeze(0)


def intervention_futures(
    intervention: str,
    *,
    predicted: torch.Tensor,
    source_latent: torch.Tensor,
    representation: SpatialRepresentation,
    env: Any,
    state: int,
    target_mean: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if intervention == "normal":
        return predicted
    if intervention == "copy_current":
        return source_latent[:, None].expand_as(predicted)
    if intervention == "true_future":
        return _true_successor_latents(
            representation, env=env, state=state, device=device
        )
    if intervention == "future_permutation":
        index = torch.as_tensor(FUTURE_PERMUTATION, device=device, dtype=torch.long)
        return predicted.index_select(1, index)
    if intervention == "future_zero":
        if target_mean.shape != (source_latent.shape[1],):
            raise ValueError("future target mean shape differs from latent channels")
        return target_mean[None, None, :, None, None].expand_as(predicted)
    raise ValueError(f"unsupported intervention: {intervention}")


def build_air_action_fn(
    model: AIRWorkspaceModel,
    representation: SpatialRepresentation,
    *,
    checkpoint: dict[str, Any],
    iterations: int,
    intervention: str,
    action_protocol: str,
    device: torch.device,
) -> Callable[[Any, np.ndarray, int, int | None], tuple[int, dict[str, float]]]:
    moments = checkpoint.get("future_target_channel_moments", {})
    target_mean = torch.as_tensor(
        moments.get("mean", []), dtype=torch.float32, device=device
    )

    def action_fn(
        env: Any,
        observation: np.ndarray,
        state: int,
        previous: int | None,
    ) -> tuple[int, dict[str, float]]:
        started = time.perf_counter()
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            source = representation.planning_latent(tensor)
            mask = _spatial_mask(source)
            output = model(source, iterations=iterations, valid_mask=mask)[-1]
            require_finite_output(output)
            predicted = output.predicted_future
            if predicted is None:
                raise RuntimeError("AIR final output omitted future fields")
            future = intervention_futures(
                intervention,
                predicted=predicted,
                source_latent=source,
                representation=representation,
                env=env,
                state=state,
                target_mean=target_mean,
                device=device,
            )
            if intervention == "normal":
                energy = output.energy
            else:
                _, energy = model.score_external_futures(output, future, mask)
        if not bool(torch.isfinite(energy).all()):
            raise FloatingPointError("AIR intervention produced non-finite energy")
        candidates = candidate_actions(env, state, previous, action_protocol)
        action = min(
            candidates,
            key=lambda value: float(energy[0, ACTION_TO_SLOT[value]]),
        )
        return int(action), {
            "inference_seconds": time.perf_counter() - started,
            "inference_calls": 1.0,
        }

    return action_fn


def build_source_action_fn(
    planner: torch.nn.Module,
    representation: SpatialRepresentation,
    *,
    static_output: PlannerOutput | None,
    iterations: int,
    action_protocol: str,
    device: torch.device,
) -> Callable[[Any, np.ndarray, int, int | None], tuple[int, dict[str, float]]]:
    def action_fn(
        env: Any,
        observation: np.ndarray,
        state: int,
        previous: int | None,
    ) -> tuple[int, dict[str, float]]:
        started = time.perf_counter()
        output = static_output
        calls = 0.0
        if output is None:
            tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                latent = representation.planning_latent(tensor)
                output = planner(latent, iterations=iterations)[-1]
            calls = 1.0
        if output is None:
            raise RuntimeError("source planner output is unavailable")
        for value in (output.policy_logits, output.value, output.valid_logits):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("source planner produced a non-finite output")
        action = choose_source_action(
            output,
            env=env,
            state=state,
            previous=previous,
            action_protocol=action_protocol,
        )
        return action, {
            "inference_seconds": time.perf_counter() - started,
            "inference_calls": calls,
        }

    return action_fn


def evaluate_entries(
    entries: list[dict[str, Any]],
    *,
    method: str,
    iterations: int,
    action_protocol: str,
    intervention: str,
    max_steps: int,
    device: torch.device,
    air_model: AIRWorkspaceModel | None = None,
    air_checkpoint: dict[str, Any] | None = None,
    source_planner: torch.nn.Module | None = None,
    representation: SpatialRepresentation,
    progress_every: int = 0,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, entry in enumerate(entries):
        task_started = time.perf_counter()
        static_output: PlannerOutput | None = None
        static_inference_seconds = 0.0
        if method in {"j0_static", "j1_static"}:
            if source_planner is None:
                raise ValueError("source static method requires a planner")
            env = validate_manifest_entry(entry, check_bfs=False)
            observation = observe_state(env, int(entry["start_cell"]))
            tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            static_started = time.perf_counter()
            with torch.no_grad():
                latent = representation.planning_latent(tensor)
                static_output = source_planner(latent, iterations=iterations)[-1]
            static_inference_seconds = time.perf_counter() - static_started
        if method in AIR_METHODS:
            if air_model is None or air_checkpoint is None:
                raise ValueError("AIR method requires its model and checkpoint")
            action_fn = build_air_action_fn(
                air_model,
                representation,
                checkpoint=air_checkpoint,
                iterations=iterations,
                intervention=intervention,
                action_protocol=action_protocol,
                device=device,
            )
        else:
            if intervention != "normal":
                raise ValueError(
                    "source bridge methods do not support future interventions"
                )
            if source_planner is None:
                raise ValueError("source method requires a planner")
            action_fn = build_source_action_fn(
                source_planner,
                representation,
                static_output=static_output,
                iterations=iterations,
                action_protocol=action_protocol,
                device=device,
            )
        row = run_navigation_with_diagnostics(
            entry,
            action_fn=action_fn,
            max_steps=max_steps,
        )
        row["elapsed_seconds"] = time.perf_counter() - task_started
        if static_output is not None:
            row["auxiliary"]["inference_seconds"] = (
                float(row["auxiliary"].get("inference_seconds", 0.0))
                + static_inference_seconds
            )
            row["auxiliary"]["inference_calls"] = (
                float(row["auxiliary"].get("inference_calls", 0.0)) + 1.0
            )
        row["failure_reason"] = classify_failure(row)
        rows.append(row)
        if progress_every > 0 and (index + 1) % progress_every == 0:
            sr = float(np.mean([float(item["success"]) for item in rows]))
            print(f"{method} K={iterations} {index + 1}/{len(entries)} SR={sr:.4f}")
    return rows, time.perf_counter() - started


def _validate_request(
    args: argparse.Namespace,
    config: Any,
    *,
    formal: bool,
) -> None:
    if args.seed not in config.seeds:
        raise ValueError("evaluation seed is outside the locked system seeds")
    if args.k not in config.evaluation.k_values:
        raise ValueError("K is outside the locked evaluation curve")
    if args.method == "j0_static" and args.k != 4:
        raise ValueError("j0_static must use its locked feedforward depth K=4")
    if args.method in {"j1_static", "j1_receding"} and args.intervention != "normal":
        raise ValueError("source planners cannot receive AIR future interventions")
    if args.method == "j1_static" and args.k != 128:
        raise ValueError("j1_static bridge is locked to K=128")
    if args.intervention != "normal":
        if args.method != "air0_jepa" or args.split_role != "air_early":
            raise ValueError(
                "future interventions are restricted to AIR0-jepa early210"
            )
        if args.k not in {16, 128} or args.action_protocol != "unmasked":
            raise ValueError("future interventions require K16/K128 unmasked")
    if args.split_role == "historical":
        if args.method not in {"j0_static", "j1_static"}:
            raise ValueError("historical split is only for static bridge parity")
        if args.action_protocol != "unmasked" or args.intervention != "normal":
            raise ValueError(
                "historical bridge parity requires normal unmasked semantics"
            )
    if not formal or args.split_role in {"air_select", "air_final"}:
        return
    matrix = expected_matrix(config)
    if args.split_role == "preflight":
        raise ValueError("formal preflight evaluation is not part of the locked matrix")
    if args.split_role == "historical":
        records = matrix["historical_bridges"]
        identity = {
            "method": args.method,
            "seed": args.seed,
            "k": args.k,
            "action_protocol": args.action_protocol,
        }
    elif args.split_role == "air_dev":
        key = (
            "air_dev_unmasked"
            if args.action_protocol == "unmasked"
            else "air_dev_corrected"
        )
        records = matrix[key]
        identity = {
            "method": args.method,
            "seed": args.seed,
            "k": args.k,
            "action_protocol": args.action_protocol,
        }
        if args.intervention != "normal":
            raise ValueError("AIR_dev formal cells do not permit future interventions")
    elif args.split_role == "air_early":
        key = (
            "air_early_interventions"
            if args.method == "air0_jepa"
            else "air_early_context"
        )
        records = matrix[key]
        identity = {
            "method": args.method,
            "seed": args.seed,
            "k": args.k,
            "action_protocol": args.action_protocol,
            "intervention": args.intervention,
        }
    else:
        raise ValueError(
            f"formal split role is not executable in AIR0: {args.split_role}"
        )
    if not any(
        all(record.get(field) == value for field, value in identity.items())
        for record in records
    ):
        raise ValueError(
            f"formal evaluation cell is absent from the protocol matrix: {identity}"
        )


def main() -> None:
    args = parse_args()
    if args.smoke_limit < 0:
        raise ValueError("smoke-limit cannot be negative")
    if args.smoke_limit > 0 and not args.output:
        raise ValueError("smoke evaluation requires an explicit non-formal --output")
    config = load_config(args.config)
    formal = args.smoke_limit == 0
    _validate_request(args, config, formal=formal)
    if args.output:
        output = resolve_path(args.output)
    elif args.intervention != "normal":
        output = resolve_path(config.paths.run_root) / (
            f"results/air_early/interventions/air0_jepa/seed{args.seed}_"
            f"unmasked_k{args.k}_{args.intervention}.json"
        )
    else:
        output = format_template(
            config.paths.result_template,
            split_role=args.split_role,
            method=args.method,
            seed=args.seed,
            action_protocol=args.action_protocol,
            k=args.k,
        )
    prepare_new_output(output)
    if formal and args.allow_dirty_worktree:
        raise ValueError("formal evaluation cannot allow a dirty worktree")
    require_clean_worktree(allow_dirty=not formal and args.allow_dirty_worktree)
    protocol_lock = verify_protocol_lock(config)
    package_lock = verify_package_lock(config)
    source_lock = verify_source_lock(config)
    manifest = manifest_for_role(config, args.split_role)
    entries = read_jsonl(manifest)
    if args.smoke_limit > 0:
        entries = entries[: args.smoke_limit]
    if not entries:
        raise ValueError("evaluation selected no tasks")
    device = resolve_device(args.device)
    if formal:
        require_h800_device(device)
    set_seed(args.seed + 101, deterministic=True)

    air_model: AIRWorkspaceModel | None = None
    air_checkpoint: dict[str, Any] | None = None
    source_planner: torch.nn.Module | None = None
    if args.method in AIR_METHODS:
        checkpoint_path = format_template(
            config.paths.air_checkpoint_template,
            method=args.method,
            seed=args.seed,
        )
        air_model, air_checkpoint = load_air_checkpoint(
            checkpoint_path,
            config=config,
            method=args.method,
            seed=args.seed,
            device=device,
            require_formal=True,
        )
        for key, expected in (
            ("protocol_sha256", protocol_lock["protocol_sha256"]),
            ("package_sha256", package_lock["package_sha256"]),
            ("source_lock_sha256", source_lock["source_lock_sha256"]),
        ):
            if air_checkpoint.get(key) != expected:
                raise ValueError(f"AIR checkpoint {key} differs from current lock")
        representation, _ = load_frozen_representation(
            config,
            seed=args.seed,
            device=device,
            source_lock=source_lock,
        )
        checkpoint_sha256 = sha256_file(checkpoint_path)
    else:
        source_planner, representation, _ = load_source_planner(
            config,
            seed=args.seed,
            method=args.method,
            device=device,
            source_lock=source_lock,
        )
        key = "j0" if args.method == "j0_static" else "j1"
        checkpoint_sha256 = source_lock["records"][str(args.seed)][key]["file_sha256"]

    rows, elapsed = evaluate_entries(
        entries,
        method=args.method,
        iterations=args.k,
        action_protocol=args.action_protocol,
        intervention=args.intervention,
        max_steps=config.evaluation.max_steps,
        device=device,
        air_model=air_model,
        air_checkpoint=air_checkpoint,
        source_planner=source_planner,
        representation=representation,
        progress_every=args.progress_every,
    )
    if len(rows) != len(entries):
        raise RuntimeError("evaluator dropped task rows")
    row_ids = [str(row["task_id"]) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise RuntimeError("evaluator produced duplicate task rows")
    payload = {
        "schema": "air-jepa-stage0-evaluation-v1",
        "metadata": {
            "experiment_id": config.experiment_id,
            "method": args.method,
            "seed": args.seed,
            "k": args.k,
            "split_role": args.split_role,
            "evidence_role": (
                "ORACLE_INTERVENTION"
                if args.intervention == "true_future"
                else (
                    "MECHANISM_DIAGNOSTIC"
                    if (
                        args.intervention != "normal"
                        or args.action_protocol == "corrected"
                    )
                    else (
                        "HISTORICAL_BRIDGE"
                        if args.split_role == "historical"
                        else (
                            "EARLY_SIGNAL"
                            if args.split_role == "air_early"
                            else "PRIMARY_PROVISIONAL"
                        )
                    )
                )
            ),
            "action_protocol": args.action_protocol,
            "intervention": args.intervention,
            "future_permutation": (
                list(FUTURE_PERMUTATION)
                if args.intervention == "future_permutation"
                else None
            ),
            "task_count": len(rows),
            "max_steps": config.evaluation.max_steps,
            "manifest": relative_path(manifest),
            "manifest_sha256": sha256_file(manifest),
            "checkpoint_sha256": checkpoint_sha256,
            "protocol_sha256": protocol_lock["protocol_sha256"],
            "package_sha256": package_lock["package_sha256"],
            "source_lock_sha256": source_lock["source_lock_sha256"],
            "git_commit": git_commit(),
            "git_dirty": git_worktree_dirty(),
            "code_fingerprint": code_fingerprint(),
            "runtime": runtime_metadata(device),
            "elapsed_seconds": elapsed,
            "formal": formal,
        },
        "navigation": summarize_rows(
            rows,
            seen_max_size=config.evaluation.seen_max_size,
            max_steps=config.evaluation.max_steps,
        ),
        "task_rows": rows,
    }
    atomic_json_dump(output, payload)
    print(
        f"saved={relative_path(output)} "
        f"SR={payload['navigation']['overall']['sr']:.4f} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
