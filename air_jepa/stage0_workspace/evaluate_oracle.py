#!/usr/bin/env python3
"""Evaluate the exact BFS ceiling on AIR_dev under max_steps=128."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from air_jepa.stage0_workspace.checkpoints import verify_source_lock
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    code_fingerprint,
    git_commit,
    git_worktree_dirty,
    load_config,
    prepare_new_output,
    relative_path,
    require_clean_worktree,
    resolve_path,
    runtime_metadata,
    set_seed,
    sha256_file,
)
from air_jepa.stage0_workspace.evaluate import (
    classify_failure,
    run_navigation_with_diagnostics,
)
from air_jepa.stage0_workspace.protocol import (
    verify_package_lock,
    verify_protocol_lock,
)
from diagnostics.common import ACTION_IDS, bfs_distances_from, next_state
from spatial_jepa_planning.common import (
    read_jsonl,
    summarize_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def build_oracle_action_fn() -> Any:
    cached_env: Any | None = None
    cached_distances: np.ndarray | None = None

    def action_fn(
        env: Any,
        _observation: np.ndarray,
        state: int,
        _previous: int | None,
    ) -> tuple[int, dict[str, float]]:
        nonlocal cached_distances, cached_env
        started = time.perf_counter()
        if env is not cached_env:
            cached_distances = bfs_distances_from(
                env._maze_mask,
                int(env._goal_position),
                int(env.config.width),
            )
            cached_env = env
        if cached_distances is None:
            raise RuntimeError("oracle BFS distance cache was not initialized")
        action = min(
            ACTION_IDS,
            key=lambda candidate: int(
                cached_distances[next_state(env, int(state), int(candidate))]
            ),
        )
        return int(action), {
            "inference_seconds": time.perf_counter() - started,
            "inference_calls": 1.0,
        }

    return action_fn


def main() -> None:
    args = parse_args()
    require_clean_worktree(allow_dirty=False)
    config = load_config(args.config)
    output = (
        resolve_path(args.output)
        if args.output
        else resolve_path(config.paths.run_root)
        / "results/air_dev/oracle_bfs/seed0_unmasked_k0.json"
    )
    prepare_new_output(output)
    protocol = verify_protocol_lock(config)
    package = verify_package_lock(config)
    source = verify_source_lock(config)
    set_seed(0, deterministic=True)
    manifest = resolve_path(config.paths.air_dev_manifest)
    entries = read_jsonl(manifest)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    action_fn = build_oracle_action_fn()
    for index, entry in enumerate(entries):
        task_started = time.perf_counter()
        row = run_navigation_with_diagnostics(
            entry,
            action_fn=action_fn,
            max_steps=config.evaluation.max_steps,
        )
        row["elapsed_seconds"] = time.perf_counter() - task_started
        row["failure_reason"] = classify_failure(row)
        rows.append(row)
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            sr = float(np.mean([float(value["success"]) for value in rows]))
            print(f"oracle_bfs {index + 1}/{len(entries)} SR={sr:.4f}")
    payload = {
        "schema": "air-jepa-stage0-evaluation-v1",
        "metadata": {
            "experiment_id": config.experiment_id,
            "method": "oracle_bfs",
            "seed": 0,
            "k": 0,
            "split_role": "air_dev",
            "evidence_role": "EVALUATOR_ORACLE",
            "action_protocol": "unmasked",
            "intervention": "normal",
            "task_count": len(rows),
            "max_steps": config.evaluation.max_steps,
            "manifest": relative_path(manifest),
            "manifest_sha256": sha256_file(manifest),
            "checkpoint_sha256": None,
            "protocol_sha256": protocol["protocol_sha256"],
            "package_sha256": package["package_sha256"],
            "source_lock_sha256": source["source_lock_sha256"],
            "git_commit": git_commit(),
            "git_dirty": git_worktree_dirty(),
            "code_fingerprint": code_fingerprint(),
            "runtime": runtime_metadata(),
            "elapsed_seconds": time.perf_counter() - started,
            "formal": True,
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
        f"saved={relative_path(output)} SR={payload['navigation']['overall']['sr']:.4f}"
    )


if __name__ == "__main__":
    main()
