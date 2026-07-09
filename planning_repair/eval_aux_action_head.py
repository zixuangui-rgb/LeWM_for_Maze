#!/usr/bin/env python3
"""Evaluate the embedding-level action head as a model-free local planner."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.common import next_state
from hdwm.planning import _bfs_shortest_path
from planning_repair.common import (
    ACTION_TO_SLOT,
    SLOT_TO_ACTION,
    create_env,
    encode_single_observation,
    grouped_limit,
    json_dump,
    load_backbone_from_repair_ckpt,
    observe_state,
    read_jsonl,
    set_agent_state,
    set_seed,
    summarize_navigation,
    valid_moving_actions,
)
from planning_repair.heads import load_aux_heads
from scripts.eval.eval_setb_distance_head_fixed import MAX_STEPS, manifest_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate aux action head navigation.")
    parser.add_argument("--manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument(
        "--output",
        default="planning_repair_runs/aux_action_head/results.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-size", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def choose_action(
    model: torch.nn.Module,
    aux_heads: torch.nn.Module,
    env: Any,
    state: int,
    previous: int | None,
    size: int,
    device: torch.device,
) -> int:
    obs = observe_state(env, state)
    embedding = encode_single_observation(model, obs, size, device)
    with torch.no_grad():
        logits = aux_heads(embedding)["action_logits"].squeeze(0).squeeze(0)
    valid_actions = valid_moving_actions(env, state, previous)
    if not valid_actions:
        return int(SLOT_TO_ACTION[0])
    mask = torch.full_like(logits, float("-inf"))
    for action in valid_actions:
        mask[ACTION_TO_SLOT[int(action)]] = 0.0
    slot = int((logits + mask).argmax().item())
    return int(SLOT_TO_ACTION[slot])


def run_episode(
    model: torch.nn.Module,
    aux_heads: torch.nn.Module,
    entry: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    env = create_env(entry)
    start, goal, opt = manifest_task(entry, env)
    size = int(entry["maze_size"])
    set_agent_state(env, start)
    cur = start
    previous: int | None = None
    path_length = 0
    stuck_steps = 0
    invalid_actions = 0
    metric_wrong_steps = 0
    last = cur

    for _ in range(MAX_STEPS):
        if cur == goal:
            break
        action = choose_action(model, aux_heads, env, cur, previous, size, device)
        nxt = next_state(env, cur, action)
        best_next_dist = None
        action_next_dist = _bfs_shortest_path(env._maze_mask, nxt, goal, env.config.width)
        for valid_action in valid_moving_actions(env, cur, previous):
            cand = next_state(env, cur, valid_action)
            dist = _bfs_shortest_path(env._maze_mask, cand, goal, env.config.width)
            if dist is not None:
                best_next_dist = dist if best_next_dist is None else min(best_next_dist, dist)
        if (
            best_next_dist is not None
            and action_next_dist is not None
            and int(action_next_dist) > int(best_next_dist)
        ):
            metric_wrong_steps += 1

        prev = cur
        _, _, _, _, info = env.step(action)
        cur = int(info["state"])
        previous = prev
        path_length += 1
        if cur == prev and action != 0:
            invalid_actions += 1
        if cur == last:
            stuck_steps += 1
        last = cur

    final_bfs = _bfs_shortest_path(env._maze_mask, cur, goal, env.config.width)
    success = cur == goal
    return {
        "success": success,
        "path_length": path_length,
        "stuck_steps": stuck_steps,
        "invalid_actions": invalid_actions,
        "metric_wrong_steps": metric_wrong_steps,
        "final_bfs_distance": 0 if success else (final_bfs if final_bfs is not None else -1),
        "op_len": opt,
        "maze_size": size,
        "spl": opt / max(path_length, opt) if success else 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    model, data = load_backbone_from_repair_ckpt(args.model_ckpt, device)
    aux_heads = load_aux_heads(data, device)
    if aux_heads is None:
        raise ValueError("checkpoint does not contain aux heads")
    entries = grouped_limit(
        read_jsonl(args.manifest),
        max_per_size=args.max_per_size,
        limit=args.limit,
    )
    print("=" * 80)
    print("EVALUATE AUX ACTION HEAD")
    print("=" * 80)
    print(f"entries={len(entries)} model={args.model_ckpt} device={device}")

    rows: list[dict[str, Any]] = []
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    t0 = time.time()
    for idx, entry in enumerate(entries):
        row = run_episode(model, aux_heads, entry, device)
        rows.append(row)
        by_size[int(row["maze_size"])].append(row)
        if (idx + 1) % args.progress_every == 0:
            print(
                f"  {idx + 1:>4d}/{len(entries)} "
                f"SR={summarize_navigation(rows)['sr']:.4f}",
                flush=True,
            )

    summary = summarize_navigation(rows)
    summary["time"] = float(time.time() - t0)
    summary["avg_metric_wrong_steps"] = float(
        sum(row["metric_wrong_steps"] for row in rows) / max(len(rows), 1)
    )
    output = {
        "metadata": {
            "manifest": args.manifest,
            "model_ckpt": args.model_ckpt,
            "limit": args.limit,
            "max_per_size": args.max_per_size,
            "seed": args.seed,
        },
        "summary": summary,
        "by_size": {
            str(size): summarize_navigation(size_rows)
            for size, size_rows in sorted(by_size.items())
        },
        "rows": rows,
    }
    json_dump(args.output, output)
    print(
        f"SR={summary['sr']:.4f} SPL={summary['spl']:.4f} "
        f"stuck={summary['stuck_rate']:.4f} invalid={summary['invalid_rate']:.4f}"
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
