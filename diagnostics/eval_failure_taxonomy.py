#!/usr/bin/env python3
"""Classify navigation failures into actionable categories.

This diagnostic is meant to turn a raw SR number into an engineering/research
signal. For each eval task it compares model-free local scoring with
predictor-greedy scoring, then tags likely failure causes.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.common import (
    ACTION_IDS,
    add_common_args,
    create_env,
    encode_observations,
    ensure_dir,
    load_distance_head,
    load_lewm,
    load_qrl_head,
    next_state,
    observe_state,
    read_jsonl,
    run_dir,
    select_entries,
    set_agent_state,
    size_bucket,
    verify_holdout,
    write_json,
)
from hdwm.planning import _bfs_shortest_path


HISTORY_SIZE = 3
MAX_STEPS = 128
ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate navigation failure taxonomy.")
    add_common_args(parser)
    parser.add_argument("--scorer", choices=["latent_l2", "distance_head", "qrl"], default="latent_l2")
    parser.add_argument("--distance-head-ckpt", default=None)
    parser.add_argument("--qrl-ckpt", default=None)
    parser.add_argument("--max-eval-per-size", type=int, default=100)
    parser.add_argument("--long-path-threshold", type=int, default=40)
    return parser.parse_args()


def build_scorer(args: argparse.Namespace, device: torch.device) -> ScoreFn:
    if args.scorer == "latent_l2":
        return lambda z, g: F.mse_loss(z, g, reduction="none").sum(dim=-1)
    if args.scorer == "distance_head":
        if not args.distance_head_ckpt:
            raise ValueError("--distance-head-ckpt is required for distance_head scorer")
        head = load_distance_head(args.distance_head_ckpt, device)
        return lambda z, g, head=head: head(z, g)
    if args.scorer == "qrl":
        if not args.qrl_ckpt:
            raise ValueError("--qrl-ckpt is required for qrl scorer")
        qrl = load_qrl_head(args.qrl_ckpt, device)
        return lambda z, g, qrl=qrl: qrl(z, g)
    raise ValueError(args.scorer)


def valid_actions(env: Any, state: int, previous: int | None = None) -> list[int]:
    actions: list[int] = []
    non_backtracking: list[int] = []
    for action in ACTION_IDS:
        nxt = next_state(env, state, action)
        if nxt != state:
            actions.append(action)
            if previous is None or nxt != previous:
                non_backtracking.append(action)
    return non_backtracking or actions


def optimal_actions(env: Any, state: int, goal: int, previous: int | None = None) -> set[int]:
    actions = valid_actions(env, state, previous)
    scored: list[tuple[int, int]] = []
    for action in actions:
        nxt = next_state(env, state, action)
        dist = _bfs_shortest_path(env._maze_mask, nxt, goal, env.config.width)
        if dist is not None:
            scored.append((action, int(dist)))
    if not scored:
        return set()
    best = min(dist for _, dist in scored)
    return {action for action, dist in scored if dist == best}


def model_free_action(
    model: torch.nn.Module,
    scorer: ScoreFn,
    env: Any,
    state: int,
    goal_emb: torch.Tensor,
    size: int,
    device: torch.device,
    previous: int | None = None,
) -> int:
    best_action = -1
    best_score = float("inf")
    for action in valid_actions(env, state, previous):
        nxt = next_state(env, state, action)
        emb = encode_observations(model, [observe_state(env, nxt)], size, device)
        score = float(scorer(emb, goal_emb).item())
        if score < best_score:
            best_score = score
            best_action = action
    return best_action


def predictor_action(
    model: torch.nn.Module,
    scorer: ScoreFn,
    env: Any,
    ctx_emb: torch.Tensor,
    goal_emb: torch.Tensor,
    state: int,
    device: torch.device,
    previous: int | None = None,
) -> int:
    actions = valid_actions(env, state, previous)
    if not actions:
        return -1
    num_actions = env.config.action_vocab_size
    ctx_emb_rep = ctx_emb.expand(num_actions, -1, -1)
    ctx_act_rep = torch.full((num_actions, HISTORY_SIZE - 1), num_actions - 1, dtype=torch.long, device=device)
    ctx_act_rep[:, -1] = torch.arange(num_actions, device=device)
    with torch.no_grad():
        pred = model.predictor(ctx_emb_rep, ctx_act_rep)[:, -1, :]
        goal_rep = goal_emb.expand(num_actions, -1)
        scores = scorer(pred, goal_rep)
    mask = torch.full_like(scores, float("inf"))
    mask[torch.as_tensor(actions, dtype=torch.long, device=device)] = 0.0
    return int((scores + mask).argmin().item())


def run_episode(
    model: torch.nn.Module,
    scorer: ScoreFn,
    env: Any,
    start: int,
    goal: int,
    size: int,
    device: torch.device,
) -> dict[str, Any]:
    goal_emb = encode_observations(model, [observe_state(env, goal)], size, device)
    start_obs = set_agent_state(env, start)
    start_emb = encode_observations(model, [start_obs], size, device)
    ctx_emb = start_emb.unsqueeze(0).repeat(1, HISTORY_SIZE, 1)

    cur = start
    previous: int | None = None
    visited = [cur]
    metric_wrong_steps = 0
    predictor_wrong_steps = 0
    predictor_disagrees_with_model_free = 0
    invalid_actions = 0

    for _ in range(MAX_STEPS):
        if cur == goal:
            break
        opt = optimal_actions(env, cur, goal, previous)
        mf_action = model_free_action(model, scorer, env, cur, goal_emb, size, device, previous)
        pred_action = predictor_action(model, scorer, env, ctx_emb, goal_emb.squeeze(0), cur, device, previous)
        if opt and mf_action not in opt:
            metric_wrong_steps += 1
        if opt and mf_action in opt and pred_action not in opt:
            predictor_wrong_steps += 1
        if mf_action != pred_action:
            predictor_disagrees_with_model_free += 1

        action = pred_action if pred_action > 0 else mf_action
        prev = cur
        obs, _, _, _, info = env.step(action)
        cur = int(info["state"])
        if cur == prev and action != 0:
            invalid_actions += 1
        previous = prev
        visited.append(cur)
        new_emb = encode_observations(model, [obs], size, device)
        ctx_emb = torch.cat([ctx_emb[:, 1:], new_emb.unsqueeze(0)], dim=1)

    success = cur == goal
    final_bfs = _bfs_shortest_path(env._maze_mask, cur, goal, env.config.width)
    repeats = len(visited) - len(set(visited))
    return {
        "success": success,
        "path_length": len(visited) - 1,
        "final_bfs_distance": 0 if success else (int(final_bfs) if final_bfs is not None else -1),
        "metric_wrong_steps": metric_wrong_steps,
        "predictor_wrong_steps": predictor_wrong_steps,
        "predictor_disagrees_with_model_free": predictor_disagrees_with_model_free,
        "invalid_actions": invalid_actions,
        "repeat_states": repeats,
        "visited_unique": len(set(visited)),
    }


def assign_failure_tags(row: dict[str, Any], long_path_threshold: int, seen_max_size: int) -> list[str]:
    if row["success"]:
        return ["success"]
    tags: list[str] = []
    if row["maze_size"] > seen_max_size:
        tags.append("ood_size")
    if row["optimal_path_length"] >= long_path_threshold:
        tags.append("long_path")
    if row["metric_wrong_steps"] > 0:
        tags.append("metric_wrong")
    if row["predictor_wrong_steps"] > 0:
        tags.append("predictor_wrong")
    if row["repeat_states"] >= 4:
        tags.append("loop_or_cycle")
    if row["invalid_actions"] > 0:
        tags.append("validity_failure")
    if not tags:
        tags.append("unclassified")
    return tags


def summarize(rows: list[dict[str, Any]], seen_max_size: int) -> dict[str, Any]:
    if not rows:
        return {}
    successes = sum(1 for row in rows if row["success"])
    tag_counter: Counter[str] = Counter()
    for row in rows:
        tag_counter.update(row["failure_tags"])
    return {
        "sr": successes / len(rows),
        "n": len(rows),
        "tag_counts": dict(tag_counter),
        "tag_rates": {tag: count / len(rows) for tag, count in sorted(tag_counter.items())},
        "avg_path_length": float(np.mean([row["path_length"] for row in rows])),
        "avg_final_bfs_distance": float(np.mean([row["final_bfs_distance"] for row in rows if row["final_bfs_distance"] >= 0])),
        "seen_max_size": seen_max_size,
    }


def main() -> None:
    args = parse_args()
    out = ensure_dir(run_dir(args))
    metrics_dir = ensure_dir(out / "metrics")
    train_entries = read_jsonl(args.train_manifest)
    eval_entries_all = read_jsonl(args.eval_manifest)
    verify_holdout(train_entries, eval_entries_all)
    eval_entries = select_entries(eval_entries_all, args.max_eval_per_size, args.seed + 401)

    device = torch.device(args.device)
    model = load_lewm(args.model_ckpt, device)
    scorer = build_scorer(args, device)

    print("=" * 80)
    print("EVALUATE FAILURE TAXONOMY")
    print("=" * 80)
    print(f"scorer={args.scorer} entries={len(eval_entries)}")

    rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(eval_entries):
        env = create_env(entry)
        start = int(entry["start_cell"])
        goal = int(entry["goal_cell"])
        opt = _bfs_shortest_path(env._maze_mask, start, goal, env.config.width)
        if opt is None:
            continue
        row = run_episode(model, scorer, env, start, goal, int(entry["maze_size"]), device)
        row.update(
            {
                "maze_size": int(entry["maze_size"]),
                "bucket": size_bucket(int(entry["maze_size"]), args.seen_max_size),
                "topology_seed": int(entry["topology_seed"]),
                "task_hash": entry.get("task_hash"),
                "start": start,
                "goal": goal,
                "optimal_path_length": int(opt),
            }
        )
        row["failure_tags"] = assign_failure_tags(row, args.long_path_threshold, args.seen_max_size)
        rows.append(row)
        if (idx + 1) % 50 == 0 or idx + 1 == len(eval_entries):
            print(f"{idx + 1:>4d}/{len(eval_entries)} SR={summarize(rows, args.seen_max_size)['sr']:.3f}", flush=True)

    by_size: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_size[str(row["maze_size"])].append(row)
        by_bucket[str(row["bucket"])].append(row)

    output = {
        "metadata": {
            "model_ckpt": args.model_ckpt,
            "scorer": args.scorer,
            "distance_head_ckpt": args.distance_head_ckpt,
            "qrl_ckpt": args.qrl_ckpt,
            "eval_manifest": args.eval_manifest,
            "max_eval_per_size": args.max_eval_per_size,
            "long_path_threshold": args.long_path_threshold,
            "seed": args.seed,
        },
        "summary": {
            "overall": summarize(rows, args.seen_max_size),
            "by_bucket": {bucket: summarize(group, args.seen_max_size) for bucket, group in sorted(by_bucket.items())},
            "by_size": {size: summarize(group, args.seen_max_size) for size, group in sorted(by_size.items(), key=lambda kv: int(kv[0]))},
        },
        "rows": rows,
    }
    write_json(metrics_dir / "failure_taxonomy.json", output)
    print(f"Saved: {metrics_dir / 'failure_taxonomy.json'}")


if __name__ == "__main__":
    main()
