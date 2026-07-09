#!/usr/bin/env python3
"""Evaluate corrected Set B latent-L2 navigation baselines.

This script uses only the LeWM backbone. It is the clean baseline for comparing
DistanceHead/QRL against latent L2 under the same corrected action selection:

- no STAY action in candidate sets
- wall/no-op actions are masked
- immediate backtracking is avoided when another moving action exists
- tasks come directly from the fixed eval manifest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.eval_setb_distance_head_fixed import (
    HISTORY_SIZE,
    MAX_STEPS,
    create_env,
    encode_obs,
    manifest_task,
    moving_actions,
    observe_state,
    run_cem,
    set_agent_state,
    summarize,
)
from scripts.train.train_dim256 import Unisize256


def load_model(model_ckpt: Path, device: torch.device) -> Unisize256:
    ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
    model = Unisize256(ckpt["model_config"], max_size=31).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def l2_score(terminal: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(terminal, goal, reduction="none").sum(dim=-1)


def run_model_free_greedy_l2(
    model: Unisize256,
    env: Any,
    start: int,
    goal: int,
    maze_size: int,
    device: torch.device,
) -> dict[str, Any]:
    goal_obs = observe_state(env, goal)
    goal_emb = encode_obs(model, goal_obs, maze_size, device)

    set_agent_state(env, start)
    cur = start
    previous: int | None = None
    path_length = 0
    stuck_steps = 0
    invalid_actions = 0
    last = cur

    for _ in range(MAX_STEPS):
        if cur == goal:
            break
        candidates = moving_actions(env, cur, previous)
        if not candidates:
            candidates = list(range(1, env.config.action_vocab_size))

        best_action = candidates[0]
        best_score = float("inf")
        for action in candidates:
            next_state = int(env._next_state(cur, env._decode_action(action)))
            obs = observe_state(env, next_state)
            next_emb = encode_obs(model, obs, maze_size, device)
            score = float(l2_score(next_emb.squeeze(1), goal_emb.squeeze(1)).item())
            if score < best_score:
                best_score = score
                best_action = action

        prev = cur
        _, _, _, _, info = env.step(best_action)
        cur = int(info["state"])
        previous = prev
        path_length += 1
        if cur == prev and best_action != 0:
            invalid_actions += 1
        if cur == last:
            stuck_steps += 1
        last = cur

    final_bfs = None if cur == goal else manifest_final_bfs(env, cur, goal)
    success = cur == goal
    return {
        "success": success,
        "path_length": path_length,
        "stuck_steps": stuck_steps,
        "invalid_actions": invalid_actions,
        "final_bfs_distance": 0 if success else (final_bfs if final_bfs is not None else -1),
    }


def run_predictor_greedy_l2(
    model: Unisize256,
    env: Any,
    start: int,
    goal: int,
    maze_size: int,
    device: torch.device,
) -> dict[str, Any]:
    num_actions = env.config.action_vocab_size
    start_obs = set_agent_state(env, start)
    start_emb = encode_obs(model, start_obs, maze_size, device)
    ctx_emb = start_emb.repeat(1, HISTORY_SIZE, 1)
    ctx_act = torch.full((1, HISTORY_SIZE), num_actions - 1, dtype=torch.long, device=device)

    goal_obs = observe_state(env, goal)
    goal_emb = encode_obs(model, goal_obs, maze_size, device)

    set_agent_state(env, start)
    cur = start
    previous: int | None = None
    path_length = 0
    stuck_steps = 0
    invalid_actions = 0
    last = cur

    for _ in range(MAX_STEPS):
        if cur == goal:
            break
        ctx_emb_rep = ctx_emb.expand(num_actions, -1, -1)
        ctx_act_rep = ctx_act[:, :-1].repeat(num_actions, 1)
        ctx_act_rep[:, -1] = torch.arange(num_actions, device=device)
        with torch.no_grad():
            pred_emb = model.predictor(ctx_emb_rep, ctx_act_rep)[:, -1, :]
            goal_rep = goal_emb.expand(num_actions, -1, -1).squeeze(1)
            scores = l2_score(pred_emb, goal_rep)

        valid_actions = moving_actions(env, cur, previous)
        if valid_actions:
            mask = torch.full_like(scores, float("inf"))
            mask[torch.tensor(valid_actions, dtype=torch.long, device=device)] = 0.0
            scores = scores + mask
        action = int(scores.argmin())

        prev = cur
        obs, _, _, _, info = env.step(action)
        cur = int(info["state"])
        previous = prev
        path_length += 1
        if cur == prev and action != 0:
            invalid_actions += 1
        if cur == last:
            stuck_steps += 1
        last = cur

        new_emb = encode_obs(model, obs, maze_size, device)
        ctx_emb = torch.cat([ctx_emb[:, 1:], new_emb], dim=1)
        ctx_act = torch.cat(
            [ctx_act[:, 1:], torch.tensor([[action]], dtype=torch.long, device=device)],
            dim=1,
        )

    final_bfs = None if cur == goal else manifest_final_bfs(env, cur, goal)
    success = cur == goal
    return {
        "success": success,
        "path_length": path_length,
        "stuck_steps": stuck_steps,
        "invalid_actions": invalid_actions,
        "final_bfs_distance": 0 if success else (final_bfs if final_bfs is not None else -1),
    }


def manifest_final_bfs(env: Any, state: int, goal: int) -> int | None:
    from hdwm.planning import _bfs_shortest_path

    dist = _bfs_shortest_path(env._maze_mask, state, goal, env.config.width)
    return None if dist is None else int(dist)


def filtered_entries(entries: list[dict[str, Any]], max_per_size: int, limit: int) -> list[dict[str, Any]]:
    if max_per_size > 0:
        counts: dict[int, int] = defaultdict(int)
        selected: list[dict[str, Any]] = []
        for entry in entries:
            size = int(entry["maze_size"])
            if counts[size] >= max_per_size:
                continue
            counts[size] += 1
            selected.append(entry)
        entries = selected
    if limit > 0:
        entries = entries[:limit]
    return entries


def evaluate_method(
    method: str,
    entries: list[dict[str, Any]],
    model: Unisize256,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for idx, entry in enumerate(entries):
        env = create_env(entry)
        start, goal, opt = manifest_task(entry, env)
        maze_size = int(entry["maze_size"])
        seed = args.seed * 10000 + idx
        if method == "model_free_greedy":
            row = run_model_free_greedy_l2(model, env, start, goal, maze_size, device)
        elif method == "predictor_greedy":
            row = run_predictor_greedy_l2(model, env, start, goal, maze_size, device)
        elif method == "cem_l2":
            row = run_cem(
                model,
                l2_score,
                env,
                start,
                goal,
                maze_size,
                device,
                seed,
                args.horizon,
                args.num_candidates,
                args.cem_iters,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        row["op_len"] = opt
        row["maze_size"] = maze_size
        row["spl"] = opt / max(int(row["path_length"]), opt) if row["success"] else 0.0
        rows.append(row)
        by_size[maze_size].append(row)
        if (idx + 1) % args.progress_every == 0:
            print(f"  [{method}] {idx + 1:>4d}/{len(entries)} SR={summarize(rows)['sr']:.4f}")
    result = summarize(rows)
    result["time"] = float(time.time() - t0)
    result["by_size"] = {str(size): summarize(size_rows) for size, size_rows in sorted(by_size.items())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--model-ckpt", default="checkpoints/backbones/unisize_dim256_clean_20260702.pt")
    parser.add_argument("--output", default="results/set_b_multisize/latent_l2_corrected_eval.json")
    parser.add_argument("--methods", default="model_free_greedy,predictor_greedy,cem_l2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-size", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--cem-iters", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    with open(args.manifest) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    entries = filtered_entries(entries, args.max_per_size, args.limit)
    print(f"Entries: {len(entries)}, sizes={sorted({int(entry['maze_size']) for entry in entries})}")
    print(f"Model: {args.model_ckpt}")
    print(f"Device: {device}")

    model = load_model(Path(args.model_ckpt), device)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    results: dict[str, Any] = {
        "manifest": args.manifest,
        "model_ckpt": args.model_ckpt,
        "methods": {},
    }
    for method in methods:
        print(f"\n[{method}]")
        results["methods"][method] = evaluate_method(method, entries, model, device, args)
        summary = results["methods"][method]
        print(
            f"  SR={summary['sr']:.4f} SPL={summary['spl']:.4f} "
            f"stuck={summary['stuck_rate']:.4f} invalid={summary['invalid_rate']:.4f} "
            f"S/F={summary['num_success']}/{summary['num_failure']} time={summary['time']:.0f}s"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
