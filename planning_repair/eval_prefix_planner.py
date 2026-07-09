#!/usr/bin/env python3
"""Evaluate the action-prefix predictor as a non-recursive latent planner."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdwm.planning import _bfs_shortest_path
from planning_repair.common import (
    ACTION_IDS,
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
from planning_repair.heads import load_aux_heads, load_prefix_predictor
from scripts.eval.eval_setb_distance_head_fixed import MAX_STEPS, manifest_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate action-prefix planning.")
    parser.add_argument("--manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument(
        "--output",
        default="planning_repair_runs/prefix_planner/results.json",
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--terminal-scorer", choices=["latent_l2", "aux_bfs"], default="latent_l2")
    parser.add_argument("--score-all-prefixes", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-size", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def sample_action_sequences(
    *,
    rng: np.random.Generator,
    valid_first_actions: list[int],
    horizon: int,
    num_candidates: int,
) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not valid_first_actions:
        valid_first_actions = [int(ACTION_IDS[0])]
    sequences = rng.choice(
        np.asarray(ACTION_IDS, dtype=np.int64),
        size=(num_candidates, horizon),
        replace=True,
    )
    sequences[:, 0] = rng.choice(
        np.asarray(valid_first_actions, dtype=np.int64),
        size=num_candidates,
        replace=True,
    )
    # Ensure every legal first action is represented at least once.
    for idx, action in enumerate(valid_first_actions[:num_candidates]):
        sequences[idx, 0] = int(action)
    return sequences.astype(np.int64)


def score_prefixes(
    *,
    prefix_predictor: torch.nn.Module,
    aux_heads: torch.nn.Module | None,
    current_embedding: torch.Tensor,
    goal_embedding: torch.Tensor,
    action_sequences: np.ndarray,
    terminal_scorer: str,
    score_all_prefixes: bool,
    device: torch.device,
) -> torch.Tensor:
    actions = torch.as_tensor(action_sequences, dtype=torch.long, device=device)
    z0 = current_embedding.squeeze(0).squeeze(0).expand(actions.shape[0], -1)
    with torch.no_grad():
        pred = prefix_predictor(z0, actions)
        candidates = pred if score_all_prefixes else pred[:, -1:, :]
        if terminal_scorer == "latent_l2":
            goal = goal_embedding.squeeze(0).squeeze(0).view(1, 1, -1)
            scores = F.mse_loss(
                candidates,
                goal.expand(candidates.shape[0], candidates.shape[1], -1),
                reduction="none",
            ).sum(dim=-1)
        elif terminal_scorer == "aux_bfs":
            if aux_heads is None:
                raise ValueError("aux_bfs scorer requires aux heads in checkpoint")
            flat = candidates.reshape(-1, candidates.shape[-1])
            scores = aux_heads(flat)["bfs_distance_norm"].view(candidates.shape[:2])
        else:
            raise ValueError(terminal_scorer)
    return scores.min(dim=1).values


def run_episode(
    *,
    model: torch.nn.Module,
    prefix_predictor: torch.nn.Module,
    aux_heads: torch.nn.Module | None,
    entry: dict[str, Any],
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, Any]:
    env = create_env(entry)
    start, goal, opt = manifest_task(entry, env)
    size = int(entry["maze_size"])
    set_agent_state(env, start)
    goal_embedding = encode_single_observation(model, observe_state(env, goal), size, device)

    cur = start
    previous: int | None = None
    path_length = 0
    stuck_steps = 0
    invalid_actions = 0
    last = cur

    for _ in range(MAX_STEPS):
        if cur == goal:
            break
        current_embedding = encode_single_observation(
            model,
            observe_state(env, cur),
            size,
            device,
        )
        valid_first = valid_moving_actions(env, cur, previous)
        action_sequences = sample_action_sequences(
            rng=rng,
            valid_first_actions=valid_first,
            horizon=args.horizon,
            num_candidates=args.num_candidates,
        )
        scores = score_prefixes(
            prefix_predictor=prefix_predictor,
            aux_heads=aux_heads,
            current_embedding=current_embedding,
            goal_embedding=goal_embedding,
            action_sequences=action_sequences,
            terminal_scorer=args.terminal_scorer,
            score_all_prefixes=args.score_all_prefixes,
            device=device,
        )
        action = int(action_sequences[int(scores.argmin().item()), 0])
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
        "final_bfs_distance": 0 if success else (final_bfs if final_bfs is not None else -1),
        "op_len": opt,
        "maze_size": size,
        "spl": opt / max(path_length, opt) if success else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.num_candidates <= 0:
        raise ValueError("--num-candidates must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    model, data = load_backbone_from_repair_ckpt(args.model_ckpt, device)
    prefix_predictor = load_prefix_predictor(data, device)
    if prefix_predictor is None:
        raise ValueError("checkpoint does not contain a prefix predictor")
    aux_heads = load_aux_heads(data, device)
    entries = grouped_limit(
        read_jsonl(args.manifest),
        max_per_size=args.max_per_size,
        limit=args.limit,
    )
    rng = np.random.default_rng(args.seed)
    print("=" * 80)
    print("EVALUATE ACTION-PREFIX PLANNER")
    print("=" * 80)
    print(
        f"entries={len(entries)} horizon={args.horizon} "
        f"candidates={args.num_candidates} scorer={args.terminal_scorer}"
    )

    rows: list[dict[str, Any]] = []
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    t0 = time.time()
    for idx, entry in enumerate(entries):
        row = run_episode(
            model=model,
            prefix_predictor=prefix_predictor,
            aux_heads=aux_heads,
            entry=entry,
            args=args,
            rng=rng,
            device=device,
        )
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
    output = {
        "metadata": {
            "manifest": args.manifest,
            "model_ckpt": args.model_ckpt,
            "horizon": args.horizon,
            "num_candidates": args.num_candidates,
            "terminal_scorer": args.terminal_scorer,
            "score_all_prefixes": args.score_all_prefixes,
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
