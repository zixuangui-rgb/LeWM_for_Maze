#!/usr/bin/env python3
"""Evaluate alignment between learned metric scores and maze geometry.

This script answers the question that the final report raised but did not fully
diagnose: a score can regress global BFS distance well yet still fail to rank
local actions correctly. We therefore report both global and local metrics.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
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
    all_pairs_bfs,
    create_env,
    encode_observations,
    ensure_dir,
    free_cells,
    load_distance_head,
    load_lewm,
    load_qrl_head,
    next_state,
    observe_state,
    pearson_corr,
    read_jsonl,
    run_dir,
    select_entries,
    size_bucket,
    spearman_corr,
    verify_holdout,
    write_json,
)


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate metric/BFS alignment.")
    add_common_args(parser)
    parser.add_argument("--distance-head-ckpt", default=None)
    parser.add_argument("--qrl-ckpt", default=None)
    parser.add_argument("--max-eval-per-size", type=int, default=100)
    parser.add_argument("--states-per-maze", type=int, default=24)
    parser.add_argument("--pairs-per-maze", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def build_scores(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> dict[str, ScoreFn]:
    scorers: dict[str, ScoreFn] = {
        "latent_l2": lambda z, g: F.mse_loss(z, g, reduction="none").sum(dim=-1),
    }
    if args.distance_head_ckpt:
        head = load_distance_head(args.distance_head_ckpt, device)
        scorers["distance_head"] = lambda z, g, head=head: head(z, g)
    if args.qrl_ckpt:
        qrl = load_qrl_head(args.qrl_ckpt, device)
        scorers["qrl"] = lambda z, g, qrl=qrl: qrl(z, g)
    return scorers


def score_pairs(
    scorer: ScoreFn,
    latents: torch.Tensor,
    pair_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    latents = latents.to(device)
    with torch.no_grad():
        for start in range(0, len(pair_indices), batch_size):
            pairs = pair_indices[start : start + batch_size]
            z1 = latents[torch.as_tensor(pairs[:, 0], dtype=torch.long, device=device)]
            z2 = latents[torch.as_tensor(pairs[:, 1], dtype=torch.long, device=device)]
            scores.append(scorer(z1, z2).detach().cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.asarray([], dtype=np.float32)


def local_action_metrics(
    scorer: ScoreFn,
    latents: torch.Tensor,
    env: Any,
    cells: np.ndarray,
    cell_to_idx: dict[int, int],
    sampled_state_indices: np.ndarray,
    goal_idx: int,
    bfs: np.ndarray,
    device: torch.device,
) -> dict[str, list[float]]:
    top1: list[float] = []
    pairwise: list[float] = []
    margins: list[float] = []
    latents = latents.to(device)
    goal_latent = latents[goal_idx : goal_idx + 1]

    for state_idx in sampled_state_indices.tolist():
        if state_idx == goal_idx or bfs[state_idx, goal_idx] <= 0:
            continue
        candidates: list[tuple[int, int, float]] = []
        for action in ACTION_IDS:
            nxt = next_state(env, int(cells[state_idx]), action)
            nxt_idx = cell_to_idx.get(nxt, -1)
            if nxt_idx < 0 or nxt == int(cells[state_idx]):
                continue
            dist = float(bfs[nxt_idx, goal_idx])
            if dist < 0:
                continue
            candidates.append((action, nxt_idx, dist))
        if len(candidates) < 2:
            continue
        next_ids = [item[1] for item in candidates]
        dists = np.asarray([item[2] for item in candidates], dtype=np.float32)
        best_dist = float(dists.min())
        optimal = np.isclose(dists, best_dist)
        with torch.no_grad():
            z = latents[torch.as_tensor(next_ids, dtype=torch.long, device=device)]
            g = goal_latent.expand(z.shape[0], -1)
            scores = scorer(z, g).detach().cpu().numpy()
        best_score_idx = int(np.argmin(scores))
        top1.append(float(optimal[best_score_idx]))

        good_bad_total = 0
        good_bad_correct = 0
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if dists[i] < dists[j]:
                    good_bad_total += 1
                    good_bad_correct += int(scores[i] < scores[j])
        if good_bad_total > 0:
            pairwise.append(good_bad_correct / good_bad_total)
        best_scores = scores[optimal]
        bad_scores = scores[~optimal]
        if len(best_scores) and len(bad_scores):
            margins.append(float(bad_scores.min() - best_scores.min()))

    return {"top1": top1, "pairwise": pairwise, "margin": margins}


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {}
    for key in ["pearson", "spearman", "local_top1", "local_pairwise", "local_margin"]:
        vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    out["num_mazes"] = len(rows)
    return out


def main() -> None:
    args = parse_args()
    out = ensure_dir(run_dir(args))
    metrics_dir = ensure_dir(out / "metrics")
    train_entries = read_jsonl(args.train_manifest)
    eval_entries_all = read_jsonl(args.eval_manifest)
    verify_holdout(train_entries, eval_entries_all)
    eval_entries = select_entries(eval_entries_all, args.max_eval_per_size, args.seed + 101)

    device = torch.device(args.device)
    model = load_lewm(args.model_ckpt, device)
    scorers = build_scores(args, model, device)
    rng = np.random.default_rng(args.seed + 202)

    print("=" * 80)
    print("EVALUATE METRIC ALIGNMENT")
    print("=" * 80)
    print(f"scorers={list(scorers)} entries={len(eval_entries)}")

    rows_by_scorer: dict[str, list[dict[str, Any]]] = {name: [] for name in scorers}
    t0 = time.time()

    for entry_idx, entry in enumerate(eval_entries):
        env = create_env(entry)
        size = int(entry["maze_size"])
        cells = free_cells(env)
        if len(cells) < 2:
            continue
        cell_to_idx = {int(cell): i for i, cell in enumerate(cells.tolist())}
        observations = [observe_state(env, int(cell)) for cell in cells.tolist()]
        latents = encode_observations(model, observations, size, device).detach().cpu()
        bfs = all_pairs_bfs(env._maze_mask, cells, env.config.width)

        pair_count = min(args.pairs_per_maze, max(1, len(cells) * 2))
        pair_indices = rng.integers(0, len(cells), size=(pair_count, 2), dtype=np.int64)
        valid = pair_indices[:, 0] != pair_indices[:, 1]
        pair_indices = pair_indices[valid]
        pair_dists = bfs[pair_indices[:, 0], pair_indices[:, 1]].astype(np.float32)
        valid = pair_dists >= 0
        pair_indices = pair_indices[valid]
        pair_dists = pair_dists[valid]

        sampled_state_indices = rng.choice(
            np.arange(len(cells)),
            size=min(args.states_per_maze, len(cells)),
            replace=False,
        )
        goal = int(entry.get("goal_cell", env._goal_position))
        goal_idx = cell_to_idx.get(goal, int(rng.integers(0, len(cells))))

        for name, scorer in scorers.items():
            scores = score_pairs(scorer, latents, pair_indices, device, args.batch_size)
            local = local_action_metrics(
                scorer,
                latents,
                env,
                cells,
                cell_to_idx,
                sampled_state_indices,
                goal_idx,
                bfs,
                device,
            )
            row = {
                "scorer": name,
                "maze_size": size,
                "bucket": size_bucket(size, args.seen_max_size),
                "topology_seed": int(entry["topology_seed"]),
                "pearson": pearson_corr(scores, pair_dists),
                "spearman": spearman_corr(scores, pair_dists),
                "local_top1": float(np.mean(local["top1"])) if local["top1"] else float("nan"),
                "local_pairwise": float(np.mean(local["pairwise"])) if local["pairwise"] else float("nan"),
                "local_margin": float(np.mean(local["margin"])) if local["margin"] else float("nan"),
                "num_pairs": int(len(pair_dists)),
                "num_local_states": int(len(local["top1"])),
            }
            rows_by_scorer[name].append(row)

        if (entry_idx + 1) % 25 == 0 or entry_idx + 1 == len(eval_entries):
            print(f"{entry_idx + 1:>4d}/{len(eval_entries)} entries elapsed={time.time() - t0:.1f}s", flush=True)

    summary: dict[str, Any] = {}
    for name, rows in rows_by_scorer.items():
        by_size: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_size[str(row["maze_size"])].append(row)
            by_bucket[str(row["bucket"])].append(row)
        summary[name] = {
            "overall": summarize_group(rows),
            "by_bucket": {bucket: summarize_group(group) for bucket, group in sorted(by_bucket.items())},
            "by_size": {size: summarize_group(group) for size, group in sorted(by_size.items(), key=lambda kv: int(kv[0]))},
        }

    output = {
        "metadata": {
            "model_ckpt": args.model_ckpt,
            "distance_head_ckpt": args.distance_head_ckpt,
            "qrl_ckpt": args.qrl_ckpt,
            "eval_manifest": args.eval_manifest,
            "max_eval_per_size": args.max_eval_per_size,
            "states_per_maze": args.states_per_maze,
            "pairs_per_maze": args.pairs_per_maze,
            "seed": args.seed,
        },
        "summary": summary,
        "rows": rows_by_scorer,
    }
    write_json(metrics_dir / "metric_alignment.json", output)
    print(f"Saved: {metrics_dir / 'metric_alignment.json'}")


if __name__ == "__main__":
    main()
