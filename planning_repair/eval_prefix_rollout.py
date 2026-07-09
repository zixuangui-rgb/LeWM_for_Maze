#!/usr/bin/env python3
"""Evaluate direct multi-horizon rollout quality of the action-prefix predictor."""

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

from diagnostics.common import (
    all_pairs_bfs,
    create_env,
    encode_observations,
    free_cells,
    observe_state,
    read_jsonl,
    size_bucket,
    verify_holdout,
)
from planning_repair.common import (
    grouped_limit,
    json_dump,
    load_backbone_from_repair_ckpt,
    require_trained_component,
    set_seed,
)
from planning_repair.heads import load_prefix_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prefix predictor rollout.")
    parser.add_argument("--train-manifest", default="data/splits/unisize_train_manifest.jsonl")
    parser.add_argument("--eval-manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument(
        "--output",
        default="planning_repair_runs/prefix_rollout/results.json",
    )
    parser.add_argument("--horizons", default="1,2,3,5")
    parser.add_argument("--episodes-per-entry", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-size", type=int, default=40)
    parser.add_argument("--seen-max-size", type=int, default=21)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--allow-untrained-prefix",
        action="store_true",
        help="Only for smoke tests; do not use for scientific comparisons.",
    )
    return parser.parse_args()


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not horizons or min(horizons) <= 0:
        raise ValueError("--horizons must contain positive integers")
    return horizons


def nearest_state_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    all_latents: torch.Tensor,
    true_cell_idx: int,
    bfs: np.ndarray,
) -> dict[str, float]:
    dists = F.mse_loss(all_latents, pred.expand_as(all_latents), reduction="none").sum(dim=1)
    nn_idx = int(dists.argmin().item())
    bfs_err = float(bfs[nn_idx, true_cell_idx]) if bfs[nn_idx, true_cell_idx] >= 0 else float("nan")
    return {
        "nn_exact": float(nn_idx == true_cell_idx),
        "nn_bfs_error": bfs_err,
        "nn_index": float(nn_idx),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    out: dict[str, float] = {}
    for key in ["latent_mse", "cosine", "nn_exact", "nn_bfs_error"]:
        vals = np.asarray([row[key] for row in rows if np.isfinite(row[key])], dtype=np.float64)
        out[key] = float(vals.mean()) if vals.size else float("nan")
    out["n"] = float(len(rows))
    return out


def evaluate_entry(
    *,
    model: torch.nn.Module,
    prefix_predictor: torch.nn.Module,
    entry: dict[str, Any],
    horizons: list[int],
    episodes_per_entry: int,
    seen_max_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    env = create_env(entry)
    size = int(entry["maze_size"])
    max_horizon = max(horizons)
    seq_len = max_horizon + 1
    cells = free_cells(env)
    cell_to_idx = {int(cell): i for i, cell in enumerate(cells.tolist())}
    all_obs = [observe_state(env, int(cell)) for cell in cells.tolist()]
    all_latents = encode_observations(model, all_obs, size, device).detach()
    bfs = all_pairs_bfs(env._maze_mask, cells, env.config.width)

    rows: list[dict[str, Any]] = []
    for episode in range(episodes_per_entry):
        batch = env.sample_sequence(batch_size=1, sequence_length=seq_len)
        obs = batch.observations[0].cpu().numpy()
        actions = batch.actions[:, :max_horizon].to(device)
        states = batch.states[0].cpu().numpy().astype(np.int64)
        true_latents = encode_observations(
            model,
            [obs[t] for t in range(seq_len)],
            size,
            device,
        ).detach()

        with torch.no_grad():
            pred = prefix_predictor(true_latents[0:1], actions)

        for horizon in horizons:
            true = true_latents[horizon]
            pred_h = pred[0, horizon - 1]
            true_cell_idx = cell_to_idx.get(int(states[horizon]), -1)
            if true_cell_idx < 0:
                continue
            nn = nearest_state_metrics(pred_h, true, all_latents, true_cell_idx, bfs)
            rows.append(
                {
                    "mode": "prefix_direct",
                    "horizon": int(horizon),
                    "maze_size": size,
                    "bucket": size_bucket(size, seen_max_size),
                    "topology_seed": int(entry["topology_seed"]),
                    "episode": int(episode),
                    "latent_mse": float(F.mse_loss(pred_h, true).item()),
                    "cosine": float(
                        F.cosine_similarity(pred_h.view(1, -1), true.view(1, -1)).item()
                    ),
                    "nn_exact": nn["nn_exact"],
                    "nn_bfs_error": nn["nn_bfs_error"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    horizons = parse_horizons(args.horizons)
    device = torch.device(args.device)
    train_entries = read_jsonl(args.train_manifest)
    eval_entries_all = read_jsonl(args.eval_manifest)
    verify_holdout(train_entries, eval_entries_all)
    entries = grouped_limit(
        eval_entries_all,
        max_per_size=args.max_per_size,
        limit=args.limit,
    )

    model, data = load_backbone_from_repair_ckpt(args.model_ckpt, device)
    require_trained_component(
        data,
        component="prefix",
        allow_untrained=args.allow_untrained_prefix,
    )
    prefix_predictor = load_prefix_predictor(data, device)
    if prefix_predictor is None:
        raise ValueError("checkpoint does not contain a prefix predictor")
    max_supported = int(prefix_predictor.config.max_horizon)
    if max(horizons) > max_supported:
        raise ValueError(f"requested horizon {max(horizons)} exceeds trained max {max_supported}")

    print("=" * 80)
    print("EVALUATE PREFIX ROLLOUT")
    print("=" * 80)
    print(f"entries={len(entries)} horizons={horizons} model={args.model_ckpt} device={device}")

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for idx, entry in enumerate(entries):
        rows.extend(
            evaluate_entry(
                model=model,
                prefix_predictor=prefix_predictor,
                entry=entry,
                horizons=horizons,
                episodes_per_entry=args.episodes_per_entry,
                seen_max_size=args.seen_max_size,
                device=device,
            )
        )
        if (idx + 1) % args.progress_every == 0 or idx + 1 == len(entries):
            print(f"{idx + 1:>4d}/{len(entries)} elapsed={time.time() - t0:.1f}s", flush=True)

    by_horizon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    by_size: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        horizon = str(row["horizon"])
        by_horizon[horizon].append(row)
        by_bucket[str(row["bucket"])][horizon].append(row)
        by_size[str(row["maze_size"])][horizon].append(row)

    output = {
        "metadata": {
            "train_manifest": args.train_manifest,
            "eval_manifest": args.eval_manifest,
            "model_ckpt": args.model_ckpt,
            "horizons": horizons,
            "episodes_per_entry": args.episodes_per_entry,
            "limit": args.limit,
            "max_per_size": args.max_per_size,
            "seed": args.seed,
        },
        "summary": {
            "overall": {
                horizon: summarize_rows(group)
                for horizon, group in sorted(by_horizon.items(), key=lambda kv: int(kv[0]))
            },
            "by_bucket": {
                bucket: {
                    horizon: summarize_rows(group)
                    for horizon, group in sorted(groups.items(), key=lambda kv: int(kv[0]))
                }
                for bucket, groups in sorted(by_bucket.items())
            },
            "by_size": {
                size: {
                    horizon: summarize_rows(group)
                    for horizon, group in sorted(groups.items(), key=lambda kv: int(kv[0]))
                }
                for size, groups in sorted(by_size.items(), key=lambda kv: int(kv[0]))
            },
        },
        "rows": rows,
    }
    json_dump(args.output, output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
