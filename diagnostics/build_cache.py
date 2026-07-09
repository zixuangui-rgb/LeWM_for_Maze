#!/usr/bin/env python3
"""Build a reusable feature cache for Maze-JEPA diagnostics.

The cache is the foundation of the diagnostic suite. It fixes the examples used
by all probes so later experiments can compare models under the same protocol:

- same train/eval manifest split;
- same topology-holdout verification;
- same sampled states per maze;
- same labels for position, valid actions, optimal actions, and BFS distance.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnostics.common import (
    ACTION_IDS,
    add_common_args,
    create_env,
    ensure_dir,
    extract_layers_batch,
    bfs_distances_from,
    free_cells,
    load_lewm,
    observe_state,
    optimal_action_mask,
    read_jsonl,
    run_dir,
    select_entries,
    size_bucket,
    valid_action_mask,
    verify_holdout,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Maze-JEPA diagnostic feature cache.")
    add_common_args(parser)
    parser.add_argument(
        "--layers",
        default="spatial_flat,spatial_pool,encoded,embedding",
        help="Comma-separated diagnostic layers to cache.",
    )
    parser.add_argument("--max-train-per-size", type=int, default=80)
    parser.add_argument("--max-eval-per-size", type=int, default=100)
    parser.add_argument("--states-per-maze", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def empty_split() -> dict[str, Any]:
    return {
        "features": defaultdict(dict),
        "labels": defaultdict(dict),
        "meta": defaultdict(list),
    }


def append_size_block(
    split_cache: dict[str, Any],
    size: int,
    layer_blocks: dict[str, torch.Tensor],
    labels: dict[str, np.ndarray],
    metas: list[dict[str, Any]],
) -> None:
    size_key = str(size)
    for layer, values in layer_blocks.items():
        existing = split_cache["features"][layer].get(size_key)
        split_cache["features"][layer][size_key] = values if existing is None else torch.cat([existing, values], dim=0)
    for name, values in labels.items():
        existing = split_cache["labels"][name].get(size_key)
        tensor = torch.as_tensor(values)
        split_cache["labels"][name][size_key] = tensor if existing is None else torch.cat([existing, tensor], dim=0)
    split_cache["meta"][size_key].extend(metas)


def sample_states(cells: np.ndarray, states_per_maze: int, rng: np.random.Generator) -> np.ndarray:
    if states_per_maze <= 0 or states_per_maze >= len(cells):
        return cells.copy()
    return rng.choice(cells, size=states_per_maze, replace=False).astype(np.int64)


def build_split_cache(
    *,
    model: torch.nn.Module,
    entries: list[dict[str, Any]],
    split_name: str,
    layers: list[str],
    states_per_maze: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    seen_max_size: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    split_cache = empty_split()
    t0 = time.time()

    for entry_idx, entry in enumerate(entries):
        size = int(entry["maze_size"])
        env = create_env(entry)
        cells = free_cells(env)
        chosen_states = sample_states(cells, states_per_maze, rng)
        goal = int(entry.get("goal_cell", env._goal_position))
        goal_dists = bfs_distances_from(env._maze_mask, goal, env.config.width)
        reachable_goal_dists = goal_dists[goal_dists >= 0]
        max_dist = max(1.0, float(reachable_goal_dists.max() if reachable_goal_dists.size else 1.0))

        observations: list[np.ndarray] = []
        labels: dict[str, list[Any]] = {
            "agent_x": [],
            "agent_y": [],
            "goal_x": [],
            "goal_y": [],
            "valid_action": [],
            "optimal_action": [],
            "optimal_action_mask": [],
            "bfs_distance": [],
            "bfs_distance_norm": [],
        }
        metas: list[dict[str, Any]] = []

        for state in chosen_states.tolist():
            obs = observe_state(env, int(state))
            opt_mask, opt_cls, cur_dist = optimal_action_mask(env, int(state), goal)
            observations.append(obs)
            labels["agent_x"].append(int(state % size))
            labels["agent_y"].append(int(state // size))
            labels["goal_x"].append(int(goal % size))
            labels["goal_y"].append(int(goal // size))
            labels["valid_action"].append(valid_action_mask(env, int(state)))
            labels["optimal_action"].append(int(opt_cls))
            labels["optimal_action_mask"].append(opt_mask)
            labels["bfs_distance"].append(float(cur_dist))
            labels["bfs_distance_norm"].append(float(cur_dist) / max_dist)
            metas.append(
                {
                    "split": split_name,
                    "entry_index": int(entry_idx),
                    "maze_size": size,
                    "bucket": size_bucket(size, seen_max_size),
                    "topology_seed": int(entry["topology_seed"]),
                    "env_seed": int(entry.get("env_seed", 42)),
                    "layout_hash": entry.get("layout_hash"),
                    "task_hash": entry.get("task_hash"),
                    "state": int(state),
                    "goal": int(goal),
                    "bfs_distance": float(cur_dist),
                }
            )

        layer_chunks: dict[str, list[torch.Tensor]] = {layer: [] for layer in layers}
        for start in range(0, len(observations), batch_size):
            batch_obs = observations[start : start + batch_size]
            blocks = extract_layers_batch(model, batch_obs, size, device, layers)
            for layer, tensor in blocks.items():
                layer_chunks[layer].append(tensor.cpu())

        layer_blocks = {layer: torch.cat(chunks, dim=0) for layer, chunks in layer_chunks.items() if chunks}
        label_arrays = {
            "agent_x": np.asarray(labels["agent_x"], dtype=np.int64),
            "agent_y": np.asarray(labels["agent_y"], dtype=np.int64),
            "goal_x": np.asarray(labels["goal_x"], dtype=np.int64),
            "goal_y": np.asarray(labels["goal_y"], dtype=np.int64),
            "valid_action": np.asarray(labels["valid_action"], dtype=np.float32),
            "optimal_action": np.asarray(labels["optimal_action"], dtype=np.int64),
            "optimal_action_mask": np.asarray(labels["optimal_action_mask"], dtype=np.float32),
            "bfs_distance": np.asarray(labels["bfs_distance"], dtype=np.float32),
            "bfs_distance_norm": np.asarray(labels["bfs_distance_norm"], dtype=np.float32),
        }
        append_size_block(split_cache, size, layer_blocks, label_arrays, metas)

        if (entry_idx + 1) % 25 == 0 or entry_idx + 1 == len(entries):
            print(
                f"[{split_name}] {entry_idx + 1:>4d}/{len(entries)} entries "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    return {
        "features": {layer: dict(by_size) for layer, by_size in split_cache["features"].items()},
        "labels": {name: dict(by_size) for name, by_size in split_cache["labels"].items()},
        "meta": dict(split_cache["meta"]),
    }


def main() -> None:
    args = parse_args()
    out = ensure_dir(run_dir(args))
    cache_dir = ensure_dir(out / "feature_cache")
    layers = [item.strip() for item in args.layers.split(",") if item.strip()]

    train_entries_all = read_jsonl(args.train_manifest)
    eval_entries_all = read_jsonl(args.eval_manifest)
    overlap_counts = verify_holdout(train_entries_all, eval_entries_all)
    train_entries = select_entries(train_entries_all, args.max_train_per_size, args.seed)
    eval_entries = select_entries(eval_entries_all, args.max_eval_per_size, args.seed + 1)

    print("=" * 80)
    print("BUILD DIAGNOSTIC FEATURE CACHE")
    print("=" * 80)
    print(f"layers={layers}")
    print(f"train_entries={len(train_entries)} eval_entries={len(eval_entries)} overlap={overlap_counts}")

    device = torch.device(args.device)
    model = load_lewm(args.model_ckpt, device)

    cache = {
        "metadata": {
            "model_ckpt": args.model_ckpt,
            "train_manifest": args.train_manifest,
            "eval_manifest": args.eval_manifest,
            "layers": layers,
            "max_train_per_size": args.max_train_per_size,
            "max_eval_per_size": args.max_eval_per_size,
            "states_per_maze": args.states_per_maze,
            "seed": args.seed,
            "seen_max_size": args.seen_max_size,
            "action_ids": list(ACTION_IDS),
            "holdout": overlap_counts,
        },
        "splits": {},
    }
    cache["splits"]["train"] = build_split_cache(
        model=model,
        entries=train_entries,
        split_name="train",
        layers=layers,
        states_per_maze=args.states_per_maze,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 10,
        seen_max_size=args.seen_max_size,
    )
    cache["splits"]["eval"] = build_split_cache(
        model=model,
        entries=eval_entries,
        split_name="eval",
        layers=layers,
        states_per_maze=args.states_per_maze,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 20,
        seen_max_size=args.seen_max_size,
    )

    cache_path = cache_dir / "features.pt"
    torch.save(cache, cache_path)
    manifest = {
        "cache_path": str(cache_path),
        "metadata": cache["metadata"],
        "sizes": {
            split: sorted(int(size) for size in cache["splits"][split]["meta"].keys())
            for split in ["train", "eval"]
        },
        "num_examples": {
            split: {
                size: len(cache["splits"][split]["meta"][size])
                for size in sorted(cache["splits"][split]["meta"])
            }
            for split in ["train", "eval"]
        },
    }
    write_json(cache_dir / "manifest.json", manifest)
    print(f"Saved cache: {cache_path}")
    print(f"Saved manifest: {cache_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
