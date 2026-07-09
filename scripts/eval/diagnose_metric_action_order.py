#!/usr/bin/env python3
"""Diagnose whether a metric head ranks local navigation actions correctly.

Global BFS regression can improve while closed-loop SR stays flat. This script
measures the quantity that greedy navigation actually needs: for a state and a
goal, does the metric assign the best score to an action whose next state has
minimal true BFS distance to the goal?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdwm.metric_heads.distance_head import DistanceHead
from hdwm.metric_heads.qrl_head import QRLHead
from scripts.eval.eval_setb_distance_head_fixed import create_env
from scripts.train.train_dim256 import Unisize256
from scripts.train.train_distance_head_v2 import all_pairs_bfs, encode_batch, observe_state


class MetricHead(Protocol):
    def __call__(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor: ...


def load_model_from_metric_or_base(
    model_ckpt: Path,
    metric_data: dict[str, Any],
    device: torch.device,
) -> tuple[Unisize256, str]:
    if "model_state_dict" in metric_data:
        model = Unisize256(metric_data["model_config"], max_size=31).to(device)
        model.load_state_dict(metric_data["model_state_dict"], strict=True)
        source = "metric_checkpoint"
    else:
        ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
        model = Unisize256(ckpt["model_config"], max_size=31).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        source = str(model_ckpt)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, source


def load_head(path: Path, head_type: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    data = torch.load(path, map_location=device, weights_only=False)
    cfg = data.get("config", {})
    if head_type == "distance":
        head: nn.Module = DistanceHead(
            latent_dim=int(cfg.get("latent_dim", 256)),
            hidden_dims=cfg.get("hidden_dims", [256, 128]),
            input_mode=cfg.get("input_mode", "concat"),
        ).to(device)
    elif head_type == "qrl":
        head = QRLHead(
            latent_dim=int(cfg.get("latent_dim", 256)),
            hidden_dims=cfg.get("hidden_dims", [256, 128]),
            temperature=float(cfg.get("temperature", 0.1)),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(device)
    else:
        raise ValueError(f"unknown head_type: {head_type}")
    head.load_state_dict(data["head_state_dict"], strict=True)
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    return head, data


def valid_candidates(next_indices: np.ndarray, state_idx: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for action in range(1, next_indices.shape[1]):
        next_idx = int(next_indices[state_idx, action])
        if next_idx >= 0 and next_idx != state_idx:
            candidates.append((action, next_idx))
    return candidates


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "n": 0.0,
            "top1": 0.0,
            "improve": 0.0,
            "non_improve": 0.0,
            "mean_regret": 0.0,
            "mean_abs_error": 0.0,
        }
    return {
        "n": float(len(rows)),
        "top1": float(np.mean([row["top1"] for row in rows])),
        "improve": float(np.mean([row["improve"] for row in rows])),
        "non_improve": float(np.mean([1.0 - row["improve"] for row in rows])),
        "mean_regret": float(np.mean([row["regret"] for row in rows])),
        "mean_abs_error": float(np.mean([row["abs_error"] for row in rows])),
    }


def evaluate_entry(
    entry: dict[str, Any],
    model: Unisize256,
    head: MetricHead,
    device: torch.device,
    states_per_maze: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    env = create_env(entry)
    cells = np.flatnonzero((~env._maze_mask).reshape(-1)).astype(np.int64).tolist()
    cell_to_idx = {cell: idx for idx, cell in enumerate(cells)}
    observations = [observe_state(env, cell) for cell in cells]
    latents = encode_batch(model, observations, int(entry["maze_size"]), device).to(device)
    bfs = all_pairs_bfs(env._maze_mask, cells, env.config.width)

    next_indices = np.full((len(cells), env.config.action_vocab_size), -1, dtype=np.int32)
    for idx, cell in enumerate(cells):
        for action in range(env.config.action_vocab_size):
            next_cell = int(env._next_state(cell, env._decode_action(action)))
            next_indices[idx, action] = cell_to_idx.get(next_cell, -1)

    goal_idx = cell_to_idx[int(entry["goal_cell"])]
    available = np.asarray([idx for idx in range(len(cells)) if idx != goal_idx and bfs[idx, goal_idx] > 0])
    if states_per_maze > 0 and available.size > states_per_maze:
        state_indices = rng.choice(available, size=states_per_maze, replace=False).tolist()
    else:
        state_indices = available.tolist()

    rows: list[dict[str, float]] = []
    goal = latents[goal_idx].unsqueeze(0)
    with torch.no_grad():
        for state_idx in state_indices:
            candidates = valid_candidates(next_indices, state_idx)
            if len(candidates) < 2:
                continue
            next_ids = [next_idx for _, next_idx in candidates]
            true_dists = np.asarray([bfs[next_idx, goal_idx] for next_idx in next_ids], dtype=np.float32)
            if (true_dists < 0).any():
                continue
            best_true = float(np.min(true_dists))
            scores = head(latents[next_ids], goal.expand(len(next_ids), -1))
            pred_slot = int(scores.argmin().item())
            pred_true = float(true_dists[pred_slot])
            cur_true = float(bfs[state_idx, goal_idx])
            pred_score = float(scores[pred_slot].item())
            rows.append(
                {
                    "top1": float(np.isclose(pred_true, best_true)),
                    "improve": float(pred_true < cur_true),
                    "regret": float(pred_true - best_true),
                    "abs_error": abs(pred_score - pred_true),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--model-ckpt", default="checkpoints/unisize_dim256.pt")
    parser.add_argument("--head-ckpt", required=True)
    parser.add_argument("--head-type", choices=["distance", "qrl"], required=True)
    parser.add_argument("--output", default="results/set_b_multisize/metric_action_order_diagnosis.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-size", type=int, default=0)
    parser.add_argument("--states-per-maze", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    head, metric_data = load_head(Path(args.head_ckpt), args.head_type, device)
    model, model_source = load_model_from_metric_or_base(Path(args.model_ckpt), metric_data, device)
    with open(args.manifest) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if args.max_per_size > 0:
        counts: dict[int, int] = defaultdict(int)
        balanced_entries: list[dict[str, Any]] = []
        for entry in entries:
            size = int(entry["maze_size"])
            if counts[size] >= args.max_per_size:
                continue
            counts[size] += 1
            balanced_entries.append(entry)
        entries = balanced_entries
    if args.limit > 0:
        entries = entries[: args.limit]

    by_size: dict[int, list[dict[str, float]]] = defaultdict(list)
    all_rows: list[dict[str, float]] = []
    for idx, entry in enumerate(entries):
        rows = evaluate_entry(entry, model, head, device, args.states_per_maze, rng)
        size = int(entry["maze_size"])
        by_size[size].extend(rows)
        all_rows.extend(rows)
        if (idx + 1) % args.progress_every == 0:
            print(f"{idx + 1}/{len(entries)} local_top1={summarize(all_rows)['top1']:.4f}")

    result = {
        "manifest": args.manifest,
        "head_ckpt": args.head_ckpt,
        "head_type": args.head_type,
        "model_source": model_source,
        "states_per_maze": args.states_per_maze,
        "overall": summarize(all_rows),
        "by_size": {str(size): summarize(rows) for size, rows in sorted(by_size.items())},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["overall"], indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
