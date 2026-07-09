#!/usr/bin/env python3
"""Evaluate B2: short-horizon receding CEM on fixed Set-B tasks."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planning_repair.common import (
    create_env,
    grouped_limit,
    json_dump,
    load_backbone_from_repair_ckpt,
    read_jsonl,
    require_trained_component,
    set_seed,
    summarize_navigation,
)
from planning_repair.heads import load_aux_heads
from scripts.eval.eval_setb_distance_head_fixed import manifest_task, run_cem
from diagnostics.common import load_distance_head


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate short-horizon receding CEM.")
    parser.add_argument("--manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--distance-head-ckpt", default=None)
    parser.add_argument(
        "--output",
        default="planning_repair_runs/b2_receding/results.json",
    )
    parser.add_argument(
        "--scorers",
        default="latent_l2",
        help="Comma-separated: latent_l2,distance_head,aux_bfs",
    )
    parser.add_argument("--horizons", default="3,5,8,12")
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--cem-iters", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-size", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--allow-untrained-aux",
        action="store_true",
        help="Only for smoke tests; do not use for scientific comparisons.",
    )
    return parser.parse_args()


def parse_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons:
        raise ValueError("--horizons must contain at least one integer")
    return horizons


def build_scorers(
    names: list[str],
    checkpoint_data: dict[str, Any],
    device: torch.device,
    distance_head_ckpt: str | None,
    allow_untrained_aux: bool,
) -> dict[str, ScoreFn]:
    scorers: dict[str, ScoreFn] = {}
    aux = load_aux_heads(checkpoint_data, device)
    distance_head = None
    for name in names:
        if name == "latent_l2":
            scorers[name] = lambda z, g: F.mse_loss(z, g, reduction="none").sum(dim=-1)
        elif name == "distance_head":
            if not distance_head_ckpt:
                raise ValueError("--distance-head-ckpt is required for distance_head")
            if distance_head is None:
                distance_head = load_distance_head(distance_head_ckpt, device)
            scorers[name] = lambda z, g, head=distance_head: head(z, g)
        elif name == "aux_bfs":
            if aux is None:
                raise ValueError("model checkpoint does not contain aux heads for aux_bfs")
            require_trained_component(
                checkpoint_data,
                component="aux_bfs",
                allow_untrained=allow_untrained_aux,
            )
            scorers[name] = lambda z, g, aux=aux: aux(z)["bfs_distance_norm"]
        else:
            raise ValueError(f"unknown scorer: {name}")
    return scorers


def evaluate(
    *,
    entries: list[dict[str, Any]],
    model: torch.nn.Module,
    scorer: ScoreFn,
    horizon: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    t0 = time.time()
    for idx, entry in enumerate(entries):
        env = create_env(entry)
        start, goal, opt = manifest_task(entry, env)
        size = int(entry["maze_size"])
        row = run_cem(
            model,
            scorer,
            env,
            start,
            goal,
            size,
            device,
            args.seed * 10000 + idx,
            horizon,
            args.num_candidates,
            args.cem_iters,
        )
        row["op_len"] = opt
        row["maze_size"] = size
        row["spl"] = opt / max(int(row["path_length"]), opt) if row["success"] else 0.0
        rows.append(row)
        by_size[size].append(row)
        if (idx + 1) % args.progress_every == 0:
            print(
                f"  horizon={horizon} {idx + 1:>4d}/{len(entries)} "
                f"SR={summarize_navigation(rows)['sr']:.4f}",
                flush=True,
            )
    summary = summarize_navigation(rows)
    summary["time"] = float(time.time() - t0)
    summary["by_size"] = {
        str(size): summarize_navigation(size_rows)
        for size, size_rows in sorted(by_size.items())
    }
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    model, checkpoint_data = load_backbone_from_repair_ckpt(args.model_ckpt, device)
    scorers = build_scorers(
        parse_names(args.scorers),
        checkpoint_data,
        device,
        args.distance_head_ckpt,
        args.allow_untrained_aux,
    )
    horizons = parse_horizons(args.horizons)
    entries = grouped_limit(
        read_jsonl(args.manifest),
        max_per_size=args.max_per_size,
        limit=args.limit,
    )
    print("=" * 80)
    print("EVALUATE B2 RECEDING HORIZON")
    print("=" * 80)
    print(f"entries={len(entries)} horizons={horizons} scorers={list(scorers)} device={device}")

    results: dict[str, Any] = {
        "metadata": {
            "manifest": args.manifest,
            "model_ckpt": args.model_ckpt,
            "distance_head_ckpt": args.distance_head_ckpt,
            "horizons": horizons,
            "num_candidates": args.num_candidates,
            "cem_iters": args.cem_iters,
            "limit": args.limit,
            "max_per_size": args.max_per_size,
            "seed": args.seed,
        },
        "results": {},
    }
    for scorer_name, scorer in scorers.items():
        results["results"][scorer_name] = {}
        for horizon in horizons:
            print(f"\n[{scorer_name}] horizon={horizon}")
            summary = evaluate(
                entries=entries,
                model=model,
                scorer=scorer,
                horizon=horizon,
                args=args,
                device=device,
            )
            results["results"][scorer_name][str(horizon)] = summary
            print(
                f"  SR={summary['sr']:.4f} SPL={summary['spl']:.4f} "
                f"stuck={summary['stuck_rate']:.4f} invalid={summary['invalid_rate']:.4f}"
            )

    json_dump(args.output, results)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
