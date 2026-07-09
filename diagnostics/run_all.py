#!/usr/bin/env python3
"""Run the full Maze-JEPA diagnostic benchmark.

This orchestrator intentionally shells out to the individual scripts. That
keeps every stage independently runnable and makes failed stages easy to resume.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Maze-JEPA diagnostics.")
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--train-manifest", default="data/splits/unisize_train_manifest.jsonl")
    parser.add_argument("--eval-manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="diagnostics_runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seen-max-size", type=int, default=21)
    parser.add_argument("--distance-head-ckpt", default=None)
    parser.add_argument("--qrl-ckpt", default=None)
    parser.add_argument("--max-train-per-size", type=int, default=80)
    parser.add_argument("--max-eval-per-size", type=int, default=100)
    parser.add_argument("--states-per-maze", type=int, default=24)
    parser.add_argument("--probe-epochs", type=int, default=25)
    parser.add_argument("--rollout-episodes-per-entry", type=int, default=1)
    parser.add_argument("--stages", default="cache,probes,metric,rollout,failure,report")
    return parser.parse_args()


def base_args(args: argparse.Namespace) -> list[str]:
    return [
        "--model-ckpt",
        args.model_ckpt,
        "--train-manifest",
        args.train_manifest,
        "--eval-manifest",
        args.eval_manifest,
        "--run-id",
        args.run_id,
        "--out-dir",
        args.out_dir,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--seen-max-size",
        str(args.seen_max_size),
    ]


def run_stage(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "diagnostics" / script), *extra]
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    common = base_args(args)

    if "cache" in stages:
        run_stage(
            "build_cache.py",
            [
                *common,
                "--max-train-per-size",
                str(args.max_train_per_size),
                "--max-eval-per-size",
                str(args.max_eval_per_size),
                "--states-per-maze",
                str(args.states_per_maze),
            ],
        )
    if "probes" in stages:
        run_stage(
            "train_probes.py",
            [
                *common,
                "--epochs",
                str(args.probe_epochs),
            ],
        )
    if "metric" in stages:
        extra = [
            *common,
            "--max-eval-per-size",
            str(args.max_eval_per_size),
            "--states-per-maze",
            str(args.states_per_maze),
        ]
        if args.distance_head_ckpt:
            extra.extend(["--distance-head-ckpt", args.distance_head_ckpt])
        if args.qrl_ckpt:
            extra.extend(["--qrl-ckpt", args.qrl_ckpt])
        run_stage("eval_metric_alignment.py", extra)
    if "rollout" in stages:
        run_stage(
            "eval_predictor_rollout.py",
            [
                *common,
                "--max-eval-per-size",
                str(args.max_eval_per_size),
                "--episodes-per-entry",
                str(args.rollout_episodes_per_entry),
            ],
        )
    if "failure" in stages:
        # Failure taxonomy uses one primary scorer per run. Metric alignment is
        # the stage that compares all scorers side-by-side.
        scorer = "qrl" if args.qrl_ckpt else ("distance_head" if args.distance_head_ckpt else "latent_l2")
        extra = [
            *common,
            "--max-eval-per-size",
            str(args.max_eval_per_size),
            "--scorer",
            scorer,
        ]
        if scorer == "distance_head":
            extra.extend(["--distance-head-ckpt", args.distance_head_ckpt])
        if scorer == "qrl":
            extra.extend(["--qrl-ckpt", args.qrl_ckpt])
        run_stage("eval_failure_taxonomy.py", extra)
    if "report" in stages:
        run_stage(
            "generate_report.py",
            [
                "--run-id",
                args.run_id,
                "--out-dir",
                args.out_dir,
            ],
        )


if __name__ == "__main__":
    main()
