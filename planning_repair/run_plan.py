#!/usr/bin/env python3
"""Run the planning-repair experiment plan from a JSON config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run planning-repair stages.")
    parser.add_argument("--config", default="planning_repair/configs/default.json")
    parser.add_argument(
        "--stages",
        default="p0,p1,diagnostics,aux_eval,prefix_eval",
        help="Comma-separated subset: p0,p1,diagnostics,aux_eval,prefix_eval",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def add_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def stage_p0(cfg: dict[str, Any]) -> list[str]:
    p = cfg["paths"]
    e = cfg["p0_receding"]
    cmd = [sys.executable, "planning_repair/eval_b2_receding.py"]
    add_arg(cmd, "--manifest", p["eval_manifest"])
    add_arg(cmd, "--model-ckpt", p["baseline_ckpt"])
    add_arg(cmd, "--output", p["p0_output"])
    add_arg(cmd, "--scorers", e.get("scorers", "latent_l2"))
    add_arg(cmd, "--horizons", e.get("horizons", "3,5,8,12"))
    add_arg(cmd, "--num-candidates", e.get("num_candidates", 64))
    add_arg(cmd, "--cem-iters", e.get("cem_iters", 1))
    add_arg(cmd, "--max-per-size", e.get("max_per_size", 0))
    add_arg(cmd, "--limit", e.get("limit", 0))
    add_arg(cmd, "--device", cfg.get("device", "cuda"))
    add_arg(cmd, "--seed", cfg.get("seed", 42))
    if e.get("distance_head_ckpt"):
        add_arg(cmd, "--distance-head-ckpt", e["distance_head_ckpt"])
    return cmd


def stage_p1(cfg: dict[str, Any]) -> list[str]:
    p = cfg["paths"]
    t = cfg["train"]
    cmd = [sys.executable, "planning_repair/train_planning_aligned.py"]
    add_arg(cmd, "--train-manifest", p["train_manifest"])
    add_arg(cmd, "--eval-manifest", p["eval_manifest"])
    add_arg(cmd, "--init-model-ckpt", p.get("baseline_ckpt"))
    add_arg(cmd, "--output", p["repair_ckpt"])
    for key, flag in [
        ("steps", "--steps"),
        ("batch_size", "--batch-size"),
        ("seq_len", "--seq-len"),
        ("lr", "--lr"),
        ("weight_decay", "--weight-decay"),
        ("log_every", "--log-every"),
        ("lambda_emb_agent", "--lambda-emb-agent"),
        ("lambda_emb_goal", "--lambda-emb-goal"),
        ("lambda_valid", "--lambda-valid"),
        ("lambda_action", "--lambda-action"),
        ("lambda_bfs", "--lambda-bfs"),
        ("lambda_reach", "--lambda-reach"),
        ("lambda_prefix", "--lambda-prefix"),
        ("prefix_horizon", "--prefix-horizon"),
    ]:
        if key in t:
            add_arg(cmd, flag, t[key])
    add_arg(cmd, "--reach-budgets", t.get("reach_budgets", "1,3,5,8,12"))
    add_arg(cmd, "--device", cfg.get("device", "cuda"))
    add_arg(cmd, "--seed", cfg.get("seed", 42))
    return cmd


def stage_diagnostics(cfg: dict[str, Any]) -> list[str]:
    p = cfg["paths"]
    d = cfg["diagnostics"]
    cmd = [sys.executable, "diagnostics/run_all.py"]
    add_arg(cmd, "--model-ckpt", p["repair_ckpt"])
    add_arg(cmd, "--train-manifest", p["train_manifest"])
    add_arg(cmd, "--eval-manifest", p["eval_manifest"])
    add_arg(cmd, "--run-id", d.get("run_id", "planning_repair"))
    add_arg(cmd, "--out-dir", d.get("out_dir", "diagnostics_runs"))
    add_arg(cmd, "--device", cfg.get("device", "cuda"))
    add_arg(cmd, "--seed", cfg.get("seed", 42))
    for key, flag in [
        ("max_train_per_size", "--max-train-per-size"),
        ("max_eval_per_size", "--max-eval-per-size"),
        ("states_per_maze", "--states-per-maze"),
        ("probe_epochs", "--probe-epochs"),
    ]:
        if key in d:
            add_arg(cmd, flag, d[key])
    return cmd


def stage_aux_eval(cfg: dict[str, Any]) -> list[str]:
    p = cfg["paths"]
    e = cfg["aux_eval"]
    cmd = [sys.executable, "planning_repair/eval_aux_action_head.py"]
    add_arg(cmd, "--manifest", p["eval_manifest"])
    add_arg(cmd, "--model-ckpt", p["repair_ckpt"])
    add_arg(cmd, "--output", p["aux_eval_output"])
    add_arg(cmd, "--max-per-size", e.get("max_per_size", 0))
    add_arg(cmd, "--limit", e.get("limit", 0))
    add_arg(cmd, "--device", cfg.get("device", "cuda"))
    add_arg(cmd, "--seed", cfg.get("seed", 42))
    return cmd


def stage_prefix_eval(cfg: dict[str, Any]) -> list[str]:
    p = cfg["paths"]
    e = cfg["prefix_eval"]
    cmd = [sys.executable, "planning_repair/eval_prefix_planner.py"]
    add_arg(cmd, "--manifest", p["eval_manifest"])
    add_arg(cmd, "--model-ckpt", p["repair_ckpt"])
    add_arg(cmd, "--output", p["prefix_eval_output"])
    add_arg(cmd, "--horizon", e.get("horizon", 5))
    add_arg(cmd, "--num-candidates", e.get("num_candidates", 128))
    add_arg(cmd, "--terminal-scorer", e.get("terminal_scorer", "latent_l2"))
    add_arg(cmd, "--score-all-prefixes", e.get("score_all_prefixes", False))
    add_arg(cmd, "--max-per-size", e.get("max_per_size", 0))
    add_arg(cmd, "--limit", e.get("limit", 0))
    add_arg(cmd, "--device", cfg.get("device", "cuda"))
    add_arg(cmd, "--seed", cfg.get("seed", 42))
    return cmd


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    builders = {
        "p0": stage_p0,
        "p1": stage_p1,
        "diagnostics": stage_diagnostics,
        "aux_eval": stage_aux_eval,
        "prefix_eval": stage_prefix_eval,
    }
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    for stage in stages:
        if stage not in builders:
            raise ValueError(f"unknown stage: {stage}")
        run(builders[stage](cfg), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

