#!/usr/bin/env python3
"""Run the planning-repair experiment plan from a JSON config."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


TRAIN_ARG_MAP = [
    ("steps", "--steps"),
    ("batch_size", "--batch-size"),
    ("seq_len", "--seq-len"),
    ("lr", "--lr"),
    ("weight_decay", "--weight-decay"),
    ("log_every", "--log-every"),
    ("grad_clip", "--grad-clip"),
    ("lambda_sigreg", "--lambda-sigreg"),
    ("lambda_encoded_abs", "--lambda-encoded-abs"),
    ("lambda_encoded_rel", "--lambda-encoded-rel"),
    ("lambda_encoded_goal", "--lambda-encoded-goal"),
    ("lambda_emb_agent", "--lambda-emb-agent"),
    ("lambda_emb_goal", "--lambda-emb-goal"),
    ("lambda_valid", "--lambda-valid"),
    ("lambda_action", "--lambda-action"),
    ("lambda_bfs", "--lambda-bfs"),
    ("lambda_reach", "--lambda-reach"),
    ("lambda_prefix", "--lambda-prefix"),
    ("prefix_horizon", "--prefix-horizon"),
    ("prefix_hidden_dim", "--prefix-hidden-dim"),
    ("prefix_layers", "--prefix-layers"),
    ("prefix_dropout", "--prefix-dropout"),
    ("aux_hidden_dim", "--aux-hidden-dim"),
    ("aux_dropout", "--aux-dropout"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run planning-repair stages.")
    parser.add_argument("--config", default="planning_repair/configs/default.json")
    parser.add_argument(
        "--stages",
        default=(
            "p0,baseline_diagnostics,train_variants,diagnostics_variants,"
            "aux_eval_variants,prefix_rollout_variants,prefix_eval_variants,summary"
        ),
        help=(
            "Comma-separated subset. Single-checkpoint stages: p0,p1,diagnostics,"
            "aux_eval,prefix_eval. Matrix stages: baseline_diagnostics,"
            "train_variants,diagnostics_variants,aux_eval_variants,"
            "prefix_rollout_variants,prefix_eval_variants,summary,full_matrix."
        ),
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


def print_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(shlex.quote(str(part)) for part in cmd), flush=True)


def run_commands(commands: list[list[str]], *, dry_run: bool) -> None:
    for cmd in commands:
        print_cmd(cmd)
        if not dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


def enabled_variants(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        variant
        for variant in cfg.get("variants", [])
        if bool(variant.get("enabled", True))
    ]


def variant_name(variant: dict[str, Any]) -> str:
    name = str(variant.get("name", "")).strip()
    if not name:
        raise ValueError("each variant must have a non-empty name")
    return name


def merged_train_cfg(cfg: dict[str, Any], variant: dict[str, Any] | None = None) -> dict[str, Any]:
    train = dict(cfg.get("train", {}))
    if variant is not None:
        train.update(variant.get("train", {}))
    return train


def variant_ckpt(cfg: dict[str, Any], variant: dict[str, Any]) -> str:
    name = variant_name(variant)
    return str(
        variant.get(
            "repair_ckpt",
            cfg.get("paths", {}).get(
                "variant_ckpt_template",
                "checkpoints/planning_repair/{name}.pt",
            ).format(name=name),
        )
    )


def variant_run_id(variant: dict[str, Any]) -> str:
    return str(variant.get("diagnostics_run_id", f"planning_repair_{variant_name(variant)}"))


def variant_output(variant: dict[str, Any], key: str, subdir: str) -> str:
    outputs = variant.get("outputs", {})
    if key in outputs:
        return str(outputs[key])
    return f"planning_repair_runs/{variant_name(variant)}/{subdir}/results.json"


def loss_enabled(train_cfg: dict[str, Any], key: str) -> bool:
    return float(train_cfg.get(key, 0.0) or 0.0) > 0.0


def build_train_cmd(
    cfg: dict[str, Any],
    *,
    output: str,
    train_cfg: dict[str, Any],
    variant: str,
) -> list[str]:
    p = cfg["paths"]
    cmd = [sys.executable, "planning_repair/train_planning_aligned.py"]
    add_arg(cmd, "--train-manifest", p["train_manifest"])
    add_arg(cmd, "--eval-manifest", p["eval_manifest"])
    add_arg(cmd, "--init-model-ckpt", p.get("baseline_ckpt"))
    add_arg(cmd, "--output", output)
    add_arg(cmd, "--variant-name", variant)
    for key, flag in TRAIN_ARG_MAP:
        if key in train_cfg:
            add_arg(cmd, flag, train_cfg[key])
    add_arg(cmd, "--reach-budgets", train_cfg.get("reach_budgets", "1,3,5,8,12"))
    add_arg(cmd, "--device", cfg.get("device", "cuda"))
    add_arg(cmd, "--seed", cfg.get("seed", 42))
    return cmd


def build_diagnostics_cmd(
    cfg: dict[str, Any],
    *,
    model_ckpt: str,
    run_id: str,
    diagnostics_cfg: dict[str, Any],
) -> list[str]:
    p = cfg["paths"]
    cmd = [sys.executable, "diagnostics/run_all.py"]
    add_arg(cmd, "--model-ckpt", model_ckpt)
    add_arg(cmd, "--train-manifest", p["train_manifest"])
    add_arg(cmd, "--eval-manifest", p["eval_manifest"])
    add_arg(cmd, "--run-id", run_id)
    add_arg(cmd, "--out-dir", diagnostics_cfg.get("out_dir", "diagnostics_runs"))
    add_arg(cmd, "--device", cfg.get("device", "cuda"))
    add_arg(cmd, "--seed", cfg.get("seed", 42))
    for key, flag in [
        ("max_train_per_size", "--max-train-per-size"),
        ("max_eval_per_size", "--max-eval-per-size"),
        ("states_per_maze", "--states-per-maze"),
        ("probe_epochs", "--probe-epochs"),
        ("rollout_episodes_per_entry", "--rollout-episodes-per-entry"),
        ("stages", "--stages"),
    ]:
        if key in diagnostics_cfg:
            add_arg(cmd, flag, diagnostics_cfg[key])
    return cmd


def stage_p0(cfg: dict[str, Any]) -> list[list[str]]:
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
    return [cmd]


def stage_p1(cfg: dict[str, Any]) -> list[list[str]]:
    return [
        build_train_cmd(
            cfg,
            output=cfg["paths"]["repair_ckpt"],
            train_cfg=merged_train_cfg(cfg),
            variant="single_planning_aligned",
        )
    ]


def stage_train_variants(cfg: dict[str, Any]) -> list[list[str]]:
    commands: list[list[str]] = []
    for variant in enabled_variants(cfg):
        commands.append(
            build_train_cmd(
                cfg,
                output=variant_ckpt(cfg, variant),
                train_cfg=merged_train_cfg(cfg, variant),
                variant=variant_name(variant),
            )
        )
    return commands


def stage_baseline_diagnostics(cfg: dict[str, Any]) -> list[list[str]]:
    d = dict(cfg.get("diagnostics", {}))
    d.update(cfg.get("baseline_diagnostics", {}))
    return [
        build_diagnostics_cmd(
            cfg,
            model_ckpt=cfg["paths"]["baseline_ckpt"],
            run_id=d.get("run_id", "planning_repair_baseline"),
            diagnostics_cfg=d,
        )
    ]


def stage_diagnostics(cfg: dict[str, Any]) -> list[list[str]]:
    d = cfg["diagnostics"]
    return [
        build_diagnostics_cmd(
            cfg,
            model_ckpt=cfg["paths"]["repair_ckpt"],
            run_id=d.get("run_id", "planning_repair"),
            diagnostics_cfg=d,
        )
    ]


def stage_diagnostics_variants(cfg: dict[str, Any]) -> list[list[str]]:
    d = cfg["diagnostics"]
    return [
        build_diagnostics_cmd(
            cfg,
            model_ckpt=variant_ckpt(cfg, variant),
            run_id=variant_run_id(variant),
            diagnostics_cfg=d,
        )
        for variant in enabled_variants(cfg)
    ]


def stage_aux_eval(cfg: dict[str, Any]) -> list[list[str]]:
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
    return [cmd]


def stage_aux_eval_variants(cfg: dict[str, Any]) -> list[list[str]]:
    e = cfg["aux_eval"]
    commands: list[list[str]] = []
    for variant in enabled_variants(cfg):
        train_cfg = merged_train_cfg(cfg, variant)
        if not loss_enabled(train_cfg, "lambda_action"):
            continue
        cmd = [sys.executable, "planning_repair/eval_aux_action_head.py"]
        add_arg(cmd, "--manifest", cfg["paths"]["eval_manifest"])
        add_arg(cmd, "--model-ckpt", variant_ckpt(cfg, variant))
        add_arg(cmd, "--output", variant_output(variant, "aux_eval_output", "aux_action_head"))
        add_arg(cmd, "--max-per-size", e.get("max_per_size", 0))
        add_arg(cmd, "--limit", e.get("limit", 0))
        add_arg(cmd, "--device", cfg.get("device", "cuda"))
        add_arg(cmd, "--seed", cfg.get("seed", 42))
        commands.append(cmd)
    return commands


def stage_prefix_eval(cfg: dict[str, Any]) -> list[list[str]]:
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
    return [cmd]


def stage_prefix_eval_variants(cfg: dict[str, Any]) -> list[list[str]]:
    e = cfg["prefix_eval"]
    commands: list[list[str]] = []
    for variant in enabled_variants(cfg):
        train_cfg = merged_train_cfg(cfg, variant)
        if not loss_enabled(train_cfg, "lambda_prefix"):
            continue
        cmd = [sys.executable, "planning_repair/eval_prefix_planner.py"]
        add_arg(cmd, "--manifest", cfg["paths"]["eval_manifest"])
        add_arg(cmd, "--model-ckpt", variant_ckpt(cfg, variant))
        add_arg(cmd, "--output", variant_output(variant, "prefix_eval_output", "prefix_planner"))
        add_arg(cmd, "--horizon", e.get("horizon", 5))
        add_arg(cmd, "--num-candidates", e.get("num_candidates", 128))
        add_arg(cmd, "--terminal-scorer", e.get("terminal_scorer", "latent_l2"))
        add_arg(cmd, "--score-all-prefixes", e.get("score_all_prefixes", False))
        add_arg(cmd, "--max-per-size", e.get("max_per_size", 0))
        add_arg(cmd, "--limit", e.get("limit", 0))
        add_arg(cmd, "--device", cfg.get("device", "cuda"))
        add_arg(cmd, "--seed", cfg.get("seed", 42))
        commands.append(cmd)
    return commands


def stage_prefix_rollout_variants(cfg: dict[str, Any]) -> list[list[str]]:
    e = cfg.get("prefix_rollout", {})
    commands: list[list[str]] = []
    for variant in enabled_variants(cfg):
        train_cfg = merged_train_cfg(cfg, variant)
        if not loss_enabled(train_cfg, "lambda_prefix"):
            continue
        cmd = [sys.executable, "planning_repair/eval_prefix_rollout.py"]
        add_arg(cmd, "--train-manifest", cfg["paths"]["train_manifest"])
        add_arg(cmd, "--eval-manifest", cfg["paths"]["eval_manifest"])
        add_arg(cmd, "--model-ckpt", variant_ckpt(cfg, variant))
        add_arg(cmd, "--output", variant_output(variant, "prefix_rollout_output", "prefix_rollout"))
        add_arg(cmd, "--horizons", e.get("horizons", "1,2,3,5"))
        add_arg(cmd, "--episodes-per-entry", e.get("episodes_per_entry", 1))
        add_arg(cmd, "--max-per-size", e.get("max_per_size", 40))
        add_arg(cmd, "--limit", e.get("limit", 0))
        add_arg(cmd, "--seen-max-size", e.get("seen_max_size", 21))
        add_arg(cmd, "--device", cfg.get("device", "cuda"))
        add_arg(cmd, "--seed", cfg.get("seed", 42))
        commands.append(cmd)
    return commands


def stage_summary(cfg: dict[str, Any]) -> list[list[str]]:
    output = cfg.get("summary", {}).get(
        "output",
        "planning_repair_runs/ablation_summary.md",
    )
    return [
        [
            sys.executable,
            "planning_repair/summarize_ablation.py",
            "--config",
            str(cfg.get("_config_path", "planning_repair/configs/default.json")),
            "--output",
            output,
        ]
    ]


def expand_full_matrix() -> list[str]:
    return [
        "p0",
        "baseline_diagnostics",
        "train_variants",
        "diagnostics_variants",
        "aux_eval_variants",
        "prefix_rollout_variants",
        "prefix_eval_variants",
        "summary",
    ]


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    cfg["_config_path"] = args.config
    builders = {
        "p0": stage_p0,
        "p1": stage_p1,
        "diagnostics": stage_diagnostics,
        "aux_eval": stage_aux_eval,
        "prefix_eval": stage_prefix_eval,
        "baseline_diagnostics": stage_baseline_diagnostics,
        "train_variants": stage_train_variants,
        "diagnostics_variants": stage_diagnostics_variants,
        "aux_eval_variants": stage_aux_eval_variants,
        "prefix_rollout_variants": stage_prefix_rollout_variants,
        "prefix_eval_variants": stage_prefix_eval_variants,
        "summary": stage_summary,
    }
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    expanded: list[str] = []
    for stage in stages:
        expanded.extend(expand_full_matrix() if stage == "full_matrix" else [stage])
    for stage in expanded:
        if stage not in builders:
            raise ValueError(f"unknown stage: {stage}")
        commands = builders[stage](cfg)
        if not commands:
            print(f"\n# stage {stage}: no commands selected", flush=True)
            continue
        run_commands(commands, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
