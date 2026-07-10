#!/usr/bin/env python3
"""Config-driven orchestration for the complete spatial-planning matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


TRAIN_FLAGS = {
    "stage": "--stage",
    "input_mode": "--input-mode",
    "encoder_mode": "--encoder-mode",
    "steps": "--steps",
    "map_batch_size": "--map-batch-size",
    "trajectories_per_map": "--trajectories-per-map",
    "seq_len": "--seq-len",
    "lr": "--lr",
    "encoder_lr_multiplier": "--encoder-lr-multiplier",
    "weight_decay": "--weight-decay",
    "scheduler": "--scheduler",
    "grad_clip": "--grad-clip",
    "log_every": "--log-every",
    "spatial_dim": "--spatial-dim",
    "planning_dim": "--planning-dim",
    "encoder_blocks": "--encoder-blocks",
    "predictor_blocks": "--predictor-blocks",
    "ema_momentum": "--ema-momentum",
    "sigreg_num_proj": "--sigreg-num-proj",
    "sigreg_max_tokens": "--sigreg-max-tokens",
    "lambda_prediction": "--lambda-prediction",
    "lambda_sigreg": "--lambda-sigreg",
    "lambda_variance": "--lambda-variance",
    "lambda_covariance": "--lambda-covariance",
    "lambda_map_wall": "--lambda-map-wall",
    "lambda_map_agent": "--lambda-map-agent",
    "lambda_map_goal": "--lambda-map-goal",
    "lambda_map_valid": "--lambda-map-valid",
    "planner_type": "--planner-type",
    "planner_hidden_dim": "--planner-hidden-dim",
    "planner_depth": "--planner-depth",
    "train_iterations": "--train-iterations",
    "iteration_schedule": "--iteration-schedule",
    "deep_supervision_every": "--deep-supervision-every",
    "distance_scale": "--distance-scale",
    "lambda_value": "--lambda-value",
    "lambda_action": "--lambda-action",
    "lambda_valid": "--lambda-valid",
    "lambda_bellman": "--lambda-bellman",
    "lambda_gap": "--lambda-gap",
    "lambda_convergence": "--lambda-convergence",
    "gap_margin": "--gap-margin",
    "lambda_joint_representation": "--lambda-joint-representation",
    "lambda_planner_map": "--lambda-planner-map",
    "gradient_audit_every": "--gradient-audit-every",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="spatial_jepa_planning/configs/default.json"
    )
    parser.add_argument(
        "--stages",
        default="audit,anchors,train_representations,eval_representations,train_planners,eval_planners,summary",
        help=(
            "Comma-separated: audit,smoke,train_representations,eval_representations,"
            "train_planners,anchors,eval_planners,summary,full"
        ),
    )
    parser.add_argument(
        "--variants", default="", help="Optional comma-separated names."
    )
    parser.add_argument(
        "--seeds", default="", help="Optional comma-separated seed override."
    )
    parser.add_argument(
        "--device",
        default="",
        help="Override config device (auto, cpu, cuda, cuda:N, or mps).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_config(path: str | Path) -> dict[str, Any]:
    with open(path) as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def add_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def format_path(template: str, *, name: str, seed: int) -> str:
    return template.format(name=name, seed=seed)


def selected_items(
    items: list[dict[str, Any]],
    names: set[str],
) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if item.get("enabled", True) and (not names or str(item["name"]) in names)
    ]


def merged(defaults: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    output = dict(defaults)
    output.update(variant.get("train", {}))
    return output


def build_train_command(
    config: dict[str, Any],
    train_config: dict[str, Any],
    *,
    variant_name: str,
    output: str,
    seed: int,
    representation_ckpt: str | None = None,
) -> list[str]:
    paths = config["paths"]
    protocol = config["protocol"]
    command = [sys.executable, "spatial_jepa_planning/train.py"]
    add_arg(command, "--variant-name", variant_name)
    add_arg(command, "--train-manifest", paths["train_manifest"])
    add_arg(command, "--eval-manifest", paths["eval_manifest"])
    add_arg(command, "--output", output)
    add_arg(command, "--representation-ckpt", representation_ckpt)
    add_arg(command, "--seed", seed)
    add_arg(command, "--device", config["device"])
    add_arg(command, "--max-steps", protocol["max_steps"])
    for key, flag in TRAIN_FLAGS.items():
        if key in train_config:
            add_arg(command, flag, train_config[key])
    if train_config.get("recall") is False:
        command.append("--no-recall")
    if train_config.get("deterministic", True):
        command.append("--deterministic")
    return command


def build_eval_command(
    config: dict[str, Any],
    *,
    mode: str,
    output: str,
    training_seed: int | None,
    planner_ckpt: str | None = None,
    representation_ckpt: str | None = None,
    decision_source: str = "policy",
    iterations: str | None = None,
) -> list[str]:
    paths = config["paths"]
    protocol = config["protocol"]
    evaluation = config["evaluation"]
    command = [sys.executable, "spatial_jepa_planning/evaluate.py"]
    add_arg(command, "--mode", mode)
    add_arg(command, "--planner-ckpt", planner_ckpt)
    add_arg(command, "--representation-ckpt", representation_ckpt)
    add_arg(command, "--train-manifest", paths["train_manifest"])
    add_arg(command, "--manifest", paths["eval_manifest"])
    add_arg(command, "--output", output)
    add_arg(command, "--iterations", iterations or evaluation["iterations"])
    add_arg(command, "--decision-source", decision_source)
    add_arg(command, "--action-selection", protocol["action_selection"])
    add_arg(command, "--max-steps", protocol["max_steps"])
    add_arg(command, "--max-per-size", protocol.get("eval_max_per_size", 0))
    add_arg(command, "--limit", protocol.get("eval_limit", 0))
    add_arg(command, "--seen-max-size", protocol["seen_max_size"])
    add_arg(command, "--progress-every", evaluation.get("progress_every", 100))
    add_arg(
        command,
        "--field-states-per-maze",
        evaluation.get("field_states_per_maze", 24),
    )
    add_arg(
        command,
        "--field-pairs-per-maze",
        evaluation.get("field_pairs_per_maze", 128),
    )
    add_arg(
        command,
        "--decoded-action-selection",
        evaluation.get("decoded_action_selection", "predicted"),
    )
    add_arg(command, "--seed", evaluation.get("seed", 42))
    add_arg(command, "--training-seed", training_seed)
    add_arg(command, "--device", config["device"])
    return command


def stage_commands(
    stage: str,
    config: dict[str, Any],
    config_path: str,
    names: set[str],
    seeds: list[int],
) -> list[list[str]]:
    paths = config["paths"]
    representations = selected_items(config.get("representations", []), names)
    planners = selected_items(config.get("planners", []), names)
    if stage == "audit":
        return [
            [
                sys.executable,
                "spatial_jepa_planning/audit_protocol.py",
                "--config",
                config_path,
                "--protocol-lock",
                paths["protocol_lock"],
                "--output",
                paths["audit_output"],
            ]
        ]
    if stage == "smoke":
        return [[sys.executable, "spatial_jepa_planning/smoke_test.py"]]
    if stage == "train_representations":
        commands: list[list[str]] = []
        for variant in representations:
            train_config = merged(config["representation_defaults"], variant)
            for seed in seeds:
                output = format_path(
                    paths["representation_ckpt_template"],
                    name=str(variant["name"]),
                    seed=seed,
                )
                commands.append(
                    build_train_command(
                        config,
                        train_config,
                        variant_name=str(variant["name"]),
                        output=output,
                        seed=seed,
                    )
                )
        return commands
    if stage == "eval_representations":
        commands = []
        for variant in representations:
            for seed in seeds:
                checkpoint = format_path(
                    paths["representation_ckpt_template"],
                    name=str(variant["name"]),
                    seed=seed,
                )
                output = format_path(
                    paths["representation_eval_template"],
                    name=str(variant["name"]),
                    seed=seed,
                )
                commands.append(
                    build_eval_command(
                        config,
                        mode="decoded_bfs",
                        representation_ckpt=checkpoint,
                        output=output,
                        training_seed=seed,
                    )
                )
        return commands
    if stage == "train_planners":
        commands = []
        rep_defaults = config["representation_defaults"]
        for variant in planners:
            train_config = dict(rep_defaults)
            train_config.update(config["planner_defaults"])
            train_config.update(variant.get("train", {}))
            train_config["input_mode"] = variant["input_mode"]
            for seed in seeds:
                representation_ckpt = None
                if variant["input_mode"] == "spatial_jepa":
                    representation_ckpt = format_path(
                        paths["representation_ckpt_template"],
                        name=str(variant["representation"]),
                        seed=seed,
                    )
                output = format_path(
                    paths["planner_ckpt_template"],
                    name=str(variant["name"]),
                    seed=seed,
                )
                commands.append(
                    build_train_command(
                        config,
                        train_config,
                        variant_name=str(variant["name"]),
                        output=output,
                        seed=seed,
                        representation_ckpt=representation_ckpt,
                    )
                )
        return commands
    if stage == "anchors":
        return [
            build_eval_command(
                config,
                mode="oracle_bfs",
                output=paths["oracle_bfs_output"],
                training_seed=None,
            ),
            build_eval_command(
                config,
                mode="oracle_vi",
                output=paths["oracle_vi_output"],
                training_seed=None,
                iterations=config["evaluation"]["oracle_vi_iterations"],
            ),
        ]
    if stage == "eval_planners":
        commands = []
        for variant in planners:
            for seed in seeds:
                checkpoint = format_path(
                    paths["planner_ckpt_template"],
                    name=str(variant["name"]),
                    seed=seed,
                )
                output = format_path(
                    paths["planner_eval_template"],
                    name=str(variant["name"]),
                    seed=seed,
                )
                commands.append(
                    build_eval_command(
                        config,
                        mode="learned",
                        planner_ckpt=checkpoint,
                        output=output,
                        training_seed=seed,
                        decision_source=str(variant["decision_source"]),
                    )
                )
        return commands
    if stage == "summary":
        return [
            [
                sys.executable,
                "spatial_jepa_planning/summarize.py",
                "--config",
                config_path,
                "--output",
                paths["summary_output"],
            ]
        ]
    raise ValueError(f"unknown stage: {stage}")


def print_command(command: list[str]) -> None:
    print("\n$ " + " ".join(shlex.quote(part) for part in command), flush=True)


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    if args.device:
        config["device"] = args.device
    names = {item.strip() for item in args.variants.split(",") if item.strip()}
    known_names = {
        str(item["name"])
        for section in ("representations", "planners")
        for item in config.get(section, [])
        if item.get("enabled", True)
    }
    unknown_names = sorted(names - known_names)
    if unknown_names:
        raise ValueError(f"unknown or disabled variants: {unknown_names}")
    seeds = (
        [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        if args.seeds
        else [int(seed) for seed in config["seeds"]]
    )
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "seed override contains duplicates and would overwrite outputs"
        )
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    expanded: list[str] = []
    full = [
        "audit",
        "anchors",
        "train_representations",
        "eval_representations",
        "train_planners",
        "eval_planners",
        "summary",
    ]
    for stage in stages:
        expanded.extend(full if stage == "full" else [stage])
    for stage in expanded:
        commands = stage_commands(stage, config, args.config, names, seeds)
        if not commands:
            print(f"\n# stage {stage}: no selected commands")
            continue
        for command in commands:
            print_command(command)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
