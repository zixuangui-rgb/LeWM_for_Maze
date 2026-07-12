#!/usr/bin/env python3
"""Run or print the complete, fixed final-closure experiment matrix."""

from __future__ import annotations

import argparse
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from final_closure.common import load_config

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="final_closure/configs/default.json")
    parser.add_argument(
        "--stages",
        default="full",
        help=(
            "Comma-separated: audit,smoke,train,development,confirmatory,"
            "summary,figures,full"
        ),
    )
    parser.add_argument("--baselines", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rerun-execution-failures",
        action="store_true",
        help=(
            "Overwrite existing outputs only after documenting an objective "
            "run failure."
        ),
    )
    return parser.parse_args()


def format_path(template: str, **values: Any) -> str:
    return template.format(**values)


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def execute(
    command: list[str],
    *,
    output: str | None,
    dry_run: bool,
    rerun: bool,
) -> None:
    if output and Path(output).exists() and not rerun:
        print(f"SKIP existing {output}")
        return
    if rerun and "--overwrite" not in command:
        command.append("--overwrite")
    print(command_text(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def parse_names(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_seeds(value: str, defaults: list[int]) -> list[int]:
    if not value.strip():
        return defaults
    selected = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(set(selected)) != len(selected):
        raise ValueError("seed override contains duplicates")
    unknown = set(selected) - set(defaults)
    if unknown:
        raise ValueError(
            f"seed override is outside the locked matrix: {sorted(unknown)}"
        )
    return selected


def randomized_jobs(
    baselines: list[dict[str, Any]],
    seeds: list[int],
    *,
    run_order_seed: int,
) -> list[tuple[dict[str, Any], int]]:
    jobs = [(baseline, seed) for baseline in baselines for seed in seeds]
    random.Random(run_order_seed).shuffle(jobs)
    return jobs


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)
    requested = parse_names(args.stages)
    if "full" in requested:
        requested = {
            "audit",
            "smoke",
            "train",
            "development",
            "confirmatory",
            "summary",
            "figures",
        }
    valid = {
        "audit",
        "smoke",
        "train",
        "development",
        "confirmatory",
        "summary",
        "figures",
    }
    unknown = requested - valid
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    selected_names = parse_names(args.baselines)
    known_names = {str(item["name"]) for item in config["baselines"]}
    if selected_names - known_names:
        raise ValueError(f"unknown baselines: {sorted(selected_names - known_names)}")
    baselines = [
        item
        for item in config["baselines"]
        if not selected_names or item["name"] in selected_names
    ]
    seeds = parse_seeds(args.seeds, [int(value) for value in config["seeds"]])
    jobs = randomized_jobs(
        baselines,
        seeds,
        run_order_seed=int(config["protocol"]["run_order_seed"]),
    )
    if requested & {"summary", "figures"}:
        if seeds != [int(value) for value in config["seeds"]]:
            raise ValueError("summary/figures require the complete locked seed matrix")
        if selected_names and selected_names != known_names:
            raise ValueError("summary/figures require both fixed baselines")
    device = args.device or config["device"]
    python = sys.executable
    if "audit" in requested:
        output = config["paths"]["audit_output"]
        execute(
            [
                python,
                "-m",
                "final_closure.audit_protocol",
                "--config",
                args.config,
                "--output",
                output,
            ],
            output=output,
            dry_run=args.dry_run,
            rerun=args.rerun_execution_failures,
        )
    if "smoke" in requested:
        execute(
            [python, "-m", "final_closure.smoke_test"],
            output=None,
            dry_run=args.dry_run,
            rerun=False,
        )
    if "train" in requested:
        for baseline, seed in jobs:
            output = format_path(
                config["paths"]["checkpoint_template"],
                name=baseline["name"],
                seed=seed,
            )
            execute(
                [
                    python,
                    "-m",
                    "final_closure.train",
                    "--config",
                    args.config,
                    "--baseline",
                    baseline["name"],
                    "--seed",
                    str(seed),
                    "--output",
                    output,
                    "--device",
                    device,
                ],
                output=output,
                dry_run=args.dry_run,
                rerun=args.rerun_execution_failures,
            )
    for split_role in ("development", "confirmatory"):
        if split_role not in requested:
            continue
        result_template = config["paths"][f"{split_role}_result_template"]
        for baseline, seed in jobs:
            checkpoint = format_path(
                config["paths"]["checkpoint_template"],
                name=baseline["name"],
                seed=seed,
            )
            for action_selection in (
                config["protocol"]["primary_action_selection"],
                *config["protocol"]["diagnostic_action_selections"],
            ):
                output = format_path(
                    result_template,
                    name=baseline["name"],
                    seed=seed,
                    action_selection=action_selection,
                )
                execute(
                    [
                        python,
                        "-m",
                        "final_closure.evaluate",
                        "--config",
                        args.config,
                        "--baseline",
                        baseline["name"],
                        "--training-seed",
                        str(seed),
                        "--checkpoint",
                        checkpoint,
                        "--split-role",
                        split_role,
                        "--action-selection",
                        action_selection,
                        "--output",
                        output,
                        "--device",
                        device,
                    ],
                    output=output,
                    dry_run=args.dry_run,
                    rerun=args.rerun_execution_failures,
                )
    if "summary" in requested:
        output = config["paths"]["closure_gate"]
        execute(
            [
                python,
                "-m",
                "final_closure.summarize",
                "--config",
                args.config,
            ],
            output=output,
            dry_run=args.dry_run,
            rerun=args.rerun_execution_failures,
        )
    if "figures" in requested:
        execute(
            [
                python,
                "-m",
                "final_closure.plot_results",
                "--config",
                args.config,
            ],
            output=str(Path(config["paths"]["figure_dir"]) / "primary_results.png"),
            dry_run=args.dry_run,
            rerun=args.rerun_execution_failures,
        )


if __name__ == "__main__":
    main()
