#!/usr/bin/env python3
"""Run or print the complete, fixed final-closure experiment matrix."""

from __future__ import annotations

import argparse
import random
import shlex
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from final_closure.common import (
    RERUN_REASONS,
    load_checkpoint,
    load_config,
    load_json,
    read_jsonl,
    sha256_file,
)

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
    parser.add_argument("--action-selections", default="")
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
    parser.add_argument("--rerun-reason", choices=RERUN_REASONS, default="")
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
    rerun_reason: str,
    validator: Callable[[], None] | None = None,
) -> None:
    if output and Path(output).exists() and not rerun:
        if validator is not None:
            validator()
        print(f"SKIP validated existing {output}")
        return
    if rerun and "--overwrite" not in command:
        command.extend(["--overwrite", "--rerun-reason", rerun_reason])
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
    config, lock = load_config(args.config)
    requested = parse_names(args.stages)
    requested_full = "full" in requested
    if requested_full:
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
    action_selections = parse_names(args.action_selections)
    allowed_action_selections = {
        str(config["protocol"]["primary_action_selection"]),
        *(str(value) for value in config["protocol"]["diagnostic_action_selections"]),
    }
    if action_selections - allowed_action_selections:
        raise ValueError(
            "unknown action selections: "
            f"{sorted(action_selections - allowed_action_selections)}"
        )
    selected_action_selections = [
        value
        for value in (
            config["protocol"]["primary_action_selection"],
            *config["protocol"]["diagnostic_action_selections"],
        )
        if not action_selections or value in action_selections
    ]
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
    if args.rerun_execution_failures:
        if not args.rerun_reason:
            raise ValueError("objective output replacement requires --rerun-reason")
        if requested_full or len(requested) != 1:
            raise ValueError(
                "objective replacement requires exactly one explicit stage"
            )
        stage = next(iter(requested))
        if stage == "smoke":
            raise ValueError("smoke tests do not create replaceable formal output")
        if stage == "figures":
            raise ValueError("figures are replaced only through the summary stage")
        if stage in {"train", "development", "confirmatory"}:
            if len(baselines) != 1 or len(seeds) != 1:
                raise ValueError(
                    "training/evaluation replacement requires exactly one baseline "
                    "and one seed"
                )
        if (
            stage in {"development", "confirmatory"}
            and len(selected_action_selections) != 1
        ):
            raise ValueError(
                "evaluation replacement requires exactly one --action-selections value"
            )
    elif args.rerun_reason:
        raise ValueError("--rerun-reason requires --rerun-execution-failures")
    device = args.device or config["device"]
    python = sys.executable
    from final_closure.summarize import (
        validate_baseline_result,
        validate_protocol_audit,
        validate_records_against_manifest,
    )
    from final_closure.verify_closure import verify_closure_gate

    def validate_audit(path: str) -> None:
        validate_protocol_audit(load_json(path), config=config, lock=lock)

    def validate_checkpoint(path: str, baseline: dict[str, Any], seed: int) -> None:
        checkpoint = load_checkpoint(
            path,
            config=config,
            lock=lock,
            name=str(baseline["name"]),
            seed=seed,
            strict_provenance=True,
        )
        if checkpoint.get("formal_run") is not True:
            raise ValueError(f"existing checkpoint is diagnostic: {path}")
        if checkpoint.get("training_config") != baseline["train"]:
            raise ValueError(f"existing checkpoint training config is stale: {path}")

    def validate_result(
        path: str,
        baseline: dict[str, Any],
        seed: int,
        split_role: str,
        action_selection: str,
    ) -> None:
        data = load_json(path)
        record = validate_baseline_result(
            data,
            config=config,
            lock=lock,
            baseline=baseline,
            seed=seed,
            split_role=split_role,
            action_selection=action_selection,
        )
        entries = read_jsonl(config["paths"][f"{split_role}_manifest"])
        validate_records_against_manifest({str(baseline["name"]): [record]}, entries)
        checkpoint_path = format_path(
            config["paths"]["checkpoint_template"],
            name=baseline["name"],
            seed=seed,
        )
        validate_checkpoint(checkpoint_path, baseline, seed)
        if data["metadata"].get("checkpoint_sha256") != sha256_file(checkpoint_path):
            raise ValueError(f"existing result references the wrong checkpoint: {path}")

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
            rerun_reason=args.rerun_reason,
            validator=partial(validate_audit, output),
        )
    if "smoke" in requested:
        execute(
            [python, "-m", "final_closure.smoke_test"],
            output=None,
            dry_run=args.dry_run,
            rerun=False,
            rerun_reason="",
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
                rerun_reason=args.rerun_reason,
                validator=partial(validate_checkpoint, output, baseline, seed),
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
            for action_selection in selected_action_selections:
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
                    rerun_reason=args.rerun_reason,
                    validator=partial(
                        validate_result,
                        output,
                        baseline,
                        seed,
                        split_role,
                        action_selection,
                    ),
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
            rerun_reason=args.rerun_reason,
            validator=partial(verify_closure_gate, args.config),
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
            rerun_reason=args.rerun_reason,
            validator=(
                partial(verify_closure_gate, args.config)
                if Path(config["paths"]["closure_gate"]).exists()
                else None
            ),
        )


if __name__ == "__main__":
    main()
