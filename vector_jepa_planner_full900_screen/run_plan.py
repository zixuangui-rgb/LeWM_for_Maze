"""Deterministic stage scheduler for the full-900 screening protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vector_jepa_planner_full900_screen.common import (
    load_config,
    load_json,
    planner_seeds,
    resolve_path,
    result_path,
    validate_lock,
)
from vector_jepa_planner_full900_screen.methods import (
    direct_control_name,
    effective_method,
    validate_final_selection,
    validate_q1_selection,
    validate_shortlist,
)
from vector_jepa_planner_full900_screen.parity import validate_q0_gate


@dataclass(frozen=True)
class Job:
    label: str
    command: tuple[str, ...]
    output: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="vector_jepa_planner_full900_screen/configs/default.json",
    )
    parser.add_argument(
        "--stage",
        choices=("audit", "Q0", "Q1", "Q2A", "Q2B", "Q2C", "Q3", "Q4"),
        required=True,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-missing", action="store_true")
    parser.add_argument("--device")
    return parser.parse_args()


def module_command(module: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, "-m", module, *arguments)


def _device_arguments(device: str | None) -> tuple[str, ...]:
    return ("--device", device) if device else ()


def _component_jobs(
    config: Any,
    lock: dict[str, Any],
    method: Any,
    *,
    backbone_seed: int,
    planner_seed: int,
    config_path: str,
    device: str | None,
) -> list[Job]:
    if not method.component_checkpoint_required:
        return []
    train_output = resolve_path(
        config.paths.component_training_template.format(
            method=method.name,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
        )
    )
    calibrated_output = resolve_path(
        config.paths.component_checkpoint_template.format(
            method=method.name,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
        )
    )
    jobs = [
        Job(
            label=f"train:{method.name}:b{backbone_seed}:p{planner_seed}",
            command=module_command(
                "vector_jepa_planner_full900_screen.train",
                "--config",
                config_path,
                "--method",
                method.name,
                "--backbone-seed",
                str(backbone_seed),
                "--planner-seed",
                str(planner_seed),
                *_device_arguments(device),
            ),
            output=train_output,
        ),
        Job(
            label=f"calibrate:{method.name}:b{backbone_seed}:p{planner_seed}",
            command=module_command(
                "vector_jepa_planner_full900_screen.calibrate",
                "--config",
                config_path,
                "--method",
                method.name,
                "--backbone-seed",
                str(backbone_seed),
                "--planner-seed",
                str(planner_seed),
                *_device_arguments(device),
            ),
            output=calibrated_output,
        ),
    ]
    if method.stage == "P6" and method.scorer.counterexample_ranker_weight > 0.0:
        for round_index in range(1, config.training.counterexample_rounds + 1):
            output = resolve_path(
                config.paths.counterexample_round_template.format(
                    method=method.name,
                    backbone_seed=backbone_seed,
                    planner_seed=planner_seed,
                    round=round_index,
                )
            )
            jobs.append(
                Job(
                    label=(
                        f"counterexample:{method.name}:b{backbone_seed}:"
                        f"p{planner_seed}:r{round_index}"
                    ),
                    command=module_command(
                        "vector_jepa_planner_full900_screen.counterexamples",
                        "--config",
                        config_path,
                        "--method",
                        method.name,
                        "--backbone-seed",
                        str(backbone_seed),
                        "--planner-seed",
                        str(planner_seed),
                        "--round",
                        str(round_index),
                        *_device_arguments(device),
                    ),
                    output=output,
                )
            )
    return jobs


def _evaluation_jobs(
    config: Any,
    method: Any,
    *,
    backbone_seed: int,
    planner_seed: int,
    config_path: str,
    device: str | None,
) -> list[Job]:
    jobs = []
    for action in config.replication.action_selections:
        output = result_path(
            config,
            method=method.name,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
            action_selection=action,
        )
        jobs.append(
            Job(
                label=(f"eval:{method.name}:b{backbone_seed}:p{planner_seed}:{action}"),
                command=module_command(
                    "vector_jepa_planner_full900_screen.evaluate",
                    "--config",
                    config_path,
                    "--method",
                    method.name,
                    "--backbone-seed",
                    str(backbone_seed),
                    "--planner-seed",
                    str(planner_seed),
                    "--action-selection",
                    action,
                    *_device_arguments(device),
                ),
                output=output,
            )
        )
    return jobs


def _method_block(
    config: Any,
    lock: dict[str, Any],
    method_name: str,
    *,
    backbone_seed: int,
    planner_seed: int,
    config_path: str,
    device: str | None,
) -> list[Job]:
    method = effective_method(config, lock, method_name)
    return [
        *_component_jobs(
            config,
            lock,
            method,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
            config_path=config_path,
            device=device,
        ),
        *_evaluation_jobs(
            config,
            method,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
            config_path=config_path,
            device=device,
        ),
    ]


def _q0_jobs(config: Any, config_path: str, device: str | None) -> list[Job]:
    jobs: list[Job] = []
    for quick_action, reference_action in (
        ("corrected_v1", "corrected"),
        ("unmasked", "unmasked"),
    ):
        reference = (
            resolve_path(config.paths.run_root)
            / "parity"
            / (f"reference_seed42_{reference_action}.json")
        )
        candidate = result_path(
            config,
            method="b0_legacy_l2_cem",
            backbone_seed=42,
            planner_seed=0,
            action_selection=quick_action,
        )
        parity = (
            resolve_path(config.paths.run_root)
            / "parity"
            / (f"parity_{quick_action}.json")
        )
        jobs.extend(
            [
                Job(
                    label=f"q0-reference:{reference_action}",
                    command=module_command(
                        "vector_jepa_planner_full900_screen.reference_evaluate",
                        "--config",
                        config_path,
                        "--action-selection",
                        reference_action,
                        "--output",
                        str(reference),
                        *_device_arguments(device),
                    ),
                    output=reference,
                ),
                Job(
                    label=f"q0-candidate:{quick_action}",
                    command=module_command(
                        "vector_jepa_planner_full900_screen.evaluate",
                        "--config",
                        config_path,
                        "--method",
                        "b0_legacy_l2_cem",
                        "--backbone-seed",
                        "42",
                        "--planner-seed",
                        "0",
                        "--action-selection",
                        quick_action,
                        *_device_arguments(device),
                    ),
                    output=candidate,
                ),
                Job(
                    label=f"q0-parity:{quick_action}",
                    command=module_command(
                        "vector_jepa_planner_full900_screen.parity",
                        "--config",
                        config_path,
                        "--action-selection",
                        quick_action,
                        "--reference",
                        str(reference),
                        "--candidate",
                        str(candidate),
                        "--output",
                        str(parity),
                    ),
                    output=parity,
                ),
            ]
        )
    return jobs


def _phase_names(config: Any, phase: str) -> list[str]:
    return [role.name for role in config.method_roles if role.phase == phase]


def _decision_methods(config: Any, lock: dict[str, Any], key: str) -> list[str]:
    if key == "p5_advancement":
        value = validate_shortlist(config, lock)
        selected = list(value.get("shortlist", []))
    elif key == "p7_selection":
        value = validate_final_selection(config, lock)
        winner = value.get("winner")
        selected = [str(winner)] if winner is not None else []
    else:
        raise ValueError(f"unsupported decision key: {key}")
    names = ["b0_legacy_l2_cem", *selected]
    for name in selected:
        control = direct_control_name(config, lock, name)
        names.append(control)
    return list(dict.fromkeys(names))


def _randomized_blocks(
    config: Any,
    lock: dict[str, Any],
    *,
    names: list[str],
    backbone_seeds: tuple[int, ...],
    final_planner_seeds: bool,
    config_path: str,
    device: str | None,
    namespace: int,
) -> list[Job]:
    rng = np.random.default_rng(config.protocol.run_order_seed + namespace)
    jobs: list[Job] = []
    for backbone_seed in backbone_seeds:
        blocks: list[list[Job]] = []
        shuffled = list(names)
        rng.shuffle(shuffled)
        for name in shuffled:
            method = effective_method(config, lock, name)
            for planner_seed in planner_seeds(
                config, method, final=final_planner_seeds
            ):
                blocks.append(
                    _method_block(
                        config,
                        lock,
                        name,
                        backbone_seed=backbone_seed,
                        planner_seed=planner_seed,
                        config_path=config_path,
                        device=device,
                    )
                )
        rng.shuffle(blocks)
        for block in blocks:
            jobs.extend(block)
    return jobs


def stage_jobs(
    config: Any,
    lock: dict[str, Any],
    *,
    stage: str,
    config_path: str,
    device: str | None,
) -> list[Job]:
    if stage == "Q0":
        return _q0_jobs(config, config_path, device)
    validate_q0_gate(config, lock)
    if stage == "Q1":
        names = _phase_names(config, "Q1")
        return _randomized_blocks(
            config,
            lock,
            names=names,
            backbone_seeds=(42,),
            final_planner_seeds=False,
            config_path=config_path,
            device=device,
            namespace=1,
        )
    validate_q1_selection(config, lock)
    if stage in {"Q2A", "Q2B", "Q2C"}:
        names = _phase_names(config, stage)
        return _randomized_blocks(
            config,
            lock,
            names=names,
            backbone_seeds=(42,),
            final_planner_seeds=False,
            config_path=config_path,
            device=device,
            namespace={"Q2A": 2, "Q2B": 3, "Q2C": 4}[stage],
        )
    if stage == "Q3":
        names = _decision_methods(config, lock, "p5_advancement")
        if names == ["b0_legacy_l2_cem"]:
            return []
        new_seeds = tuple(config.replication.expansion_backbone_seeds[1:])
        return _randomized_blocks(
            config,
            lock,
            names=names,
            backbone_seeds=new_seeds,
            final_planner_seeds=False,
            config_path=config_path,
            device=device,
            namespace=5,
        )
    if stage == "Q4":
        names = _decision_methods(config, lock, "p7_selection")
        if names == ["b0_legacy_l2_cem"]:
            return []
        winner = effective_method(config, lock, names[1])
        direct = effective_method(
            config,
            lock,
            direct_control_name(config, lock, winner.name),
        )
        new_seeds = tuple(config.replication.final_backbone_seeds[3:])
        jobs = _randomized_blocks(
            config,
            lock,
            names=names,
            backbone_seeds=new_seeds,
            final_planner_seeds=False,
            config_path=config_path,
            device=device,
            namespace=6,
        )
        second_seed_names = [
            method.name
            for method in (winner, direct)
            if method.component_checkpoint_required
        ]
        for name in dict.fromkeys(second_seed_names):
            method = effective_method(config, lock, name)
            second_seed = config.replication.final_planner_seeds[1]
            for backbone_seed in config.replication.final_backbone_seeds:
                jobs.extend(
                    _method_block(
                        config,
                        lock,
                        method.name,
                        backbone_seed=backbone_seed,
                        planner_seed=second_seed,
                        config_path=config_path,
                        device=device,
                    )
                )
        return jobs
    raise ValueError(f"unknown stage: {stage}")


def schedule_text(
    config: Any, lock: dict[str, Any], stage: str, jobs: list[Job]
) -> str:
    stream = io.StringIO(newline="")
    fields = (
        "protocol_id",
        "quick_spec_sha256",
        "stage",
        "order",
        "label",
        "output",
        "command_sha256",
    )
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for order, job in enumerate(jobs, start=1):
        writer.writerow(
            {
                "protocol_id": config.protocol_id,
                "quick_spec_sha256": lock["quick_spec_sha256"],
                "stage": stage,
                "order": order,
                "label": job.label,
                "output": str(job.output) if job.output is not None else "",
                "command_sha256": hashlib.sha256(
                    "\0".join(job.command).encode()
                ).hexdigest(),
            }
        )
    return stream.getvalue()


def freeze_schedule(
    config: Any, lock: dict[str, Any], stage: str, jobs: list[Job]
) -> Path:
    path = resolve_path(config.paths.schedule_dir) / f"{stage}.csv"
    expected = schedule_text(config, lock, stage, jobs)
    if path.exists() and path.read_text(encoding="utf-8") != expected:
        raise ValueError(f"frozen stage schedule changed: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")
    return path


def execute(jobs: list[Job], *, dry_run: bool, resume_missing: bool) -> None:
    for job in jobs:
        if resume_missing and job.output is not None and job.output.exists():
            print(f"SKIP {job.label} output-complete", flush=True)
            continue
        print(f"RUN  {job.label}\n     {shlex.join(job.command)}", flush=True)
        if not dry_run:
            subprocess.run(job.command, check=True)


def main() -> None:
    args = parse_args()
    config_path = str(resolve_path(args.config))
    config = load_config(config_path)
    lock = load_json(config.paths.protocol_lock)
    validate_lock(config, lock)
    audit = Job(
        label="protocol-audit",
        command=module_command(
            "vector_jepa_planner_full900_screen.audit_protocol",
            "--config",
            config_path,
            "--output",
            str(
                resolve_path(config.paths.run_root)
                / f"protocol_audit_{args.stage}.json"
            ),
            "--require-checkpoints",
        ),
        output=None,
    )
    if args.stage == "audit":
        execute([audit], dry_run=args.dry_run, resume_missing=False)
        return
    jobs = stage_jobs(
        config,
        lock,
        stage=args.stage,
        config_path=config_path,
        device=args.device,
    )
    if args.execute:
        freeze_schedule(config, lock, args.stage, jobs)
    execute(
        [audit, *jobs],
        dry_run=args.dry_run,
        resume_missing=args.resume_missing,
    )
    if args.execute and args.stage == "Q0":
        validate_q0_gate(config, lock)


if __name__ == "__main__":
    main()
