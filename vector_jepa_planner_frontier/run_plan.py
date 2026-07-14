"""Deterministic stage runner with dependency order and blocked job randomization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_json_dump,
    atomic_text_dump,
    load_json,
    load_study_config,
    planner_seed_values,
    resolve_path,
    uses_counterexample_rounds,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.compat import checkpoint_path, source_protocol
from vector_jepa_planner_frontier.confirmation import load_confirmation_artifacts
from vector_jepa_planner_frontier.effective_methods import resolve_effective_method
from vector_jepa_planner_frontier.oracle_ladder import ORACLES
from vector_jepa_planner_frontier.stage_gates import (
    validate_p2_selection,
    validate_p5_advancement,
    validate_p7_selection,
    validate_p8_selection,
)


@dataclass(frozen=True)
class Job:
    label: str
    command: tuple[str, ...]
    output: Path | None
    opaque: bool = False
    required_outputs: tuple[Path, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument(
        "--stage",
        choices=(
            "audit",
            "backbones",
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
            "confirmatory",
        ),
        required=True,
    )
    parser.add_argument(
        "--split-role", choices=("development", "validation"), default="validation"
    )
    parser.add_argument("--device")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-missing", action="store_true")
    return parser.parse_args()


def module_command(module: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, "-m", module, *arguments)


def selected_methods(
    config: Any, stage: str, lock: dict[str, Any] | None = None
) -> list[Any]:
    if stage == "confirmatory":
        if lock is None:
            raise RuntimeError(
                "confirmatory methods require the frozen P8 selection record"
            )
        selection = validate_p8_selection(config, lock)
        return [
            next(method for method in config.methods if method.name == name)
            for name in selection["confirmation_methods"]
        ]
    methods = [method for method in config.methods if method.stage == stage]
    if stage == "P8" and lock is not None:
        p7 = validate_p7_selection(config, lock)
        if p7.get("selected_track_j") is None:
            methods = [
                method for method in methods if not method.name.startswith("p8_p7_")
            ]
    return methods


def source_backbone_jobs(config: Any, config_path: str) -> list[Job]:
    source_config, _ = source_protocol(config)
    legacy = {int(seed) for seed in source_config["seeds"]}
    jobs: list[Job] = []
    for backbone_seed in config.protocol.training_seeds:
        if backbone_seed in legacy:
            continue
        output = checkpoint_path(config, seed=int(backbone_seed))
        jobs.append(
            Job(
                label=f"source-backbone:seed{backbone_seed}",
                command=module_command(
                    "vector_jepa_planner_frontier.train_source_backbone",
                    "--config",
                    config_path,
                    "--backbone-seed",
                    str(backbone_seed),
                ),
                output=output,
            )
        )
    return jobs


def blocked_oracle_jobs(config: Any, *, split_role: str, config_path: str) -> list[Job]:
    """Build the complete oracle ladder in randomized backbone blocks."""

    rng = np.random.default_rng(config.protocol.run_order_seed + 1)
    backbones = list(config.protocol.training_seeds)
    rng.shuffle(backbones)
    jobs: list[Job] = []
    for raw_backbone_seed in backbones:
        backbone_seed = int(raw_backbone_seed)
        block = [
            (oracle, int(search_seed), action_selection)
            for oracle in ORACLES
            for search_seed in config.protocol.search_seeds
            for action_selection in ("unmasked", "corrected_v1")
        ]
        rng.shuffle(block)
        for oracle, search_seed, action_selection in block:
            output = resolve_path(
                config.paths.oracle_result_template.format(
                    oracle=oracle,
                    backbone_seed=backbone_seed,
                    search_seed=search_seed,
                    split=split_role,
                    action_selection=action_selection,
                )
            )
            jobs.append(
                Job(
                    label=(
                        f"oracle:{split_role}:{oracle}:backbone{backbone_seed}:"
                        f"search{search_seed}:{action_selection}"
                    ),
                    command=module_command(
                        "vector_jepa_planner_frontier.oracle_ladder",
                        "--config",
                        config_path,
                        "--oracle",
                        oracle,
                        "--backbone-seed",
                        str(backbone_seed),
                        "--search-seed",
                        str(search_seed),
                        "--split-role",
                        split_role,
                        "--action-selection",
                        action_selection,
                        "--output",
                        str(output),
                    ),
                    output=output,
                )
            )
    return jobs


def component_jobs(
    config: Any,
    method: Any,
    backbone_seed: int,
    planner_seed: int,
    config_path: str,
    lock: dict[str, Any] | None = None,
) -> list[Job]:
    if method.reuse_component_from is not None:
        return []
    effective = (
        resolve_effective_method(config, lock, method)
        if lock is not None and method.adaptive_role != "static"
        else method
    )
    if (
        not effective.component_checkpoint_required
        and effective.proposal.retrieval_weight <= 0.0
    ):
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
    jobs: list[Job] = []
    if effective.component_checkpoint_required:
        jobs.append(
            Job(
                label=(
                    f"train:{method.name}:backbone{backbone_seed}:planner{planner_seed}"
                ),
                command=module_command(
                    (
                        "vector_jepa_planner_frontier.assemble_p5"
                        if method.stage == "P5"
                        else "vector_jepa_planner_frontier.train"
                    ),
                    "--config",
                    config_path,
                    "--method",
                    method.name,
                    "--backbone-seed",
                    str(backbone_seed),
                    "--planner-seed",
                    str(planner_seed),
                ),
                output=train_output,
            )
        )
    if effective.proposal.retrieval_weight > 0.0:
        bank_output = resolve_path(
            config.paths.retrieval_bank_template.format(
                method=method.name,
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
            )
        )
        jobs.append(
            Job(
                label=(
                    f"retrieval:{method.name}:backbone{backbone_seed}:"
                    f"planner{planner_seed}"
                ),
                command=module_command(
                    "vector_jepa_planner_frontier.build_retrieval_bank",
                    "--config",
                    config_path,
                    "--method",
                    method.name,
                    "--backbone-seed",
                    str(backbone_seed),
                    "--planner-seed",
                    str(planner_seed),
                ),
                output=bank_output,
            )
        )
    if effective.component_checkpoint_required:
        jobs.append(
            Job(
                label=(
                    f"calibrate:{method.name}:backbone{backbone_seed}:"
                    f"planner{planner_seed}"
                ),
                command=module_command(
                    "vector_jepa_planner_frontier.calibrate",
                    "--config",
                    config_path,
                    "--method",
                    method.name,
                    "--backbone-seed",
                    str(backbone_seed),
                    "--planner-seed",
                    str(planner_seed),
                ),
                output=calibrated_output,
            )
        )
    if method.stage == "P6" and uses_counterexample_rounds(effective):
        for round_index in range(1, config.training.counterexample_rounds + 1):
            round_output = resolve_path(
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
                        f"counterexample:{method.name}:backbone{backbone_seed}:"
                        f"planner{planner_seed}:round{round_index}"
                    ),
                    command=module_command(
                        "vector_jepa_planner_frontier.counterexamples",
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
                    ),
                    output=round_output,
                )
            )
    return jobs


def evaluation_jobs(
    config: Any,
    methods: list[Any],
    *,
    split_role: str,
    config_path: str,
    backbone_seeds: tuple[int, ...] | None = None,
) -> list[Job]:
    jobs: list[Job] = []
    selected_backbones = backbone_seeds or config.protocol.training_seeds
    for method in methods:
        for backbone_seed in selected_backbones:
            for planner_seed in planner_seed_values(config, method):
                for search_seed in config.protocol.search_seeds:
                    for action_selection in ("unmasked", "corrected_v1"):
                        output = resolve_path(
                            config.paths.result_template.format(
                                method=method.name,
                                backbone_seed=backbone_seed,
                                planner_seed=planner_seed,
                                search_seed=search_seed,
                                split=split_role,
                                action_selection=action_selection,
                            )
                        )
                        jobs.append(
                            Job(
                                label=(
                                    f"eval:{split_role}:{method.name}:"
                                    f"backbone{backbone_seed}:"
                                    f"planner{planner_seed}:search{search_seed}:"
                                    f"{action_selection}"
                                ),
                                command=module_command(
                                    "vector_jepa_planner_frontier.evaluate",
                                    "--config",
                                    config_path,
                                    "--method",
                                    method.name,
                                    "--backbone-seed",
                                    str(backbone_seed),
                                    "--planner-seed",
                                    str(planner_seed),
                                    "--search-seed",
                                    str(search_seed),
                                    "--split-role",
                                    split_role,
                                    "--action-selection",
                                    action_selection,
                                    "--output",
                                    str(output),
                                ),
                                output=output,
                                required_outputs=(
                                    output,
                                    output.with_name(
                                        f"{output.stem}.candidate_traces.jsonl"
                                    ),
                                ),
                            )
                        )
    return jobs


def confirmatory_jobs(
    config: Any,
    lock: dict[str, Any],
    *,
    config_path: str,
) -> list[Job]:
    confirmation, mapping, schedule = load_confirmation_artifacts(
        config, lock, require_opened=False
    )
    rows = mapping.get("runs", [])
    public_ids = [row["run_id"] for row in schedule.get("runs", [])]
    if [row.get("run_id") for row in rows] != public_ids:
        raise ValueError("private mapping and public schedule disagree")
    if len(rows) != int(confirmation.get("run_count", -1)):
        raise ValueError("confirmation run count mismatch")
    jobs: list[Job] = []
    for row in rows:
        output = Path(row["opaque_output"])
        jobs.append(
            Job(
                label=f"confirmatory:{row['run_id']}",
                command=module_command(
                    "vector_jepa_planner_frontier.run_opaque",
                    "--config",
                    config_path,
                    "--run-id",
                    row["run_id"],
                ),
                output=output,
                opaque=True,
                required_outputs=(
                    output,
                    output.with_name(f"{output.stem}.candidate_traces.jsonl"),
                ),
            )
        )
    return jobs


def mark_confirmation_opened(config: Any) -> None:
    marker = resolve_path(config.paths.confirmation_opened)
    confirmation = resolve_path(config.paths.confirmation_lock)
    expected_sha = sha256_file(confirmation)
    if marker.exists():
        value = load_json(marker)
        if value.get("confirmation_lock_sha256") != expected_sha:
            raise ValueError("confirmation-opened marker belongs to another lock")
        return
    atomic_json_dump(
        marker,
        {
            "schema": "vector-jepa-confirmation-opened-v1",
            "confirmation_lock_sha256": expected_sha,
            "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def randomized_blocks(
    config: Any,
    methods: list[Any],
    *,
    config_path: str,
    lock: dict[str, Any] | None = None,
) -> list[Job]:
    rng = np.random.default_rng(config.protocol.run_order_seed)
    output: list[Job] = []
    for backbone_seed in config.protocol.training_seeds:
        pairs = [
            (method, planner_seed)
            for method in methods
            for planner_seed in planner_seed_values(config, method)
        ]
        rng.shuffle(pairs)
        for method, planner_seed in pairs:
            output.extend(
                component_jobs(
                    config,
                    method,
                    int(backbone_seed),
                    int(planner_seed),
                    config_path,
                    lock,
                )
            )
    return output


def blocked_stage_jobs(
    config: Any,
    methods: list[Any],
    *,
    stage: str,
    split_role: str,
    config_path: str,
    lock: dict[str, Any] | None = None,
) -> list[Job]:
    """Keep every backbone block complete while randomizing within each block."""

    stage_number = int(stage.removeprefix("P"))
    rng = np.random.default_rng(config.protocol.run_order_seed + stage_number)
    backbones = list(config.protocol.training_seeds)
    rng.shuffle(backbones)
    output: list[Job] = []
    for raw_backbone_seed in backbones:
        backbone_seed = int(raw_backbone_seed)
        pairs = [
            (method, int(planner_seed))
            for method in methods
            for planner_seed in planner_seed_values(config, method)
        ]
        rng.shuffle(pairs)
        for method, planner_seed in pairs:
            output.extend(
                component_jobs(
                    config,
                    method,
                    backbone_seed,
                    planner_seed,
                    config_path,
                    lock,
                )
            )
        evaluations = evaluation_jobs(
            config,
            methods,
            split_role=split_role,
            config_path=config_path,
            backbone_seeds=(backbone_seed,),
        )
        rng.shuffle(evaluations)
        output.extend(evaluations)
    return output


def stage_schedule_text(
    config: Any,
    lock: dict[str, Any],
    *,
    stage: str,
    split_role: str,
    jobs: list[Job],
) -> str:
    columns = (
        "protocol_id",
        "analysis_spec_sha256",
        "stage",
        "split_role",
        "order",
        "label",
        "output",
        "command_sha256",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for order, job in enumerate(jobs, start=1):
        command_digest = hashlib.sha256("\0".join(job.command).encode()).hexdigest()
        writer.writerow(
            {
                "protocol_id": config.protocol_id,
                "analysis_spec_sha256": analysis_spec_sha256(config, lock),
                "stage": stage,
                "split_role": split_role,
                "order": order,
                "label": job.label,
                "output": str(job.output) if job.output is not None else "",
                "command_sha256": command_digest,
            }
        )
    return stream.getvalue()


def freeze_stage_schedule(
    config: Any,
    lock: dict[str, Any],
    *,
    stage: str,
    split_role: str,
    jobs: list[Job],
) -> Path:
    path = resolve_path(config.paths.schedule_dir) / f"{stage}_{split_role}.csv"
    expected = stage_schedule_text(
        config,
        lock,
        stage=stage,
        split_role=split_role,
        jobs=jobs,
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"frozen run schedule no longer reproduces: {path}")
    else:
        atomic_text_dump(path, expected)
    return path


def execute_jobs(
    jobs: list[Job],
    *,
    dry_run: bool,
    resume_missing: bool,
    device: str | None,
) -> None:
    for job in jobs:
        required = job.required_outputs or (
            (job.output,) if job.output is not None else ()
        )
        if resume_missing and required:
            present = [path.exists() for path in required]
            if all(present):
                print(f"SKIP {job.label} outputs-complete", flush=True)
                continue
            if any(present):
                raise RuntimeError(
                    f"partial immutable output for {job.label}; use the explicit "
                    "infrastructure-failure rerun procedure"
                )
        command = list(job.command)
        if device and job.command[2] not in {
            "vector_jepa_planner_frontier.audit_protocol",
            "vector_jepa_planner_frontier.summarize",
        }:
            command.extend(["--device", device])
        if job.opaque:
            print(f"RUN  {job.label} (command blinded)", flush=True)
        else:
            print(f"RUN  {job.label}\n     {shlex.join(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    config_path = str(resolve_path(args.config))
    config = load_study_config(config_path)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    audit_output = resolve_path(config.paths.run_root) / "protocol_audit.json"
    audit = Job(
        label="protocol-audit",
        command=module_command(
            "vector_jepa_planner_frontier.audit_protocol",
            "--config",
            config_path,
            "--output",
            str(audit_output),
            "--formal",
        ),
        # Audit is rerun on every invocation, including a resumed matrix.
        output=None,
    )
    if args.stage == "audit":
        execute_jobs(
            [audit],
            dry_run=args.dry_run,
            resume_missing=args.resume_missing,
            device=args.device,
        )
        return
    if args.stage == "backbones":
        execute_jobs(
            [audit, *source_backbone_jobs(config, config_path)],
            dry_run=args.dry_run,
            resume_missing=args.resume_missing,
            device=args.device,
        )
        return
    if args.stage == "P1":
        jobs = [
            audit,
            *blocked_oracle_jobs(
                config,
                split_role=args.split_role,
                config_path=config_path,
            ),
        ]
        if args.execute:
            freeze_stage_schedule(
                config,
                lock,
                stage="P1",
                split_role=args.split_role,
                jobs=jobs[1:],
            )
        execute_jobs(
            jobs,
            dry_run=args.dry_run,
            resume_missing=args.resume_missing,
            device=args.device,
        )
        return
    if args.stage in {"P3", "P4", "P5", "P6", "P7", "P8", "confirmatory"}:
        validate_p2_selection(config, lock)
    if args.stage in {"P5", "P6", "P7", "P8", "confirmatory"}:
        validate_p5_advancement(config, lock)
    if args.stage in {"P8", "confirmatory"}:
        validate_p7_selection(config, lock)
    if args.stage == "confirmatory":
        validate_p8_selection(config, lock)
    methods = selected_methods(config, args.stage, lock)
    if not methods:
        raise ValueError(f"stage {args.stage} has no configured methods")
    jobs = [audit]
    if args.stage != "confirmatory":
        split_role = args.split_role
        jobs.extend(
            blocked_stage_jobs(
                config,
                methods,
                stage=args.stage,
                split_role=split_role,
                config_path=config_path,
                lock=lock,
            )
        )
        if args.execute:
            freeze_stage_schedule(
                config,
                lock,
                stage=args.stage,
                split_role=split_role,
                jobs=jobs[1:],
            )
    else:
        power = load_json(config.paths.confirmation_power)
        if power.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
            raise ValueError("confirmatory power record belongs to another protocol")
        if power.get("claim_status") != "adequately_powered":
            raise RuntimeError(
                "confirmatory scheduler is locked until the power gate passes"
            )
        jobs.extend(confirmatory_jobs(config, lock, config_path=config_path))
        if args.execute:
            mark_confirmation_opened(config)
        execute_jobs(
            jobs,
            dry_run=args.dry_run,
            resume_missing=args.resume_missing,
            device=args.device,
        )
        return
    execute_jobs(
        jobs,
        dry_run=args.dry_run,
        resume_missing=args.resume_missing,
        device=args.device,
    )


if __name__ == "__main__":
    main()
