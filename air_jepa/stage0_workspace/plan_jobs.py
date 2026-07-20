#!/usr/bin/env python3
"""Materialize the complete score-independent AIR0 job DAG before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from air_jepa.stage0_workspace import AIR_METHODS
from air_jepa.stage0_workspace.checkpoints import verify_source_lock
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    code_fingerprint,
    git_commit,
    load_config,
    prepare_new_output,
    relative_path,
    require_clean_worktree,
    resolve_path,
    runtime_metadata,
    signed_payload,
)
from air_jepa.stage0_workspace.protocol import (
    expected_matrix,
    verify_package_lock,
    verify_protocol_lock,
)


@dataclass(frozen=True)
class Job:
    job_id: str
    level: str
    priority: int
    resource: str
    gpu: int | None
    command: tuple[str, ...]
    dependencies: tuple[str, ...]
    expected_output: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "level": self.level,
            "priority": self.priority,
            "resource": self.resource,
            "gpu": self.gpu,
            "command": list(self.command),
            "dependencies": list(self.dependencies),
            "expected_output": self.expected_output,
        }


class JobBuilder:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.jobs: list[Job] = []
        self.ids: set[str] = set()
        self.outputs: set[str] = set()

    @staticmethod
    def _gpu(job_id: str) -> int:
        digest = hashlib.sha256(job_id.encode("ascii")).digest()
        return int.from_bytes(digest[:2], "big") % 4

    def add(
        self,
        job_id: str,
        *,
        level: str,
        module: str,
        args: list[str],
        dependencies: list[str] | tuple[str, ...] = (),
        output: str | Path | None = None,
        gpu: int | None = None,
    ) -> str:
        if job_id in self.ids:
            raise ValueError(f"duplicate job id: {job_id}")
        if any(dependency not in self.ids for dependency in dependencies):
            missing = [dep for dep in dependencies if dep not in self.ids]
            raise ValueError(
                f"job {job_id} has forward/missing dependencies: {missing}"
            )
        resource = "gpu" if gpu is not None else "cpu"
        if gpu is not None and not 0 <= gpu < self.config.worker_count:
            raise ValueError("job GPU is outside the locked four-worker range")
        output_value = relative_path(output) if output is not None else None
        if output_value is not None:
            if output_value in self.outputs:
                raise ValueError(f"two jobs target the same output: {output_value}")
            self.outputs.add(output_value)
        command = [sys.executable, "-m", module, *args]
        if gpu is not None:
            command.extend(["--device", f"cuda:{gpu}"])
        if output is not None:
            command.extend(["--output", str(resolve_path(output))])
        if level == "L0":
            priority = 0
        elif level == "TRAIN":
            priority = 30 if job_id.endswith("s44") else 10
        elif level == "L1":
            priority = 20
        elif level == "L2":
            priority = 40
        elif level == "L3":
            priority = 50
        else:
            raise ValueError(f"unknown AIR job level: {level}")
        self.jobs.append(
            Job(
                job_id=job_id,
                level=level,
                priority=priority,
                resource=resource,
                gpu=gpu,
                command=tuple(command),
                dependencies=tuple(dependencies),
                expected_output=output_value,
            )
        )
        self.ids.add(job_id)
        return job_id


def _result_path(
    config: Any,
    *,
    role: str,
    method: str,
    seed: int,
    protocol: str,
    k: int,
) -> Path:
    return resolve_path(
        config.paths.result_template.format(
            split_role=role,
            method=method,
            seed=seed,
            action_protocol=protocol,
            k=k,
        )
    )


def _evaluation_args(
    *,
    config_path: str,
    method: str,
    seed: int,
    k: int,
    role: str,
    protocol: str = "unmasked",
    intervention: str = "normal",
) -> list[str]:
    return [
        "--config",
        config_path,
        "--method",
        method,
        "--seed",
        str(seed),
        "--k",
        str(k),
        "--split-role",
        role,
        "--action-protocol",
        protocol,
        "--intervention",
        intervention,
    ]


def build_jobs(config: Any, config_path: str) -> list[Job]:
    builder = JobBuilder(config)
    audit_path = resolve_path(config.paths.audit_output)
    audit = builder.add(
        "l0_protocol_audit",
        level="L0",
        module="air_jepa.stage0_workspace.audit_protocol",
        args=["--config", config_path],
        output=audit_path,
    )
    smoke_jobs: list[str] = []
    for method, gpu in (("air0_direct", 0), ("air0_jepa", 1)):
        output = resolve_path(config.paths.run_root) / (
            f"preflight/checkpoints/{method}_seed42_1000step_smoke.pt"
        )
        smoke_jobs.append(
            builder.add(
                f"l0_smoke_{method}",
                level="L0",
                module="air_jepa.stage0_workspace.train",
                args=[
                    "--config",
                    config_path,
                    "--method",
                    method,
                    "--seed",
                    "42",
                    "--mode",
                    "smoke",
                    "--smoke-steps",
                    "1000",
                    "--smoke-k",
                    "16",
                ],
                dependencies=[audit],
                output=output,
                gpu=gpu,
            )
        )
    benchmark = builder.add(
        "l0_benchmark",
        level="L0",
        module="air_jepa.stage0_workspace.benchmark",
        args=["--config", config_path],
        dependencies=[audit],
        output=resolve_path(config.paths.benchmark_output),
        gpu=2,
    )

    historical_jobs: list[str] = []
    for seed in config.seeds:
        for method, k in (("j0_static", 4), ("j1_static", 128)):
            job_id = f"bridge_historical_{method}_s{seed}"
            historical_jobs.append(
                builder.add(
                    job_id,
                    level="L0",
                    module="air_jepa.stage0_workspace.evaluate",
                    args=_evaluation_args(
                        config_path=config_path,
                        method=method,
                        seed=seed,
                        k=k,
                        role="historical",
                    ),
                    dependencies=[audit],
                    output=_result_path(
                        config,
                        role="historical",
                        method=method,
                        seed=seed,
                        protocol="unmasked",
                        k=k,
                    ),
                    gpu=builder._gpu(job_id),
                )
            )
    bridge_audit = builder.add(
        "l0_bridge_parity",
        level="L0",
        module="air_jepa.stage0_workspace.audit_bridges",
        args=["--config", config_path],
        dependencies=[*historical_jobs, *smoke_jobs, benchmark],
        output=resolve_path(config.paths.run_root)
        / "audits/historical_bridge_parity.json",
    )

    train_jobs: dict[tuple[str, int], str] = {}
    fixed_gpus = {
        ("air0_direct", 42): 0,
        ("air0_jepa", 42): 1,
        ("air0_direct", 43): 2,
        ("air0_jepa", 43): 3,
        ("air0_direct", 44): 1,
        ("air0_jepa", 44): 0,
    }
    for seed in config.seeds:
        for method in AIR_METHODS:
            job_id = f"train_{method}_s{seed}"
            output = resolve_path(
                config.paths.air_checkpoint_template.format(method=method, seed=seed)
            )
            train_jobs[(method, seed)] = builder.add(
                job_id,
                level="TRAIN",
                module="air_jepa.stage0_workspace.train",
                args=[
                    "--config",
                    config_path,
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                    "--mode",
                    "formal",
                ],
                dependencies=[bridge_audit],
                output=output,
                gpu=fixed_gpus[(method, seed)],
            )

    def eval_dependency(method: str, seed: int) -> list[str]:
        return (
            [train_jobs[(method, seed)], bridge_audit]
            if method in AIR_METHODS
            else [bridge_audit]
        )

    l1_jobs: list[str] = []
    for method in ("j1_receding", "air0_direct", "air0_jepa"):
        for k in (16, 128):
            job_id = f"l1_early_{method}_s42_k{k}"
            l1_jobs.append(
                builder.add(
                    job_id,
                    level="L1",
                    module="air_jepa.stage0_workspace.evaluate",
                    args=_evaluation_args(
                        config_path=config_path,
                        method=method,
                        seed=42,
                        k=k,
                        role="air_early",
                    ),
                    dependencies=eval_dependency(method, 42),
                    output=_result_path(
                        config,
                        role="air_early",
                        method=method,
                        seed=42,
                        protocol="unmasked",
                        k=k,
                    ),
                    gpu=builder._gpu(job_id),
                )
            )
    for k in (16, 128):
        for intervention in INTERVENTIONS_WITHOUT_NORMAL:
            job_id = f"l1_intervention_s42_k{k}_{intervention}"
            output = resolve_path(config.paths.run_root) / (
                f"results/air_early/interventions/air0_jepa/seed42_"
                f"unmasked_k{k}_{intervention}.json"
            )
            l1_jobs.append(
                builder.add(
                    job_id,
                    level="L1",
                    module="air_jepa.stage0_workspace.evaluate",
                    args=_evaluation_args(
                        config_path=config_path,
                        method="air0_jepa",
                        seed=42,
                        k=k,
                        role="air_early",
                        intervention=intervention,
                    ),
                    dependencies=eval_dependency("air0_jepa", 42),
                    output=output,
                    gpu=builder._gpu(job_id),
                )
            )
    early_diagnostic = builder.add(
        "l1_diagnostic_s42",
        level="L1",
        module="air_jepa.stage0_workspace.diagnose",
        args=[
            "--config",
            config_path,
            "--seed",
            "42",
            "--split-role",
            "air_early",
        ],
        dependencies=eval_dependency("air0_jepa", 42),
        output=resolve_path(config.paths.run_root)
        / "diagnostics/air_early/air0_jepa/seed42.json",
        gpu=3,
    )
    l1_release = builder.add(
        "l1_release",
        level="L1",
        module="air_jepa.stage0_workspace.summarize",
        args=["--config", config_path, "--level", "l1"],
        dependencies=[*l1_jobs, early_diagnostic],
        output=resolve_path(config.paths.release_template.format(level="l1")),
    )

    l2_jobs: list[str] = []
    for seed in config.seeds:
        for method in ("j1_receding", "air0_direct", "air0_jepa"):
            job_id = f"l2_primary_{method}_s{seed}_k128"
            l2_jobs.append(
                builder.add(
                    job_id,
                    level="L2",
                    module="air_jepa.stage0_workspace.evaluate",
                    args=_evaluation_args(
                        config_path=config_path,
                        method=method,
                        seed=seed,
                        k=128,
                        role="air_dev",
                    ),
                    dependencies=eval_dependency(method, seed),
                    output=_result_path(
                        config,
                        role="air_dev",
                        method=method,
                        seed=seed,
                        protocol="unmasked",
                        k=128,
                    ),
                    gpu=builder._gpu(job_id),
                )
            )
        diagnostic_id = f"l2_diagnostic_s{seed}"
        l2_jobs.append(
            builder.add(
                diagnostic_id,
                level="L2",
                module="air_jepa.stage0_workspace.diagnose",
                args=[
                    "--config",
                    config_path,
                    "--seed",
                    str(seed),
                    "--split-role",
                    "air_dev",
                ],
                dependencies=eval_dependency("air0_jepa", seed),
                output=resolve_path(
                    config.paths.diagnostic_template.format(
                        method="air0_jepa", seed=seed
                    )
                ),
                gpu=builder._gpu(diagnostic_id),
            )
        )
    l2_release = builder.add(
        "l2_release",
        level="L2",
        module="air_jepa.stage0_workspace.summarize",
        args=["--config", config_path, "--level", "l2"],
        dependencies=[*l2_jobs, l1_release],
        output=resolve_path(config.paths.release_template.format(level="l2")),
    )

    oracle_output = (
        resolve_path(config.paths.run_root)
        / "results/air_dev/oracle_bfs/seed0_unmasked_k0.json"
    )
    oracle_job = builder.add(
        "l3_oracle_bfs",
        level="L3",
        module="air_jepa.stage0_workspace.evaluate_oracle",
        args=["--config", config_path],
        dependencies=[bridge_audit],
        output=oracle_output,
    )
    l3_jobs: list[str] = [oracle_job]
    for seed in config.seeds:
        for method in ("j1_receding", "air0_direct", "air0_jepa"):
            for k in config.evaluation.k_values:
                if k == 128:
                    continue
                job_id = f"l3_curve_{method}_s{seed}_k{k}"
                l3_jobs.append(
                    builder.add(
                        job_id,
                        level="L3",
                        module="air_jepa.stage0_workspace.evaluate",
                        args=_evaluation_args(
                            config_path=config_path,
                            method=method,
                            seed=seed,
                            k=k,
                            role="air_dev",
                        ),
                        dependencies=eval_dependency(method, seed),
                        output=_result_path(
                            config,
                            role="air_dev",
                            method=method,
                            seed=seed,
                            protocol="unmasked",
                            k=k,
                        ),
                        gpu=builder._gpu(job_id),
                    )
                )
        for method, k in (("j0_static", 4), ("j1_static", 128)):
            job_id = f"l3_static_{method}_s{seed}"
            l3_jobs.append(
                builder.add(
                    job_id,
                    level="L3",
                    module="air_jepa.stage0_workspace.evaluate",
                    args=_evaluation_args(
                        config_path=config_path,
                        method=method,
                        seed=seed,
                        k=k,
                        role="air_dev",
                    ),
                    dependencies=[bridge_audit],
                    output=_result_path(
                        config,
                        role="air_dev",
                        method=method,
                        seed=seed,
                        protocol="unmasked",
                        k=k,
                    ),
                    gpu=builder._gpu(job_id),
                )
            )
        for method in (
            "j0_static",
            "j1_static",
            "j1_receding",
            "air0_direct",
            "air0_jepa",
        ):
            k = 4 if method == "j0_static" else 128
            job_id = f"l3_corrected_{method}_s{seed}"
            l3_jobs.append(
                builder.add(
                    job_id,
                    level="L3",
                    module="air_jepa.stage0_workspace.evaluate",
                    args=_evaluation_args(
                        config_path=config_path,
                        method=method,
                        seed=seed,
                        k=k,
                        role="air_dev",
                        protocol="corrected",
                    ),
                    dependencies=eval_dependency(method, seed),
                    output=_result_path(
                        config,
                        role="air_dev",
                        method=method,
                        seed=seed,
                        protocol="corrected",
                        k=k,
                    ),
                    gpu=builder._gpu(job_id),
                )
            )
    for seed in (43, 44):
        for k in (16, 128):
            normal_job_id = f"l3_intervention_s{seed}_k{k}_normal"
            l3_jobs.append(
                builder.add(
                    normal_job_id,
                    level="L3",
                    module="air_jepa.stage0_workspace.evaluate",
                    args=_evaluation_args(
                        config_path=config_path,
                        method="air0_jepa",
                        seed=seed,
                        k=k,
                        role="air_early",
                    ),
                    dependencies=eval_dependency("air0_jepa", seed),
                    output=_result_path(
                        config,
                        role="air_early",
                        method="air0_jepa",
                        seed=seed,
                        protocol="unmasked",
                        k=k,
                    ),
                    gpu=builder._gpu(normal_job_id),
                )
            )
            for intervention in INTERVENTIONS_WITHOUT_NORMAL:
                job_id = f"l3_intervention_s{seed}_k{k}_{intervention}"
                output = resolve_path(config.paths.run_root) / (
                    f"results/air_early/interventions/air0_jepa/seed{seed}_"
                    f"unmasked_k{k}_{intervention}.json"
                )
                l3_jobs.append(
                    builder.add(
                        job_id,
                        level="L3",
                        module="air_jepa.stage0_workspace.evaluate",
                        args=_evaluation_args(
                            config_path=config_path,
                            method="air0_jepa",
                            seed=seed,
                            k=k,
                            role="air_early",
                            intervention=intervention,
                        ),
                        dependencies=eval_dependency("air0_jepa", seed),
                        output=output,
                        gpu=builder._gpu(job_id),
                    )
                )
    builder.add(
        "l3_release",
        level="L3",
        module="air_jepa.stage0_workspace.summarize",
        args=["--config", config_path, "--level", "l3"],
        dependencies=[*l3_jobs, l2_release],
        output=resolve_path(config.paths.release_template.format(level="l3")),
    )
    return builder.jobs


INTERVENTIONS_WITHOUT_NORMAL = (
    "copy_current",
    "true_future",
    "future_permutation",
    "future_zero",
)


def _command_options(job: Job) -> tuple[str, dict[str, str]]:
    command = list(job.command)
    try:
        module_index = command.index("-m") + 1
    except ValueError as error:
        raise ValueError(f"job lacks a Python module command: {job.job_id}") from error
    if module_index >= len(command):
        raise ValueError(f"job module command is incomplete: {job.job_id}")
    module = command[module_index]
    values = command[module_index + 1 :]
    options: dict[str, str] = {}
    cursor = 0
    while cursor < len(values):
        flag = values[cursor]
        if not flag.startswith("--") or cursor + 1 >= len(values):
            raise ValueError(f"job has an unparseable command: {job.job_id}")
        if flag in options:
            raise ValueError(f"job repeats command option {flag}: {job.job_id}")
        options[flag] = values[cursor + 1]
        cursor += 2
    return module, options


def scientific_matrix_from_jobs(jobs: list[Job], config: Any) -> dict[str, Any]:
    """Recover every scientific cell from executable commands for exact auditing."""

    matrix = {key: [] for key in expected_matrix(config)}
    for job in jobs:
        module, options = _command_options(job)
        if module.endswith(".train"):
            if options.get("--mode") != "formal":
                continue
            matrix["train"].append(
                {
                    "method": options["--method"],
                    "seed": int(options["--seed"]),
                    "steps": config.training.steps,
                }
            )
            continue
        if module.endswith(".diagnose"):
            role = options["--split-role"]
            key = "air_early_diagnostics" if role == "air_early" else "diagnostics"
            matrix[key].append(
                {
                    "method": "air0_jepa",
                    "seed": int(options["--seed"]),
                    "k": config.evaluation.primary_k,
                    "states_per_maze": config.evaluation.local_states_per_maze,
                }
            )
            continue
        if module.endswith(".evaluate_oracle"):
            matrix["evaluator_oracle"].append(
                {
                    "method": "oracle_bfs",
                    "split_role": "air_dev",
                    "action_protocol": "unmasked",
                    "max_steps": config.evaluation.max_steps,
                }
            )
            continue
        if not module.endswith(".evaluate"):
            continue
        role = options["--split-role"]
        method = options["--method"]
        protocol = options["--action-protocol"]
        intervention = options["--intervention"]
        row: dict[str, Any] = {
            "method": method,
            "seed": int(options["--seed"]),
            "k": int(options["--k"]),
            "action_protocol": protocol,
        }
        if role == "historical":
            key = "historical_bridges"
        elif role == "air_dev":
            key = "air_dev_unmasked" if protocol == "unmasked" else "air_dev_corrected"
        elif role == "air_early":
            key = (
                "air_early_interventions"
                if method == "air0_jepa"
                else "air_early_context"
            )
            row["intervention"] = intervention
        else:
            raise ValueError(f"job uses an unauthorized scientific role: {job.job_id}")
        matrix[key].append(row)
    return matrix


def _cell_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    )


def validate_jobs(jobs: list[Job], config: Any) -> None:
    if len(jobs) != len({job.job_id for job in jobs}):
        raise ValueError("job DAG contains duplicate IDs")
    positions = {job.job_id: index for index, job in enumerate(jobs)}
    for job in jobs:
        if any(
            positions[dependency] >= positions[job.job_id]
            for dependency in job.dependencies
        ):
            raise ValueError(f"job DAG is not topologically ordered at {job.job_id}")
    formal_train = [job for job in jobs if job.job_id.startswith("train_air0_")]
    if len(formal_train) != 6:
        raise ValueError("job DAG must contain six formal AIR training cells")
    primary = [job for job in jobs if job.job_id.startswith("l2_primary_")]
    if len(primary) != 9:
        raise ValueError("job DAG must contain nine three-seed K128 primary cells")
    curves = [job for job in jobs if job.job_id.startswith("l3_curve_")]
    if len(curves) != 54:
        raise ValueError("job DAG must contain 54 non-K128 recurrent curve cells")
    oracle = [job for job in jobs if job.job_id == "l3_oracle_bfs"]
    if len(oracle) != 1:
        raise ValueError("job DAG must contain one AIR_dev BFS oracle cell")
    if config.worker_count != 4:
        raise ValueError("AIR0 job DAG assumes exactly four GPU workers")
    scientific_cells = sum(len(values) for values in expected_matrix(config).values())
    orchestration_jobs = 8  # audit, two smokes, benchmark, bridge, three releases
    if scientific_cells != 135 or len(jobs) != scientific_cells + orchestration_jobs:
        raise ValueError(
            "job DAG no longer matches the 135-cell protocol plus eight locked gates"
        )
    executable_matrix = scientific_matrix_from_jobs(jobs, config)
    expected = expected_matrix(config)
    for key, expected_rows in expected.items():
        if _cell_counter(executable_matrix[key]) != _cell_counter(expected_rows):
            raise ValueError(
                f"job DAG scientific cells differ from protocol matrix section {key}"
            )
    priorities = {job.job_id: job.priority for job in jobs}
    if not all(
        priorities[f"train_{method}_s44"] > priorities["l1_early_air0_jepa_s42_k128"]
        for method in AIR_METHODS
    ):
        raise ValueError("seed44 training must yield scheduling priority to L1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_clean_worktree(allow_dirty=False)
    config = load_config(args.config)
    protocol = verify_protocol_lock(config)
    package = verify_package_lock(config)
    source = verify_source_lock(config)
    jobs = build_jobs(config, str(resolve_path(args.config)))
    validate_jobs(jobs, config)
    payload = signed_payload(
        {
            "schema": "air-jepa-stage0-job-plan-v1",
            "experiment_id": config.experiment_id,
            "score_independent": True,
            "automatic_continue_after_quicklooks": True,
            "job_count": len(jobs),
            "protocol_sha256": protocol["protocol_sha256"],
            "package_sha256": package["package_sha256"],
            "source_lock_sha256": source["source_lock_sha256"],
            "git_commit": git_commit(),
            "code_fingerprint": code_fingerprint(),
            "runtime": runtime_metadata(),
            "jobs": [job.as_dict() for job in jobs],
        },
        "job_plan_sha256",
    )
    output = args.output or (str(config.paths.run_root) + "/job_plan.json")
    prepare_new_output(output)
    atomic_json_dump(output, payload)
    print(f"saved={relative_path(output)} jobs={len(jobs)}")


if __name__ == "__main__":
    main()
