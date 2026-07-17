"""Run a generated job DAG across independent local CUDA workers."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from distance_head_study.common import (
    atomic_json_dump,
    canonical_json_sha256,
    load_json,
    load_study_config,
    read_jsonl,
    require_clean_worktree,
    resolve_path,
    sha256_file,
    source_backbone_path,
    validate_backbone_protocol_binding,
)
from distance_head_study.gates import (
    load_signed_artifact,
    require_evaluation_gate,
    require_seed_released,
)
from distance_head_study.plan_jobs import (
    Job,
    _existing_cache,
    _existing_candidate_bank,
    _existing_diagnostic,
    _existing_head,
    _existing_result,
    _validate_job_graph,
    job_plan_metadata_path,
)
from distance_head_study.protocol import verify_protocol_lock


def _load_job_plan(path: str) -> tuple[dict[str, Job], dict[str, Any]]:
    resolved = resolve_path(path)
    rows = read_jsonl(resolved)
    jobs: dict[str, Job] = {}
    expected_fields = {
        "job_id",
        "command",
        "dependencies",
        "outputs",
        "gpu_required",
    }
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError("job row fields differ from the locked schema")
        if not isinstance(row["job_id"], str) or not row["job_id"]:
            raise ValueError("job ID must be a nonempty string")
        if row["job_id"] in jobs:
            raise ValueError(f"duplicate job ID: {row['job_id']}")
        if (
            not isinstance(row["command"], list)
            or not all(isinstance(value, str) for value in row["command"])
            or not isinstance(row["dependencies"], list)
            or not all(isinstance(value, str) for value in row["dependencies"])
            or not isinstance(row["outputs"], list)
            or not all(isinstance(value, str) for value in row["outputs"])
            or not isinstance(row["gpu_required"], bool)
        ):
            raise ValueError(f"job row has invalid value types: {row['job_id']}")
        jobs[row["job_id"]] = Job(
            job_id=row["job_id"],
            command=tuple(row["command"]),
            dependencies=tuple(row["dependencies"]),
            outputs=tuple(row["outputs"]),
            gpu_required=row["gpu_required"],
        )
    _validate_job_graph(jobs)
    metadata = load_json(job_plan_metadata_path(resolved))
    signature = metadata.get("job_plan_metadata_sha256")
    unsigned = {
        key: value
        for key, value in metadata.items()
        if key != "job_plan_metadata_sha256"
    }
    if signature != canonical_json_sha256(unsigned):
        raise ValueError("job plan metadata signature mismatch")
    expected = (
        metadata.get("schema") == "distance-head-job-plan-v1"
        and metadata.get("job_plan_path") == resolved.as_posix()
        and metadata.get("job_plan_sha256") == sha256_file(resolved)
        and int(metadata.get("job_count", -1)) == len(jobs)
        and metadata.get("job_ids") == list(jobs)
    )
    if not expected:
        raise ValueError("job plan differs from its signed metadata")
    release_path = metadata.get("seed_release_path")
    if not isinstance(release_path, str) or sha256_file(release_path) != metadata.get(
        "seed_release_file_sha256"
    ):
        raise ValueError("job plan seed release changed")
    config_path = metadata.get("config_path")
    if not isinstance(config_path, str) or sha256_file(config_path) != metadata.get(
        "config_sha256"
    ):
        raise ValueError("job plan config binding changed")
    config = load_study_config(config_path)
    lock = verify_protocol_lock(config)
    if (
        metadata.get("protocol_id") != config.protocol_id
        or metadata.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
        or metadata.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
    ):
        raise ValueError("job plan uses another protocol lock")
    return jobs, metadata


def _completion_path(logs: Path, job_id: str) -> Path:
    return logs / f"{job_id}.complete.json"


def _write_completion(
    logs: Path, job: Job, *, plan_sha256: str, return_code: int
) -> None:
    outputs = {path: sha256_file(resolve_path(path)) for path in job.outputs}
    payload = {
        "schema": "distance-head-job-completion-v1",
        "job_plan_sha256": plan_sha256,
        "job_id": job.job_id,
        "return_code": int(return_code),
        "output_hashes": outputs,
    }
    payload["completion_sha256"] = canonical_json_sha256(payload)
    atomic_json_dump(_completion_path(logs, job.job_id), payload)


def _verify_completion(logs: Path, job: Job, *, plan_sha256: str) -> None:
    path = _completion_path(logs, job.job_id)
    payload = load_json(path)
    signature = payload.get("completion_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "completion_sha256"
    }
    if signature != canonical_json_sha256(unsigned):
        raise ValueError(f"job completion signature mismatch: {path}")
    if (
        payload.get("schema") != "distance-head-job-completion-v1"
        or payload.get("job_plan_sha256") != plan_sha256
        or payload.get("job_id") != job.job_id
        or int(payload.get("return_code", -1)) != 0
    ):
        raise ValueError(f"job completion metadata differs: {path}")
    expected_hashes = payload.get("output_hashes")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(
        job.outputs
    ):
        raise ValueError(f"job completion outputs differ: {path}")
    for output, expected_hash in expected_hashes.items():
        if sha256_file(resolve_path(output)) != expected_hash:
            raise ValueError(f"completed job output changed: {output}")


def _write_state(
    path: Path, *, plan_sha256: str, jobs: dict[str, dict[str, Any]]
) -> None:
    payload = {
        "schema": "distance-head-job-state-v1",
        "job_plan_sha256": plan_sha256,
        "jobs": jobs,
    }
    payload["state_sha256"] = canonical_json_sha256(payload)
    atomic_json_dump(path, payload)


def _load_state(
    path: Path, *, plan_sha256: str, job_ids: set[str]
) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    signature = payload.get("state_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "state_sha256"}
    if signature != canonical_json_sha256(unsigned):
        raise ValueError("job executor state signature mismatch")
    jobs = payload.get("jobs")
    if (
        payload.get("schema") != "distance-head-job-state-v1"
        or payload.get("job_plan_sha256") != plan_sha256
        or not isinstance(jobs, dict)
        or set(jobs) != job_ids
    ):
        raise ValueError("job executor state belongs to another plan")
    allowed = {"pending", "running", "complete", "failed"}
    if any(
        not isinstance(value, dict) or value.get("status") not in allowed
        for value in jobs.values()
    ):
        raise ValueError("job executor state contains an invalid status")
    return jobs


def _command_value(job: Job, option: str) -> str:
    try:
        index = job.command.index(option)
    except ValueError as error:
        raise ValueError(f"job {job.job_id} omits required option {option}") from error
    if index + 1 >= len(job.command) or job.command[index + 1].startswith("--"):
        raise ValueError(f"job {job.job_id} has no value for option {option}")
    return job.command[index + 1]


def _require_job_method_gate(job: Job, *, config: Any) -> None:
    method = _command_value(job, "--method")
    backbone_seed = int(_command_value(job, "--backbone-seed"))
    head_seed = int(_command_value(job, "--head-seed"))
    gate = require_evaluation_gate(
        config,
        split_role=_command_value(job, "--split-role"),
        method=method,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
    )
    if gate is None:
        require_seed_released(
            config,
            backbone_seed=backbone_seed,
            head_seed=0 if method == "b_l2_cem" else head_seed,
        )


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reset_interrupted_state(
    job_id: str,
    value: dict[str, Any],
    *,
    current_host: str,
    pid_is_alive: Any = _pid_is_alive,
) -> dict[str, Any]:
    recorded_host = value.get("host")
    if recorded_host != current_host:
        raise RuntimeError(
            f"interrupted job {job_id} was recorded on host {recorded_host!r}, not "
            f"{current_host!r}; verify it is stopped on the original host"
        )
    pid = value.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError(f"interrupted job {job_id} has no valid recorded PID")
    if pid_is_alive(pid):
        raise RuntimeError(
            f"interrupted job {job_id} still has a live recorded PID {pid}"
        )
    return {
        "status": "pending",
        "attempt": int(value.get("attempt", 1)),
        "recovered_interrupted": True,
    }


def _validate_recovered_outputs(job: Job, *, config: Any, lock: dict[str, Any]) -> None:
    """Revalidate atomic outputs before sealing an interrupted completed job."""

    if not all(resolve_path(path).is_file() for path in job.outputs):
        raise ValueError(f"recovered job outputs are not complete files: {job.job_id}")
    module = job.command[2]
    backbone_seed = (
        int(_command_value(job, "--backbone-seed"))
        if module != "distance_head_study.open_confirmation"
        else None
    )
    if module == "distance_head_study.train_backbone":
        assert backbone_seed is not None
        expected = source_backbone_path(config, backbone_seed)
        if tuple(job.outputs) != (expected.as_posix(),):
            raise ValueError("recovered backbone output path differs from its command")
        payload = torch.load(expected, map_location="cpu", weights_only=False)
        validate_backbone_protocol_binding(
            config,
            payload,
            backbone_seed=backbone_seed,
            protocol_lock=lock,
        )
        if payload.get("formal_run") is not True:
            raise ValueError("recovered backbone is not a formal checkpoint")
        return
    if module == "distance_head_study.build_cache":
        assert backbone_seed is not None
        if not _existing_cache(
            config,
            lock,
            split_role=_command_value(job, "--split-role"),
            backbone_seed=backbone_seed,
        ):
            raise ValueError("recovered cache did not validate")
        return
    if module == "distance_head_study.candidates":
        assert backbone_seed is not None
        if not _existing_candidate_bank(config, lock, backbone_seed=backbone_seed):
            raise ValueError("recovered candidate bank did not validate")
        return
    if module == "distance_head_study.train_head":
        assert backbone_seed is not None
        if not _existing_head(
            config,
            lock,
            owner=_command_value(job, "--method"),
            backbone_seed=backbone_seed,
            head_seed=int(_command_value(job, "--head-seed")),
        ):
            raise ValueError("recovered head checkpoint did not validate")
        return
    if module == "distance_head_study.diagnose":
        assert backbone_seed is not None
        _require_job_method_gate(job, config=config)
        if not _existing_diagnostic(
            config,
            lock,
            split_role=_command_value(job, "--split-role"),
            method=_command_value(job, "--method"),
            backbone_seed=backbone_seed,
            head_seed=int(_command_value(job, "--head-seed")),
        ):
            raise ValueError("recovered diagnostic did not validate")
        return
    if module == "distance_head_study.evaluate":
        assert backbone_seed is not None
        _require_job_method_gate(job, config=config)
        if not _existing_result(
            config,
            lock,
            split_role=_command_value(job, "--split-role"),
            method=_command_value(job, "--method"),
            backbone_seed=backbone_seed,
            head_seed=int(_command_value(job, "--head-seed")),
            action_protocol=_command_value(job, "--action-protocol"),
        ):
            raise ValueError("recovered evaluation did not validate")
        return
    if module == "distance_head_study.open_confirmation":
        artifact = load_signed_artifact(
            config.paths.confirm_opened,
            signature_field="confirm_open_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        if (
            artifact.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
            or artifact.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
        ):
            raise ValueError("recovered confirmation-open artifact uses another lock")
        for path, expected_hash in artifact.get("locked_checkpoint_hashes", {}).items():
            if sha256_file(path) != expected_hash:
                raise ValueError("recovered confirmation checkpoint changed")
        return
    raise ValueError(f"no recovery validator is registered for {module}")


def _validate_successful_job_outputs(
    job: Job, *, config: Any, lock: dict[str, Any]
) -> None:
    """Apply the atomic artifact validator before any successful job is sealed."""

    if not all(resolve_path(path).is_file() for path in job.outputs):
        raise ValueError(f"successful job outputs are incomplete: {job.job_id}")
    _validate_recovered_outputs(job, config=config, lock=lock)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--state", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed jobs under the identical signed plan/spec.",
    )
    parser.add_argument(
        "--retry-interrupted",
        action="store_true",
        help="Recover a stopped running job after verifying its recorded PID is gone.",
    )
    args = parser.parse_args()
    # CHECK-REQUIRED: On a scheduler-managed server, replace this local executor
    # with the site's Slurm/Kubernetes launcher while preserving each command,
    # dependency, output path, and CUDA assignment in the emitted job file.
    require_clean_worktree()
    jobs, plan_metadata = _load_job_plan(args.jobs)
    plan_sha256 = str(plan_metadata["job_plan_sha256"])
    config = load_study_config(str(plan_metadata["config_path"]))
    lock = verify_protocol_lock(config)
    current_host = socket.gethostname()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU worker is required")
    logs = resolve_path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    state_path = resolve_path(args.state)
    if state_path.exists():
        state = _load_state(
            state_path,
            plan_sha256=plan_sha256,
            job_ids=set(jobs),
        )
        for job_id, value in state.items():
            marker = _completion_path(logs, job_id)
            if marker.exists():
                _verify_completion(logs, jobs[job_id], plan_sha256=plan_sha256)
                state[job_id] = {"status": "complete", "recovered": True}
                continue
            if value.get("status") == "running":
                if not args.retry_interrupted:
                    raise RuntimeError(
                        "executor stopped while a job was running and no signed "
                        f"completion marker exists: {job_id}; inspect the PID/log, "
                        "then restart with --retry-interrupted"
                    )
                reset = _reset_interrupted_state(
                    job_id,
                    value,
                    current_host=current_host,
                )
                if all(resolve_path(path).is_file() for path in jobs[job_id].outputs):
                    _validate_recovered_outputs(jobs[job_id], config=config, lock=lock)
                    _write_completion(
                        logs,
                        jobs[job_id],
                        plan_sha256=plan_sha256,
                        return_code=0,
                    )
                    state[job_id] = {
                        "status": "complete",
                        "recovered_interrupted": True,
                        "attempt": int(value.get("attempt", 1)),
                    }
                else:
                    state[job_id] = reset
                continue
            if value.get("status") == "complete":
                raise ValueError(
                    f"complete job is missing its completion marker: {job_id}"
                )
            if value.get("status") == "failed" and args.retry_failed:
                state[job_id] = {
                    "status": "pending",
                    "attempt": int(value.get("attempt", 1)),
                }
    else:
        state = {job_id: {"status": "pending"} for job_id in jobs}
    _write_state(state_path, plan_sha256=plan_sha256, jobs=state)
    running: dict[str, tuple[subprocess.Popen[bytes], Any, str | None]] = {}
    while True:
        for job_id, (process, stream, gpu) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del running[job_id]
            job = jobs[job_id]
            outputs_exist = all(resolve_path(path).is_file() for path in job.outputs)
            validation_error = None
            if return_code == 0:
                try:
                    _validate_successful_job_outputs(job, config=config, lock=lock)
                except Exception as error:  # The failure is persisted for resumption.
                    validation_error = f"{type(error).__name__}: {error}"
            complete = return_code == 0 and validation_error is None
            if complete:
                _write_completion(
                    logs,
                    job,
                    plan_sha256=plan_sha256,
                    return_code=return_code,
                )
            state_value = {
                "status": "complete" if complete else "failed",
                "return_code": return_code,
                "gpu": gpu,
                "outputs_exist": outputs_exist,
                "attempt": int(state[job_id].get("attempt", 1)),
            }
            if validation_error is not None:
                state_value["output_validation_error"] = validation_error
            state[job_id] = state_value
            _write_state(state_path, plan_sha256=plan_sha256, jobs=state)
        if any(value["status"] == "failed" for value in state.values()):
            if not running:
                raise RuntimeError("job DAG stopped after a failed job; inspect logs")
            time.sleep(2.0)
            continue
        if all(value["status"] == "complete" for value in state.values()):
            break
        busy_gpus = {gpu for _, _, gpu in running.values() if gpu is not None}
        launched = False
        for job_id, job in jobs.items():
            if state[job_id]["status"] != "pending":
                continue
            if not all(
                state[dependency]["status"] == "complete"
                for dependency in job.dependencies
            ):
                continue
            gpu = None
            if job.gpu_required:
                gpu = next((value for value in gpus if value not in busy_gpus), None)
                if gpu is None:
                    continue
            attempt = int(state[job_id].get("attempt", 0)) + 1
            log_path = logs / f"{job_id}.attempt_{attempt:03d}.log"
            if log_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite an existing job log: {log_path}"
                )
            stream = open(log_path, "wb")
            environment = dict(os.environ)
            if gpu is not None:
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                busy_gpus.add(gpu)
            process = subprocess.Popen(
                job.command,
                cwd=resolve_path("."),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            running[job_id] = (process, stream, gpu)
            state[job_id] = {
                "status": "running",
                "pid": process.pid,
                "host": current_host,
                "gpu": gpu,
                "attempt": attempt,
            }
            _write_state(state_path, plan_sha256=plan_sha256, jobs=state)
            launched = True
            if len(running) >= len(gpus):
                break
        if not launched and not running:
            blocked = [
                job_id
                for job_id, value in state.items()
                if value["status"] == "pending"
            ]
            raise RuntimeError(f"job DAG is deadlocked: {blocked[:5]}")
        time.sleep(2.0)
    print(state_path)


if __name__ == "__main__":
    main()
