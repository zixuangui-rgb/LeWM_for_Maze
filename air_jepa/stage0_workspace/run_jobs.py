#!/usr/bin/env python3
"""Run the immutable AIR0 DAG with four fixed GPU slots and fail-fast status."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from air_jepa.stage0_workspace.checkpoints import verify_source_lock
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    load_config,
    read_json,
    relative_path,
    require_clean_worktree,
    resolve_path,
    sha256_file,
    verify_signature,
)
from air_jepa.stage0_workspace.plan_jobs import build_jobs, validate_jobs
from air_jepa.stage0_workspace.protocol import (
    verify_package_lock,
    verify_protocol_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--plan", default=None)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def _status_path(root: Path, job_id: str) -> Path:
    return root / "job_status" / f"{job_id}.json"


def _write_status(root: Path, job: dict[str, Any], **values: Any) -> None:
    atomic_json_dump(
        _status_path(root, str(job["job_id"])),
        {
            "schema": "air-jepa-stage0-job-status-v1",
            "job_id": job["job_id"],
            **values,
        },
    )


def _completed(root: Path, job: dict[str, Any]) -> bool:
    path = _status_path(root, str(job["job_id"]))
    if not path.is_file():
        return False
    status = read_json(path)
    if status.get("status") != "complete":
        return False
    expected = job.get("expected_output")
    if expected is not None:
        output = resolve_path(expected)
        if not output.is_file() or sha256_file(output) != status.get("output_sha256"):
            raise ValueError(f"completed job artifact changed: {job['job_id']}")
    return True


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def validate_job_plan_payload(
    plan: dict[str, Any],
    *,
    config: Any,
    config_path: str | Path,
    protocol_sha256: str,
    package_sha256: str,
    package_code_fingerprint: str,
    source_lock_sha256: str,
) -> dict[str, dict[str, Any]]:
    verify_signature(plan, "job_plan_sha256")
    if (
        plan.get("schema") != "air-jepa-stage0-job-plan-v1"
        or plan.get("experiment_id") != config.experiment_id
        or not plan.get("git_commit")
    ):
        raise ValueError("job plan identity/provenance is invalid")
    for key, expected in (
        ("protocol_sha256", protocol_sha256),
        ("package_sha256", package_sha256),
        ("source_lock_sha256", source_lock_sha256),
    ):
        if plan.get(key) != expected:
            raise ValueError(f"job plan {key} differs from current lock")
    if plan.get("score_independent") is not True:
        raise ValueError("runner rejects a score-adaptive job plan")
    if (
        plan.get("automatic_continue_after_quicklooks") is not True
        or plan.get("code_fingerprint") != package_code_fingerprint
    ):
        raise ValueError("job plan continuation/code contract is invalid")
    expected_jobs = build_jobs(config, str(resolve_path(config_path)))
    validate_jobs(expected_jobs, config)
    if plan.get("jobs") != [job.as_dict() for job in expected_jobs]:
        raise ValueError("job plan differs from the executable locked DAG")
    jobs = {str(job["job_id"]): job for job in plan["jobs"]}
    if len(jobs) != int(plan.get("job_count", -1)):
        raise ValueError("job plan contains duplicate job IDs")
    return jobs


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    require_clean_worktree(allow_dirty=False)
    config = load_config(args.config)
    protocol = verify_protocol_lock(config)
    package = verify_package_lock(config)
    source = verify_source_lock(config)
    plan_path = resolve_path(
        args.plan or (str(config.paths.run_root) + "/job_plan.json")
    )
    plan = read_json(plan_path)
    jobs = validate_job_plan_payload(
        plan,
        config=config,
        config_path=args.config,
        protocol_sha256=protocol["protocol_sha256"],
        package_sha256=package["package_sha256"],
        package_code_fingerprint=package["code_fingerprint"],
        source_lock_sha256=source["source_lock_sha256"],
    )
    run_root = resolve_path(config.paths.run_root)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    runner_lock = open(run_root / "runner.lock", "a+", encoding="utf-8")  # noqa: SIM115
    try:
        fcntl.flock(runner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        runner_lock.close()
        raise RuntimeError("another AIR0 runner currently owns runner.lock") from error
    runner_lock.seek(0)
    runner_lock.truncate()
    runner_lock.write(f"pid={os.getpid()} plan_sha256={sha256_file(plan_path)}\n")
    runner_lock.flush()
    os.fsync(runner_lock.fileno())
    running: dict[str, tuple[subprocess.Popen[bytes], Any, float]] = {}
    occupied_gpus: set[int] = set()
    cpu_busy = False

    while True:
        complete = {job_id for job_id, job in jobs.items() if _completed(run_root, job)}
        if len(complete) == len(jobs):
            print(
                f"complete={len(complete)}/{len(jobs)} plan={relative_path(plan_path)}"
            )
            return

        failures: list[str] = []
        for job_id, (process, log_stream, started) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log_stream.close()
            job = jobs[job_id]
            gpu = job.get("gpu")
            if gpu is None:
                cpu_busy = False
            else:
                occupied_gpus.remove(int(gpu))
            del running[job_id]
            expected = job.get("expected_output")
            output_hash = None
            if code == 0 and expected is not None:
                output = resolve_path(expected)
                if not output.is_file():
                    code = 97
                else:
                    output_hash = sha256_file(output)
            status = "complete" if code == 0 else "technical_invalid"
            _write_status(
                run_root,
                job,
                status=status,
                return_code=code,
                elapsed_seconds=time.time() - started,
                output_sha256=output_hash,
                log_path=relative_path(run_root / "logs" / f"{job_id}.log"),
            )
            if code != 0:
                failures.append(job_id)
            else:
                print(f"complete job={job_id}")
        if failures:
            for peer_id, (process, log_stream, started) in list(running.items()):
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                log_stream.close()
                _write_status(
                    run_root,
                    jobs[peer_id],
                    status="interrupted_by_peer_failure",
                    return_code=process.returncode,
                    elapsed_seconds=time.time() - started,
                    triggering_failures=failures,
                    log_path=relative_path(run_root / "logs" / f"{peer_id}.log"),
                )
            raise RuntimeError(
                "AIR0 queue stopped for technical invalidity: " + ", ".join(failures)
            )

        launched = False
        ordered_jobs = sorted(
            jobs.items(),
            key=lambda item: (int(item[1]["priority"]), item[0]),
        )
        for job_id, job in ordered_jobs:
            if job_id in complete or job_id in running:
                continue
            status_path = _status_path(run_root, job_id)
            if status_path.is_file():
                previous = read_json(status_path)
                status = previous.get("status")
                if status == "technical_invalid":
                    raise RuntimeError(
                        f"job {job_id} is technical_invalid; preserve the status and "
                        "follow the replacement protocol"
                    )
                if status == "running":
                    pid = int(previous.get("pid", -1))
                    if _process_is_alive(pid):
                        raise RuntimeError(
                            f"job {job_id} still has live pid={pid}; refusing a "
                            "duplicate launch"
                        )
                    _write_status(
                        run_root,
                        job,
                        status="interrupted_before_resume",
                        previous_pid=pid,
                        detected_at_unix=time.time(),
                    )
            dependencies = set(job["dependencies"])
            if not dependencies <= complete:
                continue
            gpu = job.get("gpu")
            if gpu is None and cpu_busy:
                continue
            if gpu is not None and int(gpu) in occupied_gpus:
                continue
            expected = job.get("expected_output")
            if expected is not None and resolve_path(expected).exists():
                raise FileExistsError(
                    "untracked pre-existing output for pending job "
                    f"{job_id}: {expected}"
                )
            log_path = run_root / "logs" / f"{job_id}.log"
            log_stream = open(log_path, "wb")  # noqa: SIM115
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = "0"
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            process = subprocess.Popen(
                [str(value) for value in job["command"]],
                cwd=resolve_path("."),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            running[job_id] = (process, log_stream, time.time())
            if gpu is None:
                cpu_busy = True
            else:
                occupied_gpus.add(int(gpu))
            _write_status(
                run_root,
                job,
                status="running",
                pid=process.pid,
                started_at_unix=time.time(),
                command=job["command"],
            )
            print(f"launched job={job_id} resource={job['resource']} gpu={gpu}")
            launched = True
        if not running and not launched:
            pending = sorted(set(jobs) - complete)
            raise RuntimeError(f"job DAG deadlocked; pending={pending[:20]}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
