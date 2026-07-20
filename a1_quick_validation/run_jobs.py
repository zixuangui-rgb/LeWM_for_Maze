"""Execute one immutable worker partition without shell interpretation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from a1_quick_validation import JOB_PLAN_SCHEMA
from a1_quick_validation.common import (
    DEFAULT_PROFILE,
    atomic_json_dump,
    canonical_json_sha256,
    load_json,
    prepare_immutable,
    require_clean_worktree,
    resolve_path,
    sha256_file,
)
from a1_quick_validation.plan_jobs import PHASES, build_plan
from a1_quick_validation.profile import verify_package_lock


def validate_plan(
    plan: dict[str, Any],
    package_lock_sha256: str,
    *,
    expected_profile_path: str | Path | None = None,
) -> None:
    if plan.get("schema") != JOB_PLAN_SCHEMA:
        raise ValueError("job plan schema mismatch")
    signature = plan.get("plan_sha256")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if signature != canonical_json_sha256(unsigned):
        raise ValueError("job plan signature mismatch")
    if plan.get("package_lock_sha256") != package_lock_sha256:
        raise ValueError("job plan uses another package lock")
    if plan.get("phase") not in PHASES:
        raise ValueError("job plan uses an unknown phase")
    profile_path = plan.get("profile_path")
    if not isinstance(profile_path, str) or not profile_path:
        raise ValueError("job plan omits its profile path")
    if expected_profile_path is not None and resolve_path(profile_path) != resolve_path(
        expected_profile_path
    ):
        raise ValueError("job plan uses another profile")
    worker_count = int(plan.get("worker_count", -1))
    identifiers = []
    for job in plan.get("jobs", []):
        identifiers.append(job.get("job_id"))
        worker = int(job.get("worker", -1))
        if worker < 0 or worker >= worker_count:
            raise ValueError("job worker index is invalid")
        commands = job.get("commands")
        if not isinstance(commands, list) or not commands:
            raise ValueError("job has no commands")
        for command in commands:
            if not isinstance(command, list) or len(command) < 6:
                raise ValueError("job command is not an argv list")
            if Path(command[0]).resolve() != Path(sys.executable).resolve():
                raise ValueError("job command uses another Python executable")
            if command[1:3] != ["-m", "a1_quick_validation.run"]:
                raise ValueError("job command escapes the locked gateway")
            if command[3] != "--profile" or resolve_path(command[4]) != resolve_path(
                profile_path
            ):
                raise ValueError("job command uses another profile")
            if any(not isinstance(value, str) for value in command):
                raise ValueError("job argv contains a non-string value")
    if len(identifiers) != len(set(identifiers)) or any(
        not value for value in identifiers
    ):
        raise ValueError("job identifiers are missing or duplicated")
    if expected_profile_path is not None:
        expected = build_plan(expected_profile_path, str(plan["phase"]))
        if plan != expected:
            raise ValueError("job plan differs from the canonical locked phase matrix")


def _render(command: list[str], device: str) -> list[str]:
    return [device if value == "{device}" else value for value in command]


def _validate_completion(
    path: Path, payload: dict[str, Any], plan: dict[str, Any], job: dict[str, Any]
) -> None:
    signature = payload.get("completion_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "completion_sha256"
    }
    if signature != canonical_json_sha256(unsigned):
        raise ValueError(f"completion signature mismatch: {path}")
    if (
        payload.get("schema") != "a1-quick-validation-job-completion-v1"
        or payload.get("profile_id") != plan.get("profile_id")
        or payload.get("phase") != plan.get("phase")
        or payload.get("plan_sha256") != plan["plan_sha256"]
        or payload.get("job_id") != job["job_id"]
        or int(payload.get("worker", -1)) != int(job["worker"])
        or payload.get("commands_sha256") != canonical_json_sha256(job["commands"])
        or payload.get("all_commands_succeeded") is not True
    ):
        raise ValueError(f"stale completion seal: {path}")
    logs = payload.get("logs")
    if not isinstance(logs, dict) or len(logs) != len(job["commands"]):
        raise ValueError(f"completion seal has the wrong log count: {path}")
    expected_log_names = {
        f"command_{index:02d}.log" for index in range(len(job["commands"]))
    }
    if {Path(value).name for value in logs} != expected_log_names:
        raise ValueError(f"completion seal has noncanonical log names: {path}")
    for log_path, expected_hash in logs.items():
        if (
            not resolve_path(log_path).exists()
            or sha256_file(log_path) != expected_hash
        ):
            raise ValueError(f"completion log changed: {log_path}")


def run_worker(
    *,
    profile_path: str | Path,
    plan_path: str | Path,
    worker_index: int,
    device: str,
) -> list[Path]:
    profile, package_lock, _ = verify_package_lock(profile_path)
    require_clean_worktree()
    plan = load_json(plan_path)
    validate_plan(
        plan,
        package_lock["package_lock_sha256"],
        expected_profile_path=profile_path,
    )
    if not 0 <= worker_index < int(plan["worker_count"]):
        raise ValueError("requested worker is outside the plan")
    completions = []
    for job in plan["jobs"]:
        if int(job["worker"]) != worker_index:
            continue
        completion = resolve_path(profile.paths.run_root) / (
            f"completions/{plan['phase']}/{job['job_id']}.json"
        )
        if completion.exists():
            payload = load_json(completion)
            _validate_completion(completion, payload, plan, job)
            completions.append(completion)
            continue
        logs = []
        for index, raw_command in enumerate(job["commands"]):
            command = _render(raw_command, device)
            log = resolve_path(profile.paths.run_root) / (
                f"logs/{plan['phase']}/{job['job_id']}/command_{index:02d}.log"
            )
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{job['job_id']}] command {index + 1}/{len(job['commands'])}")
            with open(log, "a", encoding="utf-8") as stream:
                stream.write(f"argv={command!r}\n")
                stream.flush()
                process = subprocess.run(
                    command,
                    cwd=resolve_path("."),
                    check=False,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                stream.write(f"returncode={process.returncode}\n")
            if process.returncode != 0:
                raise RuntimeError(f"job {job['job_id']} failed; inspect {log}")
            logs.append(log)
        payload = {
            "schema": "a1-quick-validation-job-completion-v1",
            "profile_id": profile.profile_id,
            "phase": plan["phase"],
            "job_id": job["job_id"],
            "worker": worker_index,
            "device": device,
            "plan_path": resolve_path(plan_path).as_posix(),
            "plan_sha256": plan["plan_sha256"],
            "commands_sha256": canonical_json_sha256(job["commands"]),
            "logs": {path.as_posix(): sha256_file(path) for path in logs},
            "all_commands_succeeded": True,
        }
        payload["completion_sha256"] = canonical_json_sha256(payload)
        prepare_immutable(completion)
        atomic_json_dump(completion, payload)
        completions.append(completion)
    return completions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    for path in run_worker(
        profile_path=args.profile,
        plan_path=args.plan,
        worker_index=args.worker_index,
        device=args.device,
    ):
        print(path)


if __name__ == "__main__":
    main()


__all__ = ["run_worker", "validate_plan"]
