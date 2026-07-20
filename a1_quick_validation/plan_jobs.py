"""Generate immutable four-worker plans for one explicitly requested phase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from a1_quick_validation import JOB_PLAN_SCHEMA, NEW_METHODS, PROMOTABLE_METHODS
from a1_quick_validation.common import (
    DEFAULT_PROFILE,
    atomic_json_dump,
    canonical_json_sha256,
    prepare_immutable,
    resolve_path,
)
from a1_quick_validation.profile import verify_package_lock
from a1_quick_validation.selection import load_q1_shortlist, load_q2_winner
from distance_head_study.common import load_study_config

PHASES = (
    "q0",
    "q1",
    "q1_select",
    "q2_gate",
    "q2_train",
    "q2_eval",
    "q2_select",
    "q3_eval",
    "q3_assess",
)


def _command(profile_path: str | Path, *values: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "a1_quick_validation.run",
        "--profile",
        str(profile_path),
        *values,
    ]


def _job(identifier: str, worker: int, commands: list[list[str]]) -> dict[str, Any]:
    if not commands:
        raise ValueError("job pipeline cannot be empty")
    return {"job_id": identifier, "worker": int(worker), "commands": commands}


def _shortlist(
    config: Any, quick_lock: dict[str, Any], package_lock_sha256: str
) -> list[str]:
    payload = load_q1_shortlist(
        config,
        quick_lock,
        package_lock_sha256=package_lock_sha256,
    )
    values = [str(value) for value in payload.get("new_methods", [])]
    if not values or len(values) > 2 or not set(values) <= set(PROMOTABLE_METHODS):
        raise ValueError("Q2 planning requires a valid Q1 shortlist")
    return values


def _winner(config: Any, quick_lock: dict[str, Any], package_lock_sha256: str) -> str:
    payload = load_q2_winner(
        config,
        quick_lock,
        package_lock_sha256=package_lock_sha256,
    )
    value = payload.get("selected_method")
    if value not in PROMOTABLE_METHODS:
        raise ValueError("Q3 planning requires a valid Q2 winner")
    return str(value)


def build_jobs(profile_path: str | Path, phase: str) -> list[dict[str, Any]]:
    profile, package_lock, quick_lock = verify_package_lock(profile_path)
    config = load_study_config(profile.paths.quick_config)
    package_lock_sha256 = str(package_lock["package_lock_sha256"])
    jobs: list[dict[str, Any]] = []
    if phase == "q0":
        commands = [_command(profile_path, "audit")]
        commands.extend(
            _command(
                profile_path,
                "rebind-cache",
                "--split-role",
                role,
                "--backbone-seed",
                "42",
            )
            for role in ("train", "cal", "screen", "select")
        )
        commands.extend(
            (
                _command(profile_path, "candidate-bank", "--backbone-seed", "42"),
                _command(profile_path, "release-seeds", "--tier", "seed1"),
                _command(
                    profile_path,
                    "import-reference",
                    "--method",
                    "b_dh_cem",
                    "--head-seed",
                    "0",
                ),
                _command(
                    profile_path,
                    "import-reference",
                    "--method",
                    "a1_log",
                    "--head-seed",
                    "0",
                ),
            )
        )
        jobs.append(_job("q0_protocol_and_inputs", 0, commands))
    elif phase == "q1":
        for index, method in enumerate(profile.q1.methods):
            commands = []
            if method in NEW_METHODS:
                commands.append(
                    _command(
                        profile_path,
                        "train",
                        "--method",
                        method,
                        "--head-seed",
                        "0",
                        "--device",
                        "{device}",
                    )
                )
            commands.extend(
                (
                    _command(
                        profile_path,
                        "diagnose",
                        "--method",
                        method,
                        "--split-role",
                        "screen",
                        "--head-seed",
                        "0",
                        "--device",
                        "{device}",
                    ),
                    _command(
                        profile_path,
                        "evaluate",
                        "--method",
                        method,
                        "--split-role",
                        "screen",
                        "--head-seed",
                        "0",
                        "--action-protocol",
                        "corrected_v1",
                        "--device",
                        "{device}",
                    ),
                )
            )
            jobs.append(_job(f"q1_{method}", index % profile.worker_count, commands))
    elif phase == "q1_select":
        jobs.append(
            _job(
                "q1_select",
                0,
                [_command(profile_path, "select", "--phase", "q1")],
            )
        )
    elif phase == "q2_gate":
        _shortlist(config, quick_lock, package_lock_sha256)
        jobs.append(
            _job(
                "q2_release_and_reference_heads",
                0,
                [
                    _command(profile_path, "release-seeds", "--tier", "seed3"),
                    _command(
                        profile_path,
                        "import-reference",
                        "--method",
                        "b_dh_cem",
                        "--head-seed",
                        "1",
                    ),
                    _command(
                        profile_path,
                        "import-reference",
                        "--method",
                        "a1_log",
                        "--head-seed",
                        "1",
                    ),
                ],
            )
        )
    elif phase == "q2_train":
        for index, method in enumerate(
            _shortlist(config, quick_lock, package_lock_sha256)
        ):
            jobs.append(
                _job(
                    f"q2_train_{method}_head1",
                    index % profile.worker_count,
                    [
                        _command(
                            profile_path,
                            "train",
                            "--method",
                            method,
                            "--head-seed",
                            "1",
                            "--device",
                            "{device}",
                        )
                    ],
                )
            )
    elif phase == "q2_eval":
        methods = [
            *profile.q2.methods,
            *_shortlist(config, quick_lock, package_lock_sha256),
        ]
        job_index = 0
        for method in methods:
            for head_seed in profile.q2.head_seeds:
                commands = [
                    _command(
                        profile_path,
                        "diagnose",
                        "--method",
                        method,
                        "--split-role",
                        "select",
                        "--head-seed",
                        str(head_seed),
                        "--device",
                        "{device}",
                    )
                ]
                commands.extend(
                    _command(
                        profile_path,
                        "evaluate",
                        "--method",
                        method,
                        "--split-role",
                        "select",
                        "--head-seed",
                        str(head_seed),
                        "--action-protocol",
                        action_protocol,
                        "--device",
                        "{device}",
                    )
                    for action_protocol in profile.q2.action_protocols
                )
                jobs.append(
                    _job(
                        f"q2_eval_{method}_head{head_seed}",
                        job_index % profile.worker_count,
                        commands,
                    )
                )
                job_index += 1
    elif phase == "q2_select":
        _shortlist(config, quick_lock, package_lock_sha256)
        jobs.append(
            _job(
                "q2_select",
                0,
                [_command(profile_path, "select", "--phase", "q2")],
            )
        )
    elif phase == "q3_eval":
        winner = _winner(config, quick_lock, package_lock_sha256)
        methods = [*profile.q3.methods, winner]
        job_index = 0
        for method in methods:
            for action_protocol in profile.q3.action_protocols:
                jobs.append(
                    _job(
                        f"q3_{method}_{action_protocol}",
                        job_index % profile.worker_count,
                        [
                            _command(
                                profile_path,
                                "evaluate",
                                "--method",
                                method,
                                "--split-role",
                                "legacy",
                                "--head-seed",
                                "0",
                                "--action-protocol",
                                action_protocol,
                                "--device",
                                "{device}",
                            )
                        ],
                    )
                )
                job_index += 1
    elif phase == "q3_assess":
        _winner(config, quick_lock, package_lock_sha256)
        jobs.append(
            _job(
                "q3_assess",
                0,
                [_command(profile_path, "select", "--phase", "q3")],
            )
        )
    else:
        raise ValueError(f"unknown phase: {phase}")
    return jobs


def build_plan(profile_path: str | Path, phase: str) -> dict[str, Any]:
    profile, package_lock, quick_lock = verify_package_lock(profile_path)
    jobs = build_jobs(profile_path, phase)
    payload = {
        "schema": JOB_PLAN_SCHEMA,
        "profile_id": profile.profile_id,
        "profile_path": resolve_path(profile_path).as_posix(),
        "phase": phase,
        "worker_count": profile.worker_count,
        "package_lock_sha256": package_lock["package_lock_sha256"],
        "protocol_lock_sha256": quick_lock["protocol_lock_sha256"],
        "jobs": jobs,
    }
    payload["plan_sha256"] = canonical_json_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    profile, _, _ = verify_package_lock(args.profile)
    output = (
        resolve_path(args.output)
        if args.output
        else resolve_path(profile.paths.run_root) / f"plans/{args.phase}.json"
    )
    prepare_immutable(output)
    atomic_json_dump(output, build_plan(args.profile, args.phase))
    print(output)


if __name__ == "__main__":
    main()


__all__ = ["PHASES", "build_jobs", "build_plan"]
