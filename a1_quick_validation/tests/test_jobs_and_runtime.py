from __future__ import annotations

import copy

import pytest

from a1_quick_validation.common import (
    DEFAULT_PROFILE,
    canonical_json_sha256,
    resolve_path,
    sha256_file,
)
from a1_quick_validation.plan_jobs import build_plan
from a1_quick_validation.profile import load_profile, verify_package_lock
from a1_quick_validation.run import (
    _redirect_diagnostics,
    _redirect_training_state,
    _validate_eval_request,
)
from a1_quick_validation.run_jobs import _validate_completion, validate_plan
from distance_head_study import diagnose as base_diagnose
from distance_head_study import train_head as base_train
from distance_head_study.common import load_study_config
from distance_head_study.protocol import verify_protocol_lock


def test_q1_plan_covers_every_method_once_and_uses_four_workers() -> None:
    _, package_lock, _ = verify_package_lock()
    plan = build_plan(DEFAULT_PROFILE, "q1")
    validate_plan(plan, package_lock["package_lock_sha256"])
    assert len(plan["jobs"]) == 7
    assert {job["worker"] for job in plan["jobs"]} == {0, 1, 2, 3}
    assert len({job["job_id"] for job in plan["jobs"]}) == 7


def test_plan_validation_rejects_gateway_escape_and_duplicate_ids() -> None:
    _, package_lock, _ = verify_package_lock()
    plan = build_plan(DEFAULT_PROFILE, "q1")
    escaped = copy.deepcopy(plan)
    escaped["jobs"][0]["commands"][0][2] = "unlocked.module"
    unsigned = {key: value for key, value in escaped.items() if key != "plan_sha256"}
    escaped["plan_sha256"] = canonical_json_sha256(unsigned)
    with pytest.raises(ValueError, match="escapes"):
        validate_plan(escaped, package_lock["package_lock_sha256"])
    duplicated = copy.deepcopy(plan)
    duplicated["jobs"][1]["job_id"] = duplicated["jobs"][0]["job_id"]
    unsigned = {key: value for key, value in duplicated.items() if key != "plan_sha256"}
    duplicated["plan_sha256"] = canonical_json_sha256(unsigned)
    with pytest.raises(ValueError, match="duplicated"):
        validate_plan(duplicated, package_lock["package_lock_sha256"])


def test_runtime_rejects_protocol_cells_outside_phase() -> None:
    profile = load_profile()
    config = load_study_config(profile.paths.quick_config)
    lock = verify_protocol_lock(config)
    with pytest.raises(ValueError, match="action protocol"):
        _validate_eval_request(
            config,
            lock,
            method="a1_log",
            split_role="screen",
            backbone_seed=42,
            head_seed=0,
            action_protocol="unmasked",
            diagnostic=False,
        )
    with pytest.raises(ValueError, match="backbone"):
        _validate_eval_request(
            config,
            lock,
            method="a1_log",
            split_role="screen",
            backbone_seed=43,
            head_seed=0,
            action_protocol="corrected_v1",
            diagnostic=False,
        )


def test_training_and_diagnostic_redirection_is_scoped() -> None:
    original_state = base_train._training_state_path
    original_resolve = base_diagnose.resolve_path
    with _redirect_training_state("a1_quick_validation_runs"):
        path = base_train._training_state_path("a1_bellman", 42, 0)
        assert path == resolve_path(
            "a1_quick_validation_runs/train_state/a1_bellman/backbone42_head0.pt"
        )
    with _redirect_diagnostics("a1_quick_validation_runs"):
        path = base_diagnose.resolve_path(
            "distance_head_study_runs/diagnostics/screen/a1_log/x.json"
        )
        assert path == resolve_path(
            "a1_quick_validation_runs/diagnostics/screen/a1_log/x.json"
        )
    assert base_train._training_state_path is original_state
    assert base_diagnose.resolve_path is original_resolve


def test_completion_seal_rechecks_log_content(tmp_path) -> None:
    plan = build_plan(DEFAULT_PROFILE, "q1")
    job = plan["jobs"][0]
    log = tmp_path / "command.log"
    log.write_text("complete\n", encoding="utf-8")
    payload = {
        "schema": "a1-quick-validation-job-completion-v1",
        "profile_id": plan["profile_id"],
        "phase": plan["phase"],
        "job_id": job["job_id"],
        "worker": job["worker"],
        "device": "cpu",
        "plan_path": "unused",
        "plan_sha256": plan["plan_sha256"],
        "commands_sha256": canonical_json_sha256(job["commands"]),
        "logs": {log.as_posix(): sha256_file(log)},
        "all_commands_succeeded": True,
    }
    payload["completion_sha256"] = canonical_json_sha256(payload)
    seal = tmp_path / "completion.json"
    _validate_completion(seal, payload, plan, job)
    log.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="log changed"):
        _validate_completion(seal, payload, plan, job)
