from __future__ import annotations

from pathlib import Path

import pytest

from distance_head_study.plan_jobs import Job, _validate_job_graph
from distance_head_study.run_jobs import (
    _load_state,
    _require_job_method_gate,
    _reset_interrupted_state,
    _validate_successful_job_outputs,
    _verify_completion,
    _write_completion,
    _write_state,
)


def _job(job_id: str, *, dependencies: tuple[str, ...] = ()) -> Job:
    return Job(
        job_id=job_id,
        command=("python", "-m", "distance_head_study.audit_protocol"),
        dependencies=dependencies,
        outputs=(f"/tmp/{job_id}.json",),
        gpu_required=False,
    )


def test_job_graph_rejects_cycles_and_shared_outputs() -> None:
    with pytest.raises(ValueError, match="cycle"):
        _validate_job_graph(
            {
                "a": _job("a", dependencies=("b",)),
                "b": _job("b", dependencies=("a",)),
            }
        )
    first = _job("a")
    second = Job(
        job_id="b",
        command=first.command,
        dependencies=(),
        outputs=first.outputs,
        gpu_required=False,
    )
    with pytest.raises(ValueError, match="share output"):
        _validate_job_graph({"a": first, "b": second})


def test_executor_state_and_completion_bind_output_hashes(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    output.write_text("one", encoding="utf-8")
    job = Job(
        job_id="test",
        command=("python", "-m", "distance_head_study.audit_protocol"),
        dependencies=(),
        outputs=(output.as_posix(),),
        gpu_required=False,
    )
    _write_completion(tmp_path, job, plan_sha256="plan", return_code=0)
    _verify_completion(tmp_path, job, plan_sha256="plan")
    state_path = tmp_path / "state.json"
    states = {"test": {"status": "complete"}}
    _write_state(state_path, plan_sha256="plan", jobs=states)
    assert _load_state(state_path, plan_sha256="plan", job_ids={"test"}) == states
    output.write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="output changed"):
        _verify_completion(tmp_path, job, plan_sha256="plan")


def test_interrupted_retry_requires_same_host_and_stopped_pid() -> None:
    state = {"status": "running", "host": "worker-a", "pid": 123, "attempt": 2}
    recovered = _reset_interrupted_state(
        "job", state, current_host="worker-a", pid_is_alive=lambda _: False
    )
    assert recovered == {
        "status": "pending",
        "attempt": 2,
        "recovered_interrupted": True,
    }
    with pytest.raises(RuntimeError, match="recorded on host"):
        _reset_interrupted_state(
            "job", state, current_host="worker-b", pid_is_alive=lambda _: False
        )
    with pytest.raises(RuntimeError, match="still has a live"):
        _reset_interrupted_state(
            "job", state, current_host="worker-a", pid_is_alive=lambda _: True
        )


def test_successful_job_requires_files_and_semantic_validation(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "artifact.json"
    job = Job(
        job_id="test",
        command=("python", "-m", "distance_head_study.audit_protocol"),
        dependencies=(),
        outputs=(output.as_posix(),),
        gpu_required=False,
    )
    with pytest.raises(ValueError, match="outputs are incomplete"):
        _validate_successful_job_outputs(job, config=object(), lock={})
    output.write_text("{}", encoding="utf-8")

    def reject_semantics(job, *, config, lock):
        del job, config, lock
        raise ValueError("semantic mismatch")

    monkeypatch.setattr(
        "distance_head_study.run_jobs._validate_recovered_outputs",
        reject_semantics,
    )
    with pytest.raises(ValueError, match="semantic mismatch"):
        _validate_successful_job_outputs(job, config=object(), lock={})


def test_job_output_revalidation_rechecks_method_gate(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    job = Job(
        job_id="diagnostic",
        command=(
            "python",
            "-m",
            "distance_head_study.diagnose",
            "--method",
            "b_l2_cem",
            "--split-role",
            "screen",
            "--backbone-seed",
            "42",
            "--head-seed",
            "7",
        ),
        dependencies=(),
        outputs=("/tmp/diagnostic.json",),
        gpu_required=True,
    )

    def no_split_gate(config, **kwargs):
        del config
        calls.append(("evaluation", int(kwargs["head_seed"])))
        return None

    def released(config, **kwargs):
        del config
        calls.append(("release", int(kwargs["head_seed"])))
        return {"release_sha256": "ok"}

    monkeypatch.setattr(
        "distance_head_study.run_jobs.require_evaluation_gate", no_split_gate
    )
    monkeypatch.setattr("distance_head_study.run_jobs.require_seed_released", released)
    _require_job_method_gate(job, config=object())
    assert calls == [("evaluation", 7), ("release", 0)]
