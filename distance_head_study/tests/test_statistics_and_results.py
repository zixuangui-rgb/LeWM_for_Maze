from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from distance_head_study.analyze import (
    _crossed_bootstrap,
    _holm,
    _one_sided_sign_flip_p,
)
from distance_head_study.common import canonical_json_sha256, sha256_file
from distance_head_study.diagnose import (
    _binary_auc,
    _candidate_drift_steps,
    _expected_calibration_error,
    _spearman,
)
from distance_head_study.evidence import diagnostic_evidence_hashes
from distance_head_study.gates import load_signed_artifact
from distance_head_study.results import (
    load_complete_rows,
    merge_shards,
    result_evidence_hashes,
)
from spatial_jepa_planning.common import summarize_rows


def _row(task_id: str, *, size: int, success: bool) -> dict:
    optimal = 4
    path_length = 5 if success else 10
    return {
        "task_id": task_id,
        "maze_size": size,
        "topology_seed": int(task_id.split("-")[-1]),
        "start_cell": 1,
        "goal_cell": 9,
        "success": success,
        "path_length": path_length,
        "optimal_length": optimal,
        "spl": optimal / path_length if success else 0.0,
        "final_bfs_distance": 0 if success else 2,
        "failure_mode": "success" if success else "timeout_inefficient",
        "loop_or_cycle": False,
        "invalid_actions": 0,
        "repeat_states": 0,
        "max_state_visits": 1,
        "proposed_invalid": 0,
        "proposed_backtrack": 0,
        "assistance_count": 0,
        "assistance_rate": 0.0,
        "plan_transitions": 100,
        "fallback_transitions": 0,
        "mean_best_cost": 1.0,
        "episode_seconds": 0.5,
    }


def _write_result(
    directory: Path,
    rows: list[dict],
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    manifest_rows: list[dict] | None = None,
    manifest_path: Path | None = None,
) -> None:
    directory.mkdir(parents=True)
    all_rows = rows if manifest_rows is None else manifest_rows
    manifest_path = manifest_path or directory / "manifest.jsonl"
    if not manifest_path.exists():
        manifest_path.write_text(
            "".join(
                json.dumps(
                    {
                        "task_hash": row["task_id"],
                        "maze_size": row["maze_size"],
                        "topology_seed": row["topology_seed"],
                        "start_cell": row["start_cell"],
                        "goal_cell": row["goal_cell"],
                        "bfs_path_length": row["optimal_length"],
                    }
                )
                + "\n"
                for row in all_rows
            ),
            encoding="utf-8",
        )
    checkpoint_path = manifest_path.parent / "backbone.pt"
    if not checkpoint_path.exists():
        checkpoint_path.write_bytes(b"fixed-backbone")
    metadata = {
        "schema": "distance-head-task-results-v1",
        "protocol_id": "procgen-maze-distance-head-staged-v1",
        "protocol_lock_sha256": "0" * 64,
        "method": {
            "name": "candidate",
            "role": "candidate",
            "uses_test_bfs": False,
            "planner": {"kind": "categorical_cem"},
        },
        "backbone_seed": 42,
        "head_seed": 0,
        "action_protocol": "corrected_v1",
        "split_role": "confirm",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "diagnostic_limit": 0,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint": {
            "backbone_path": checkpoint_path.as_posix(),
            "backbone_sha256": sha256_file(checkpoint_path),
        },
    }
    metadata["run_spec_sha256"] = canonical_json_sha256(metadata)
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (directory / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = summarize_rows(rows, seen_max_size=21, max_steps=128)
    summary["failure_modes"] = dict(
        sorted(
            (mode, sum(row["failure_mode"] == mode for row in rows))
            for mode in {row["failure_mode"] for row in rows}
        )
    )
    summary["mean_assistance_rate"] = float(
        np.mean([row["assistance_rate"] for row in rows])
    )
    summary["run_spec_sha256"] = metadata["run_spec_sha256"]
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_crossed_bootstrap_is_seed_deterministic() -> None:
    differences = [
        np.asarray([0.0, 1.0, 1.0]),
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([1.0, 1.0, 1.0]),
    ]
    seeds = list(range(200))
    first = _crossed_bootstrap(differences, seeds, familywise_upper_quantile=0.975)
    second = _crossed_bootstrap(differences, seeds, familywise_upper_quantile=0.975)
    assert first == second
    assert first["backbone_n"] == 3
    assert first["task_resampling"] == "crossed_unstratified"
    assert first["task_indices_shared_across_backbone_resamples"] is True
    significance = _one_sided_sign_flip_p(differences, seeds)
    assert significance["test"] == "exact_backbone_sign_flip"
    assert significance["one_sided_p"] == 0.125


def test_crossed_bootstrap_rejects_nonidentical_task_strata() -> None:
    differences = [np.zeros(4), np.ones(4)]
    strata = [np.asarray([9, 9, 11, 11]), np.asarray([9, 11, 9, 11])]
    with pytest.raises(ValueError, match="identical task strata"):
        _crossed_bootstrap(
            differences,
            [1, 2],
            familywise_upper_quantile=0.975,
            strata=strata,
        )


def test_crossed_bootstrap_uses_one_shared_task_draw_per_replicate() -> None:
    differences = [
        np.asarray([0.0, 10.0, 100.0, 1000.0]),
        np.asarray([1.0, 20.0, 200.0, 2000.0]),
    ]
    seed = 17
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, 2, size=2)
    shared_tasks = rng.integers(0, 4, size=4)
    expected = float(
        np.mean([differences[int(index)][shared_tasks].mean() for index in selected])
    )
    result = _crossed_bootstrap(
        differences,
        [seed],
        familywise_upper_quantile=0.975,
    )
    assert result["ci95_low"] == pytest.approx(expected)
    assert result["ci95_high"] == pytest.approx(expected)


def test_rank_and_calibration_statistics_have_known_values() -> None:
    assert _spearman(np.asarray([1, 2, 3]), np.asarray([10, 20, 30])) == 1.0
    assert (
        _binary_auc(np.asarray([0.1, 0.2, 0.8, 0.9]), np.asarray([0, 0, 1, 1])) == 1.0
    )
    assert (
        _expected_calibration_error(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
        == 0.0
    )
    adjusted = _holm({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}


def test_rollout_drift_uses_every_candidate_terminal() -> None:
    latent_bank = torch.tensor([[0.0], [1.0], [2.0]])
    predicted = torch.tensor([[0.1], [1.9]])
    endpoints = np.asarray([2, 0])
    all_pairs = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    assert _candidate_drift_steps(predicted, latent_bank, endpoints, all_pairs) == [
        2.0,
        2.0,
    ]


def test_result_loader_recomputes_signature_rows_and_summary(tmp_path: Path) -> None:
    directory = tmp_path / "result"
    rows = [
        _row("task-1", size=11, success=True),
        _row("task-2", size=23, success=False),
    ]
    _write_result(directory, rows)
    metadata, loaded = load_complete_rows(directory)
    assert metadata["run_spec_sha256"]
    assert loaded == rows
    changed = [dict(row) for row in rows]
    changed[0]["task_id"] = "task-999"
    (directory / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in changed), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="bound manifest"):
        load_complete_rows(directory)
    (directory / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    stored = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    stored["head_seed"] = 99
    (directory / "metadata.json").write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="signature mismatch"):
        load_complete_rows(directory)


def test_result_loader_requires_protocol_lock_binding(tmp_path: Path) -> None:
    directory = tmp_path / "result"
    _write_result(directory, [_row("task-1", size=11, success=True)])
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("protocol_lock_sha256")
    metadata.pop("run_spec_sha256")
    metadata["run_spec_sha256"] = canonical_json_sha256(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol-lock binding"):
        load_complete_rows(directory)


def test_result_loader_recomputes_failure_and_assistance_semantics(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "result"
    row = _row("task-1", size=11, success=False)
    _write_result(directory, [row])
    changed = dict(row)
    changed["failure_mode"] = "loop_or_cycle"
    (directory / "rows.jsonl").write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failure mode does not reproduce"):
        load_complete_rows(directory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("plan_transitions", 3_841, "transition budget"),
        ("fallback_transitions", 5, "fallback compute"),
        ("invalid_actions", 1, "invalid executed action"),
    ),
)
def test_result_loader_enforces_compute_and_action_protocol(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    directory = tmp_path / field
    row = _row("task-1", size=11, success=True)
    _write_result(directory, [row])
    changed = {**row, field: value}
    (directory / "rows.jsonl").write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_complete_rows(directory)


def test_result_evidence_binds_full_bundle_and_checkpoint(tmp_path: Path) -> None:
    directory = tmp_path / "result"
    _write_result(directory, [_row("task-1", size=11, success=True)])
    metadata, _ = load_complete_rows(directory)
    hashes = result_evidence_hashes(directory, metadata)
    assert (directory / "metadata.json").as_posix() in hashes
    assert (directory / "rows.jsonl").as_posix() in hashes
    assert (directory / "summary.json").as_posix() in hashes
    checkpoint = Path(metadata["checkpoint"]["backbone_path"])
    assert checkpoint.as_posix() in hashes
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        load_complete_rows(directory)


def test_result_evidence_expands_head_training_dependencies(tmp_path: Path) -> None:
    directory = tmp_path / "result"
    _write_result(directory, [_row("task-1", size=11, success=True)])
    bank = tmp_path / "bank.pt"
    train_index = tmp_path / "train-index.json"
    cal_index = tmp_path / "cal-index.json"
    parent = tmp_path / "parent.pt"
    bank.write_bytes(b"bank")
    train_index.write_text("{}", encoding="utf-8")
    cal_index.write_text("{}", encoding="utf-8")
    parent.write_bytes(b"parent")
    head = tmp_path / "head.pt"
    torch.save(
        {
            "candidate_bank": {
                "path": bank.as_posix(),
                "sha256": sha256_file(bank),
            },
            "cache_bindings": {
                role: {
                    "index_path": path.as_posix(),
                    "index_sha256": sha256_file(path),
                }
                for role, path in (("train", train_index), ("cal", cal_index))
            },
            "initialization": {
                "parent_checkpoint_path": parent.as_posix(),
                "parent_checkpoint_sha256": sha256_file(parent),
            },
        },
        head,
    )
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint"].update(
        {
            "head_checkpoint_path": head.as_posix(),
            "head_checkpoint_sha256": sha256_file(head),
        }
    )
    metadata.pop("run_spec_sha256")
    metadata["run_spec_sha256"] = canonical_json_sha256(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_spec_sha256"] = metadata["run_spec_sha256"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    metadata, _ = load_complete_rows(directory)
    hashes = result_evidence_hashes(directory, metadata)
    assert {
        path.as_posix() for path in (head, bank, train_index, cal_index, parent)
    } <= set(hashes)
    bank.write_bytes(b"changed")
    with pytest.raises(ValueError, match="candidate bank changed"):
        result_evidence_hashes(directory, metadata)


def test_diagnostic_evidence_binds_cache_bank_and_checkpoints(tmp_path: Path) -> None:
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text("{}", encoding="utf-8")
    backbone = tmp_path / "backbone.pt"
    head = tmp_path / "head.pt"
    bank = tmp_path / "bank.pt"
    index = tmp_path / "index.json"
    for path, content in (
        (backbone, b"backbone"),
        (head, b"head"),
        (bank, b"bank"),
    ):
        path.write_bytes(content)
    index.write_text("{}", encoding="utf-8")
    lock = {
        "analysis_spec_sha256": "a" * 64,
        "protocol_lock_sha256": "b" * 64,
    }
    payload = {
        "checkpoint": {
            "backbone_path": backbone.as_posix(),
            "backbone_sha256": sha256_file(backbone),
            "head_checkpoint_path": head.as_posix(),
            "head_checkpoint_sha256": sha256_file(head),
        },
        "candidate_bank": {
            "path": bank.as_posix(),
            "sha256": sha256_file(bank),
        },
        "cache_binding": {
            "index_path": index.as_posix(),
            "index_sha256": sha256_file(index),
            "split_role": "screen",
            "backbone_seed": 42,
            **lock,
        },
    }
    hashes = diagnostic_evidence_hashes(
        diagnostic,
        payload,
        split_role="screen",
        backbone_seed=42,
        protocol_lock=lock,
    )
    assert set(hashes) == {
        path.as_posix() for path in (diagnostic, backbone, head, bank, index)
    }
    bank.write_bytes(b"changed")
    with pytest.raises(ValueError, match="candidate bank changed"):
        diagnostic_evidence_hashes(
            diagnostic,
            payload,
            split_role="screen",
            backbone_seed=42,
            protocol_lock=lock,
        )


def test_merge_shards_is_complete_disjoint_and_self_verifying(tmp_path: Path) -> None:
    base = tmp_path / "sharded"
    all_rows = [
        _row("task-1", size=11, success=True),
        _row("task-2", size=23, success=False),
    ]
    manifest = base / "manifest.jsonl"
    base.mkdir(parents=True)
    _write_result(
        base / "shard_000_of_002",
        [all_rows[0]],
        shard_index=0,
        num_shards=2,
        manifest_rows=all_rows,
        manifest_path=manifest,
    )
    _write_result(
        base / "shard_001_of_002",
        [all_rows[1]],
        shard_index=1,
        num_shards=2,
        manifest_rows=all_rows,
        manifest_path=manifest,
    )
    merged = merge_shards(base, expected_shards=2)
    metadata, rows = load_complete_rows(merged)
    assert metadata["merged_from_shards"] == 2
    assert [row["task_id"] for row in rows] == ["task-1", "task-2"]


def test_signed_artifact_detects_dependency_tampering(tmp_path: Path) -> None:
    dependency = tmp_path / "evidence.txt"
    dependency.write_text("a", encoding="utf-8")
    from distance_head_study.common import sha256_file

    payload = {
        "protocol_id": "procgen-maze-distance-head-staged-v1",
        "input_hashes": {dependency.as_posix(): sha256_file(dependency)},
    }
    payload["signature"] = canonical_json_sha256(payload)
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    load_signed_artifact(
        artifact,
        signature_field="signature",
        expected_protocol_id="procgen-maze-distance-head-staged-v1",
        verify_hash_fields=("input_hashes",),
    )
    dependency.write_text("b", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency changed"):
        load_signed_artifact(
            artifact,
            signature_field="signature",
            expected_protocol_id="procgen-maze-distance-head-staged-v1",
            verify_hash_fields=("input_hashes",),
        )
