from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from distance_head_study import EXPERIMENT_FAMILY
from distance_head_study.common import (
    ROOT,
    canonical_json_sha256,
    experiment_code_files,
    load_json,
    load_study_config,
    merge_hash_bindings,
    read_jsonl,
    sha256_file,
    validate_backbone_protocol_binding,
    validate_confirmation_seed_freshness,
)
from distance_head_study.data import (
    CACHE_SCHEMA,
    ShardedGoalDataset,
    _topology_content_sha256,
    _validate_topology_shard,
    validate_cache_binding,
)
from distance_head_study.diagnose import _require_diagnostic_gate
from distance_head_study.evaluate import _require_method_evaluation_gate
from distance_head_study.generate_manifests import calibration_entries
from distance_head_study.protocol import audit_split_contract, build_protocol_lock


def test_calibration_split_is_a_deterministic_train_subset() -> None:
    first = calibration_entries()
    second = calibration_entries()
    assert first == second
    assert len(first) == 140
    assert {row["split_role"] for row in first} == {"cal"}
    train_hashes = {
        row["layout_hash"]
        for row in read_jsonl("data/splits/unisize_train_manifest.jsonl")
    }
    assert {row["layout_hash"] for row in first} <= train_hashes


def test_written_split_contract_has_no_topology_leakage() -> None:
    config = load_study_config("distance_head_study/configs/default.json")
    audit = audit_split_contract(config, regenerate=False)
    assert audit["screen_count"] == 140
    assert audit["select_count"] == 210
    assert audit["confirm_count"] == 900
    assert audit["stress_count"] == 150
    assert all(value == 0 for value in audit["heldout_overlap_counts"].values())


def test_full_protocol_lock_builds_from_repository_provenance() -> None:
    config = load_study_config("distance_head_study/configs/default.json")
    lock = build_protocol_lock(config, regenerate=False)
    assert lock["protocol_id"] == config.protocol_id
    assert lock["analysis_spec"]["checkpoint_selection"] == "final_step"
    assert len(lock["protocol_lock_sha256"]) == 64


def test_protocol_fingerprint_covers_transitive_scientific_dependencies() -> None:
    relative = {path.relative_to(ROOT).as_posix() for path in experiment_code_files()}
    assert {
        "scripts/train/train_dim256.py",
        "final_closure/data.py",
        "hdwm/losses.py",
        "vector_jepa_planner_frontier/common.py",
        "vector_jepa_planner_frontier/schemas.py",
    } <= relative


def test_evidence_hash_flattening_rejects_conflicting_dependencies() -> None:
    assert merge_hash_bindings({"a": "1"}, {"b": "2"}, {"a": "1"}) == {
        "a": "1",
        "b": "2",
    }
    with pytest.raises(ValueError, match="conflicting evidence"):
        merge_hash_bindings({"a": "1"}, {"a": "changed"})


def test_source_train_and_fresh_seed_namespaces_are_locked() -> None:
    config = load_study_config("distance_head_study/configs/default.json")
    source_lock = load_json(config.paths.source_lock)
    assert (
        sha256_file(config.paths.train_manifest)
        == source_lock["train_manifest"]["sha256"]
    )
    freshness = validate_confirmation_seed_freshness(config)
    assert freshness["overlap"] == []
    assert freshness["ordered_confirmation"][0] == 1001


def test_diagnostics_apply_the_same_seed_and_split_gates(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def no_split_gate(config, **kwargs):
        del config
        calls.append(("evaluation", int(kwargs["head_seed"])))
        return None

    def released(config, **kwargs):
        del config
        calls.append(("release", int(kwargs["head_seed"])))
        return {"release_sha256": "released"}

    monkeypatch.setattr(
        "distance_head_study.diagnose.require_evaluation_gate", no_split_gate
    )
    monkeypatch.setattr("distance_head_study.diagnose.require_seed_released", released)
    artifact = _require_diagnostic_gate(
        object(),
        split_role="screen",
        method="b_l2_cem",
        backbone_seed=42,
        head_seed=9,
    )
    assert artifact == {"release_sha256": "released"}
    assert calls == [("evaluation", 9), ("release", 0)]


def test_limited_evaluation_cannot_bypass_method_gate(monkeypatch) -> None:
    calls: list[str] = []

    def split_gate(config, **kwargs):
        del config, kwargs
        calls.append("evaluation")
        return {"shortlist_sha256": "locked"}

    def unexpected_release(config, **kwargs):
        del config, kwargs
        raise AssertionError("split gate should have returned directly")

    monkeypatch.setattr(
        "distance_head_study.evaluate.require_evaluation_gate", split_gate
    )
    monkeypatch.setattr(
        "distance_head_study.evaluate.require_seed_released", unexpected_release
    )
    artifact = _require_method_evaluation_gate(
        object(),
        split_role="select",
        method="candidate",
        backbone_seed=42,
        head_seed=0,
    )
    assert artifact == {"shortlist_sha256": "locked"}
    assert calls == ["evaluation"]


def test_fresh_backbone_must_match_both_study_locks() -> None:
    from final_closure.common import (
        EXPERIMENT_FAMILY as SOURCE_EXPERIMENT_FAMILY,
    )
    from final_closure.common import FORMAT_VERSION as SOURCE_FORMAT_VERSION
    from final_closure.common import (
        analysis_spec_sha256,
        baseline_config,
        training_spec_sha256,
    )

    config = load_study_config("distance_head_study/configs/default.json")
    lock = {
        "analysis_spec_sha256": "analysis",
        "protocol_lock_sha256": "protocol",
    }
    source_config = load_json(config.paths.source_config)
    source_lock = load_json(config.paths.source_lock)
    source_baseline = baseline_config(source_config, "lewm_l2_cem_seqlen2")
    fresh_source_spec = canonical_json_sha256(
        {
            "schema": "distance-head-source-backbone-v1",
            "source_protocol_id": source_config["protocol_id"],
            "source_analysis_spec_sha256": source_lock["analysis_spec_sha256"],
            "source_train_manifest_sha256": source_lock["train_manifest"]["sha256"],
            "baseline": source_baseline,
            "fresh_seed": 1001,
        }
    )
    payload = {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": 1,
        "protocol_id": config.protocol_id,
        "stage": "fresh_source_backbone",
        "baseline_name": "lewm_l2_cem_seqlen2",
        "baseline_kind": source_baseline["kind"],
        "formal_run": True,
        "training_seed": 1001,
        "source_training_spec_sha256": fresh_source_spec,
        "source_analysis_spec_sha256": source_lock["analysis_spec_sha256"],
        "source_train_manifest_sha256": source_lock["train_manifest"]["sha256"],
        "training_config": source_baseline["train"],
        "training": {"optimizer_steps": source_baseline["train"]["steps"]},
        "model_config": {},
        "model_state_dict": {},
        **lock,
    }
    assert (
        validate_backbone_protocol_binding(
            config,
            payload,
            backbone_seed=1001,
            protocol_lock=lock,
        )
        == "fresh_confirmation"
    )
    payload["protocol_lock_sha256"] = "changed"
    with pytest.raises(ValueError, match="protocol binding differs"):
        validate_backbone_protocol_binding(
            config,
            payload,
            backbone_seed=1001,
            protocol_lock=lock,
        )
    historical = {
        "experiment_family": SOURCE_EXPERIMENT_FAMILY,
        "format_version": SOURCE_FORMAT_VERSION,
        "stage": "baseline_training",
        "baseline_name": "lewm_l2_cem_seqlen2",
        "baseline_kind": source_baseline["kind"],
        "training_seed": 42,
        "formal_run": True,
        "rerun": None,
        "analysis_spec_sha256": analysis_spec_sha256(source_config, source_lock),
        "training_spec_sha256": training_spec_sha256(
            source_config,
            source_lock,
            name="lewm_l2_cem_seqlen2",
            seed=42,
        ),
        "training_config": source_baseline["train"],
        "protocol": {
            f"{role}_sha256": source_lock[role]["sha256"]
            for role in (
                "train_manifest",
                "development_manifest",
                "confirmatory_manifest",
            )
        },
        "model_config": {},
        "model_state_dict": {},
    }
    assert (
        validate_backbone_protocol_binding(
            config,
            historical,
            backbone_seed=42,
            protocol_lock=lock,
        )
        == "historical"
    )
    historical["training_seed"] = 43
    with pytest.raises(ValueError, match="historical backbone source binding"):
        validate_backbone_protocol_binding(
            config,
            historical,
            backbone_seed=42,
            protocol_lock=lock,
        )


def _cache_fixture(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"task_hash":"task"}\n', encoding="utf-8")
    backbone = tmp_path / "backbone_42.pt"
    backbone.write_bytes(b"backbone")
    shard = tmp_path / "shard.pt"
    torch.save({"metadata": {"task_hash": "task"}}, shard)
    index = {
        "schema": CACHE_SCHEMA,
        "split_role": "screen",
        "manifest_path": manifest.as_posix(),
        "manifest_sha256": sha256_file(manifest),
        "backbone_seed": 42,
        "backbone_path": backbone.as_posix(),
        "backbone_sha256": sha256_file(backbone),
        "diagnostic_limit": 0,
        "analysis_spec_sha256": "analysis",
        "protocol_lock_sha256": "protocol",
        "records": [
            {
                "position": 0,
                "task_hash": "task",
                "maze_size": 11,
                "path": shard.as_posix(),
                "sha256": sha256_file(shard),
            }
        ],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    config = load_study_config("distance_head_study/configs/default.json")
    paths = config.paths.model_copy(
        update={
            "screen_manifest": manifest,
            "checkpoint_root": tmp_path,
            "legacy_backbone_template": str(tmp_path / "backbone_{backbone_seed}.pt"),
        }
    )
    config = config.model_copy(update={"paths": paths})
    lock = {
        "analysis_spec_sha256": "analysis",
        "protocol_lock_sha256": "protocol",
        "analysis_spec": {
            "manifests": {
                "screen": {
                    "path": manifest.as_posix(),
                    "sha256": sha256_file(manifest),
                    "count": 1,
                }
            }
        },
    }
    return config, lock, index_path, backbone


def test_cache_binding_checks_manifest_backbone_and_count(tmp_path: Path) -> None:
    config, lock, index_path, _ = _cache_fixture(tmp_path)
    dataset = ShardedGoalDataset(index_path)
    record = validate_cache_binding(
        dataset,
        config,
        split_role="screen",
        backbone_seed=42,
        protocol_lock=lock,
    )
    assert record["record_count"] == 1
    assert record["index_path"] == index_path.as_posix()
    assert record["index_sha256"] == sha256_file(index_path)


def test_cache_binding_rejects_changed_backbone(tmp_path: Path) -> None:
    config, lock, index_path, backbone = _cache_fixture(tmp_path)
    backbone.write_bytes(b"changed")
    with pytest.raises(ValueError, match="backbone hash"):
        validate_cache_binding(
            ShardedGoalDataset(index_path),
            config,
            split_role="screen",
            backbone_seed=42,
            protocol_lock=lock,
        )


def test_partial_cache_shard_resume_requires_content_and_protocol_match(
    tmp_path: Path,
) -> None:
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"backbone")
    entry = {
        "task_hash": "task",
        "layout_hash": "layout",
        "maze_size": 11,
        "topology_seed": 123,
        "goal_cell": 1,
    }
    metadata = {
        "task_hash": "task",
        "layout_hash": "layout",
        "maze_size": 11,
        "topology_seed": 123,
        "goal_cell": 1,
        "backbone_path": backbone.as_posix(),
        "backbone_sha256": sha256_file(backbone),
        "analysis_spec_sha256": "analysis",
        "protocol_lock_sha256": "protocol",
        "observation_semantics": "every state rendered with manifest goal",
        "cell_count": 2,
        "goal_index": 1,
        "max_goal_distance": 1,
    }
    bfs = torch.tensor([[0, 1], [1, 0]], dtype=torch.int16)
    next_indices = torch.tensor([[0, 1, 0, 0, 0], [1, 1, 0, 1, 1]], dtype=torch.int32)
    valid_actions = torch.tensor(
        [[False, True, False, False, False], [False, False, True, False, False]]
    )
    payload = {
        "metadata": metadata,
        "entry": entry,
        "cells": torch.tensor([0, 1]),
        "observations": torch.zeros(2, 11, 11, 5, dtype=torch.uint8),
        "latents": torch.zeros(2, 256),
        "all_pairs_bfs": bfs,
        "goal_distances": torch.tensor([1, 0], dtype=torch.int16),
        "next_indices": next_indices,
        "valid_actions": valid_actions,
        "optimal_actions": valid_actions.clone(),
    }
    metadata["content_sha256"] = _topology_content_sha256(payload)
    _validate_topology_shard(
        payload,
        entry,
        backbone_path=backbone,
        analysis_spec_sha256="analysis",
        protocol_lock_sha256="protocol",
    )
    metadata["max_goal_distance"] = 2
    metadata["content_sha256"] = _topology_content_sha256(payload)
    with pytest.raises(ValueError, match="max goal distance"):
        _validate_topology_shard(
            payload,
            entry,
            backbone_path=backbone,
            analysis_spec_sha256="analysis",
            protocol_lock_sha256="protocol",
        )
    metadata["max_goal_distance"] = 1
    metadata["content_sha256"] = _topology_content_sha256(payload)
    payload["latents"][0, 0] = 1.0
    with pytest.raises(ValueError, match="content hash"):
        _validate_topology_shard(
            payload,
            entry,
            backbone_path=backbone,
            analysis_spec_sha256="analysis",
            protocol_lock_sha256="protocol",
        )
    payload["latents"][0, 0] = 0.0
    payload["goal_distances"][0] = 0
    with pytest.raises(ValueError, match="goal distances|content hash"):
        _validate_topology_shard(
            payload,
            entry,
            backbone_path=backbone,
            analysis_spec_sha256="analysis",
            protocol_lock_sha256="protocol",
        )
