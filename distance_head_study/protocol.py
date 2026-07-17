"""Fail-closed protocol locking and leakage audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from distance_head_study.common import (
    canonical_json_sha256,
    experiment_code_fingerprint,
    load_json,
    load_method_catalog,
    manifest_record,
    read_jsonl,
    resolve_path,
    sha256_file,
    validate_confirmation_seed_freshness,
)
from distance_head_study.generate_bootstrap_schedule import build_schedule
from distance_head_study.generate_manifests import (
    ROLE_SPECS,
    generate_entries,
    serialized,
    validate_entries,
)
from distance_head_study.methods import validate_static_catalog
from distance_head_study.schemas import StudyConfig


def _manifest_paths(config: StudyConfig) -> dict[str, Path]:
    return {
        "train": resolve_path(config.paths.train_manifest),
        "legacy": resolve_path(config.paths.legacy_manifest),
        "cal": resolve_path(config.paths.cal_manifest),
        "screen": resolve_path(config.paths.screen_manifest),
        "select": resolve_path(config.paths.select_manifest),
        "confirm": resolve_path(config.paths.confirm_manifest),
        "stress": resolve_path(config.paths.stress_manifest),
    }


def audit_split_contract(config: StudyConfig, *, regenerate: bool) -> dict[str, Any]:
    paths = _manifest_paths(config)
    rows = {role: read_jsonl(path) for role, path in paths.items()}
    for role in ROLE_SPECS:
        validate_entries(role, rows[role])
        if regenerate and serialized(generate_entries(role)) != paths[role].read_text(
            encoding="utf-8"
        ):
            raise ValueError(f"{role} manifest does not regenerate byte-for-byte")
    if {int(row["maze_size"]) for row in rows["screen"]} != set(range(9, 22, 2)):
        raise ValueError("D_screen contains OOD or missing development sizes")
    if {int(row["maze_size"]) for row in rows["select"]} != set(range(9, 22, 2)):
        raise ValueError("D_select contains OOD or missing development sizes")
    if len(rows["confirm"]) != 900 or {
        int(row["maze_size"]) for row in rows["confirm"]
    } != set(range(9, 26, 2)):
        raise ValueError("D_confirm is not the locked full-900 9..25 split")
    train_layouts = {str(row["layout_hash"]) for row in rows["train"]}
    cal_layouts = {str(row["layout_hash"]) for row in rows["cal"]}
    if not cal_layouts <= train_layouts:
        raise ValueError("D_cal must be a subset of train-role topologies")
    heldout_roles = ("screen", "select", "confirm", "stress")
    overlaps: dict[str, int] = {}
    for left_index, left_role in enumerate(("train", *heldout_roles)):
        left_layouts = {str(row["layout_hash"]) for row in rows[left_role]}
        left_tasks = {str(row["task_hash"]) for row in rows[left_role]}
        for right_role in ("train", *heldout_roles)[left_index + 1 :]:
            right_layouts = {str(row["layout_hash"]) for row in rows[right_role]}
            right_tasks = {str(row["task_hash"]) for row in rows[right_role]}
            key = f"{left_role}__{right_role}"
            overlap = len(left_layouts & right_layouts) + len(left_tasks & right_tasks)
            overlaps[key] = overlap
            if overlap:
                raise ValueError(
                    f"topology/task leakage between {left_role} and {right_role}"
                )
    return {
        "manifest_records": {
            role: manifest_record(path) for role, path in paths.items()
        },
        "heldout_overlap_counts": overlaps,
        "calibration_layouts_are_train_subset": True,
        "screen_count": len(rows["screen"]),
        "select_count": len(rows["select"]),
        "confirm_count": len(rows["confirm"]),
        "stress_count": len(rows["stress"]),
    }


def analysis_spec(config: StudyConfig, split_audit: dict[str, Any]) -> dict[str, Any]:
    source_lock = load_json(config.paths.source_lock)
    return {
        "schema": "distance-head-analysis-spec-v1",
        "protocol_id": config.protocol_id,
        "study_role": config.study_role,
        "splits": config.splits.model_dump(mode="json"),
        "seeds": config.seeds.model_dump(mode="json"),
        "planner": config.planner.model_dump(mode="json"),
        "training": config.training.model_dump(mode="json"),
        "analysis": config.analysis.model_dump(mode="json"),
        "method_catalog_sha256": sha256_file(config.paths.method_catalog),
        "seed_registry_sha256": sha256_file(config.paths.seed_registry),
        "baseline_provenance_sha256": sha256_file(config.paths.baseline_provenance),
        "bootstrap_schedule_sha256": sha256_file(config.paths.bootstrap_schedule),
        "source_config_sha256": sha256_file(config.paths.source_config),
        "source_lock_sha256": sha256_file(config.paths.source_lock),
        "source_analysis_spec_sha256": source_lock["analysis_spec_sha256"],
        "manifests": split_audit["manifest_records"],
        "selection_uses_ood_sizes": False,
        "checkpoint_selection": "final_step",
    }


def build_protocol_lock(
    config: StudyConfig, *, regenerate: bool = True
) -> dict[str, Any]:
    split_audit = audit_split_contract(config, regenerate=regenerate)
    catalog_audit = validate_static_catalog(
        load_method_catalog(config.paths.method_catalog)
    )
    seed_audit = validate_confirmation_seed_freshness(config)
    bootstrap_schedule = load_json(config.paths.bootstrap_schedule)
    expected_bootstrap_schedule = build_schedule(
        config.seeds.bootstrap_seed, config.analysis.bootstrap_samples
    )
    if bootstrap_schedule != expected_bootstrap_schedule:
        raise ValueError("bootstrap schedule does not regenerate exactly")
    source_lock = load_json(config.paths.source_lock)
    if source_lock["train_manifest"]["sha256"] != sha256_file(
        config.paths.train_manifest
    ):
        raise ValueError("D_train differs from the source final_closure protocol")
    provenance = load_json(config.paths.baseline_provenance)
    protocol_repair = provenance.get("protocol_repair")
    if not isinstance(protocol_repair, dict) or (
        protocol_repair.get("new_checkpoint_selection") != "final_step"
    ):
        raise ValueError("baseline provenance does not lock final-step selection")
    spec = analysis_spec(config, split_audit)
    payload = {
        "schema": "distance-head-protocol-lock-v1",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": canonical_json_sha256(spec),
        "analysis_spec": spec,
        "code_fingerprint": experiment_code_fingerprint(),
        "split_audit": split_audit,
        "catalog_audit": catalog_audit,
        "confirmation_seed_audit": seed_audit,
        "bootstrap_schedule_regenerates": True,
        "test_bfs_policy": "oracle_and_diagnostics_only",
        "confirmatory_model_selection": False,
    }
    payload["protocol_lock_sha256"] = canonical_json_sha256(payload)
    return payload


def verify_protocol_lock(
    config: StudyConfig, *, regenerate: bool = False
) -> dict[str, Any]:
    path = resolve_path(config.paths.protocol_lock)
    lock = load_json(path)
    signature = lock.get("protocol_lock_sha256")
    unsigned = {
        key: value for key, value in lock.items() if key != "protocol_lock_sha256"
    }
    if signature != canonical_json_sha256(unsigned):
        raise ValueError("protocol lock signature mismatch")
    expected = build_protocol_lock(config, regenerate=regenerate)
    if lock != expected:
        raise ValueError("protocol lock no longer matches code/config/manifests")
    return lock


__all__ = [
    "analysis_spec",
    "audit_split_contract",
    "build_protocol_lock",
    "verify_protocol_lock",
]
