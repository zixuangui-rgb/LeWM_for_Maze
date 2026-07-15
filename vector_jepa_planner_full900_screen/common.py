"""Shared provenance, locking, and path helpers for the full-900 screen."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from final_closure.common import sha256_file
from spatial_jepa_planning.common import canonical_json_sha256
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_json_dump,
    method_by_name,
    training_spec_sha256,
)
from vector_jepa_planner_frontier.common import (
    component_checkpoint_path as frontier_component_checkpoint_path,
)
from vector_jepa_planner_frontier.schemas import MethodConfig
from vector_jepa_planner_full900_screen import (
    EXPERIMENT_FAMILY,
    FORMAT_VERSION,
    PROTOCOL_ID,
)
from vector_jepa_planner_full900_screen.schemas import QuickStudyConfig

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_json(path: str | Path) -> dict[str, Any]:
    with open(resolve_path(path), encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_text_dump(path: str | Path, value: str) -> None:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: str | Path) -> QuickStudyConfig:
    return QuickStudyConfig.model_validate(load_json(path))


def role_by_name(config: QuickStudyConfig, name: str) -> Any:
    matches = [role for role in config.method_roles if role.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one role named {name!r}")
    return matches[0]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty() -> bool:
    watched = (
        "vector_jepa_planner_full900_screen",
        "vector_jepa_planner_frontier",
        "final_closure",
        "hdwm",
        "spatial_jepa_planning/common.py",
        "scripts/train/train_dim256.py",
        "data/splits",
        "pyproject.toml",
    )
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *watched],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(output.strip())


def require_clean_worktree(*, allow_dirty: bool = False) -> None:
    if git_worktree_dirty() and not allow_dirty:
        raise RuntimeError("formal full-900 runs require a clean committed worktree")


def code_fingerprint() -> str:
    files = list(PACKAGE_ROOT.rglob("*.py"))
    files.extend((ROOT / "vector_jepa_planner_frontier").rglob("*.py"))
    files.extend((ROOT / "hdwm").rglob("*.py"))
    files.extend((ROOT / "final_closure").rglob("*.py"))
    files.extend(
        [
            ROOT / "spatial_jepa_planning/common.py",
            ROOT / "scripts/train/train_dim256.py",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ]
    )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def quick_spec_sha256(config: QuickStudyConfig, lock: dict[str, Any]) -> str:
    payload = {
        "schema": "vector-jepa-full900-screen-spec-v1",
        "protocol_id": config.protocol_id,
        "replication": config.replication.model_dump(mode="json"),
        "gates": config.gates.model_dump(mode="json"),
        "method_roles": [role.model_dump(mode="json") for role in config.method_roles],
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "locked_documents": {
            key: lock[key]["sha256"]
            for key in (
                "protocol_document",
                "methods_document",
                "compatibility_document",
                "runbook_document",
                "claims_document",
                "result_schema_document",
                "readme_document",
                "implementation_audit_document",
                "handoff_document",
                "validation_document",
                "test_suite",
            )
        },
    }
    return canonical_json_sha256(payload)


def validate_lock(config: QuickStudyConfig, lock: dict[str, Any]) -> None:
    if lock.get("status") != "locked":
        raise RuntimeError("formal operation requires a completed quick-screen lock")
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol lock belongs to another study")
    if lock.get("code_fingerprint") != code_fingerprint():
        raise ValueError("quick-screen code no longer matches the protocol lock")
    records = (
        "amendments",
        "amendment_document",
        "amendment_before",
        "amendment_after",
        "protocol_document",
        "methods_document",
        "compatibility_document",
        "runbook_document",
        "claims_document",
        "result_schema_document",
        "readme_document",
        "implementation_audit_document",
        "handoff_document",
        "validation_document",
        "test_suite",
        "method_config",
        "environment_spec",
        "environment_lock",
    )
    for key in records:
        record = lock.get(key, {})
        path = record.get("path")
        expected = record.get("sha256")
        if not path or not expected or sha256_file(resolve_path(path)) != expected:
            raise ValueError(f"locked artifact hash mismatch: {key}:{path}")
    for role in (
        "train_manifest",
        "development_manifest",
        "validation_manifest",
        "confirmatory_manifest",
    ):
        record = lock.get(role, {})
        path = resolve_path(getattr(config.paths, role))
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"locked manifest hash mismatch: {role}")
    source = lock.get("source_baseline", {})
    source_records = (
        ("config", config.paths.source_config, "config_sha256"),
        ("lock", config.paths.source_lock, "lock_sha256"),
    )
    for name, configured_path, digest_key in source_records:
        recorded_path = source.get(f"{name}_path")
        expected = source.get(digest_key)
        resolved = resolve_path(configured_path)
        if (
            recorded_path != str(resolved.relative_to(ROOT))
            or not expected
            or sha256_file(resolved) != expected
        ):
            raise ValueError(f"source baseline {name} hash/path mismatch")
    if lock.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("analysis specification no longer reproduces")
    if lock.get("quick_spec_sha256") != quick_spec_sha256(config, lock):
        raise ValueError("quick-screen decision specification no longer reproduces")


def component_checkpoint_path(
    config: QuickStudyConfig,
    method: MethodConfig,
    *,
    backbone_seed: int,
    planner_seed: int,
) -> Path | None:
    if not method.component_checkpoint_required:
        return None
    if method.stage == "P6" and method.scorer.counterexample_ranker_weight > 0.0:
        return resolve_path(
            config.paths.counterexample_round_template.format(
                method=method.name,
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
                round=config.training.counterexample_rounds,
            )
        )
    return frontier_component_checkpoint_path(
        config,
        method,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
    )


def result_path(
    config: QuickStudyConfig,
    *,
    method: str,
    backbone_seed: int,
    planner_seed: int,
    action_selection: str,
) -> Path:
    return resolve_path(
        config.paths.result_template.format(
            method=method,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
            search_seed=0,
            split=config.replication.evaluation_manifest_role,
            action_selection=action_selection,
        )
    )


def planner_seeds(
    config: QuickStudyConfig, method: MethodConfig, *, final: bool
) -> tuple[int, ...]:
    if not method.component_checkpoint_required:
        return (0,)
    return (
        config.replication.final_planner_seeds
        if final
        else config.replication.screen_planner_seeds
    )


def metadata(
    config: QuickStudyConfig,
    lock: dict[str, Any],
    *,
    method: MethodConfig,
    backbone_seed: int,
    planner_seed: int | None,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "study_role": config.study_role,
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "quick_spec_sha256": quick_spec_sha256(config, lock),
        "method": method.model_dump(mode="json"),
        "method_sha256": canonical_json_sha256(method.model_dump(mode="json")),
        "backbone_seed": int(backbone_seed),
        "planner_seed": int(planner_seed) if planner_seed is not None else None,
        "search_seed": None,
        "evaluation_seed": config.protocol.evaluation_seed,
        "device": str(device),
        "git_commit": git_commit(),
        "git_dirty": git_worktree_dirty(),
        "code_fingerprint": code_fingerprint(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


__all__ = [
    "ROOT",
    "analysis_spec_sha256",
    "atomic_json_dump",
    "atomic_text_dump",
    "code_fingerprint",
    "component_checkpoint_path",
    "load_config",
    "load_json",
    "metadata",
    "method_by_name",
    "planner_seeds",
    "quick_spec_sha256",
    "require_clean_worktree",
    "resolve_path",
    "result_path",
    "role_by_name",
    "training_spec_sha256",
    "validate_lock",
]
