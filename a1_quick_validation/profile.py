"""Load and verify the inner scientific lock and outer package lock."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from a1_quick_validation import PACKAGE_LOCK_SCHEMA, PROFILE_ID
from a1_quick_validation.common import (
    DEFAULT_PACKAGE_LOCK,
    DEFAULT_PROFILE,
    canonical_json_sha256,
    load_json,
    package_files,
    relative,
    resolve_path,
    sha256_file,
)
from a1_quick_validation.schemas import QuickProfile
from distance_head_study.common import load_study_config
from distance_head_study.protocol import verify_protocol_lock


def load_profile(path: str | Path = DEFAULT_PROFILE) -> QuickProfile:
    return QuickProfile.model_validate(load_json(path))


def build_package_lock(
    profile_path: str | Path = DEFAULT_PROFILE,
    *,
    regenerate_protocols: bool = True,
) -> dict[str, Any]:
    profile = load_profile(profile_path)
    quick_config = load_study_config(profile.paths.quick_config)
    source_config = load_study_config(profile.paths.source_config)
    if resolve_path(quick_config.paths.protocol_lock) != resolve_path(
        profile.paths.quick_protocol_lock
    ):
        raise ValueError("profile and quick config point at different protocol locks")
    if resolve_path(source_config.paths.protocol_lock) != resolve_path(
        profile.paths.source_protocol_lock
    ):
        raise ValueError("profile and source config point at different protocol locks")
    if regenerate_protocols:
        quick_lock = verify_protocol_lock(quick_config, regenerate=True)
        source_lock = verify_protocol_lock(source_config, regenerate=True)
    else:
        quick_lock = _load_signed_protocol_lock(profile.paths.quick_protocol_lock)
        source_lock = _load_signed_protocol_lock(profile.paths.source_protocol_lock)
    files = {relative(path): sha256_file(path) for path in package_files()}
    payload: dict[str, Any] = {
        "schema": PACKAGE_LOCK_SCHEMA,
        "profile_id": PROFILE_ID,
        "profile_path": relative(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "quick_protocol_lock_path": relative(profile.paths.quick_protocol_lock),
        "quick_protocol_lock_sha256": sha256_file(profile.paths.quick_protocol_lock),
        "quick_analysis_spec_sha256": quick_lock["analysis_spec_sha256"],
        "source_protocol_lock_path": relative(profile.paths.source_protocol_lock),
        "source_protocol_lock_sha256": sha256_file(profile.paths.source_protocol_lock),
        "source_analysis_spec_sha256": source_lock["analysis_spec_sha256"],
        "package_files": files,
        "claim_boundary": profile.claim_boundary,
    }
    payload["package_lock_sha256"] = canonical_json_sha256(payload)
    return payload


def _load_signed_protocol_lock(path: str | Path) -> dict[str, Any]:
    lock = load_json(path)
    signature = lock.get("protocol_lock_sha256")
    unsigned = {
        key: value for key, value in lock.items() if key != "protocol_lock_sha256"
    }
    if signature != canonical_json_sha256(unsigned):
        raise ValueError(f"protocol-lock signature mismatch: {resolve_path(path)}")
    analysis = lock.get("analysis_spec")
    if not isinstance(analysis, dict) or lock.get(
        "analysis_spec_sha256"
    ) != canonical_json_sha256(analysis):
        raise ValueError(f"analysis-spec signature mismatch: {resolve_path(path)}")
    return lock


def verify_package_lock(
    profile_path: str | Path = DEFAULT_PROFILE,
    lock_path: str | Path = DEFAULT_PACKAGE_LOCK,
) -> tuple[QuickProfile, dict[str, Any], dict[str, Any]]:
    profile = load_profile(profile_path)
    lock = load_json(lock_path)
    signature = lock.get("package_lock_sha256")
    unsigned = {
        key: value for key, value in lock.items() if key != "package_lock_sha256"
    }
    if signature != canonical_json_sha256(unsigned):
        raise ValueError("quick-validation package-lock signature mismatch")
    expected = build_package_lock(profile_path, regenerate_protocols=False)
    if lock != expected:
        raise ValueError("quick-validation package changed after package locking")
    quick_lock = _load_signed_protocol_lock(profile.paths.quick_protocol_lock)
    if resolve_path(profile.paths.package_lock) != resolve_path(lock_path):
        raise ValueError("profile points at another package lock")
    return profile, lock, quick_lock


__all__ = ["build_package_lock", "load_profile", "verify_package_lock"]
