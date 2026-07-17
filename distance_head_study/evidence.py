"""Transitive file bindings for diagnostic and result-derived decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from distance_head_study.common import resolve_path, sha256_file
from distance_head_study.data import validate_recorded_cache_binding


def _bind_recorded_file(
    hashes: dict[str, str],
    path: str | Path,
    expected_sha256: Any,
    *,
    label: str,
) -> Path:
    resolved = resolve_path(path)
    if not isinstance(expected_sha256, str) or not resolved.exists():
        raise ValueError(f"{label} is missing: {resolved}")
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise ValueError(f"{label} changed: {resolved}")
    existing = hashes.get(resolved.as_posix())
    if existing is not None and existing != observed:
        raise ValueError(f"conflicting evidence hashes for {resolved}")
    hashes[resolved.as_posix()] = observed
    return resolved


def diagnostic_evidence_hashes(
    path: str | Path,
    payload: dict[str, Any],
    *,
    split_role: str,
    backbone_seed: int,
    protocol_lock: dict[str, Any],
) -> dict[str, str]:
    """Bind a diagnostic plus every external file directly used to produce it."""

    diagnostic_path = resolve_path(path)
    hashes = {diagnostic_path.as_posix(): sha256_file(diagnostic_path)}
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("diagnostic omits checkpoint provenance")
    for path_key, hash_key, label in (
        ("backbone_path", "backbone_sha256", "diagnostic backbone"),
        (
            "head_checkpoint_path",
            "head_checkpoint_sha256",
            "diagnostic head checkpoint",
        ),
    ):
        recorded_path = checkpoint.get(path_key)
        if recorded_path is not None:
            _bind_recorded_file(
                hashes,
                recorded_path,
                checkpoint.get(hash_key),
                label=label,
            )
    bank = payload.get("candidate_bank")
    if not isinstance(bank, dict) or not isinstance(bank.get("path"), str):
        raise ValueError("diagnostic omits candidate-bank provenance")
    _bind_recorded_file(
        hashes,
        bank["path"],
        bank.get("sha256"),
        label="diagnostic candidate bank",
    )
    cache_binding = payload.get("cache_binding")
    if not isinstance(cache_binding, dict):
        raise ValueError("diagnostic omits cache provenance")
    cache_index = validate_recorded_cache_binding(
        cache_binding,
        split_role=split_role,
        backbone_seed=backbone_seed,
        protocol_lock=protocol_lock,
    )
    cache_hash = sha256_file(cache_index)
    existing = hashes.get(cache_index.as_posix())
    if existing is not None and existing != cache_hash:
        raise ValueError(f"conflicting evidence hashes for {cache_index}")
    hashes[cache_index.as_posix()] = cache_hash
    return hashes


__all__ = ["diagnostic_evidence_hashes"]
