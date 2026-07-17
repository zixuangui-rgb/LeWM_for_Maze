"""Signed seed-release and sealed-split execution gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from distance_head_study.common import (
    canonical_json_sha256,
    load_json,
    resolve_path,
    sha256_file,
)
from distance_head_study.schemas import StudyConfig


def load_signed_artifact(
    path: str | Path,
    *,
    signature_field: str,
    expected_protocol_id: str | None = None,
    verify_hash_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = load_json(resolved)
    signature = payload.get(signature_field)
    unsigned = {key: value for key, value in payload.items() if key != signature_field}
    if signature != canonical_json_sha256(unsigned):
        raise ValueError(f"signed artifact is invalid: {resolved}")
    if (
        expected_protocol_id is not None
        and payload.get("protocol_id") != expected_protocol_id
    ):
        raise ValueError(f"signed artifact protocol mismatch: {resolved}")
    for field in verify_hash_fields:
        hashes = payload.get(field)
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(
                f"signed artifact has no nonempty hash map {field}: {resolved}"
            )
        for input_path, expected_hash in hashes.items():
            path = resolve_path(str(input_path))
            if not path.exists() or sha256_file(path) != expected_hash:
                raise ValueError(
                    f"signed artifact dependency changed ({field}): {path}"
                )
    return payload


def seed_release_path(config: StudyConfig, tier: str) -> Path:
    return resolve_path(config.paths.seed_release_root) / f"{tier}.json"


def _require_current_lock(config: StudyConfig, artifact: dict[str, Any]) -> None:
    lock = load_json(config.paths.protocol_lock)
    if artifact.get("analysis_spec_sha256") != lock.get("analysis_spec_sha256"):
        raise ValueError("signed gate uses another analysis specification")
    if artifact.get("protocol_lock_sha256") != lock.get("protocol_lock_sha256"):
        raise ValueError("signed gate uses another protocol lock")


def require_seed_released(
    config: StudyConfig,
    *,
    backbone_seed: int,
    head_seed: int | None = None,
) -> dict[str, Any]:
    for tier in ("seed1", "seed3", "seed10"):
        path = seed_release_path(config, tier)
        if not path.exists():
            continue
        try:
            return require_tier_released(
                config,
                tier=tier,
                backbone_seed=backbone_seed,
                head_seed=head_seed,
            )
        except RuntimeError:
            continue
    raise RuntimeError(
        f"backbone/head seed has not been released: {backbone_seed}/{head_seed}"
    )


def require_tier_released(
    config: StudyConfig,
    *,
    tier: str,
    backbone_seed: int,
    head_seed: int | None = None,
) -> dict[str, Any]:
    path = seed_release_path(config, tier)
    artifact = load_signed_artifact(
        path,
        signature_field="release_sha256",
        expected_protocol_id=config.protocol_id,
    )
    _require_current_lock(config, artifact)
    if artifact.get("tier") != tier:
        raise ValueError("seed-release tier/path mismatch")
    for prerequisite, expected_hash in artifact.get("prerequisite_hashes", {}).items():
        prerequisite_path = resolve_path(prerequisite)
        if (
            not prerequisite_path.exists()
            or sha256_file(prerequisite_path) != expected_hash
        ):
            raise ValueError(f"seed-release prerequisite changed: {prerequisite_path}")
    if backbone_seed not in [int(value) for value in artifact["backbone_seeds"]]:
        raise RuntimeError(f"backbone seed is outside released tier {tier}")
    if head_seed is not None and head_seed not in [
        int(value) for value in artifact["head_seeds"]
    ]:
        raise RuntimeError(f"head seed is outside released tier {tier}")
    return artifact


def require_evaluation_gate(
    config: StudyConfig,
    *,
    split_role: str,
    method: str,
    backbone_seed: int,
    head_seed: int,
) -> dict[str, Any] | None:
    baselines = {"b_l2_cem", "b_dh_cem"}
    if split_role == "select":
        artifact = load_signed_artifact(
            config.paths.shortlist_lock,
            signature_field="shortlist_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        _require_current_lock(config, artifact)
        allowed = set(artifact["selected_methods"]) | baselines
        if method not in allowed:
            raise RuntimeError(
                f"method is outside the locked D_select shortlist: {method}"
            )
        require_tier_released(
            config,
            tier="seed3",
            backbone_seed=backbone_seed,
            head_seed=0 if method == "b_l2_cem" else head_seed,
        )
        return artifact
    if split_role == "confirm":
        artifact = load_signed_artifact(
            config.paths.confirm_opened,
            signature_field="confirm_open_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        _require_current_lock(config, artifact)
        if method not in set(artifact["allowed_methods"]):
            raise RuntimeError(
                f"method is outside the sealed confirmation matrix: {method}"
            )
        require_tier_released(
            config,
            tier="seed10",
            backbone_seed=backbone_seed,
            head_seed=0 if method == "b_l2_cem" else head_seed,
        )
        return artifact
    if split_role == "stress":
        artifact = load_signed_artifact(
            config.paths.closure_gate,
            signature_field="closure_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        _require_current_lock(config, artifact)
        if method not in set(artifact["allowed_stress_methods"]):
            raise RuntimeError(
                "stress evaluation is restricted to locked final methods"
            )
        if backbone_seed not in [
            int(value) for value in artifact.get("backbone_seeds", [])
        ]:
            raise RuntimeError(
                "stress evaluation must reuse a confirmation backbone seed"
            )
        expected_head = 0 if method == "b_l2_cem" else int(artifact["head_seed"])
        if head_seed != expected_head:
            raise RuntimeError(
                "stress evaluation must reuse the confirmation head seed"
            )
        require_tier_released(
            config,
            tier="seed10",
            backbone_seed=backbone_seed,
            head_seed=expected_head,
        )
        return artifact
    return None


def load_confirmation_selection(
    config: StudyConfig, n_lock: dict[str, Any]
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    """Load the exact finalist/shortlist pair sealed into confirmation n."""

    finalist_value = n_lock.get("finalist_decision_path")
    shortlist_value = n_lock.get("shortlist_path")
    if not isinstance(finalist_value, str) or not isinstance(shortlist_value, str):
        raise ValueError("confirmation n lock omits selection artifact paths")
    finalist_path = resolve_path(finalist_value)
    shortlist_path = resolve_path(shortlist_value)
    finalist = load_signed_artifact(
        finalist_path,
        signature_field="decision_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    shortlist = load_signed_artifact(
        shortlist_path,
        signature_field="shortlist_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    _require_current_lock(config, finalist)
    _require_current_lock(config, shortlist)
    if finalist.get("decision_sha256") != n_lock.get(
        "finalist_decision_sha256"
    ) or shortlist.get("shortlist_sha256") != n_lock.get("shortlist_sha256"):
        raise ValueError("confirmation selection differs from its n lock")
    source = n_lock.get("selection_source")
    if source == "d_select_finalist":
        expected_finalist = (
            resolve_path(config.paths.decision_root) / "finalist_lock.json"
        )
        if (
            finalist_path != expected_finalist
            or shortlist_path != resolve_path(config.paths.shortlist_lock)
            or tuple(finalist.get("eligible_methods", ()))
            != tuple(shortlist.get("selected_methods", ()))
        ):
            raise ValueError("D_select confirmation selection paths differ")
    elif source == "screen_closure_fallback":
        selected = list(shortlist.get("selected_methods", ()))
        ranked_selected = [
            method
            for method in finalist.get("ranked_methods", ())
            if method in selected
        ]
        if (
            n_lock.get("claim_route") != "negative"
            or shortlist_path != resolve_path(config.paths.negative_shortlist_lock)
            or finalist_path
            != resolve_path(str(shortlist.get("selection_decision_path", "")))
            or finalist.get("decision_name") != "closure_selection"
            or ranked_selected != selected
        ):
            raise ValueError("screen-closure fallback selection differs")
    else:
        raise ValueError("confirmation n lock has an unknown selection source")
    return finalist_path, finalist, shortlist_path, shortlist


__all__ = [
    "load_confirmation_selection",
    "load_signed_artifact",
    "require_evaluation_gate",
    "require_seed_released",
    "require_tier_released",
    "seed_release_path",
]
