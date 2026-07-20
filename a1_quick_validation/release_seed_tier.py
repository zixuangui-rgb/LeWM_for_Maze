"""Release only the backbone/head seeds admitted by the quick profile."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from a1_quick_validation.common import (
    DEFAULT_PROFILE,
    atomic_json_dump,
    prepare_immutable,
    resolve_path,
    sha256_file,
)
from a1_quick_validation.profile import verify_package_lock
from a1_quick_validation.selection import load_q1_shortlist
from distance_head_study.common import (
    canonical_json_sha256,
    load_study_config,
    merge_hash_bindings,
)
from distance_head_study.gates import load_signed_artifact, seed_release_path
from distance_head_study.protocol import verify_protocol_lock

TIERS = ("seed1", "seed3")


def expected_seed_matrix(
    profile: Any, tier: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if tier == "seed1":
        return profile.q1.backbone_seeds, profile.q1.head_seeds
    if tier == "seed3":
        return profile.q2.backbone_seeds, profile.q2.head_seeds
    raise ValueError(f"unsupported quick seed tier: {tier}")


def _prerequisite_hashes(
    profile: Any,
    package_lock: dict[str, Any],
    config: Any,
    lock: dict[str, Any],
    *,
    tier: str,
) -> dict[str, str]:
    package_path = resolve_path(profile.paths.package_lock)
    hashes = {package_path.as_posix(): sha256_file(package_path)}
    if tier == "seed1":
        return hashes
    shortlist_path = resolve_path(config.paths.shortlist_lock)
    shortlist = load_q1_shortlist(
        config,
        lock,
        package_lock_sha256=package_lock["package_lock_sha256"],
    )
    return merge_hash_bindings(
        hashes,
        {shortlist_path.as_posix(): sha256_file(shortlist_path)},
        shortlist["input_hashes"],
    )


def release_seed_tier(profile_path: str | Path = DEFAULT_PROFILE, *, tier: str) -> Path:
    profile, package_lock, lock = verify_package_lock(profile_path)
    config = load_study_config(profile.paths.quick_config)
    verify_protocol_lock(config)
    backbones, heads = expected_seed_matrix(profile, tier)
    output = seed_release_path(config, tier)
    prepare_immutable(output)
    payload = {
        "schema": "distance-head-seed-release-v1",
        "protocol_id": config.protocol_id,
        "quick_profile_id": profile.profile_id,
        "package_lock_sha256": package_lock["package_lock_sha256"],
        "tier": tier,
        "backbone_seeds": list(backbones),
        "head_seeds": list(heads),
        "evidence_status": (
            "exploratory_single_backbone"
            if tier == "seed1"
            else "independent_split_two_head_seeds"
        ),
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "prerequisite_hashes": _prerequisite_hashes(
            profile, package_lock, config, lock, tier=tier
        ),
        "profile_exact_seed_scope": True,
        "no_performance_based_seed_skipping": True,
    }
    payload["release_sha256"] = canonical_json_sha256(payload)
    atomic_json_dump(output, payload)
    return output


def validate_seed_release(
    profile_path: str | Path = DEFAULT_PROFILE, *, tier: str
) -> dict[str, Any]:
    profile, package_lock, lock = verify_package_lock(profile_path)
    config = load_study_config(profile.paths.quick_config)
    path = seed_release_path(config, tier)
    payload = load_signed_artifact(
        path,
        signature_field="release_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("prerequisite_hashes",),
    )
    backbones, heads = expected_seed_matrix(profile, tier)
    expected = (
        payload.get("quick_profile_id") == profile.profile_id
        and payload.get("package_lock_sha256") == package_lock["package_lock_sha256"]
        and payload.get("tier") == tier
        and tuple(int(value) for value in payload.get("backbone_seeds", ()))
        == backbones
        and tuple(int(value) for value in payload.get("head_seeds", ())) == heads
        and payload.get("analysis_spec_sha256") == lock["analysis_spec_sha256"]
        and payload.get("protocol_lock_sha256") == lock["protocol_lock_sha256"]
        and payload.get("profile_exact_seed_scope") is True
        and payload.get("no_performance_based_seed_skipping") is True
    )
    if not expected:
        raise ValueError("seed release differs from the exact quick-profile scope")
    expected_prerequisites = _prerequisite_hashes(
        profile, package_lock, config, lock, tier=tier
    )
    if payload.get("prerequisite_hashes") != expected_prerequisites:
        raise ValueError("seed release prerequisite closure differs")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--tier", choices=TIERS, required=True)
    args = parser.parse_args()
    print(release_seed_tier(args.profile, tier=args.tier))


if __name__ == "__main__":
    main()


__all__ = [
    "TIERS",
    "expected_seed_matrix",
    "release_seed_tier",
    "validate_seed_release",
]
