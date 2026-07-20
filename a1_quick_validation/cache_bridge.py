"""Rebind immutable source caches to the quick lock without tensor duplication."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from a1_quick_validation.common import (
    DEFAULT_PROFILE,
    atomic_json_dump,
    prepare_immutable,
    resolve_path,
    sha256_file,
)
from a1_quick_validation.profile import load_profile
from distance_head_study.common import load_study_config, source_backbone_path
from distance_head_study.data import (
    ShardedGoalDataset,
    cache_index_path,
    validate_cache_binding,
)
from distance_head_study.protocol import verify_protocol_lock

ALLOWED_ROLES = ("train", "cal", "screen", "select")


def _verify_all_shards(dataset: ShardedGoalDataset) -> None:
    for index, record in enumerate(dataset.records):
        path = resolve_path(record["path"])
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"source cache shard {index} changed: {path}")


def rebind_cache(
    *,
    profile_path: str | Path,
    split_role: str,
    backbone_seed: int,
) -> Path:
    if split_role not in ALLOWED_ROLES:
        raise ValueError(f"cache rebinding is not allowed for {split_role}")
    profile = load_profile(profile_path)
    source_config = load_study_config(profile.paths.source_config)
    quick_config = load_study_config(profile.paths.quick_config)
    source_lock = verify_protocol_lock(source_config)
    quick_lock = verify_protocol_lock(quick_config)
    source_path = cache_index_path(
        source_config, split_role=split_role, backbone_seed=backbone_seed
    )
    source_dataset = ShardedGoalDataset(source_path)
    validate_cache_binding(
        source_dataset,
        source_config,
        split_role=split_role,
        backbone_seed=backbone_seed,
        protocol_lock=source_lock,
    )
    _verify_all_shards(source_dataset)
    source_manifest = resolve_path(
        getattr(source_config.paths, f"{split_role}_manifest")
    )
    quick_manifest = resolve_path(getattr(quick_config.paths, f"{split_role}_manifest"))
    if source_manifest != quick_manifest or sha256_file(source_manifest) != sha256_file(
        quick_manifest
    ):
        raise ValueError("source and quick cache manifests are not identical")
    source_backbone = source_backbone_path(source_config, backbone_seed)
    quick_backbone = source_backbone_path(quick_config, backbone_seed)
    if source_backbone != quick_backbone or sha256_file(source_backbone) != sha256_file(
        quick_backbone
    ):
        raise ValueError("source and quick cache backbones are not identical")
    destination = cache_index_path(
        quick_config, split_role=split_role, backbone_seed=backbone_seed
    )
    prepare_immutable(destination)
    rebound: dict[str, Any] = copy.deepcopy(source_dataset.index)
    rebound.update(
        {
            "split_role": split_role,
            "manifest_path": quick_manifest.as_posix(),
            "manifest_sha256": quick_lock["analysis_spec"]["manifests"][split_role][
                "sha256"
            ],
            "backbone_seed": int(backbone_seed),
            "backbone_path": quick_backbone.as_posix(),
            "backbone_sha256": sha256_file(quick_backbone),
            "analysis_spec_sha256": quick_lock["analysis_spec_sha256"],
            "protocol_lock_sha256": quick_lock["protocol_lock_sha256"],
            "rebound_without_tensor_copy": True,
            "rebound_from_index_path": source_path.as_posix(),
            "rebound_from_index_sha256": sha256_file(source_path),
            "rebound_from_protocol_lock_sha256": source_lock["protocol_lock_sha256"],
        }
    )
    atomic_json_dump(destination, rebound)
    rebound_dataset = ShardedGoalDataset(destination)
    validate_cache_binding(
        rebound_dataset,
        quick_config,
        split_role=split_role,
        backbone_seed=backbone_seed,
        protocol_lock=quick_lock,
    )
    if rebound_dataset.records != source_dataset.records:
        raise RuntimeError("cache rebinding unexpectedly changed shard records")
    _verify_all_shards(rebound_dataset)
    return destination


def validate_quick_cache(
    *, profile_path: str | Path, split_role: str, backbone_seed: int
) -> dict[str, Any]:
    profile = load_profile(profile_path)
    quick_config = load_study_config(profile.paths.quick_config)
    quick_lock = verify_protocol_lock(quick_config)
    path = cache_index_path(
        quick_config, split_role=split_role, backbone_seed=backbone_seed
    )
    dataset = ShardedGoalDataset(path)
    binding = validate_cache_binding(
        dataset,
        quick_config,
        split_role=split_role,
        backbone_seed=backbone_seed,
        protocol_lock=quick_lock,
    )
    if dataset.index.get("rebound_without_tensor_copy") is True:
        source_path = resolve_path(
            str(dataset.index.get("rebound_from_index_path", ""))
        )
        if not source_path.exists() or sha256_file(source_path) != dataset.index.get(
            "rebound_from_index_sha256"
        ):
            raise ValueError("source cache index changed after rebinding")
    elif any(str(key).startswith("rebound_") for key in dataset.index):
        raise ValueError("cache has an incomplete rebound provenance record")
    _verify_all_shards(dataset)
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--split-role", choices=ALLOWED_ROLES, required=True)
    parser.add_argument("--backbone-seed", type=int, required=True)
    args = parser.parse_args()
    print(
        rebind_cache(
            profile_path=args.profile,
            split_role=args.split_role,
            backbone_seed=args.backbone_seed,
        )
    )


if __name__ == "__main__":
    main()
