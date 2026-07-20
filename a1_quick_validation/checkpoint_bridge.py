"""Import exact reference heads after proving training inputs are identical."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

import torch

from a1_quick_validation.common import DEFAULT_PROFILE, prepare_immutable
from a1_quick_validation.profile import load_profile
from distance_head_study.candidates import candidate_bank_path, load_candidate_bank
from distance_head_study.common import (
    atomic_torch_save,
    canonical_json_sha256,
    head_checkpoint_path,
    hierarchical_seed,
    load_study_config,
    sha256_file,
    source_backbone_path,
)
from distance_head_study.data import (
    ShardedGoalDataset,
    cache_index_path,
    validate_cache_binding,
    validate_recorded_cache_binding,
)
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.models import build_distance_head
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.train_head import TRAIN_STATE_SCHEMA

IMPORTABLE_METHODS = ("b_dh_cem", "a1_log")


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _quick_cache_bindings(
    quick_config: Any, quick_lock: dict[str, Any], backbone_seed: int
) -> dict[str, dict[str, Any]]:
    bindings = {}
    for role in ("train", "cal"):
        dataset = ShardedGoalDataset(
            cache_index_path(quick_config, split_role=role, backbone_seed=backbone_seed)
        )
        bindings[role] = validate_cache_binding(
            dataset,
            quick_config,
            split_role=role,
            backbone_seed=backbone_seed,
            protocol_lock=quick_lock,
        )
    return bindings


def import_reference_checkpoint(
    *,
    profile_path: str | Path,
    method_name: str,
    backbone_seed: int,
    head_seed: int,
) -> Path:
    if method_name not in IMPORTABLE_METHODS:
        raise ValueError(f"checkpoint import is forbidden for {method_name}")
    profile = load_profile(profile_path)
    source_config = load_study_config(profile.paths.source_config)
    quick_config = load_study_config(profile.paths.quick_config)
    source_lock = verify_protocol_lock(source_config)
    quick_lock = verify_protocol_lock(quick_config)
    source_method, source_method_hash, _ = load_and_resolve_method(
        source_config.paths.method_catalog,
        method_name,
        decision_root=source_config.paths.decision_root,
        protocol_lock=source_lock,
    )
    quick_method, quick_method_hash, _ = load_and_resolve_method(
        quick_config.paths.method_catalog,
        method_name,
        decision_root=quick_config.paths.decision_root,
        protocol_lock=quick_lock,
    )
    if source_method != quick_method or source_method_hash != quick_method_hash:
        raise ValueError("reference method is not byte-semantically identical")
    source_path = head_checkpoint_path(
        source_config,
        method=method_name,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
    )
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    required = {
        "formal_run": True,
        "protocol_id": source_config.protocol_id,
        "analysis_spec_sha256": source_lock["analysis_spec_sha256"],
        "protocol_lock_sha256": source_lock["protocol_lock_sha256"],
        "method_sha256": source_method_hash,
        "backbone_seed": int(backbone_seed),
        "head_seed": int(head_seed),
        "checkpoint_selection": "final_step",
        "final_step": int(source_config.training.steps),
    }
    for key, expected in required.items():
        if source.get(key) != expected:
            raise ValueError(f"source checkpoint field differs: {key}")
    if source.get("method") != source_method.model_dump(mode="json"):
        raise ValueError("source checkpoint method metadata differs")
    if source.get("head_spec") != quick_method.head.model_dump(mode="json"):
        raise ValueError("source checkpoint head architecture differs")
    source_backbone = source_backbone_path(source_config, backbone_seed)
    quick_backbone = source_backbone_path(quick_config, backbone_seed)
    if source_backbone != quick_backbone:
        raise ValueError("source and quick backbone paths differ")
    if source.get("backbone_path") != source_backbone.as_posix() or source.get(
        "backbone_sha256"
    ) != sha256_file(source_backbone):
        raise ValueError("source checkpoint backbone binding differs")
    head = build_distance_head(quick_method.head)
    head.load_state_dict(source["head_state_dict"], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in head.parameters()):
        raise ValueError("source checkpoint has non-finite head parameters")
    source_bindings = source.get("cache_bindings")
    if not isinstance(source_bindings, dict) or set(source_bindings) != {
        "train",
        "cal",
    }:
        raise ValueError("source checkpoint cache bindings are incomplete")
    for role, binding in source_bindings.items():
        validate_recorded_cache_binding(
            binding,
            split_role=role,
            backbone_seed=backbone_seed,
            protocol_lock=source_lock,
        )
    quick_bindings = _quick_cache_bindings(quick_config, quick_lock, backbone_seed)
    for role in ("train", "cal"):
        source_index = ShardedGoalDataset(source_bindings[role]["index_path"])
        quick_index = ShardedGoalDataset(quick_bindings[role]["index_path"])
        source_content = [
            (record["task_hash"], record["sha256"]) for record in source_index.records
        ]
        quick_content = [
            (record["task_hash"], record["sha256"]) for record in quick_index.records
        ]
        if source_content != quick_content:
            raise ValueError(f"{role} cache tensor records differ")
    source_bank_path = source["candidate_bank"]["path"]
    if source["candidate_bank"].get("sha256") != sha256_file(source_bank_path):
        raise ValueError("source checkpoint candidate bank changed")
    source_bank_metadata, source_actions = load_candidate_bank(source_bank_path)
    quick_bank_path = candidate_bank_path(
        quick_config, split_role="train", backbone_seed=backbone_seed
    )
    quick_bank_metadata, quick_actions = load_candidate_bank(quick_bank_path)
    if not torch.equal(source_actions, quick_actions):
        raise ValueError("source and quick candidate action tensors differ")
    for field in (
        "split_role",
        "backbone_seed",
        "set_count",
        "candidate_count",
        "horizon",
        "allowed_actions",
        "schedule_seed",
        "actions_sha256",
    ):
        if source_bank_metadata.get(field) != quick_bank_metadata.get(field):
            raise ValueError(f"candidate bank scientific field differs: {field}")
    quick_candidate_binding = {
        "path": quick_bank_path.as_posix(),
        "sha256": sha256_file(quick_bank_path),
        "metadata": quick_bank_metadata,
    }
    training_rng_seed = hierarchical_seed(
        "distance-head-training-rng", backbone_seed, head_seed
    )
    training_spec = canonical_json_sha256(
        {
            "schema": TRAIN_STATE_SCHEMA,
            "analysis_spec_sha256": quick_lock["analysis_spec_sha256"],
            "protocol_lock_sha256": quick_lock["protocol_lock_sha256"],
            "method_sha256": quick_method_hash,
            "backbone_sha256": source["backbone_sha256"],
            "backbone_seed": int(backbone_seed),
            "head_seed": int(head_seed),
            "training_rng_seed": training_rng_seed,
            "steps": int(quick_config.training.steps),
            "calibrated_weights": source["calibrated_weights"],
            "candidate_bank": quick_candidate_binding,
            "cache_bindings": quick_bindings,
            "initialization": source["initialization"],
        }
    )
    rebound = copy.deepcopy(source)
    rebound.update(
        {
            "analysis_spec_sha256": quick_lock["analysis_spec_sha256"],
            "protocol_lock_sha256": quick_lock["protocol_lock_sha256"],
            "method": quick_method.model_dump(mode="json"),
            "method_sha256": quick_method_hash,
            "training_spec_sha256": training_spec,
            "cache_bindings": quick_bindings,
            "candidate_bank": quick_candidate_binding,
            "artifact_rebound_without_retraining": True,
            "rebind_provenance": {
                "source_checkpoint_path": source_path.as_posix(),
                "source_checkpoint_sha256": sha256_file(source_path),
                "source_analysis_spec_sha256": source_lock["analysis_spec_sha256"],
                "source_protocol_lock_sha256": source_lock["protocol_lock_sha256"],
                "head_state_sha256": state_dict_sha256(source["head_state_dict"]),
                "identical_method_sha256": quick_method_hash,
                "identical_cache_records": True,
                "identical_candidate_actions": True,
            },
        }
    )
    destination = head_checkpoint_path(
        quick_config,
        method=method_name,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
    )
    prepare_immutable(destination)
    atomic_torch_save(destination, rebound)
    written = torch.load(destination, map_location="cpu", weights_only=False)
    if state_dict_sha256(written["head_state_dict"]) != state_dict_sha256(
        source["head_state_dict"]
    ):
        raise RuntimeError("checkpoint import changed head parameters")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--method", choices=IMPORTABLE_METHODS, required=True)
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--head-seed", type=int, required=True)
    args = parser.parse_args()
    print(
        import_reference_checkpoint(
            profile_path=args.profile,
            method_name=args.method,
            backbone_seed=args.backbone_seed,
            head_seed=args.head_seed,
        )
    )


if __name__ == "__main__":
    main()
