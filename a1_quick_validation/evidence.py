"""Strict, transitive evidence validation for quick-stage artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from a1_quick_validation.cache_bridge import validate_quick_cache
from a1_quick_validation.checkpoint_bridge import (
    IMPORTABLE_METHODS,
    state_dict_sha256,
)
from a1_quick_validation.common import resolve_path, sha256_file
from a1_quick_validation.profile import load_profile
from distance_head_study import ACTION_IDS
from distance_head_study.candidates import candidate_bank_path, load_candidate_bank
from distance_head_study.common import (
    canonical_json_sha256,
    head_checkpoint_path,
    hierarchical_seed,
    load_study_config,
    merge_hash_bindings,
    source_backbone_path,
)
from distance_head_study.data import (
    ShardedGoalDataset,
    cache_index_path,
    validate_recorded_cache_binding,
)
from distance_head_study.evaluate import _load_models
from distance_head_study.evidence import diagnostic_evidence_hashes
from distance_head_study.gates import load_signed_artifact
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.results import (
    load_complete_rows,
    result_directory,
    result_evidence_hashes,
)
from distance_head_study.train_head import TRAIN_STATE_SCHEMA


def _bind_file(
    hashes: dict[str, str],
    path: str | Path,
    expected_sha256: Any,
    *,
    label: str,
) -> dict[str, str]:
    resolved = resolve_path(path)
    if not isinstance(expected_sha256, str) or not resolved.exists():
        raise ValueError(f"{label} is missing: {resolved}")
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise ValueError(f"{label} changed: {resolved}")
    return merge_hash_bindings(hashes, {resolved.as_posix(): observed})


def cache_binding_evidence_hashes(
    binding: dict[str, Any],
    *,
    split_role: str,
    backbone_seed: int,
    protocol_lock: dict[str, Any],
    expected_index_path: str | Path | None = None,
) -> dict[str, str]:
    """Validate a cache index, every shard, and optional rebound source index."""

    index_path = validate_recorded_cache_binding(
        binding,
        split_role=split_role,
        backbone_seed=backbone_seed,
        protocol_lock=protocol_lock,
    )
    if expected_index_path is not None and index_path != resolve_path(
        expected_index_path
    ):
        raise ValueError(f"cache index is outside its canonical path: {index_path}")
    hashes = {index_path.as_posix(): sha256_file(index_path)}
    dataset = ShardedGoalDataset(index_path)
    for index, record in enumerate(dataset.records):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"cache shard record {index} is malformed: {index_path}")
        hashes = _bind_file(
            hashes,
            record["path"],
            record.get("sha256"),
            label=f"cache shard {index}",
        )

    rebound = dataset.index.get("rebound_without_tensor_copy")
    rebound_keys = {key for key in dataset.index if str(key).startswith("rebound_")}
    if rebound is True:
        source_path = dataset.index.get("rebound_from_index_path")
        hashes = _bind_file(
            hashes,
            str(source_path or ""),
            dataset.index.get("rebound_from_index_sha256"),
            label="rebound source cache index",
        )
        source_dataset = ShardedGoalDataset(str(source_path))
        if source_dataset.records != dataset.records:
            raise ValueError("rebound cache records differ from their source index")
    elif rebound_keys:
        raise ValueError("cache has incomplete rebound provenance")
    return hashes


def quick_cache_evidence_hashes(
    profile_path: str | Path,
    *,
    split_role: str,
    backbone_seed: int,
) -> dict[str, str]:
    profile = load_profile(profile_path)
    config = load_study_config(profile.paths.quick_config)
    lock = verify_protocol_lock(config)
    binding = validate_quick_cache(
        profile_path=profile_path,
        split_role=split_role,
        backbone_seed=backbone_seed,
    )
    return cache_binding_evidence_hashes(
        binding,
        split_role=split_role,
        backbone_seed=backbone_seed,
        protocol_lock=lock,
        expected_index_path=cache_index_path(
            config, split_role=split_role, backbone_seed=backbone_seed
        ),
    )


def validate_candidate_bank(
    config: Any,
    protocol_lock: dict[str, Any],
    *,
    backbone_seed: int,
    path: str | Path | None = None,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, str]]:
    expected_path = candidate_bank_path(
        config, split_role="train", backbone_seed=backbone_seed
    )
    actual_path = expected_path if path is None else resolve_path(path)
    if actual_path != expected_path:
        raise ValueError("candidate bank is outside its canonical quick-study path")
    metadata, actions = load_candidate_bank(actual_path)
    expected = {
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": protocol_lock["analysis_spec_sha256"],
        "protocol_lock_sha256": protocol_lock["protocol_lock_sha256"],
        "split_role": "train",
        "backbone_seed": int(backbone_seed),
        "set_count": int(config.training.candidate_sets_per_backbone),
        "candidate_count": int(config.training.trajectory_candidates),
        "horizon": int(config.planner.horizon),
        "allowed_actions": list(ACTION_IDS),
        "schedule_seed": int(config.seeds.sample_schedule_seed),
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    expected_shape = (
        config.training.candidate_sets_per_backbone,
        config.training.trajectory_candidates,
        config.planner.horizon,
    )
    if tuple(actions.shape) != expected_shape:
        mismatches["shape"] = {
            "expected": list(expected_shape),
            "observed": list(actions.shape),
        }
    unique = torch.tensor(
        [torch.unique(item, dim=0).shape[0] for item in actions], dtype=torch.int64
    )
    if int(unique.min()) < int(0.95 * config.training.trajectory_candidates):
        mismatches["minimum_unique_candidates"] = {
            "expected": int(0.95 * config.training.trajectory_candidates),
            "observed": int(unique.min()),
        }
    if mismatches:
        raise ValueError(
            "candidate bank differs from the locked protocol: "
            f"{canonical_json_sha256(mismatches)}"
        )
    return metadata, actions, {actual_path.as_posix(): sha256_file(actual_path)}


def _checkpoint_payload_evidence(
    config: Any,
    lock: dict[str, Any],
    payload: dict[str, Any],
    *,
    backbone_seed: int,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    bank = payload.get("candidate_bank")
    if not isinstance(bank, dict) or not isinstance(bank.get("path"), str):
        raise ValueError("head checkpoint omits candidate-bank provenance")
    _, _, bank_hashes = validate_candidate_bank(
        config,
        lock,
        backbone_seed=backbone_seed,
        path=bank["path"],
    )
    if bank.get("sha256") != next(iter(bank_hashes.values())):
        raise ValueError("head checkpoint candidate-bank hash differs")
    hashes = merge_hash_bindings(hashes, bank_hashes)
    bindings = payload.get("cache_bindings")
    if not isinstance(bindings, dict) or set(bindings) != {"train", "cal"}:
        raise ValueError("head checkpoint cache bindings are incomplete")
    for split_role, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ValueError("head checkpoint cache binding is malformed")
        hashes = merge_hash_bindings(
            hashes,
            cache_binding_evidence_hashes(
                binding,
                split_role=split_role,
                backbone_seed=backbone_seed,
                protocol_lock=lock,
                expected_index_path=cache_index_path(
                    config,
                    split_role=split_role,
                    backbone_seed=backbone_seed,
                ),
            ),
        )
    initialization = payload.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("head checkpoint initialization provenance is malformed")
    parent = initialization.get("parent_checkpoint_path")
    if parent is not None:
        hashes = _bind_file(
            hashes,
            parent,
            initialization.get("parent_checkpoint_sha256"),
            label="head initialization parent",
        )
    return hashes


def validate_quick_checkpoint(
    profile_path: str | Path,
    *,
    method_name: str,
    backbone_seed: int,
    head_seed: int,
) -> dict[str, str]:
    """Fully load a quick checkpoint and bind its complete training lineage."""

    profile = load_profile(profile_path)
    config = load_study_config(profile.paths.quick_config)
    lock = verify_protocol_lock(config)
    method, method_hash, decisions = load_and_resolve_method(
        config.paths.method_catalog,
        method_name,
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    _, head, provenance = _load_models(
        config,
        method,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
        device=torch.device("cpu"),
        expected_analysis_spec_sha256=lock["analysis_spec_sha256"],
        expected_protocol_lock_sha256=lock["protocol_lock_sha256"],
    )
    backbone = source_backbone_path(config, backbone_seed)
    if provenance.get("backbone_path") != backbone.as_posix():
        raise ValueError("loaded checkpoint uses a noncanonical backbone path")
    hashes = {backbone.as_posix(): sha256_file(backbone)}
    if method.head is None:
        return hashes
    checkpoint_path = head_checkpoint_path(
        config,
        method=method_name,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "protocol_id": config.protocol_id,
        "formal_run": True,
        "method": method.model_dump(mode="json"),
        "method_sha256": method_hash,
        "decision_sha256s": list(decisions),
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "backbone_seed": int(backbone_seed),
        "head_seed": int(head_seed),
        "backbone_path": backbone.as_posix(),
        "backbone_sha256": sha256_file(backbone),
        "head_spec": method.head.model_dump(mode="json"),
        "checkpoint_selection": "final_step",
        "final_step": int(config.training.steps),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("quick head checkpoint metadata differs from its method cell")
    if head is None or any(
        not torch.isfinite(parameter).all() for parameter in head.parameters()
    ):
        raise ValueError("quick head checkpoint contains non-finite parameters")
    if method.training_scope.value == "frozen" and "model_state_dict" in payload:
        raise ValueError("frozen quick method unexpectedly stores a trainable backbone")
    expected_training_spec = canonical_json_sha256(
        {
            "schema": TRAIN_STATE_SCHEMA,
            "analysis_spec_sha256": lock["analysis_spec_sha256"],
            "protocol_lock_sha256": lock["protocol_lock_sha256"],
            "method_sha256": method_hash,
            "backbone_sha256": sha256_file(backbone),
            "backbone_seed": int(backbone_seed),
            "head_seed": int(head_seed),
            "training_rng_seed": hierarchical_seed(
                "distance-head-training-rng", backbone_seed, head_seed
            ),
            "steps": int(config.training.steps),
            "calibrated_weights": payload.get("calibrated_weights"),
            "candidate_bank": payload.get("candidate_bank"),
            "cache_bindings": payload.get("cache_bindings"),
            "initialization": payload.get("initialization"),
        }
    )
    if payload.get("training_spec_sha256") != expected_training_spec:
        raise ValueError("quick head training-spec hash does not reproduce")
    hashes = merge_hash_bindings(
        hashes,
        {checkpoint_path.as_posix(): sha256_file(checkpoint_path)},
        _checkpoint_payload_evidence(
            config, lock, payload, backbone_seed=backbone_seed
        ),
    )

    rebound = payload.get("artifact_rebound_without_retraining") is True
    if method_name in IMPORTABLE_METHODS and not rebound:
        raise ValueError("reference checkpoint lacks immutable import provenance")
    if method_name not in IMPORTABLE_METHODS and rebound:
        raise ValueError("new treatment checkpoint cannot be imported")
    if not rebound:
        return hashes

    source_config = load_study_config(profile.paths.source_config)
    source_lock = verify_protocol_lock(source_config)
    source_method, source_method_hash, source_decisions = load_and_resolve_method(
        source_config.paths.method_catalog,
        method_name,
        decision_root=source_config.paths.decision_root,
        protocol_lock=source_lock,
    )
    if source_method != method or source_method_hash != method_hash:
        raise ValueError("imported reference method no longer matches its source")
    rebind = payload.get("rebind_provenance")
    if not isinstance(rebind, dict):
        raise ValueError("imported reference checkpoint omits rebind provenance")
    source_path = head_checkpoint_path(
        source_config,
        method=method_name,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
    )
    if rebind.get("source_checkpoint_path") != source_path.as_posix():
        raise ValueError("imported reference checkpoint source path differs")
    hashes = _bind_file(
        hashes,
        source_path,
        rebind.get("source_checkpoint_sha256"),
        label="source reference checkpoint",
    )
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_expected = {
        "protocol_id": source_config.protocol_id,
        "formal_run": True,
        "method": source_method.model_dump(mode="json"),
        "method_sha256": source_method_hash,
        "decision_sha256s": list(source_decisions),
        "analysis_spec_sha256": source_lock["analysis_spec_sha256"],
        "protocol_lock_sha256": source_lock["protocol_lock_sha256"],
        "backbone_seed": int(backbone_seed),
        "head_seed": int(head_seed),
        "checkpoint_selection": "final_step",
        "final_step": int(source_config.training.steps),
    }
    if any(source.get(key) != value for key, value in source_expected.items()):
        raise ValueError("source reference checkpoint metadata differs")
    source_state_hash = state_dict_sha256(source["head_state_dict"])
    if (
        source_state_hash != state_dict_sha256(payload["head_state_dict"])
        or rebind.get("head_state_sha256") != source_state_hash
        or rebind.get("source_analysis_spec_sha256")
        != source_lock["analysis_spec_sha256"]
        or rebind.get("source_protocol_lock_sha256")
        != source_lock["protocol_lock_sha256"]
        or rebind.get("identical_method_sha256") != method_hash
        or rebind.get("identical_cache_records") is not True
        or rebind.get("identical_candidate_actions") is not True
    ):
        raise ValueError("reference checkpoint import proof does not reproduce")
    source_backbone = source_backbone_path(source_config, backbone_seed)
    if source_backbone != backbone or source.get("backbone_sha256") != sha256_file(
        source_backbone
    ):
        raise ValueError("source and quick reference backbones differ")
    hashes = merge_hash_bindings(
        hashes,
        _checkpoint_payload_evidence(
            source_config,
            source_lock,
            source,
            backbone_seed=backbone_seed,
        ),
    )
    _, quick_actions, _ = validate_candidate_bank(
        config, lock, backbone_seed=backbone_seed
    )
    _, source_actions, _ = validate_candidate_bank(
        source_config, source_lock, backbone_seed=backbone_seed
    )
    if not torch.equal(source_actions, quick_actions):
        raise ValueError("source and quick reference candidate actions differ")
    return hashes


def validate_result_cell(
    config: Any,
    protocol_lock: dict[str, Any],
    *,
    split_role: str,
    method_name: str,
    backbone_seed: int,
    head_seed: int,
    action_protocol: str,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    directory = result_directory(
        config,
        split_role=split_role,
        method=method_name,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
        action_protocol=action_protocol,
    )
    metadata, rows = load_complete_rows(directory)
    method, method_hash, decisions = load_and_resolve_method(
        config.paths.method_catalog,
        method_name,
        decision_root=config.paths.decision_root,
        protocol_lock=protocol_lock,
    )
    manifest = protocol_lock["analysis_spec"]["manifests"][split_role]
    expected = (
        metadata.get("analysis_spec_sha256") == protocol_lock["analysis_spec_sha256"]
        and metadata.get("protocol_lock_sha256")
        == protocol_lock["protocol_lock_sha256"]
        and metadata.get("split_role") == split_role
        and metadata.get("method") == method.model_dump(mode="json")
        and metadata.get("method_sha256") == method_hash
        and metadata.get("decision_sha256s") == list(decisions)
        and int(metadata.get("backbone_seed", -1)) == backbone_seed
        and int(metadata.get("head_seed", -1)) == head_seed
        and metadata.get("action_protocol") == action_protocol
        and metadata.get("manifest_sha256") == manifest["sha256"]
        and int(metadata.get("diagnostic_limit", -1)) == 0
        and int(metadata.get("num_shards", -1)) == 1
        and metadata.get("shard_index") == 0
        and len(rows) == int(manifest["count"])
    )
    if not expected:
        raise ValueError(f"result cell differs from the quick protocol: {directory}")
    return directory, metadata, rows, result_evidence_hashes(directory, metadata)


def load_validated_diagnostic(
    config: Any,
    protocol_lock: dict[str, Any],
    path: str | Path,
    *,
    split_role: str,
    method_name: str,
    backbone_seed: int,
    head_seed: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = resolve_path(path)
    payload = load_signed_artifact(
        resolved,
        signature_field="diagnostic_sha256",
        expected_protocol_id=config.protocol_id,
    )
    _, method_hash, decisions = load_and_resolve_method(
        config.paths.method_catalog,
        method_name,
        decision_root=config.paths.decision_root,
        protocol_lock=protocol_lock,
    )
    expected = (
        payload.get("analysis_spec_sha256") == protocol_lock["analysis_spec_sha256"]
        and payload.get("protocol_lock_sha256") == protocol_lock["protocol_lock_sha256"]
        and payload.get("split_role") == split_role
        and payload.get("method") == method_name
        and payload.get("method_sha256") == method_hash
        and payload.get("decision_sha256s") == list(decisions)
        and int(payload.get("backbone_seed", -1)) == backbone_seed
        and int(payload.get("head_seed", -1)) == head_seed
        and int(payload.get("sample_count", -1))
        == config.analysis.diagnostic_batches * config.training.effective_batch_size
        and int(payload.get("cache_binding", {}).get("diagnostic_limit", -1)) == 0
    )
    if not expected:
        raise ValueError(f"diagnostic differs from the quick protocol: {resolved}")
    hashes = diagnostic_evidence_hashes(
        resolved,
        payload,
        split_role=split_role,
        backbone_seed=backbone_seed,
        protocol_lock=protocol_lock,
    )
    cache_binding = payload.get("cache_binding")
    if not isinstance(cache_binding, dict):
        raise ValueError("diagnostic cache binding is malformed")
    hashes = merge_hash_bindings(
        hashes,
        cache_binding_evidence_hashes(
            cache_binding,
            split_role=split_role,
            backbone_seed=backbone_seed,
            protocol_lock=protocol_lock,
            expected_index_path=cache_index_path(
                config,
                split_role=split_role,
                backbone_seed=backbone_seed,
            ),
        ),
    )
    bank = payload.get("candidate_bank")
    if not isinstance(bank, dict) or not isinstance(bank.get("path"), str):
        raise ValueError("diagnostic candidate-bank binding is malformed")
    _, _, bank_hashes = validate_candidate_bank(
        config,
        protocol_lock,
        backbone_seed=backbone_seed,
        path=bank["path"],
    )
    if bank.get("sha256") != next(iter(bank_hashes.values())):
        raise ValueError("diagnostic candidate-bank hash differs")
    return payload, merge_hash_bindings(hashes, bank_hashes)


__all__ = [
    "cache_binding_evidence_hashes",
    "load_validated_diagnostic",
    "quick_cache_evidence_hashes",
    "validate_candidate_bank",
    "validate_quick_checkpoint",
    "validate_result_cell",
]
