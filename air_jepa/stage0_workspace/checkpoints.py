"""Strict source/AIR checkpoint loading and representation identity checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from air_jepa.stage0_workspace import EXPERIMENT_ID, FORMAT_VERSION
from air_jepa.stage0_workspace.common import (
    SOURCE_LOCK_SCHEMA,
    format_template,
    read_json,
    relative_path,
    resolve_path,
    sha256_file,
    state_dict_sha256,
    verify_signature,
)
from air_jepa.stage0_workspace.models import AIRWorkspaceModel
from air_jepa.stage0_workspace.schemas import Stage0Config
from spatial_jepa_planning import EXPERIMENT_FAMILY as SOURCE_FAMILY
from spatial_jepa_planning import FORMAT_VERSION as SOURCE_FORMAT_VERSION
from spatial_jepa_planning.common import (
    load_planner_checkpoint,
    load_representation_checkpoint,
)
from spatial_jepa_planning.models import SpatialRepresentation


def _load_payload(path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(resolve_path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a dictionary: {path}")
    return payload


def _training_seed(payload: dict[str, Any]) -> int:
    return int(payload.get("protocol", {}).get("seed", -1))


def _optimizer_steps(payload: dict[str, Any]) -> int:
    accounting = payload.get("training_accounting", {})
    return int(accounting.get("optimizer_steps", -1))


def _validate_source_protocol(
    payload: dict[str, Any],
    *,
    seed: int,
    config: Stage0Config,
    label: str,
) -> None:
    protocol = payload.get("protocol", {})
    if int(protocol.get("seed", -1)) != seed:
        raise ValueError(f"{label} source protocol seed mismatch")
    expected_manifests = {
        "train_manifest_sha256": config.paths.train_manifest,
        "development_manifest_sha256": config.paths.historical_development_manifest,
        "eval_manifest_sha256": config.paths.historical_confirmatory_manifest,
    }
    for key, path in expected_manifests.items():
        if protocol.get(key) != sha256_file(path):
            raise ValueError(f"{label} source protocol {key} mismatch")
    if protocol.get("git_dirty") is not False:
        raise ValueError(f"{label} source checkpoint was created from a dirty worktree")
    if not protocol.get("git_commit") or not protocol.get("code_fingerprint"):
        raise ValueError(f"{label} source checkpoint lacks code provenance")


def _validate_j1_compute_accounting(payload: dict[str, Any]) -> None:
    planner_parameters = int(payload.get("planner_parameter_count", -1))
    representation_parameters = int(
        payload.get("representation_planning_parameter_count", -1)
    )
    total_parameters = int(payload.get("total_inference_parameter_count", -1))
    if (
        planner_parameters <= 0
        or representation_parameters <= 0
        or total_parameters != planner_parameters + representation_parameters
    ):
        raise ValueError("J1 source parameter accounting is absent or inconsistent")
    representation_macs = payload.get("representation_inference_conv_macs", {})
    planner_macs = payload.get("planner_inference_conv_macs", {})
    for size in (21, 25):
        size_key = str(size)
        if int(representation_macs.get(size_key, -1)) <= 0:
            raise ValueError(f"J1 source lacks representation MACs for size {size}")
        curve = planner_macs.get(size_key, {})
        points = sorted((int(key), int(value)) for key, value in curve.items())
        if len(points) < 2 or 128 not in {key for key, _ in points}:
            raise ValueError(f"J1 source lacks an auditable K curve for size {size}")
        if any(k <= 0 or value <= 0 for k, value in points):
            raise ValueError(f"J1 source has invalid MAC values for size {size}")
        (first_k, first_value), (second_k, second_value) = points[:2]
        delta_k = second_k - first_k
        delta_value = second_value - first_value
        if delta_k <= 0 or delta_value <= 0 or delta_value % delta_k:
            raise ValueError(
                f"J1 source MAC curve is not integral-affine at size {size}"
            )
        per_iteration = delta_value // delta_k
        fixed = first_value - first_k * per_iteration
        if fixed < 0 or any(value != fixed + k * per_iteration for k, value in points):
            raise ValueError(f"J1 source MAC curve is not affine at size {size}")


def validate_source_representation_payload(
    payload: dict[str, Any],
    *,
    seed: int,
    config: Stage0Config,
) -> str:
    if payload.get("experiment_family") != SOURCE_FAMILY:
        raise ValueError("source representation has the wrong experiment family")
    if int(payload.get("format_version", -1)) != SOURCE_FORMAT_VERSION:
        raise ValueError("source representation format version mismatch")
    if payload.get("stage") != "representation":
        raise ValueError("source representation checkpoint is not representation stage")
    _validate_source_protocol(
        payload,
        seed=seed,
        config=config,
        label="representation",
    )
    if payload.get("variant_name") != "spatial_info_sigreg":
        raise ValueError("unexpected source representation variant")
    if _optimizer_steps(payload) != config.training.steps:
        raise ValueError("source representation is not the locked final 30k checkpoint")
    state = payload.get("representation_state_dict")
    if not isinstance(state, dict):
        raise ValueError("source representation state dict is missing")
    return state_dict_sha256(state)


def validate_source_planner_payload(
    payload: dict[str, Any],
    *,
    seed: int,
    variant: str,
    representation_state_sha256: str,
    config: Stage0Config,
) -> None:
    if payload.get("experiment_family") != SOURCE_FAMILY:
        raise ValueError(f"{variant} has the wrong experiment family")
    if int(payload.get("format_version", -1)) != SOURCE_FORMAT_VERSION:
        raise ValueError(f"{variant} source format version mismatch")
    if payload.get("stage") != "planner" or payload.get("input_mode") != "spatial_jepa":
        raise ValueError(f"{variant} must be a frozen spatial_jepa planner checkpoint")
    if payload.get("variant_name") != variant:
        raise ValueError(f"planner variant mismatch: expected {variant}")
    _validate_source_protocol(
        payload,
        seed=seed,
        config=config,
        label=variant,
    )
    if _optimizer_steps(payload) != config.training.steps:
        raise ValueError(f"{variant} is not the locked final 30k checkpoint")
    training_args = payload.get("training_args", {})
    if training_args.get("encoder_mode") != "frozen":
        raise ValueError(f"{variant} must have encoder_mode=frozen")
    planner_type = payload.get("planner_config", {}).get("planner_type")
    expected_type = "feedforward_dilated" if variant.startswith("j0_") else "iterative"
    if planner_type != expected_type:
        raise ValueError(f"{variant} planner type must be {expected_type}")
    embedded = payload.get("representation_state_dict")
    if not isinstance(embedded, dict):
        raise ValueError(f"{variant} lacks its embedded representation state")
    if state_dict_sha256(embedded) != representation_state_sha256:
        raise ValueError(
            f"{variant} does not embed the exact seed-matched representation"
        )
    if variant == "j1_spatial_iterative_frozen":
        _validate_j1_compute_accounting(payload)


def build_source_lock_payload(config: Stage0Config) -> dict[str, Any]:
    from air_jepa.stage0_workspace.common import signed_payload

    records: dict[str, Any] = {}
    analysis_specs: set[str] = set()
    source_code_fingerprints: set[str] = set()
    for seed in config.seeds:
        rep_path = format_template(
            config.paths.representation_checkpoint_template, seed=seed
        )
        j0_path = format_template(config.paths.j0_checkpoint_template, seed=seed)
        j1_path = format_template(config.paths.j1_checkpoint_template, seed=seed)
        j0_result = format_template(
            config.paths.historical_j0_result_template, seed=seed
        )
        j1_result = format_template(
            config.paths.historical_j1_result_template, seed=seed
        )
        required = (rep_path, j0_path, j1_path, j0_result, j1_result)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "source-lock inputs are missing:\n" + "\n".join(missing)
            )
        rep_payload = _load_payload(rep_path, torch.device("cpu"))
        rep_state_hash = validate_source_representation_payload(
            rep_payload, seed=seed, config=config
        )
        j0_payload = _load_payload(j0_path, torch.device("cpu"))
        j1_payload = _load_payload(j1_path, torch.device("cpu"))
        validate_source_planner_payload(
            j0_payload,
            seed=seed,
            variant="j0_spatial_feedforward",
            representation_state_sha256=rep_state_hash,
            config=config,
        )
        validate_source_planner_payload(
            j1_payload,
            seed=seed,
            variant="j1_spatial_iterative_frozen",
            representation_state_sha256=rep_state_hash,
            config=config,
        )
        analysis_spec = rep_payload.get("analysis_spec_sha256")
        if not analysis_spec or any(
            payload.get("analysis_spec_sha256") != analysis_spec
            for payload in (j0_payload, j1_payload)
        ):
            raise ValueError(
                "source representation/J0/J1 analysis-spec hashes do not match"
            )
        analysis_specs.add(str(analysis_spec))
        code_fingerprints = {
            str(payload["protocol"]["code_fingerprint"])
            for payload in (rep_payload, j0_payload, j1_payload)
        }
        if len(code_fingerprints) != 1:
            raise ValueError(
                "source representation/J0/J1 code fingerprints do not match"
            )
        source_code_fingerprints.update(code_fingerprints)
        records[str(seed)] = {
            "representation": {
                "path": relative_path(rep_path),
                "file_sha256": sha256_file(rep_path),
                "state_sha256": rep_state_hash,
            },
            "j0": {
                "path": relative_path(j0_path),
                "file_sha256": sha256_file(j0_path),
                "embedded_representation_state_sha256": state_dict_sha256(
                    j0_payload["representation_state_dict"]
                ),
            },
            "j1": {
                "path": relative_path(j1_path),
                "file_sha256": sha256_file(j1_path),
                "embedded_representation_state_sha256": state_dict_sha256(
                    j1_payload["representation_state_dict"]
                ),
            },
            "historical_j0_result": {
                "path": relative_path(j0_result),
                "file_sha256": sha256_file(j0_result),
            },
            "historical_j1_result": {
                "path": relative_path(j1_result),
                "file_sha256": sha256_file(j1_result),
            },
        }
    if len(analysis_specs) != 1 or len(source_code_fingerprints) != 1:
        raise ValueError(
            "source seeds do not share one analysis spec and source code fingerprint"
        )
    return signed_payload(
        {
            "schema": SOURCE_LOCK_SCHEMA,
            "experiment_id": config.experiment_id,
            "seeds": list(config.seeds),
            "analysis_spec_sha256": next(iter(analysis_specs)),
            "source_code_fingerprint": next(iter(source_code_fingerprints)),
            "records": records,
        },
        "source_lock_sha256",
    )


def verify_source_lock(config: Stage0Config, *, deep: bool = False) -> dict[str, Any]:
    payload = read_json(config.paths.source_lock)
    if not isinstance(payload, dict) or payload.get("schema") != SOURCE_LOCK_SCHEMA:
        raise ValueError("invalid AIR source lock schema")
    verify_signature(payload, "source_lock_sha256")
    if payload.get("experiment_id") != config.experiment_id:
        raise ValueError("source lock belongs to another AIR experiment")
    if tuple(payload.get("seeds", ())) != config.seeds:
        raise ValueError("source lock seed set differs from the AIR protocol")
    for seed in config.seeds:
        record = payload.get("records", {}).get(str(seed))
        if not isinstance(record, dict):
            raise ValueError(f"source lock lacks seed {seed}")
        for key in (
            "representation",
            "j0",
            "j1",
            "historical_j0_result",
            "historical_j1_result",
        ):
            item = record.get(key)
            if not isinstance(item, dict):
                raise ValueError(f"source lock seed {seed} lacks {key}")
            path = resolve_path(item.get("path", ""))
            if not path.is_file() or sha256_file(path) != item.get("file_sha256"):
                raise ValueError(f"source file changed after locking: {path}")
    if deep:
        expected = build_source_lock_payload(config)
        if payload != expected:
            raise ValueError(
                "source checkpoints/results changed after source lock creation"
            )
    return payload


def load_frozen_representation(
    config: Stage0Config,
    *,
    seed: int,
    device: torch.device,
    source_lock: dict[str, Any],
) -> tuple[SpatialRepresentation, dict[str, Any]]:
    record = source_lock["records"][str(seed)]["representation"]
    path = resolve_path(record["path"])
    if sha256_file(path) != record["file_sha256"]:
        raise ValueError("representation file hash changed after source lock")
    model, payload = load_representation_checkpoint(path, device)
    state_hash = validate_source_representation_payload(
        payload, seed=seed, config=config
    )
    if state_hash != record["state_sha256"]:
        raise ValueError("representation tensor hash changed after source lock")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, payload


def load_source_planner(
    config: Stage0Config,
    *,
    seed: int,
    method: str,
    device: torch.device,
    source_lock: dict[str, Any],
) -> tuple[torch.nn.Module, SpatialRepresentation, dict[str, Any]]:
    if method not in {"j0_static", "j1_static", "j1_receding"}:
        raise ValueError(f"not a source planner method: {method}")
    key = "j0" if method == "j0_static" else "j1"
    record = source_lock["records"][str(seed)][key]
    path = resolve_path(record["path"])
    if sha256_file(path) != record["file_sha256"]:
        raise ValueError(f"{key} checkpoint file hash changed after source lock")
    planner, representation, payload = load_planner_checkpoint(path, device)
    if representation is None:
        raise ValueError(f"{key} checkpoint did not load a spatial representation")
    variant = "j0_spatial_feedforward" if key == "j0" else "j1_spatial_iterative_frozen"
    representation_hash = source_lock["records"][str(seed)]["representation"][
        "state_sha256"
    ]
    validate_source_planner_payload(
        payload,
        seed=seed,
        variant=variant,
        representation_state_sha256=representation_hash,
        config=config,
    )
    return planner, representation, payload


def save_air_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    output = resolve_path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite AIR checkpoint: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with open(temporary, "xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_air_checkpoint(
    path: str | Path,
    *,
    config: Stage0Config,
    method: str,
    seed: int,
    device: torch.device,
    require_formal: bool = False,
) -> tuple[AIRWorkspaceModel, dict[str, Any]]:
    payload = _load_payload(path, device)
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("not an AIR0 workspace checkpoint")
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("AIR checkpoint format version mismatch")
    if payload.get("method") != method or int(payload.get("seed", -1)) != seed:
        raise ValueError("AIR checkpoint method/seed mismatch")
    if int(payload.get("optimizer_steps", -1)) != config.training.steps:
        raise ValueError("formal AIR evaluation requires the final 30k checkpoint")
    if require_formal and (
        payload.get("formal") is not True
        or payload.get("checkpoint_role") != "final_step"
    ):
        raise ValueError("formal AIR consumer requires a signed final checkpoint")
    if payload.get("config") != config.model_dump(mode="json", by_alias=True):
        raise ValueError("AIR checkpoint config differs from the locked config")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("AIR checkpoint model state is missing")
    if state_dict_sha256(state) != payload.get("model_state_sha256"):
        raise ValueError("AIR checkpoint tensor signature is invalid")
    model = AIRWorkspaceModel(config.model).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, payload


__all__ = [
    "build_source_lock_payload",
    "load_air_checkpoint",
    "load_frozen_representation",
    "load_source_planner",
    "save_air_checkpoint",
    "validate_source_planner_payload",
    "validate_source_representation_payload",
    "verify_source_lock",
]
