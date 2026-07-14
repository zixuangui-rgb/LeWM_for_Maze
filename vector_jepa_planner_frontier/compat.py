"""Compatibility boundary to the frozen final_closure Vector-JEPA baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from final_closure.common import (
    baseline_config,
    load_checkpoint,
    load_config,
    sha256_file,
)
from final_closure.evaluate import load_model
from final_closure.models import build_lewm, serialize_lewm_config
from vector_jepa_planner_frontier import EXPERIMENT_FAMILY, FORMAT_VERSION
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    experiment_code_fingerprint,
    resolve_path,
)
from vector_jepa_planner_frontier.schemas import MethodConfig, PlannerKind, StudyConfig

SOURCE_BASELINE_NAME = "lewm_l2_cem_seqlen2"


def source_protocol(
    config: StudyConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_config_path = resolve_path(config.paths.source_config)
    source_lock_path = resolve_path(config.paths.source_lock)
    source_config, source_lock = load_config(source_config_path)
    if resolve_path(source_config["paths"]["protocol_lock"]) != source_lock_path:
        raise ValueError("source config does not point to the declared source lock")
    return source_config, source_lock


def validate_source_contract(
    config: StudyConfig, lock: dict[str, Any]
) -> dict[str, Any]:
    source_config, source_lock = source_protocol(config)
    expected = lock["source_baseline"]
    if expected["name"] != SOURCE_BASELINE_NAME:
        raise ValueError("source baseline name is not the frozen Vector-JEPA B0")
    if (
        sha256_file(resolve_path(config.paths.source_config))
        != expected["config_sha256"]
    ):
        raise ValueError("source final_closure config hash mismatch")
    if sha256_file(resolve_path(config.paths.source_lock)) != expected["lock_sha256"]:
        raise ValueError("source final_closure lock hash mismatch")
    baseline = baseline_config(source_config, SOURCE_BASELINE_NAME)
    planner = baseline["planner"]
    exact = {
        "history_size": 3,
        "context_action_initialization": "repeat_action_id_4",
        "horizon": 12,
        "num_candidates": 64,
        "num_elites": 8,
        "cem_iters": 1,
        "momentum": 0.1,
        "replan_every_step": True,
        "allowed_actions": [1, 2, 3, 4],
        "cem_seed_schedule": "historical_eval_seed_task_step",
        "score": "squared_latent_l2",
    }
    if planner != exact:
        raise ValueError("source Vector-JEPA planner no longer matches historical B0")
    if source_lock["train_manifest"]["sha256"] != lock["train_manifest"]["sha256"]:
        raise ValueError(
            "new study and source baseline do not share the training split"
        )
    return baseline


def validate_b0_method(method: MethodConfig) -> None:
    if method.planner.kind != PlannerKind.LEGACY_CEM:
        raise ValueError("requested method is not the historical B0")
    if method.name != "b0_legacy_l2_cem":
        raise ValueError("historical B0 must retain its preregistered method name")


def checkpoint_path(config: StudyConfig, *, seed: int) -> Path:
    return resolve_path(
        config.paths.checkpoint_template.format(name=SOURCE_BASELINE_NAME, seed=seed)
    )


def load_source_lewm(
    config: StudyConfig,
    lock: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    baseline = validate_source_contract(config, lock)
    source_config, source_lock = source_protocol(config)
    path = checkpoint_path(config, seed=seed)
    if seed in {int(value) for value in source_config["seeds"]}:
        checkpoint = load_checkpoint(
            path,
            config=source_config,
            lock=source_lock,
            name=SOURCE_BASELINE_NAME,
            seed=seed,
            strict_provenance=False,
        )
        model = load_model(baseline, checkpoint, device)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("experiment_family") != EXPERIMENT_FAMILY:
            raise ValueError("extended source checkpoint belongs to another experiment")
        if int(checkpoint.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError("unsupported extended source-checkpoint format")
        if checkpoint.get("stage") != "source_backbone_extension":
            raise ValueError("formal runs require a full extended source checkpoint")
        if checkpoint.get("source_baseline_name") != SOURCE_BASELINE_NAME:
            raise ValueError("extended checkpoint does not implement the frozen B0")
        if int(checkpoint.get("backbone_seed", -1)) != seed:
            raise ValueError("extended source checkpoint seed mismatch")
        if checkpoint.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
            raise ValueError("extended source checkpoint analysis-spec mismatch")
        if (
            checkpoint.get("source_config_sha256")
            != lock["source_baseline"]["config_sha256"]
        ):
            raise ValueError("extended source checkpoint source-config mismatch")
        if (
            checkpoint.get("source_lock_sha256")
            != lock["source_baseline"]["lock_sha256"]
        ):
            raise ValueError("extended source checkpoint source-lock mismatch")
        if checkpoint.get("train_manifest_sha256") != lock["train_manifest"]["sha256"]:
            raise ValueError("extended source checkpoint train-manifest mismatch")
        if checkpoint.get("training_config") != baseline["train"]:
            raise ValueError("extended source checkpoint changed frozen LeWM training")
        protocol = checkpoint.get("protocol", {})
        if protocol.get("git_dirty") is not False:
            raise ValueError("extended source checkpoint was trained from dirty code")
        if protocol.get("code_fingerprint") != experiment_code_fingerprint():
            raise ValueError("extended source checkpoint code fingerprint mismatch")
        model, model_config = build_lewm(baseline["train"])
        if checkpoint.get("model_config") != serialize_lewm_config(model_config):
            raise ValueError("extended source checkpoint model-config mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    if checkpoint.get("protocol", {}).get("git_dirty") is not False:
        raise ValueError(
            "source Vector-JEPA checkpoint was trained from a dirty worktree"
        )
    return model, checkpoint, path


__all__ = [
    "SOURCE_BASELINE_NAME",
    "checkpoint_path",
    "load_source_lewm",
    "source_protocol",
    "validate_b0_method",
    "validate_source_contract",
]
