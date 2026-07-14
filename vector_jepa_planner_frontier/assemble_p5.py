"""Deterministically assemble effective P5 from frozen P3/P4 checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from final_closure.common import sha256_file
from vector_jepa_planner_frontier import EXPERIMENT_FAMILY, FORMAT_VERSION
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_torch_save,
    component_checkpoint_path,
    load_json,
    load_study_config,
    method_by_name,
    prepare_formal_output,
    protocol_metadata,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    training_spec_sha256,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.compat import load_source_lewm
from vector_jepa_planner_frontier.effective_methods import (
    RADICAL_METHODS,
    resolve_effective_method,
)
from vector_jepa_planner_frontier.heads import HeadConfig, required_head_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--method", default="p5_track_f_all_hard_memory")
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--planner-seed", type=int, required=True)
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-reason", default="")
    return parser.parse_args()


def _validated_source_checkpoint(
    path: Path,
    *,
    config: Any,
    lock: dict[str, Any],
    method: Any,
    backbone_seed: int,
    planner_seed: int,
    source_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        value.get("experiment_family") != EXPERIMENT_FAMILY
        or int(value.get("format_version", -1)) != FORMAT_VERSION
        or value.get("stage") != "component_calibration"
    ):
        raise ValueError(f"P5 source is not a calibrated component: {path}")
    if (
        value.get("method_name") != method.name
        or int(value.get("backbone_seed", -1)) != backbone_seed
        or int(value.get("planner_seed", -1)) != planner_seed
    ):
        raise ValueError(f"P5 source checkpoint label mismatch: {path}")
    if value.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError(f"P5 source analysis mismatch: {path}")
    if value.get("training_spec_sha256") != training_spec_sha256(
        config,
        lock,
        method=method,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
    ):
        raise ValueError(f"P5 source training-spec mismatch: {path}")
    if value.get("source_checkpoint_sha256") != source_sha256:
        raise ValueError(f"P5 source uses another Vector-JEPA backbone: {path}")
    protocol = value.get("protocol", {})
    if (
        protocol.get("git_dirty") is not False
        or protocol.get("code_fingerprint") != lock["code_fingerprint"]
    ):
        raise ValueError(f"P5 source provenance mismatch: {path}")
    return value


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    require_clean_worktree(allow_dirty=False)
    base = method_by_name(config, args.method)
    if base.stage != "P5":
        raise ValueError("assembly is valid only for the locked P5 method")
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside the locked matrix")
    if args.planner_seed not in config.protocol.planner_seeds:
        raise ValueError("planner seed lies outside the locked matrix")
    method = resolve_effective_method(config, lock, base)
    output = resolve_path(
        args.output
        or config.paths.component_training_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        )
    )
    rerun = prepare_formal_output(
        output, overwrite=args.overwrite, rerun_reason=args.rerun_reason
    )
    device = resolve_device(args.device or config.device)
    _, _, source_path = load_source_lewm(
        config, lock, seed=args.backbone_seed, device=device
    )
    source_sha256 = sha256_file(source_path)
    p5_decision = load_json(config.paths.p5_advancement)
    source_names = [str(p5_decision["selected_p3_cell"])]
    radical_name = p5_decision.get("selected_radical")
    if radical_name is not None:
        source_names.append(RADICAL_METHODS[str(radical_name)])

    source_records: list[dict[str, Any]] = []
    source_states: list[dict[str, dict[str, torch.Tensor]]] = []
    head_config: dict[str, Any] | None = None
    for source_name in source_names:
        source_base = method_by_name(config, source_name)
        source_method = resolve_effective_method(config, lock, source_base)
        path = component_checkpoint_path(
            config,
            source_base,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        )
        if path is None:
            if required_head_names(source_method):
                raise ValueError(
                    f"P5 source has heads but no checkpoint: {source_name}"
                )
            source_records.append(
                {
                    "method": source_name,
                    "path": None,
                    "sha256": None,
                    "head_names": [],
                }
            )
            source_states.append({})
            continue
        checkpoint = _validated_source_checkpoint(
            path,
            config=config,
            lock=lock,
            method=source_method,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            source_sha256=source_sha256,
        )
        candidate_config = dict(checkpoint["head_config"])
        if head_config is None:
            head_config = candidate_config
        elif candidate_config != head_config:
            raise ValueError("P5 sources use incompatible head architectures")
        states = dict(checkpoint["head_state_dicts"])
        source_states.append(states)
        source_records.append(
            {
                "method": source_name,
                "path": str(path),
                "sha256": sha256_file(path),
                "head_names": sorted(states),
            }
        )

    required = required_head_names(method)
    merged: dict[str, dict[str, torch.Tensor]] = {}
    ownership: dict[str, str] = {}
    for source_name, states in zip(source_names, source_states, strict=True):
        for name, state in states.items():
            if name in required and name not in merged:
                merged[name] = state
                ownership[name] = source_name
    if set(merged) != required:
        raise ValueError(
            "P5 assembly cannot supply exact required heads: "
            f"missing={sorted(required - set(merged))}, "
            f"extra={sorted(set(merged) - required)}"
        )
    if head_config is None:
        head_config = HeadConfig(
            latent_dim=256,
            hidden_dim=512,
            action_count=4,
            horizon=method.planner.horizon,
            reachability_bins=config.training.reachability_bins,
        ).to_dict()
    payload = {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": FORMAT_VERSION,
        "stage": "component_training",
        "assembly_stage": "p5_deterministic_assembly",
        "diagnostic": False,
        "method_name": method.name,
        "track": method.track,
        "seed": args.backbone_seed,
        "backbone_seed": args.backbone_seed,
        "planner_seed": args.planner_seed,
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "training_spec_sha256": training_spec_sha256(
            config,
            lock,
            method=method,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        ),
        "train_manifest_sha256": lock["train_manifest"]["sha256"],
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "initialization_parent": None,
        "initialization_parents": source_records,
        "head_ownership": ownership,
        "head_config": head_config,
        "head_state_dicts": merged,
        "model_state_dict": None,
        "training_summary": {
            "steps": 0,
            "locked_steps": 0,
            "deterministic_assembly": True,
        },
        "validation_metrics": {},
        "protocol": protocol_metadata(
            config,
            lock,
            method=method,
            seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            device=device,
        ),
        "rerun": rerun,
    }
    atomic_torch_save(output, payload)


if __name__ == "__main__":
    main()
