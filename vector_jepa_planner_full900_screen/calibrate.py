"""Calibrate planner heads on the topology-disjoint 700-task validation split."""

from __future__ import annotations

import argparse

import torch

from final_closure.common import read_jsonl, sha256_file
from vector_jepa_planner_frontier import (
    EXPERIMENT_FAMILY as FRONTIER_ARTIFACT_FAMILY,
)
from vector_jepa_planner_frontier import FORMAT_VERSION as FRONTIER_FORMAT_VERSION
from vector_jepa_planner_frontier.calibrate import calibrate_heads, load_heads
from vector_jepa_planner_frontier.common import (
    atomic_torch_save,
    prepare_formal_output,
    resolve_device,
    set_seed,
)
from vector_jepa_planner_frontier.compat import load_source_lewm
from vector_jepa_planner_frontier.heads import required_head_names
from vector_jepa_planner_full900_screen.common import (
    analysis_spec_sha256,
    load_config,
    load_json,
    method_by_name,
    require_clean_worktree,
    resolve_path,
    training_spec_sha256,
    validate_lock,
)
from vector_jepa_planner_full900_screen.methods import effective_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="vector_jepa_planner_full900_screen/configs/default.json",
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--planner-seed", type=int, required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-reason", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_lock(config, lock)
    method = effective_method(config, lock, method_by_name(config, args.method))
    if not method.component_checkpoint_required:
        raise ValueError("selected method has no component to calibrate")
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside historical seeds 42-51")
    if args.planner_seed not in config.protocol.planner_seeds:
        raise ValueError("planner seed lies outside the locked nested seeds")
    if args.allow_dirty_worktree:
        raise ValueError("formal calibration never permits a dirty worktree")
    input_path = resolve_path(
        args.input
        or config.paths.component_training_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        )
    )
    output_path = resolve_path(
        args.output
        or config.paths.component_checkpoint_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        )
    )
    require_clean_worktree()
    rerun = prepare_formal_output(
        output_path, overwrite=args.overwrite, rerun_reason=args.rerun_reason
    )
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_family") != FRONTIER_ARTIFACT_FAMILY:
        raise ValueError("input is not a compatible frontier component checkpoint")
    if int(checkpoint.get("format_version", -1)) != FRONTIER_FORMAT_VERSION:
        raise ValueError("unsupported component checkpoint version")
    if checkpoint.get("stage") != "component_training":
        raise ValueError("calibration input must be a formal training checkpoint")
    expected_training = training_spec_sha256(
        config,
        lock,
        method=method,
        backbone_seed=args.backbone_seed,
        planner_seed=args.planner_seed,
    )
    if checkpoint.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("training checkpoint analysis-spec mismatch")
    if checkpoint.get("training_spec_sha256") != expected_training:
        raise ValueError("training checkpoint training-spec mismatch")
    if (
        checkpoint.get("protocol", {}).get("code_fingerprint")
        != lock["code_fingerprint"]
    ):
        raise ValueError("training checkpoint code fingerprint mismatch")

    device = resolve_device(args.device or config.device)
    set_seed(config.training.calibration_seed, deterministic=True)
    model, _, source_path = load_source_lewm(
        config, lock, seed=args.backbone_seed, device=device
    )
    if checkpoint.get("source_checkpoint_sha256") != sha256_file(source_path):
        raise ValueError("training and calibration loaded different backbones")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = load_heads(checkpoint, device)
    if set(modules) != required_head_names(method):
        raise ValueError("training checkpoint does not contain the exact method heads")
    validation_path = resolve_path(config.paths.validation_manifest)
    if sha256_file(validation_path) != lock["validation_manifest"]["sha256"]:
        raise ValueError("validation manifest hash mismatch")
    validation_entries = read_jsonl(validation_path)
    if len(validation_entries) != config.protocol.validation_count:
        raise ValueError("validation manifest count mismatch")
    metrics = calibrate_heads(
        model,
        modules,
        validation_entries,
        horizon=method.planner.horizon,
        history_size=method.planner.history_size,
        rollout_semantics=method.planner.rollout_semantics,
        batch_size=config.training.transition_batch_size,
        chunk_batch_size=config.training.proposal_batch_size,
        dts_batch_size=config.training.dts_batch_size,
        batches=config.training.calibration_batches,
        seed=config.training.calibration_seed,
        required_join_precision=method.memory.required_validation_precision,
        device=device,
    )
    payload = dict(checkpoint)
    payload.update(
        {
            "stage": "component_calibration",
            "source_training_checkpoint": str(input_path),
            "source_training_checkpoint_sha256": sha256_file(input_path),
            "validation_manifest_sha256": sha256_file(validation_path),
            "validation_metrics": metrics,
            "rerun": rerun,
        }
    )
    atomic_torch_save(output_path, payload)


if __name__ == "__main__":
    main()
