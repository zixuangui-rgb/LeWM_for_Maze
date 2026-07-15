"""Train one frozen-backbone planner component under the locked quick protocol."""

from __future__ import annotations

import argparse

from final_closure.common import sha256_file
from vector_jepa_planner_frontier import (
    EXPERIMENT_FAMILY as FRONTIER_ARTIFACT_FAMILY,
)
from vector_jepa_planner_frontier import FORMAT_VERSION as FRONTIER_FORMAT_VERSION
from vector_jepa_planner_frontier.common import (
    atomic_torch_save,
    hierarchical_seed,
    prepare_formal_output,
    resolve_device,
    set_seed,
)
from vector_jepa_planner_frontier.heads import required_head_names
from vector_jepa_planner_frontier.train import (
    locked_training_steps,
    train_components,
)
from vector_jepa_planner_full900_screen import EXPERIMENT_FAMILY
from vector_jepa_planner_full900_screen.common import (
    analysis_spec_sha256,
    load_config,
    load_json,
    metadata,
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
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--diagnostic", action="store_true")
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
        raise ValueError("selected method has no trainable planner component")
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside historical seeds 42-51")
    if args.planner_seed not in config.protocol.planner_seeds:
        raise ValueError("planner seed lies outside the locked nested seeds")

    head_names = required_head_names(method)
    locked_steps = locked_training_steps(config, method, head_names)
    steps = args.steps or locked_steps
    if args.steps is not None and not args.diagnostic:
        raise ValueError("formal training cannot override the locked step budget")
    if args.diagnostic and not (0 < steps < locked_steps):
        raise ValueError(
            "diagnostic steps must be positive and below the locked budget"
        )
    if args.allow_dirty_worktree and not args.diagnostic:
        raise ValueError("dirty worktrees are allowed only for isolated diagnostics")
    output = resolve_path(
        args.output
        or config.paths.component_training_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        )
    )
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    rerun = prepare_formal_output(
        output, overwrite=args.overwrite, rerun_reason=args.rerun_reason
    )
    device = resolve_device(args.device or config.device)
    set_seed(
        hierarchical_seed(
            "full900-planner-component",
            args.backbone_seed,
            args.planner_seed,
        ),
        deterministic=True,
    )
    model, modules, training_summary, source_path = train_components(
        config,
        lock,
        method,
        backbone_seed=args.backbone_seed,
        planner_seed=args.planner_seed,
        device=device,
        steps=steps,
    )
    payload = {
        "experiment_family": FRONTIER_ARTIFACT_FAMILY,
        "study_experiment_family": EXPERIMENT_FAMILY,
        "format_version": FRONTIER_FORMAT_VERSION,
        "stage": "component_training_diagnostic"
        if args.diagnostic
        else "component_training",
        "diagnostic": bool(args.diagnostic),
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
        "train_manifest_sha256": sha256_file(resolve_path(config.paths.train_manifest)),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "initialization_parent": training_summary["initialization_parent"],
        "joint_counterexample_provenance": training_summary[
            "joint_counterexample_provenance"
        ],
        "head_config": training_summary["head_config"],
        "head_state_dicts": {
            name: module.state_dict() for name, module in modules.items()
        },
        "model_state_dict": model.state_dict() if method.track == "J" else None,
        "training_summary": training_summary["losses"],
        "validation_metrics": {},
        "protocol": metadata(
            config,
            lock,
            method=method,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            device=device,
        ),
        "rerun": rerun,
    }
    atomic_torch_save(output, payload)


if __name__ == "__main__":
    main()
