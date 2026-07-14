"""Train protocol-identical LeWM backbones beyond the ten legacy checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from final_closure.common import read_jsonl, sha256_file
from final_closure.train import cpu_state_dict, train_lewm
from vector_jepa_planner_frontier import EXPERIMENT_FAMILY, FORMAT_VERSION
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_torch_save,
    load_json,
    load_study_config,
    method_by_name,
    prepare_formal_output,
    protocol_metadata,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    set_seed,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.compat import (
    SOURCE_BASELINE_NAME,
    checkpoint_path,
    source_protocol,
    validate_source_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--diagnostic-steps", type=int, default=0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-reason", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    validate_source_contract(config, lock)
    source_config, _ = source_protocol(config)
    legacy_seeds = {int(seed) for seed in source_config["seeds"]}
    if args.backbone_seed in legacy_seeds:
        raise ValueError("legacy backbone seeds must use their frozen old checkpoint")
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside the locked frontier matrix")
    if args.diagnostic_steps < 0:
        raise ValueError("diagnostic steps cannot be negative")
    if args.allow_dirty_worktree and args.diagnostic_steps == 0:
        raise ValueError("dirty worktrees are allowed only for isolated diagnostics")
    output = resolve_path(
        args.output or checkpoint_path(config, seed=args.backbone_seed)
    )
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    rerun = prepare_formal_output(
        output,
        overwrite=args.overwrite,
        rerun_reason=args.rerun_reason,
    )
    baseline = validate_source_contract(config, lock)
    train_config = baseline["train"]
    locked_steps = int(train_config["steps"])
    steps = int(args.diagnostic_steps or locked_steps)
    if args.diagnostic_steps and args.output is None:
        raise ValueError("diagnostic backbone training requires an isolated --output")
    device = resolve_device(args.device or config.device)
    set_seed(args.backbone_seed, deterministic=True)
    train_path = resolve_path(config.paths.train_manifest)
    entries = read_jsonl(train_path)
    if len(entries) != int(lock["train_manifest"]["count"]):
        raise ValueError("training manifest count differs from the frontier lock")
    model, training = train_lewm(
        entries,
        train_config,
        seed=args.backbone_seed,
        device=device,
        steps=steps,
        sigreg_num_proj=int(train_config["sigreg_num_proj"]),
    )
    model_config = training.pop("model_config")
    b0 = method_by_name(config, "b0_legacy_l2_cem")
    payload = {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": FORMAT_VERSION,
        "stage": (
            "source_backbone_extension_diagnostic"
            if args.diagnostic_steps
            else "source_backbone_extension"
        ),
        "source_baseline_name": SOURCE_BASELINE_NAME,
        "training_seed": int(args.backbone_seed),
        "backbone_seed": int(args.backbone_seed),
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "source_config_sha256": sha256_file(resolve_path(config.paths.source_config)),
        "source_lock_sha256": sha256_file(resolve_path(config.paths.source_lock)),
        "train_manifest_sha256": sha256_file(train_path),
        "training_config": train_config,
        "locked_steps": locked_steps,
        "actual_steps": steps,
        "model_config": model_config,
        "model_state_dict": cpu_state_dict(model),
        "training": training,
        "protocol": protocol_metadata(
            config,
            lock,
            method=b0,
            seed=args.backbone_seed,
            device=device,
        ),
        "rerun": rerun,
    }
    atomic_torch_save(output, payload)
    print(f"saved source backbone seed={args.backbone_seed} to {Path(output)}")


if __name__ == "__main__":
    main()
