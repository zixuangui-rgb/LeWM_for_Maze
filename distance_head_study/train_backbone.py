"""Train fresh confirmation LeWM backbones with the exact source recipe."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from distance_head_study.common import (
    atomic_torch_save,
    canonical_json_sha256,
    load_json,
    load_study_config,
    read_jsonl,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    set_seed,
    sha256_file,
    source_backbone_path,
    validate_confirmation_seed_freshness,
)
from distance_head_study.gates import require_seed_released
from distance_head_study.protocol import verify_protocol_lock
from final_closure.common import baseline_config
from final_closure.train import cpu_state_dict, train_lewm


def _initialize_backbone_rng(seed: int) -> None:
    """Match the source training entrypoint before model construction."""

    set_seed(seed, deterministic=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--diagnostic-steps", type=int, default=0)
    parser.add_argument("--diagnostic-sigreg-num-proj", type=int, default=0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


def _source_training_spec(
    source_config: dict[str, Any],
    source_lock: dict[str, Any],
    *,
    seed: int,
) -> str:
    baseline = baseline_config(source_config, "lewm_l2_cem_seqlen2")
    return canonical_json_sha256(
        {
            "schema": "distance-head-source-backbone-v1",
            "source_protocol_id": source_config["protocol_id"],
            "source_analysis_spec_sha256": source_lock["analysis_spec_sha256"],
            "source_train_manifest_sha256": source_lock["train_manifest"]["sha256"],
            "baseline": baseline,
            "fresh_seed": int(seed),
        }
    )


def main() -> None:
    args = parse_args()
    if args.diagnostic_steps < 0 or args.diagnostic_sigreg_num_proj < 0:
        raise ValueError("diagnostic overrides must be non-negative")
    diagnostic = bool(args.diagnostic_steps or args.diagnostic_sigreg_num_proj)
    if diagnostic and not args.allow_dirty_worktree:
        raise ValueError(
            "diagnostic backbone training requires the explicit dirty flag"
        )
    if diagnostic and args.diagnostic_steps <= 0:
        raise ValueError(
            "diagnostic backbone training requires an explicit positive step limit"
        )
    config = load_study_config(args.config)
    freshness = validate_confirmation_seed_freshness(config)
    if args.backbone_seed not in config.seeds.ordered_confirmation_backbones:
        raise ValueError(
            "fresh backbone seed is outside the preregistered ordered list"
        )
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    lock = verify_protocol_lock(config)
    if not diagnostic:
        require_seed_released(config, backbone_seed=args.backbone_seed)
    source_config = load_json(config.paths.source_config)
    source_lock = load_json(config.paths.source_lock)
    if (
        sha256_file(config.paths.train_manifest)
        != source_lock["train_manifest"]["sha256"]
    ):
        raise ValueError(
            "source training manifest hash differs from final_closure lock"
        )
    baseline = baseline_config(source_config, "lewm_l2_cem_seqlen2")
    train_config = baseline["train"]
    entries = read_jsonl(config.paths.train_manifest)
    device = resolve_device(args.device or config.device)
    steps = int(args.diagnostic_steps or train_config["steps"])
    projections = int(
        args.diagnostic_sigreg_num_proj or train_config["sigreg_num_proj"]
    )
    _initialize_backbone_rng(args.backbone_seed)
    model, training = train_lewm(
        entries,
        train_config,
        seed=args.backbone_seed,
        device=device,
        steps=steps,
        sigreg_num_proj=projections,
    )
    output = (
        resolve_path(
            "distance_head_study_runs/smoke/checkpoints/backbones/"
            f"backbone{args.backbone_seed}_steps{steps}_proj{projections}.pt"
        )
        if diagnostic
        else source_backbone_path(config, args.backbone_seed)
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fresh backbone: {output}")
    payload = {
        "experiment_family": "procgen_maze_distance_head_study",
        "format_version": 1,
        "protocol_id": config.protocol_id,
        "stage": "fresh_source_backbone",
        "baseline_name": baseline["name"],
        "baseline_kind": baseline["kind"],
        "training_seed": int(args.backbone_seed),
        "formal_run": not diagnostic,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "source_training_spec_sha256": _source_training_spec(
            source_config,
            source_lock,
            seed=args.backbone_seed,
        ),
        "source_analysis_spec_sha256": source_lock["analysis_spec_sha256"],
        "source_train_manifest_sha256": source_lock["train_manifest"]["sha256"],
        "fresh_seed_audit": freshness,
        "training_config": train_config,
        "model_config": training.pop("model_config"),
        "model_state_dict": cpu_state_dict(model),
        "training": training,
    }
    atomic_torch_save(output, payload)
    print(Path(output))


if __name__ == "__main__":
    main()
