"""Build immutable goal-consistent cache shards for one backbone and split."""

from __future__ import annotations

import argparse

from distance_head_study.common import (
    load_study_config,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    source_backbone_path,
)
from distance_head_study.data import build_cache
from distance_head_study.protocol import verify_protocol_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument(
        "--split-role",
        choices=("train", "cal", "screen", "select", "confirm", "stress"),
        required=True,
    )
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--diagnostic-limit", type=int, default=0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.diagnostic_limit < 0:
        raise ValueError("diagnostic limit must be non-negative")
    if args.diagnostic_limit and not args.allow_dirty_worktree:
        raise ValueError("partial caches are diagnostic and require the explicit flag")
    config = load_study_config(args.config)
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    lock = verify_protocol_lock(config)
    manifest = getattr(config.paths, f"{args.split_role}_manifest")
    backbone = source_backbone_path(config, args.backbone_seed)
    if not backbone.exists():
        raise FileNotFoundError(backbone)
    output = build_cache(
        config,
        split_role=args.split_role,
        manifest_path=manifest,
        backbone_seed=args.backbone_seed,
        backbone_path=backbone,
        device=resolve_device(args.device or config.device),
        analysis_spec_sha256=lock["analysis_spec_sha256"],
        protocol_lock_sha256=lock["protocol_lock_sha256"],
        diagnostic_limit=args.diagnostic_limit,
        output_path=(
            resolve_path(
                "distance_head_study_runs/smoke/cache/"
                f"{args.split_role}/backbone{args.backbone_seed}_"
                f"limit{args.diagnostic_limit}/index.json"
            )
            if args.diagnostic_limit
            else None
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
