"""Merge independently evaluated task shards after strict metadata checks."""

from __future__ import annotations

import argparse

from distance_head_study.common import load_study_config
from distance_head_study.results import merge_shards, result_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--method", required=True)
    parser.add_argument("--split-role", required=True)
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--head-seed", type=int, default=0)
    parser.add_argument("--action-protocol", required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    base = result_directory(
        config,
        split_role=args.split_role,
        method=args.method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
        action_protocol=args.action_protocol,
    )
    print(merge_shards(base, expected_shards=args.expected_shards))


if __name__ == "__main__":
    main()
