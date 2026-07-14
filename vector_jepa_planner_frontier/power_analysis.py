"""Freeze confirmatory backbone count from paired validation-pilot variance."""

from __future__ import annotations

import argparse
import math
from statistics import NormalDist

import numpy as np

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_json_dump,
    load_json,
    load_study_config,
    method_by_name,
    require_clean_worktree,
    resolve_path,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.stage_gates import (
    validate_p5_advancement,
    validate_p8_selection,
)
from vector_jepa_planner_frontier.validation_results import (
    load_validation_seed_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--comparison-count", type=int, choices=(2, 4), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def required_backbone_count(
    *,
    overall_std: float,
    large_size_std: float,
    comparison_count: int,
    alpha: float = 0.05,
    power: float = 0.80,
    minimum_effect: float = 0.05,
) -> dict[str, int | float]:
    if comparison_count not in (2, 4):
        raise ValueError("comparison_count must be 2 or 4")
    if min(overall_std, large_size_std) < 0.0 or minimum_effect <= 0.0:
        raise ValueError("power-analysis scales must be non-negative")
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha / (2.0 * comparison_count))
    power_quantile = normal.inv_cdf(power)

    def count(standard_deviation: float) -> int:
        return int(
            math.ceil(
                ((critical + power_quantile) * standard_deviation / minimum_effect) ** 2
            )
        )

    ood_proxy_std = 1.5 * large_size_std
    overall_count = count(overall_std)
    ood_count = count(ood_proxy_std)
    return {
        "z_critical": float(critical),
        "z_power": float(power_quantile),
        "overall_std": float(overall_std),
        "large_size_std": float(large_size_std),
        "ood_proxy_std": float(ood_proxy_std),
        "n_overall": overall_count,
        "n_ood": ood_count,
        "required_backbones": max(20, overall_count, ood_count),
    }


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    if lock.get("status") != "locked":
        raise RuntimeError("power analysis requires a completed protocol lock")
    validate_p5_advancement(config, lock)
    p8_selection = validate_p8_selection(config, lock)
    require_clean_worktree(allow_dirty=False)
    output_path = resolve_path(args.output)
    if output_path != resolve_path(config.paths.confirmation_power):
        raise ValueError("power output must use the locked confirmation_power path")
    if output_path.exists():
        raise FileExistsError("confirmatory power record is immutable")
    candidate_config = method_by_name(config, args.candidate)
    if (
        not candidate_config.confirmatory_eligible
        or args.candidate != p8_selection["selected_track_f"]
        or args.comparison_count != int(p8_selection["comparison_count"])
    ):
        raise ValueError("power inputs must exactly match the frozen P8 decision")
    baseline = "b0_legacy_l2_cem"
    overall_differences: list[float] = []
    large_differences: list[float] = []
    pilot_seeds = tuple(config.protocol.training_seeds[:8])
    for seed in pilot_seeds:
        candidate = load_validation_seed_rows(
            config,
            lock,
            method=args.candidate,
            backbone_seed=int(seed),
        )
        reference = load_validation_seed_rows(
            config,
            lock,
            method=baseline,
            backbone_seed=int(seed),
        )
        candidate_by_id = {str(row["task_id"]): row for row in candidate}
        reference_by_id = {str(row["task_id"]): row for row in reference}
        if set(candidate_by_id) != set(reference_by_id):
            raise ValueError("pilot methods do not contain identical validation tasks")
        identifiers = sorted(candidate_by_id)
        overall_differences.append(
            float(
                np.mean(
                    [
                        float(candidate_by_id[key]["success"])
                        - float(reference_by_id[key]["success"])
                        for key in identifiers
                    ]
                )
            )
        )
        large = [
            key
            for key in identifiers
            if int(candidate_by_id[key]["maze_size"]) in (19, 21)
        ]
        large_differences.append(
            float(
                np.mean(
                    [
                        float(candidate_by_id[key]["success"])
                        - float(reference_by_id[key]["success"])
                        for key in large
                    ]
                )
            )
        )
    power = required_backbone_count(
        overall_std=float(np.std(overall_differences, ddof=1)),
        large_size_std=float(np.std(large_differences, ddof=1)),
        comparison_count=args.comparison_count,
    )
    available = len(config.protocol.training_seeds)
    payload = {
        "schema": "vector-jepa-confirmatory-power-v1",
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "candidate": args.candidate,
        "baseline": baseline,
        "action_selection": config.protocol.primary_action_selection,
        "comparison_count": args.comparison_count,
        "p8_selection": {
            "path": str(resolve_path(config.paths.p8_selection)),
            "sha256": sha256_file(resolve_path(config.paths.p8_selection)),
        },
        "pilot_backbone_seeds": list(pilot_seeds),
        "pilot_seed_count": len(pilot_seeds),
        "overall_seed_differences": overall_differences,
        "large_size_seed_differences": large_differences,
        **power,
        "available_backbones": available,
        "claim_status": (
            "adequately_powered"
            if available >= int(power["required_backbones"])
            else "exploratory_only"
        ),
    }
    atomic_json_dump(output_path, payload)


if __name__ == "__main__":
    main()
