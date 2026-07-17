"""Baseline-only model-seed variance estimate for confirmation sample size."""

from __future__ import annotations

import argparse
import math
from statistics import NormalDist

import numpy as np

from distance_head_study.common import (
    atomic_json_dump,
    canonical_json_sha256,
    load_study_config,
    merge_hash_bindings,
    require_clean_worktree,
    resolve_path,
)
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.results import (
    load_complete_rows,
    result_directory,
    result_evidence_hashes,
)


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if len(parsed) < 3 or len(parsed) != len(set(parsed)):
        raise ValueError("baseline-only power analysis needs at least three backbones")
    return parsed


def _complete(path):
    return path if (path / "rows.jsonl").exists() else path / "merged"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--split-role", default="legacy")
    parser.add_argument("--baseline", default="b_dh_cem")
    parser.add_argument("--backbone-seeds", required=True)
    parser.add_argument("--head-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    lock = verify_protocol_lock(config)
    if (
        args.split_role != "legacy"
        or args.baseline != "b_dh_cem"
        or args.head_seed != 0
    ):
        raise ValueError("power analysis is locked to legacy b_dh_cem head seed 0")
    backbones = _parse_ints(args.backbone_seeds)
    if not set(backbones) <= set(config.seeds.historical_backbones):
        raise ValueError("power analysis may use historical backbone seeds only")
    input_hashes = {}
    overall = []
    ood = []
    for seed in backbones:
        directory = _complete(
            result_directory(
                config,
                split_role=args.split_role,
                method=args.baseline,
                backbone_seed=seed,
                head_seed=args.head_seed,
                action_protocol="corrected_v1",
            )
        )
        metadata, rows = load_complete_rows(directory)
        expected = (
            metadata.get("analysis_spec_sha256") == lock["analysis_spec_sha256"]
            and metadata.get("protocol_lock_sha256") == lock["protocol_lock_sha256"]
            and metadata.get("split_role") == args.split_role
            and metadata.get("method", {}).get("name") == args.baseline
            and int(metadata.get("backbone_seed", -1)) == seed
            and int(metadata.get("head_seed", -1)) == args.head_seed
            and metadata.get("action_protocol") == "corrected_v1"
            and int(metadata.get("diagnostic_limit", -1)) == 0
        )
        if not expected:
            raise ValueError(
                f"power-analysis result differs from protocol: {directory}"
            )
        overall.append(float(np.mean([float(row["success"]) for row in rows])))
        ood_rows = [row for row in rows if int(row["maze_size"]) > 21]
        if not ood_rows:
            raise ValueError("baseline-only power source has no OOD tasks")
        ood.append(float(np.mean([float(row["success"]) for row in ood_rows])))
        input_hashes = merge_hash_bindings(
            input_hashes, result_evidence_hashes(directory, metadata)
        )
    z_power = NormalDist().inv_cdf(config.analysis.required_power)

    def required(values: list[float], effect: float, family_size: int) -> int:
        baseline_seed_sd = float(np.std(values, ddof=1))
        # Candidate-baseline covariance is unavailable before candidate results.
        # Treat the two model-seed effects as independent, a conservative
        # baseline-only proxy unless pairing later reduces the variance.
        difference_sd = math.sqrt(2.0) * baseline_seed_sd
        z_alpha = NormalDist().inv_cdf(
            1.0 - config.analysis.familywise_alpha / family_size
        )
        return int(math.ceil(((z_alpha + z_power) * difference_sd / effect) ** 2))

    recommendations = {}
    for route, family_size in (("positive", 4), ("negative", 8)):
        raw_required = max(
            required(overall, config.analysis.minimum_overall_delta, family_size),
            required(ood, config.analysis.minimum_ood_delta, family_size),
        )
        recommendations[route] = max(
            config.analysis.minimum_confirmation_backbones, raw_required
        )
    if max(recommendations.values()) > len(config.seeds.ordered_confirmation_backbones):
        raise RuntimeError(
            "ordered fresh-seed registry is shorter than required power n"
        )
    payload = {
        "schema": "distance-head-baseline-only-power-v1",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "baseline_only": True,
        "baseline": args.baseline,
        "source_split_role": args.split_role,
        "backbone_seeds": list(backbones),
        "overall_seed_sr": overall,
        "ood_seed_sr": ood,
        "familywise_alpha": config.analysis.familywise_alpha,
        "positive_family_size": 4,
        "negative_family_size": 8,
        "target_power": config.analysis.required_power,
        "overall_mei": config.analysis.minimum_overall_delta,
        "ood_mei": config.analysis.minimum_ood_delta,
        "recommended_confirmation_n_positive": recommendations["positive"],
        "recommended_confirmation_n_negative": recommendations["negative"],
        "input_hashes": input_hashes,
        "does_not_use_candidate_effect": True,
        "variance_proxy": "sqrt(2) times historical baseline model-seed SR SD",
    }
    payload["power_sha256"] = canonical_json_sha256(payload)
    output = resolve_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite power analysis: {output}")
    atomic_json_dump(output, payload)
    print(output)


if __name__ == "__main__":
    main()
