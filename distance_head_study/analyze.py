"""Crossed paired analysis with backbone seeds as independent replications."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from distance_head_study.common import (
    atomic_json_dump,
    canonical_json_sha256,
    load_json,
    load_study_config,
    merge_hash_bindings,
    require_clean_worktree,
    resolve_path,
    sha256_file,
)
from distance_head_study.gates import load_signed_artifact, require_evaluation_gate
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.results import (
    load_complete_rows,
    result_directory,
    result_evidence_hashes,
)

ANALYSIS_SCHEMA = "distance-head-crossed-analysis-v1"


@dataclass(frozen=True)
class TaskTable:
    task_ids: tuple[str, ...]
    maze_sizes: np.ndarray
    success: np.ndarray
    spl: np.ndarray
    loop: np.ndarray
    invalid_rate: np.ndarray
    assistance_rate: np.ndarray
    predictor_transitions_per_step: np.ndarray
    episode_seconds_per_step: np.ndarray


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("seed list must be nonempty and unique")
    return parsed


def _complete_directory(path: Path) -> Path:
    if (path / "rows.jsonl").exists():
        return path
    if (path / "merged" / "rows.jsonl").exists():
        return path / "merged"
    raise FileNotFoundError(f"no complete direct or merged result at {path}")


def _head_averaged_table(
    config: Any,
    *,
    split_role: str,
    method: str,
    backbone_seed: int,
    head_seeds: tuple[int, ...],
    action_protocol: str,
    analysis_spec_sha256: str,
    protocol_lock_sha256: str,
    confirmation_gate: dict[str, Any] | None = None,
) -> tuple[TaskTable, dict[str, str]]:
    resolved_method, method_hash, decision_hashes = load_and_resolve_method(
        config.paths.method_catalog,
        method,
        decision_root=config.paths.decision_root,
        protocol_lock={
            "analysis_spec_sha256": analysis_spec_sha256,
            "protocol_lock_sha256": protocol_lock_sha256,
        },
    )
    rows_by_head: list[dict[str, dict[str, Any]]] = []
    input_hashes: dict[str, str] = {}
    for head_seed in head_seeds:
        directory = _complete_directory(
            result_directory(
                config,
                split_role=split_role,
                method=method,
                backbone_seed=backbone_seed,
                head_seed=head_seed,
                action_protocol=action_protocol,
            )
        )
        metadata, rows = load_complete_rows(directory)
        if metadata["method"]["name"] != method:
            raise ValueError("result method metadata mismatch")
        if (
            metadata.get("method") != resolved_method.model_dump(mode="json")
            or metadata.get("method_sha256") != method_hash
            or metadata.get("decision_sha256s") != list(decision_hashes)
        ):
            raise ValueError("result effective method provenance mismatch")
        if int(metadata["backbone_seed"]) != backbone_seed:
            raise ValueError("result backbone metadata mismatch")
        if int(metadata["head_seed"]) != head_seed:
            raise ValueError("result head-seed metadata mismatch")
        if metadata.get("split_role") != split_role:
            raise ValueError("result split-role metadata mismatch")
        if metadata.get("analysis_spec_sha256") != analysis_spec_sha256:
            raise ValueError("result analysis-lock metadata mismatch")
        if metadata.get("protocol_lock_sha256") != protocol_lock_sha256:
            raise ValueError("result protocol-lock metadata mismatch")
        if metadata["action_protocol"] != action_protocol:
            raise ValueError("result action protocol mismatch")
        if confirmation_gate is not None:
            if metadata.get("analysis_spec_sha256") != confirmation_gate.get(
                "analysis_spec_sha256"
            ):
                raise ValueError("confirmation result uses another analysis lock")
            if metadata.get("protocol_lock_sha256") != confirmation_gate.get(
                "protocol_lock_sha256"
            ):
                raise ValueError("confirmation result uses another protocol lock")
            checkpoint = metadata.get("checkpoint", {})
            locked_hashes = confirmation_gate["locked_checkpoint_hashes"]
            for path_key, hash_key in (
                ("backbone_path", "backbone_sha256"),
                ("head_checkpoint_path", "head_checkpoint_sha256"),
            ):
                path = checkpoint.get(path_key)
                if path is not None and locked_hashes.get(path) != checkpoint.get(
                    hash_key
                ):
                    raise ValueError("confirmation result used an unsealed checkpoint")
        mapping = {str(row["task_id"]): row for row in rows}
        rows_by_head.append(mapping)
        input_hashes = merge_hash_bindings(
            input_hashes, result_evidence_hashes(directory, metadata)
        )
    identifiers = tuple(sorted(rows_by_head[0]))
    if any(set(mapping) != set(identifiers) for mapping in rows_by_head[1:]):
        raise ValueError("head-seed result files do not contain identical tasks")

    def average(name: str) -> np.ndarray:
        values = []
        for identifier in identifiers:
            per_head = []
            for mapping in rows_by_head:
                row = mapping[identifier]
                if name == "invalid_rate":
                    value = int(row["invalid_actions"]) / max(
                        int(row["path_length"]), 1
                    )
                elif name == "predictor_transitions_per_step":
                    value = (
                        int(row["plan_transitions"]) + int(row["fallback_transitions"])
                    ) / max(int(row["path_length"]), 1)
                elif name == "episode_seconds_per_step":
                    value = float(row["episode_seconds"]) / max(
                        int(row["path_length"]), 1
                    )
                else:
                    value = row[name]
                per_head.append(float(value))
            values.append(float(np.mean(per_head)))
        return np.asarray(values, dtype=np.float64)

    first = rows_by_head[0]
    table = TaskTable(
        task_ids=identifiers,
        maze_sizes=np.asarray([int(first[key]["maze_size"]) for key in identifiers]),
        success=average("success"),
        spl=average("spl"),
        loop=average("loop_or_cycle"),
        invalid_rate=average("invalid_rate"),
        assistance_rate=average("assistance_rate"),
        predictor_transitions_per_step=average("predictor_transitions_per_step"),
        episode_seconds_per_step=average("episode_seconds_per_step"),
    )
    return table, input_hashes


def _paired_difference(
    candidate: TaskTable,
    baseline: TaskTable,
    field: str,
    *,
    ood_only: bool,
) -> np.ndarray:
    if candidate.task_ids != baseline.task_ids:
        raise ValueError("candidate and baseline task IDs differ")
    mask = (
        candidate.maze_sizes > 21
        if ood_only
        else np.ones(len(candidate.task_ids), dtype=bool)
    )
    if ood_only and not bool(mask.any()):
        raise ValueError("requested OOD endpoint on a split without sizes above 21")
    return getattr(candidate, field)[mask] - getattr(baseline, field)[mask]


def _crossed_bootstrap(
    differences: list[np.ndarray],
    replicate_seeds: list[int],
    *,
    familywise_upper_quantile: float,
    strata: list[np.ndarray] | None = None,
) -> dict[str, float | int | str | bool]:
    if len(differences) < 2:
        raise ValueError("confirmatory crossed bootstrap needs at least two backbones")
    reference_shape = differences[0].shape
    if not reference_shape or reference_shape[0] == 0:
        raise ValueError("confirmatory bootstrap needs nonempty task differences")
    if any(values.shape != reference_shape for values in differences):
        raise ValueError("crossed bootstrap needs identical tasks for every backbone")
    observed = float(np.mean([values.mean() for values in differences]))
    if strata is not None:
        if len(strata) != len(differences) or any(
            labels.shape != values.shape
            for labels, values in zip(strata, differences, strict=True)
        ):
            raise ValueError("bootstrap strata do not align with paired differences")
        if any(not np.array_equal(labels, strata[0]) for labels in strata[1:]):
            raise ValueError("crossed bootstrap needs identical task strata")
    replicates = np.empty(len(replicate_seeds), dtype=np.float64)
    seed_count = len(differences)
    task_count = int(reference_shape[0])
    reference_strata = None if strata is None else strata[0]
    for replicate_index, seed in enumerate(replicate_seeds):
        rng = np.random.default_rng(seed)
        selected = rng.integers(0, seed_count, size=seed_count)
        if reference_strata is None:
            tasks = rng.integers(0, task_count, size=task_count)
        else:
            tasks = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for label in np.unique(reference_strata)
                    for indices in [np.flatnonzero(reference_strata == label)]
                ]
            )
        seed_means = [
            float(differences[int(seed_index)][tasks].mean()) for seed_index in selected
        ]
        replicates[replicate_index] = float(np.mean(seed_means))
    return {
        "backbone_n": seed_count,
        "task_n_per_backbone": int(len(differences[0])),
        "mean_delta": observed,
        "ci95_low": float(np.quantile(replicates, 0.025)),
        "ci95_high": float(np.quantile(replicates, 0.975)),
        "one_sided_upper95": float(np.quantile(replicates, 0.95)),
        "one_sided_upper_familywise": float(
            np.quantile(replicates, familywise_upper_quantile)
        ),
        "task_resampling": "crossed_stratified_by_maze_size"
        if strata is not None
        else "crossed_unstratified",
        "task_indices_shared_across_backbone_resamples": True,
    }


def _one_sided_sign_flip_p(
    differences: list[np.ndarray], replicate_seeds: list[int]
) -> dict[str, float | int | str]:
    """Test a positive mean effect at the independent backbone-seed level."""

    seed_effects = np.asarray(
        [float(values.mean()) for values in differences], dtype=np.float64
    )
    observed = float(seed_effects.mean())
    count = len(seed_effects)
    tolerance = 1e-15
    if count <= 16:
        total = 1 << count
        extreme = 0
        bit_positions = np.arange(count, dtype=np.uint64)
        for mask in range(total):
            bits = (np.uint64(mask) >> bit_positions) & np.uint64(1)
            signs = np.where(bits == 1, 1.0, -1.0)
            extreme += int(float(np.mean(seed_effects * signs)) >= observed - tolerance)
        p_value = extreme / total
        mode = "exact_backbone_sign_flip"
        replicates = total
    else:
        if not replicate_seeds:
            raise ValueError("Monte Carlo sign-flip test needs locked seeds")
        extreme = 0
        for seed in replicate_seeds:
            rng = np.random.default_rng(seed)
            signs = rng.choice(np.asarray((-1.0, 1.0)), size=count)
            extreme += int(float(np.mean(seed_effects * signs)) >= observed - tolerance)
        replicates = len(replicate_seeds)
        p_value = (extreme + 1) / (replicates + 1)
        mode = "monte_carlo_backbone_sign_flip"
    return {
        "one_sided_p": float(p_value),
        "test": mode,
        "replicates": int(replicates),
        "observed_mean": observed,
    }


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baselines", default="b_dh_cem,b_l2_cem")
    parser.add_argument(
        "--split-role", choices=("select", "confirm", "stress"), required=True
    )
    parser.add_argument("--backbone-seeds", required=True)
    parser.add_argument("--head-seeds", default="0")
    parser.add_argument(
        "--family-size-override",
        type=int,
        default=0,
        help="Total preregistered primary contrasts across jointly closed analyses.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    lock = verify_protocol_lock(config)
    backbones = _parse_ints(args.backbone_seeds)
    head_seeds = _parse_ints(args.head_seeds)
    baselines = tuple(
        item.strip() for item in args.baselines.split(",") if item.strip()
    )
    if not baselines or len(baselines) != len(set(baselines)):
        raise ValueError("baseline list must be nonempty and unique")
    if baselines != ("b_dh_cem", "b_l2_cem"):
        raise ValueError("formal analysis requires both locked baselines in order")
    if args.candidate in baselines:
        raise ValueError("candidate must differ from both locked baselines")
    methods = (args.candidate, *baselines)
    confirmation_gate = None
    if args.split_role == "confirm":
        confirmation_gate = load_signed_artifact(
            config.paths.confirm_opened,
            signature_field="confirm_open_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        if (
            confirmation_gate.get("analysis_spec_sha256")
            != lock["analysis_spec_sha256"]
            or confirmation_gate.get("protocol_lock_sha256")
            != lock["protocol_lock_sha256"]
        ):
            raise ValueError("confirmation gate uses another protocol lock")
        if args.candidate not in confirmation_gate["allowed_methods"]:
            raise ValueError("candidate is outside the sealed confirmation matrix")
        if list(backbones) != confirmation_gate["backbone_seeds"]:
            raise ValueError("analysis backbones differ from the confirmation seal")
        if head_seeds != (int(confirmation_gate["head_seed"]),):
            raise ValueError("analysis head seed differs from the confirmation seal")
        required_family = 4 if confirmation_gate["claim_route"] == "positive" else 8
        if args.family_size_override not in (0, required_family):
            raise ValueError("family-size override differs from the claim route")
        if required_family == 8 and args.family_size_override != 8:
            raise ValueError(
                "negative confirmation analysis must declare family size 8"
            )
    schedule = load_json(config.paths.bootstrap_schedule)
    signature = schedule.get("schedule_sha256")
    unsigned = {
        key: value for key, value in schedule.items() if key != "schedule_sha256"
    }
    if signature != canonical_json_sha256(unsigned):
        raise ValueError("bootstrap schedule signature mismatch")
    replicate_seeds = [int(value) for value in schedule["replicate_seeds"]]
    tables: dict[tuple[str, int, str], TaskTable] = {}
    gate_hashes: dict[str, str] = {}
    if confirmation_gate is not None:
        gate_hashes = merge_hash_bindings(
            {
                resolve_path(config.paths.confirm_opened).as_posix(): sha256_file(
                    config.paths.confirm_opened
                )
            },
            confirmation_gate["input_hashes"],
        )
    elif args.split_role in ("select", "stress"):
        gate_path = resolve_path(
            config.paths.shortlist_lock
            if args.split_role == "select"
            else config.paths.closure_gate
        )
        gate_hashes = {gate_path.as_posix(): sha256_file(gate_path)}
        for method in methods:
            method_heads = (0,) if method == "b_l2_cem" else head_seeds
            for backbone in backbones:
                for head_seed in method_heads:
                    gate = require_evaluation_gate(
                        config,
                        split_role=args.split_role,
                        method=method,
                        backbone_seed=backbone,
                        head_seed=head_seed,
                    )
                    if gate is None:
                        raise RuntimeError("sealed analysis split has no active gate")
                    gate_hashes = merge_hash_bindings(gate_hashes, gate["input_hashes"])
    input_hashes: dict[str, str] = merge_hash_bindings(
        {
            resolve_path(config.paths.bootstrap_schedule).as_posix(): sha256_file(
                config.paths.bootstrap_schedule
            )
        },
        gate_hashes,
    )
    for method in methods:
        method_heads = (0,) if method == "b_l2_cem" else head_seeds
        for backbone in backbones:
            for protocol in ("corrected_v1", "unmasked"):
                table, hashes = _head_averaged_table(
                    config,
                    split_role=args.split_role,
                    method=method,
                    backbone_seed=backbone,
                    head_seeds=method_heads,
                    action_protocol=protocol,
                    analysis_spec_sha256=lock["analysis_spec_sha256"],
                    protocol_lock_sha256=lock["protocol_lock_sha256"],
                    confirmation_gate=confirmation_gate,
                )
                tables[(method, backbone, protocol)] = table
                input_hashes = merge_hash_bindings(input_hashes, hashes)
    endpoints: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    natural_endpoint_count = len(baselines) * (1 if args.split_role == "select" else 2)
    if args.family_size_override and args.family_size_override < natural_endpoint_count:
        raise ValueError("family-size override cannot be smaller than this analysis")
    endpoint_count = int(args.family_size_override or natural_endpoint_count)
    upper_quantile = 1.0 - config.analysis.familywise_alpha / endpoint_count
    for baseline in baselines:
        for endpoint, ood in (("overall", False), ("ood", True)):
            if ood and args.split_role == "select":
                continue
            differences = [
                _paired_difference(
                    tables[(args.candidate, backbone, "corrected_v1")],
                    tables[(baseline, backbone, "corrected_v1")],
                    "success",
                    ood_only=ood,
                )
                for backbone in backbones
            ]
            strata = []
            for backbone in backbones:
                table = tables[(args.candidate, backbone, "corrected_v1")]
                mask = (
                    table.maze_sizes > 21
                    if ood
                    else np.ones(len(table.task_ids), dtype=bool)
                )
                strata.append(table.maze_sizes[mask])
            name = f"{args.candidate}__vs__{baseline}__corrected_{endpoint}_sr"
            result = _crossed_bootstrap(
                differences,
                replicate_seeds,
                familywise_upper_quantile=upper_quantile,
                strata=strata,
            )
            result.update(_one_sided_sign_flip_p(differences, replicate_seeds))
            endpoints[name] = result
            p_values[name] = float(result["one_sided_p"])
    adjusted = (
        _holm(p_values)
        if endpoint_count == len(p_values)
        else {
            name: min(1.0, endpoint_count * value) for name, value in p_values.items()
        }
    )
    for name, value in adjusted.items():
        endpoints[name]["familywise_adjusted_p"] = value
        endpoints[name]["adjustment"] = (
            "holm" if endpoint_count == len(p_values) else "bonferroni_joint_family"
        )

    secondary: dict[str, Any] = {}
    for baseline in baselines:
        for protocol in ("corrected_v1", "unmasked"):
            for field in (
                "success",
                "spl",
                "loop",
                "invalid_rate",
                "assistance_rate",
                "predictor_transitions_per_step",
                "episode_seconds_per_step",
            ):
                differences = [
                    _paired_difference(
                        tables[(args.candidate, backbone, protocol)],
                        tables[(baseline, backbone, protocol)],
                        field,
                        ood_only=False,
                    )
                    for backbone in backbones
                ]
                secondary[f"{args.candidate}__vs__{baseline}__{protocol}_{field}"] = {
                    "mean_delta": float(
                        np.mean([values.mean() for values in differences])
                    ),
                    "per_backbone_delta": [
                        float(values.mean()) for values in differences
                    ],
                }
    reference = "b_dh_cem"
    positive = False
    positive_checks: dict[str, bool] = {}
    treatment_pass = False
    frontier_pass = False
    if args.split_role == "confirm" and reference in baselines:
        overall_name = f"{args.candidate}__vs__{reference}__corrected_overall_sr"
        ood_name = f"{args.candidate}__vs__{reference}__corrected_ood_sr"
        treatment_checks = {
            "overall_mei": endpoints[overall_name]["mean_delta"]
            >= config.analysis.minimum_overall_delta,
            "ood_mei": endpoints[ood_name]["mean_delta"]
            >= config.analysis.minimum_ood_delta,
            "overall_superiority": endpoints[overall_name]["ci95_low"] > 0
            and endpoints[overall_name]["familywise_adjusted_p"]
            < config.analysis.familywise_alpha,
            "ood_superiority": endpoints[ood_name]["ci95_low"] > 0
            and endpoints[ood_name]["familywise_adjusted_p"]
            < config.analysis.familywise_alpha,
            "unmasked_noninferiority": secondary[
                f"{args.candidate}__vs__{reference}__unmasked_success"
            ]["mean_delta"]
            >= -config.analysis.max_secondary_drop,
            "spl_noninferiority": secondary[
                f"{args.candidate}__vs__{reference}__corrected_v1_spl"
            ]["mean_delta"]
            >= -config.analysis.max_secondary_drop,
        }
        treatment_pass = all(treatment_checks.values())
        frontier_reference = "b_l2_cem"
        frontier_overall = (
            f"{args.candidate}__vs__{frontier_reference}__corrected_overall_sr"
        )
        frontier_ood = f"{args.candidate}__vs__{frontier_reference}__corrected_ood_sr"
        frontier_checks = {
            "l2_overall_superiority": endpoints[frontier_overall]["ci95_low"] > 0
            and endpoints[frontier_overall]["familywise_adjusted_p"]
            < config.analysis.familywise_alpha,
            "l2_ood_superiority": endpoints[frontier_ood]["ci95_low"] > 0
            and endpoints[frontier_ood]["familywise_adjusted_p"]
            < config.analysis.familywise_alpha,
            "l2_unmasked_noninferiority": secondary[
                f"{args.candidate}__vs__{frontier_reference}__unmasked_success"
            ]["mean_delta"]
            >= -config.analysis.max_secondary_drop,
            "l2_spl_noninferiority": secondary[
                f"{args.candidate}__vs__{frontier_reference}__corrected_v1_spl"
            ]["mean_delta"]
            >= -config.analysis.max_secondary_drop,
        }
        positive_checks = {**treatment_checks, **frontier_checks}
        frontier_pass = treatment_pass and all(frontier_checks.values())
        positive = frontier_pass
    output = {
        "schema": ANALYSIS_SCHEMA,
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "split_role": args.split_role,
        "candidate": args.candidate,
        "baselines": list(baselines),
        "backbone_seeds": list(backbones),
        "head_seeds_nested_within_backbone": list(head_seeds),
        "independent_unit": "backbone_training_seed",
        "familywise_primary_count": endpoint_count,
        "bootstrap_schedule_sha256": signature,
        "input_evidence_hashes": input_hashes,
        "primary_endpoints": endpoints,
        "secondary": secondary,
        "positive_claim_checks": positive_checks,
        "distance_head_treatment_pass": treatment_pass,
        "new_vector_jepa_frontier_pass": frontier_pass,
        "positive_claim_pass": positive,
        "negative_claim_requires_closure_and_two_finalists": True,
    }
    output["analysis_sha256"] = canonical_json_sha256(output)
    path = resolve_path(args.output)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {path}")
    atomic_json_dump(path, output)
    print(Path(path))


if __name__ == "__main__":
    main()
