"""Validate, aggregate, and bootstrap planner-frontier result files."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from final_closure.common import (
    sha256_file,
    validate_task_rows,
)
from spatial_jepa_planning.common import canonical_json_sha256
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_json_dump,
    atomic_text_dump,
    load_json,
    load_study_config,
    planner_seed_values,
    resolve_path,
    validate_finite_tree,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.effective_methods import (
    effective_method_sha256,
    resolve_effective_method,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument(
        "--split-role",
        choices=("development", "validation", "confirmatory"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="")
    parser.add_argument("--stage", choices=("P2", "P3", "P4", "P5", "P6", "P7", "P8"))
    parser.add_argument("--diagnostic-allow-missing", action="store_true")
    return parser.parse_args()


def result_path(
    config: Any,
    *,
    method: str,
    backbone_seed: int,
    planner_seed: int,
    search_seed: int,
    split_role: str,
    action_selection: str,
) -> Path:
    return resolve_path(
        config.paths.result_template.format(
            method=method,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
            search_seed=search_seed,
            split=split_role,
            action_selection=action_selection,
        )
    )


def load_result(
    path: Path,
    *,
    analysis_hash: str,
    method: str,
    backbone_seed: int,
    planner_seed: int,
    search_seed: int,
    split_role: str,
    action_selection: str,
    expected_count: int,
    expected_manifest_sha256: str,
    expected_code_fingerprint: str,
    expected_method_sha256: str | None = None,
) -> dict[str, Any]:
    value = load_json(path)
    metadata = value.get("metadata", {})
    if metadata.get("analysis_spec_sha256") != analysis_hash:
        raise ValueError(f"analysis-spec mismatch: {path}")
    if metadata.get("method", {}).get("name") != method:
        raise ValueError(f"method label mismatch: {path}")
    if (
        expected_method_sha256 is not None
        and canonical_json_sha256(metadata.get("method", {})) != expected_method_sha256
    ):
        raise ValueError(f"effective method specification mismatch: {path}")
    if int(metadata.get("backbone_seed", -1)) != backbone_seed:
        raise ValueError(f"backbone seed label mismatch: {path}")
    expected_planner = planner_seed if planner_seed != 0 else None
    if metadata.get("planner_seed") != expected_planner:
        raise ValueError(f"planner seed label mismatch: {path}")
    if int(metadata.get("search_seed", -1)) != search_seed:
        raise ValueError(f"search seed label mismatch: {path}")
    if value.get("split_role") != split_role:
        raise ValueError(f"split label mismatch: {path}")
    if value.get("action_selection") != action_selection:
        raise ValueError(f"action-selection label mismatch: {path}")
    if metadata.get("git_dirty") is not False:
        raise ValueError(f"formal result was produced from a dirty worktree: {path}")
    if metadata.get("code_fingerprint") != expected_code_fingerprint:
        raise ValueError(f"result code fingerprint mismatch: {path}")
    if value.get("manifest", {}).get("sha256") != expected_manifest_sha256:
        raise ValueError(f"manifest hash mismatch: {path}")
    candidate_artifact = value.get("candidate_traces", {})
    candidate_path = Path(str(candidate_artifact.get("path", "")))
    if not candidate_path.is_absolute():
        candidate_path = resolve_path(candidate_path)
    if not candidate_path.is_file():
        raise FileNotFoundError(
            f"candidate-trace artifact is missing: {candidate_path}"
        )
    if sha256_file(candidate_path) != candidate_artifact.get("sha256"):
        raise ValueError(f"candidate-trace hash mismatch: {candidate_path}")
    validate_task_rows(value.get("tasks"), expected_count)
    return value


AVERAGED_TASK_METRICS = (
    "success",
    "spl",
    "invalid_actions",
    "loop_or_cycle",
    "revisit_rate",
    "two_cycle_rate",
    "assistance_rate",
    "invalid_correction_rate",
    "backtrack_correction_rate",
    "dead_end_recovery_rate",
    "final_bfs_distance",
)


def average_nested_task_rows(
    runs: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not runs:
        raise ValueError("cannot average an empty nested run block")
    by_run = [{str(row["task_id"]): row for row in rows} for rows in runs]
    identifiers = sorted(by_run[0])
    if any(sorted(rows) != identifiers for rows in by_run[1:]):
        raise ValueError("nested runs do not share identical task IDs")
    output: list[dict[str, Any]] = []
    for identifier in identifiers:
        source = by_run[0][identifier]
        row = {
            "task_id": identifier,
            "maze_size": int(source["maze_size"]),
            "shortest_path_bin": source["shortest_path_bin"],
            "dead_end_density": float(source["dead_end_density"]),
            "junction_count": int(source["junction_count"]),
            "mean_corridor_length": float(source["mean_corridor_length"]),
        }
        for metric in AVERAGED_TASK_METRICS:
            row[metric] = float(
                np.mean([float(run[identifier][metric]) for run in by_run])
            )
        row["decision_count"] = float(
            np.mean([float(run[identifier]["decision_count"]) for run in by_run])
        )
        auxiliary_names = sorted(
            {
                str(name)
                for run in by_run
                for name in run[identifier].get("auxiliary", {})
            }
        )
        row["auxiliary"] = {
            name: float(
                np.mean(
                    [
                        float(run[identifier].get("auxiliary", {}).get(name, 0.0))
                        for run in by_run
                    ]
                )
            )
            for name in auxiliary_names
        }
        output.append(row)
    return output


def collapse_nested_records(
    raw: dict[
        str,
        dict[str, dict[int, dict[int, dict[int, list[dict[str, Any]]]]]],
    ],
    *,
    config: Any,
    methods: dict[str, Any],
    backbone_seeds: tuple[int, ...],
) -> dict[str, dict[str, list[list[dict[str, Any]]]]]:
    """Average search then planner seeds inside each backbone block."""

    collapsed: dict[str, dict[str, list[list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for method_name, by_action in raw.items():
        method = methods[method_name]
        expected_planners = planner_seed_values(config, method)
        for action_selection, by_backbone in by_action.items():
            for backbone_seed in backbone_seeds:
                if backbone_seed not in by_backbone:
                    continue
                block = by_backbone[backbone_seed]
                planner_averages: list[list[dict[str, Any]]] = []
                for planner_seed in expected_planners:
                    by_search = block.get(planner_seed, {})
                    if set(by_search) != set(config.protocol.search_seeds):
                        raise ValueError(
                            "nested block is missing a preregistered search seed"
                        )
                    planner_averages.append(
                        average_nested_task_rows(
                            [by_search[seed] for seed in config.protocol.search_seeds]
                        )
                    )
                collapsed[method_name][action_selection].append(
                    average_nested_task_rows(planner_averages)
                )
    return collapsed


def _subset_rows(rows: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    if subset == "overall":
        selected = rows
    elif subset == "ood":
        selected = [row for row in rows if int(row["maze_size"]) > 21]
    elif subset == "large_proxy_19_21":
        selected = [row for row in rows if int(row["maze_size"]) in (19, 21)]
    else:
        raise ValueError(f"unknown analysis subset: {subset}")
    if not selected:
        raise ValueError(f"analysis subset contains no tasks: {subset}")
    return selected


def nested_metric_cube(
    raw: dict[
        str,
        dict[str, dict[int, dict[int, dict[int, list[dict[str, Any]]]]]],
    ],
    *,
    config: Any,
    method: Any,
    backbone_seeds: tuple[int, ...],
    action_selection: str,
    metric: str,
    subset: str,
) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    """Return [backbone, planner seed, task] after averaging search seeds."""

    if method.name not in raw or action_selection not in raw[method.name]:
        raise ValueError(f"missing nested records for {method.name}/{action_selection}")
    by_backbone = raw[method.name][action_selection]
    if set(by_backbone) != set(backbone_seeds):
        raise ValueError(f"nested backbone block mismatch for {method.name}")
    planner_seeds = planner_seed_values(config, method)
    backbone_values: list[np.ndarray] = []
    canonical_ids: tuple[str, ...] | None = None
    canonical_strata: tuple[str, ...] | None = None
    for backbone_seed in backbone_seeds:
        by_planner = by_backbone[backbone_seed]
        if set(by_planner) != set(planner_seeds):
            raise ValueError(f"nested planner-seed block mismatch for {method.name}")
        planner_values: list[np.ndarray] = []
        for planner_seed in planner_seeds:
            by_search = by_planner[planner_seed]
            if set(by_search) != set(config.protocol.search_seeds):
                raise ValueError(f"nested search-seed block mismatch for {method.name}")
            rows = _subset_rows(
                average_nested_task_rows(
                    [by_search[seed] for seed in config.protocol.search_seeds]
                ),
                subset,
            )
            by_id = {str(row["task_id"]): row for row in rows}
            identifiers = tuple(sorted(by_id))
            strata = tuple(str(by_id[key]["maze_size"]) for key in identifiers)
            if canonical_ids is None:
                canonical_ids = identifiers
                canonical_strata = strata
            elif identifiers != canonical_ids or strata != canonical_strata:
                raise ValueError("nested runs disagree on task IDs or maze-size strata")
            planner_values.append(
                np.asarray(
                    [float(by_id[key][metric]) for key in identifiers],
                    dtype=np.float64,
                )
            )
        backbone_values.append(np.stack(planner_values))
    assert canonical_ids is not None and canonical_strata is not None
    return (
        np.stack(backbone_values),
        tuple(int(seed) for seed in planner_seeds),
        canonical_ids,
        canonical_strata,
    )


def nested_paired_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    alpha: float,
    seed: int,
    strata: tuple[str, ...],
    pair_planner_seeds: bool,
) -> dict[str, Any]:
    """Crossed bootstrap with planner seeds nested inside backbone seeds."""

    if (
        candidate.ndim != 3
        or baseline.ndim != 3
        or candidate.shape[0] != baseline.shape[0]
        or candidate.shape[2] != baseline.shape[2]
        or candidate.shape[2] != len(strata)
        or candidate.shape[0] == 0
        or candidate.shape[2] == 0
    ):
        raise ValueError("nested bootstrap arrays are not aligned")
    if samples <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("nested bootstrap samples/alpha are invalid")
    if pair_planner_seeds and candidate.shape[1] != baseline.shape[1]:
        raise ValueError("paired planner resampling requires equal seed counts")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("nested bootstrap arrays contain non-finite values")
    rng = np.random.default_rng(seed)
    backbone_count, candidate_planners, task_count = candidate.shape
    baseline_planners = baseline.shape[1]
    stratum_array = np.asarray(strata)
    stratum_indices = [
        np.flatnonzero(stratum_array == value) for value in sorted(set(stratum_array))
    ]
    draws = np.empty(samples, dtype=np.float64)
    batch_size = min(128, samples)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        backbone_indices = rng.integers(backbone_count, size=(count, backbone_count))
        selected_candidate = candidate[backbone_indices]
        selected_baseline = baseline[backbone_indices]
        candidate_indices = rng.integers(
            candidate_planners,
            size=(count, backbone_count, candidate_planners),
        )
        if pair_planner_seeds:
            baseline_indices = candidate_indices
        else:
            baseline_indices = rng.integers(
                baseline_planners,
                size=(count, backbone_count, baseline_planners),
            )
        candidate_mean = np.take_along_axis(
            selected_candidate,
            candidate_indices[..., None],
            axis=2,
        ).mean(axis=2)
        baseline_mean = np.take_along_axis(
            selected_baseline,
            baseline_indices[..., None],
            axis=2,
        ).mean(axis=2)
        difference = (candidate_mean - baseline_mean).mean(axis=1)
        selected_tasks = np.concatenate(
            [
                rng.choice(indices, size=(count, len(indices)), replace=True)
                for indices in stratum_indices
            ],
            axis=1,
        )
        if selected_tasks.shape != (count, task_count):
            raise AssertionError("stratified task bootstrap changed task count")
        draws[start : start + count] = np.take_along_axis(
            difference, selected_tasks, axis=1
        ).mean(axis=1)
    delta = float((candidate.mean(axis=1) - baseline.mean(axis=1)).mean(axis=(0, 1)))
    return {
        "delta": delta,
        "ci_low": float(np.quantile(draws, alpha / 2.0)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "alpha": float(alpha),
        "bootstrap_samples": int(samples),
        "backbone_count": int(backbone_count),
        "candidate_planner_seed_count": int(candidate_planners),
        "baseline_planner_seed_count": int(baseline_planners),
        "unique_task_count": int(task_count),
        "maze_size_stratum_count": int(len(stratum_indices)),
        "backbone_resampling": "paired_across_methods",
        "planner_seed_resampling": (
            "paired_within_backbone"
            if pair_planner_seeds
            else "independent_within_backbone"
        ),
        "search_seed_handling": "averaged_before_resampling",
        "task_resampling": "paired_by_task_id_within_maze_size",
    }


def nested_paired_effect(
    raw: dict[
        str,
        dict[str, dict[int, dict[int, dict[int, list[dict[str, Any]]]]]],
    ],
    *,
    config: Any,
    methods: dict[str, Any],
    backbone_seeds: tuple[int, ...],
    candidate: str,
    baseline: str,
    action_selection: str,
    baseline_action_selection: str | None = None,
    metric: str,
    subset: str,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    candidate_cube, candidate_planners, candidate_ids, candidate_strata = (
        nested_metric_cube(
            raw,
            config=config,
            method=methods[candidate],
            backbone_seeds=backbone_seeds,
            action_selection=action_selection,
            metric=metric,
            subset=subset,
        )
    )
    baseline_cube, baseline_planners, baseline_ids, baseline_strata = (
        nested_metric_cube(
            raw,
            config=config,
            method=methods[baseline],
            backbone_seeds=backbone_seeds,
            action_selection=baseline_action_selection or action_selection,
            metric=metric,
            subset=subset,
        )
    )
    if candidate_ids != baseline_ids or candidate_strata != baseline_strata:
        raise ValueError("paired methods do not contain identical task strata")
    return nested_paired_bootstrap(
        candidate_cube,
        baseline_cube,
        samples=config.analysis.bootstrap_samples,
        alpha=alpha,
        seed=seed,
        strata=candidate_strata,
        pair_planner_seeds=candidate_planners == baseline_planners,
    )


def nested_variance_rows(
    raw: dict[
        str,
        dict[str, dict[int, dict[int, dict[int, list[dict[str, Any]]]]]],
    ],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, by_action in sorted(raw.items()):
        for action_selection, by_backbone in sorted(by_action.items()):
            backbone_means: list[float] = []
            planner_within: list[float] = []
            search_within: list[float] = []
            for by_planner in by_backbone.values():
                planner_means: list[float] = []
                for by_search in by_planner.values():
                    search_means = [
                        float(np.mean([float(row["success"]) for row in rows]))
                        for rows in by_search.values()
                    ]
                    planner_means.append(float(np.mean(search_means)))
                    if len(search_means) > 1:
                        search_within.append(float(np.var(search_means, ddof=1)))
                backbone_means.append(float(np.mean(planner_means)))
                if len(planner_means) > 1:
                    planner_within.append(float(np.var(planner_means, ddof=1)))
            output.append(
                {
                    "method": method,
                    "action_selection": action_selection,
                    "backbone_count": len(backbone_means),
                    "backbone_mean_sr_variance": (
                        float(np.var(backbone_means, ddof=1))
                        if len(backbone_means) > 1
                        else 0.0
                    ),
                    "mean_planner_seed_variance_within_backbone": (
                        float(np.mean(planner_within)) if planner_within else 0.0
                    ),
                    "mean_search_seed_variance_within_planner": (
                        float(np.mean(search_within)) if search_within else 0.0
                    ),
                }
            )
    return output


def candidate_mechanism_summary(
    result: dict[str, Any],
    *,
    method: str,
    backbone_seed: int,
    planner_seed: int,
    search_seed: int,
    action_selection: str,
) -> dict[str, Any]:
    artifact = result["candidate_traces"]
    path = Path(str(artifact["path"]))
    if not path.is_absolute():
        path = resolve_path(path)
    totals: defaultdict[str, float] = defaultdict(float)
    correlations: list[float] = []
    identifiers: set[tuple[str, int]] = set()
    counts_by_size: defaultdict[int, int] = defaultdict(int)
    count = 0
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_finite_tree(row, label=f"{path}:{line_number}")
            if row.get("schema") != "vector-jepa-candidate-trace-v1":
                raise ValueError(f"unknown candidate-trace schema: {path}")
            if row.get("analysis_only_no_action_influence") is not True:
                raise ValueError("candidate truth labels were not marked post-hoc only")
            if row.get("diagnostic_rescore_excluded_from_planner_budget") is not True:
                raise ValueError(
                    "candidate diagnostic rescoring polluted planner compute"
                )
            identifier = (str(row["task_id"]), int(row["step"]))
            if identifier in identifiers:
                raise ValueError(f"duplicate candidate decision trace: {identifier}")
            identifiers.add(identifier)
            counts_by_size[int(row["maze_size"])] += 1
            metrics = row["metrics"]
            count += 1
            for name in (
                "first_action_coverage_at_k",
                "goal_reaching_coverage_at_k",
                "selection_accuracy",
                "selection_regret",
                "false_optimistic",
                "selected_invalid",
                "selected_short_cycle",
                "selected_no_progress",
                "generated_candidate_count",
                "unique_candidate_count",
                "effective_k",
                "selected_true_progress",
                "oracle_best_progress",
                "unique_route_ratio",
                "pairwise_normalized_edit_distance",
                "frequency_effective_sample_size",
                "entropy_effective_sample_size",
            ):
                totals[name] += float(metrics[name])
            for length, covered in metrics["prefix_coverage_at_k"].items():
                totals[f"prefix_coverage_at_{length}"] += float(covered)
            correlation = metrics.get("predicted_true_distance_spearman")
            if correlation is not None:
                correlations.append(float(correlation))
    if count != int(artifact.get("decision_record_count", -1)):
        raise ValueError("candidate-trace record count does not match its manifest")
    sampling = artifact.get("sampling", {})
    if (
        sampling.get("mode") != "exact_stratified_bottom_hash_replay"
        or float(sampling.get("target_fraction", -1.0)) != 0.1
        or artifact.get("replay_verified") is not True
        or artifact.get("replay_compute_excluded_from_planner_budget") is not True
    ):
        raise ValueError("candidate traces were not produced by the formal replay")
    expected_by_size = {
        int(size): int(record["selected_count"])
        for size, record in sampling.get("strata", {}).items()
    }
    if dict(counts_by_size) != expected_by_size:
        raise ValueError("candidate-trace size strata do not match the sample record")
    if count == 0:
        raise ValueError(
            "formal candidate-trace artifact contains no sampled decisions"
        )
    output = {
        "method": method,
        "backbone_seed": int(backbone_seed),
        "planner_seed": int(planner_seed),
        "search_seed": int(search_seed),
        "action_selection": action_selection,
        "sampled_decisions": count,
        **{name: value / count for name, value in sorted(totals.items())},
        "predicted_true_distance_spearman": (
            float(np.mean(correlations)) if correlations else None
        ),
        "correlation_decisions": len(correlations),
    }
    return output


def aggregate_mechanism_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["method"]), str(row["action_selection"]))].append(row)
    output: list[dict[str, Any]] = []
    excluded = {
        "method",
        "backbone_seed",
        "planner_seed",
        "search_seed",
        "action_selection",
        "sampled_decisions",
        "correlation_decisions",
        "predicted_true_distance_spearman",
    }
    for (method, action_selection), rows in sorted(grouped.items()):
        decision_count = sum(int(row["sampled_decisions"]) for row in rows)
        correlation_count = sum(int(row["correlation_decisions"]) for row in rows)
        metric_names = sorted(set(rows[0]) - excluded)
        aggregate: dict[str, Any] = {
            "method": method,
            "action_selection": action_selection,
            "run_count": len(rows),
            "sampled_decisions": decision_count,
        }
        for name in metric_names:
            aggregate[name] = (
                sum(float(row[name]) * int(row["sampled_decisions"]) for row in rows)
                / decision_count
            )
        aggregate["predicted_true_distance_spearman"] = (
            sum(
                float(row["predicted_true_distance_spearman"])
                * int(row["correlation_decisions"])
                for row in rows
                if row["predicted_true_distance_spearman"] is not None
            )
            / correlation_count
            if correlation_count
            else None
        )
        aggregate["correlation_decisions"] = correlation_count
        output.append(aggregate)
    return output


def mean_metric(seed_rows: list[list[dict[str, Any]]], metric: str) -> float:
    values = [float(row[metric]) for rows in seed_rows for row in rows]
    if not values:
        raise ValueError("cannot aggregate an empty result collection")
    return float(np.mean(values))


def primary_rows(
    records: dict[str, dict[str, list[list[dict[str, Any]]]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, by_action in sorted(records.items()):
        for action_selection, seed_rows in sorted(by_action.items()):
            output.append(
                {
                    "method": method,
                    "action_selection": action_selection,
                    "backbone_count": len(seed_rows),
                    "task_count_per_backbone": len(seed_rows[0]),
                    "sr": mean_metric(seed_rows, "success"),
                    "spl": mean_metric(seed_rows, "spl"),
                    "invalid_actions_per_episode": float(
                        np.mean(
                            [
                                float(row.get("invalid_actions", 0))
                                for rows in seed_rows
                                for row in rows
                            ]
                        )
                    ),
                    "loop_rate": mean_metric(seed_rows, "loop_or_cycle"),
                    "revisit_rate": mean_metric(seed_rows, "revisit_rate"),
                    "two_cycle_rate": mean_metric(seed_rows, "two_cycle_rate"),
                    "assistance_rate": mean_metric(seed_rows, "assistance_rate"),
                }
            )
    return output


def per_size_rows(
    records: dict[str, dict[str, list[list[dict[str, Any]]]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, by_action in sorted(records.items()):
        for action_selection, seed_rows in sorted(by_action.items()):
            sizes = sorted({int(row["maze_size"]) for row in seed_rows[0]})
            for size in sizes:
                tasks_per_backbone = sum(
                    int(row["maze_size"]) == size for row in seed_rows[0]
                )
                rows = [
                    row
                    for seed in seed_rows
                    for row in seed
                    if int(row["maze_size"]) == size
                ]
                output.append(
                    {
                        "method": method,
                        "action_selection": action_selection,
                        "maze_size": size,
                        "ood": size > 21,
                        "sr": float(np.mean([float(row["success"]) for row in rows])),
                        "spl": float(np.mean([float(row["spl"]) for row in rows])),
                        "backbone_count": len(seed_rows),
                        "unique_task_count": tasks_per_backbone,
                        "nested_row_count": len(rows),
                    }
                )
    return output


def structural_strata_rows(
    records: dict[str, dict[str, list[list[dict[str, Any]]]]],
) -> list[dict[str, Any]]:
    if not records:
        return []
    first_method = next(iter(sorted(records)))
    first_action = next(iter(sorted(records[first_method])))
    reference = records[first_method][first_action][0]
    structural_metrics = (
        "dead_end_density",
        "junction_count",
        "mean_corridor_length",
    )
    boundaries = {
        metric: np.quantile(
            np.asarray([float(row[metric]) for row in reference]),
            [0.25, 0.5, 0.75],
        )
        for metric in structural_metrics
    }
    output: list[dict[str, Any]] = []
    for method, by_action in sorted(records.items()):
        for action_selection, seed_rows in sorted(by_action.items()):
            rows = [row for seed in seed_rows for row in seed]
            categorical: list[tuple[str, str, list[dict[str, Any]]]] = []
            for label in ("le16", "17_32", "33_64", "65_128", "gt128"):
                categorical.append(
                    (
                        "shortest_path_bin",
                        label,
                        [row for row in rows if row["shortest_path_bin"] == label],
                    )
                )
            for metric in structural_metrics:
                cuts = boundaries[metric]
                for quartile in range(4):
                    selected = [
                        row
                        for row in rows
                        if int(
                            np.searchsorted(
                                cuts,
                                float(row[metric]),
                                side="right",
                            )
                        )
                        == quartile
                    ]
                    categorical.append((metric, f"q{quartile + 1}", selected))
            for stratum, level, selected in categorical:
                if not selected:
                    continue
                output.append(
                    {
                        "method": method,
                        "action_selection": action_selection,
                        "stratum": stratum,
                        "level": level,
                        "sr": float(
                            np.mean([float(row["success"]) for row in selected])
                        ),
                        "spl": float(np.mean([float(row["spl"]) for row in selected])),
                        "backbone_count": len(seed_rows),
                        "unique_task_count": len(
                            {str(row["task_id"]) for row in selected}
                        ),
                        "nested_row_count": len(selected),
                    }
                )
    return output


def paired_effects(
    raw: dict[
        str,
        dict[str, dict[int, dict[int, dict[int, list[dict[str, Any]]]]]],
    ],
    *,
    candidates: list[str],
    config: Any,
    methods: dict[str, Any],
    backbone_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    baseline = "b0_legacy_l2_cem"
    comparisons = [name for name in candidates if name != baseline]
    primary_contrast_count = max(2 * len(comparisons), 1)
    alpha = config.analysis.familywise_alpha / primary_contrast_count
    output: list[dict[str, Any]] = []
    primary_protocol = config.protocol.primary_action_selection
    baseline_cube = raw[baseline][primary_protocol]
    first_backbone = next(iter(baseline_cube.values()))
    first_planner = next(iter(first_backbone.values()))
    first_rows = next(iter(first_planner.values()))
    has_true_ood = any(int(row["maze_size"]) > 21 for row in first_rows)
    subsets = ("overall", "ood" if has_true_ood else "large_proxy_19_21")
    for index, candidate in enumerate(comparisons):
        for subset_index, subset in enumerate(subsets):
            effect = nested_paired_effect(
                raw,
                config=config,
                methods=methods,
                backbone_seeds=backbone_seeds,
                candidate=candidate,
                baseline=baseline,
                action_selection=primary_protocol,
                metric="success",
                subset=subset,
                alpha=alpha,
                seed=(config.analysis.bootstrap_seed + index * 100 + subset_index),
            )
            output.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": "success",
                    "subset": subset,
                    "action_selection": primary_protocol,
                    "familywise_alpha": config.analysis.familywise_alpha,
                    "comparison_alpha": alpha,
                    "minimum_effect_of_interest": 0.05,
                    **effect,
                }
            )
    return output


def paired_backbone_differences(
    candidate_rows: list[list[dict[str, Any]]],
    baseline_rows: list[list[dict[str, Any]]],
    *,
    metric: str,
) -> np.ndarray:
    if len(candidate_rows) != len(baseline_rows) or not candidate_rows:
        raise ValueError("paired seed differences require matched backbone lists")
    differences: list[float] = []
    for candidate, baseline in zip(candidate_rows, baseline_rows, strict=True):
        candidate_by_id = {str(row["task_id"]): row for row in candidate}
        baseline_by_id = {str(row["task_id"]): row for row in baseline}
        if set(candidate_by_id) != set(baseline_by_id):
            raise ValueError("paired contrast contains different task IDs")
        differences.append(
            float(
                np.mean(
                    [
                        float(candidate_by_id[key][metric])
                        - float(baseline_by_id[key][metric])
                        for key in sorted(candidate_by_id)
                    ]
                )
            )
        )
    return np.asarray(differences, dtype=np.float64)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Exact two-sided randomization test at the independent backbone level."""

    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size == 0 or values.size > 24 or not np.isfinite(values).all():
        raise ValueError("exact sign-flip test requires 1 to 24 finite backbones")
    observed = abs(float(values.mean()))
    total = 1 << values.size
    extreme = 0
    bit_positions = np.arange(values.size, dtype=np.uint64)
    for start in range(0, total, 8192):
        stop = min(start + 8192, total)
        masks = np.arange(start, stop, dtype=np.uint64).reshape(-1, 1)
        signs = 1.0 - 2.0 * ((masks >> bit_positions) & 1).astype(np.float64)
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
    return float(extreme / total)


def holm_adjust(rows: list[dict[str, Any]], *, alpha: float) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["p_raw"]))
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * float(rows[index]["p_raw"]))
        running = max(running, adjusted)
        rows[index]["p_holm"] = running
        rows[index]["holm_reject"] = bool(running <= alpha)


def confirmatory_primary_effects(
    records: dict[str, dict[str, list[list[dict[str, Any]]]]],
    raw: dict[
        str,
        dict[str, dict[int, dict[int, dict[int, list[dict[str, Any]]]]]],
    ],
    config: Any,
    methods: dict[str, Any],
    backbone_seeds: tuple[int, ...],
    primary_family: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(primary_family) not in (2, 4):
        raise ValueError("confirmatory primary family must contain two or four rows")
    required = {
        str(row[key]) for row in primary_family for key in ("candidate", "baseline")
    }
    if not required <= set(records):
        raise ValueError("confirmatory primary family is missing a locked method")
    output: list[dict[str, Any]] = []
    action_selection = config.protocol.primary_action_selection
    family_size = len(primary_family)
    for contrast_index, contrast in enumerate(primary_family):
        hypothesis = str(contrast["hypothesis"])
        candidate = str(contrast["candidate"])
        baseline = str(contrast["baseline"])
        subset = str(contrast["subset"])
        if subset not in ("overall", "ood"):
            raise ValueError("confirmatory subset must be overall or ood")
        candidate_rows = records[candidate][action_selection]
        baseline_rows = records[baseline][action_selection]
        if subset == "ood":
            candidate_rows = [
                [row for row in seed if int(row["maze_size"]) > 21]
                for seed in candidate_rows
            ]
            baseline_rows = [
                [row for row in seed if int(row["maze_size"]) > 21]
                for seed in baseline_rows
            ]
        effect = nested_paired_effect(
            raw,
            config=config,
            methods=methods,
            backbone_seeds=backbone_seeds,
            candidate=candidate,
            baseline=baseline,
            action_selection=action_selection,
            metric="success",
            subset=subset,
            alpha=config.analysis.familywise_alpha / family_size,
            seed=config.analysis.bootstrap_seed + 120_000 + contrast_index,
        )
        seed_differences = paired_backbone_differences(
            candidate_rows, baseline_rows, metric="success"
        )
        output.append(
            {
                "hypothesis": hypothesis,
                "candidate": candidate,
                "baseline": baseline,
                "metric": "success",
                "subset": subset,
                "action_selection": action_selection,
                "family_size": family_size,
                "familywise_alpha": config.analysis.familywise_alpha,
                "comparison_alpha": config.analysis.familywise_alpha / family_size,
                "minimum_effect_of_interest": 0.05,
                "p_test": "exact_two_sided_backbone_sign_flip",
                "p_raw": exact_sign_flip_pvalue(seed_differences),
                "backbone_differences": seed_differences.tolist(),
                **effect,
            }
        )
    holm_adjust(output, alpha=config.analysis.familywise_alpha)
    return output


def paired_outcome_tables(
    raw: dict[
        str,
        dict[str, dict[int, dict[int, dict[int, list[dict[str, Any]]]]]],
    ],
    *,
    config: Any,
    methods: dict[str, Any],
    backbone_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, by_action in sorted(raw.items()):
        if set(by_action) != {"corrected_v1", "unmasked"}:
            continue
        counts: defaultdict[str, int] = defaultdict(int)
        planner_seeds = planner_seed_values(config, methods[method])
        task_count: int | None = None
        for backbone_seed in backbone_seeds:
            for planner_seed in planner_seeds:
                for search_seed in config.protocol.search_seeds:
                    unmasked = by_action["unmasked"][backbone_seed][planner_seed][
                        search_seed
                    ]
                    assisted = by_action["corrected_v1"][backbone_seed][planner_seed][
                        search_seed
                    ]
                    unmasked_by_id = {str(row["task_id"]): row for row in unmasked}
                    assisted_by_id = {str(row["task_id"]): row for row in assisted}
                    if set(unmasked_by_id) != set(assisted_by_id):
                        raise ValueError(
                            "assistance pairing requires identical task IDs"
                        )
                    task_count = task_count or len(unmasked_by_id)
                    if len(unmasked_by_id) != task_count:
                        raise ValueError(
                            "assistance runs contain different task counts"
                        )
                    for identifier in unmasked_by_id:
                        left_value = unmasked_by_id[identifier]["success"]
                        right_value = assisted_by_id[identifier]["success"]
                        if not isinstance(left_value, bool) or not isinstance(
                            right_value, bool
                        ):
                            raise ValueError(
                                "paired outcome table requires raw boolean success"
                            )
                        left = "success" if left_value else "fail"
                        right = "success" if right_value else "fail"
                        counts[f"unmasked_{left}__assisted_{right}"] += 1
        total = sum(counts.values())
        if total == 0 or task_count is None:
            raise ValueError("paired outcome table contains no episodes")
        output.append(
            {
                "method": method,
                "backbone_count": len(backbone_seeds),
                "planner_seed_count": len(planner_seeds),
                "search_seed_count": len(config.protocol.search_seeds),
                "task_count_per_run": task_count,
                "paired_episode_count": total,
                **dict(sorted(counts.items())),
                "rescue_rate": counts["unmasked_fail__assisted_success"] / total,
                "harm_rate": counts["unmasked_success__assisted_fail"] / total,
            }
        )
    return output


def assistance_effects(
    raw: dict[
        str,
        dict[str, dict[int, dict[int, dict[int, list[dict[str, Any]]]]]],
    ],
    config: Any,
    *,
    methods: dict[str, Any],
    backbone_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, (method, by_action) in enumerate(sorted(raw.items())):
        if set(by_action) != {"corrected_v1", "unmasked"}:
            continue
        for metric in ("success", "spl"):
            effect = nested_paired_effect(
                raw,
                config=config,
                methods=methods,
                backbone_seeds=backbone_seeds,
                candidate=method,
                baseline=method,
                action_selection="corrected_v1",
                baseline_action_selection="unmasked",
                metric=metric,
                subset="overall",
                alpha=0.05,
                seed=config.analysis.bootstrap_seed + 50_000 + index * 100,
            )
            output.append(
                {
                    "method": method,
                    "contrast": "corrected_v1_minus_unmasked",
                    "metric": metric,
                    **effect,
                }
            )
    return output


def factorial_effects(
    raw: dict[
        str,
        dict[str, dict[int, dict[int, dict[int, list[dict[str, Any]]]]]],
    ],
    config: Any,
    *,
    methods: dict[str, Any],
    backbone_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    names = sorted(name for name in raw if name.startswith("p3_factorial_"))
    if len(names) != 16:
        return []
    output: list[dict[str, Any]] = []
    factors = {"verifier": "v", "reachability": "r", "proposal": "p", "memory": "m"}
    action_selection = config.protocol.primary_action_selection
    contrasts: list[tuple[str, dict[str, float], str]] = []
    for label, code in factors.items():
        contrasts.append(
            (
                label,
                {
                    name: (1.0 / 8.0 if f"{code}1" in name else -1.0 / 8.0)
                    for name in names
                },
                "main_effect",
            )
        )
    factor_items = list(factors.items())
    for left_index, (left_label, left_code) in enumerate(factor_items):
        for right_label, right_code in factor_items[left_index + 1 :]:
            weights: dict[str, float] = {}
            for name in names:
                code = name.rsplit("_", 1)[-1]
                left = f"{left_code}1" in code
                right = f"{right_code}1" in code
                weights[name] = 0.25 if left == right else -0.25
            contrasts.append(
                (
                    f"{left_label}_x_{right_label}",
                    weights,
                    "second_order_interaction",
                )
            )
    for contrast_index, (label, weights, contrast_type) in enumerate(contrasts):
        for metric in ("success", "spl"):
            contrast_cube: np.ndarray | None = None
            canonical_planners: tuple[int, ...] | None = None
            canonical_ids: tuple[str, ...] | None = None
            canonical_strata: tuple[str, ...] | None = None
            for name in names:
                cube, planners, identifiers, strata = nested_metric_cube(
                    raw,
                    config=config,
                    method=methods[name],
                    backbone_seeds=backbone_seeds,
                    action_selection=action_selection,
                    metric=metric,
                    subset="overall",
                )
                if canonical_planners is None:
                    canonical_planners = planners
                    canonical_ids = identifiers
                    canonical_strata = strata
                    contrast_cube = np.zeros_like(cube)
                elif (
                    planners != canonical_planners
                    or identifiers != canonical_ids
                    or strata != canonical_strata
                ):
                    raise ValueError("factorial cells are not nested-pair compatible")
                assert contrast_cube is not None
                contrast_cube += weights[name] * cube
            assert contrast_cube is not None and canonical_strata is not None
            effect = nested_paired_bootstrap(
                contrast_cube,
                np.zeros_like(contrast_cube),
                samples=config.analysis.bootstrap_samples,
                alpha=config.analysis.familywise_alpha / len(contrasts),
                seed=(
                    config.analysis.bootstrap_seed
                    + 80_000
                    + contrast_index * 100
                    + (1 if metric == "spl" else 0)
                ),
                strata=canonical_strata,
                pair_planner_seeds=True,
            )
            output.append(
                {
                    "factor": label,
                    "contrast_type": contrast_type,
                    "metric": metric,
                    "action_selection": action_selection,
                    "interaction_scale": (
                        "difference_in_differences"
                        if contrast_type == "second_order_interaction"
                        else "high_minus_low"
                    ),
                    **effect,
                }
            )
    return output


def csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = sorted({key for row in rows for key in row})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def markdown_report(
    split_role: str,
    primary: list[dict[str, Any]],
    effects: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Vector-JEPA Planner Frontier: {split_role}",
        "",
        "> Generated from hash-validated task-level records. "
        "Oracle diagnostics are excluded.",
        "",
        "## Primary results",
        "",
        "| Method | Protocol | SR | SPL | Loop rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['method']} | {row['action_selection']} | "
            f"{row['sr']:.4f} | {row['spl']:.4f} | {row['loop_rate']:.4f} |"
        )
    lines.extend(["", "## Paired effects", ""])
    for row in effects:
        lines.append(
            f"- {row['candidate']} minus {row['baseline']} ({row['metric']}): "
            f"{row['delta']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    if lock.get("status") != "locked":
        raise RuntimeError("summary requires a completed protocol lock")
    analysis_hash = analysis_spec_sha256(config, lock)
    expected_count = int(lock[f"{args.split_role}_manifest"]["count"])
    expected_manifest_sha256 = lock[f"{args.split_role}_manifest"]["sha256"]
    summary_backbones = tuple(config.protocol.training_seeds)
    confirmation_marker: dict[str, Any] | None = None
    if args.split_role == "confirmatory":
        confirmation_path = resolve_path(config.paths.confirmation_lock)
        marker_path = resolve_path(config.paths.confirmation_unblinded)
        if not marker_path.is_file():
            raise RuntimeError(
                "confirmatory results cannot be summarized before complete unblinding"
            )
        confirmation = load_json(confirmation_path)
        confirmation_marker = load_json(marker_path)
        if confirmation_marker.get("analysis_spec_sha256") != analysis_hash:
            raise ValueError("unblinding marker belongs to another analysis")
        if confirmation_marker.get("confirmation_lock_sha256") != sha256_file(
            confirmation_path
        ):
            raise ValueError("unblinding marker references another confirmation lock")
        if confirmation_marker.get("mapping_sha256") != confirmation.get(
            "mapping_sha256"
        ):
            raise ValueError("unblinding mapping hash mismatch")
        if int(confirmation_marker.get("run_count", -1)) != int(
            confirmation.get("run_count", -2)
        ):
            raise ValueError("unblinding marker is incomplete")
        published = confirmation_marker.get("results", [])
        if len(published) != int(confirmation.get("run_count", -1)):
            raise ValueError("unblinding marker omits named result records")
        for record in published:
            formal = Path(str(record.get("formal_result", "")))
            if not formal.is_file() or sha256_file(formal) != record.get(
                "formal_result_sha256"
            ):
                raise ValueError(f"named confirmatory result hash mismatch: {formal}")
        summary_backbones = tuple(int(seed) for seed in confirmation["backbone_seeds"])
    if args.methods and args.stage:
        raise ValueError("choose either --methods or --stage, not both")
    if args.methods:
        names = [name.strip() for name in args.methods.split(",") if name.strip()]
    elif args.stage:
        names = [method.name for method in config.methods if method.stage == args.stage]
    elif args.split_role == "confirmatory":
        names = list(confirmation["methods"])
    else:
        names = [method.name for method in config.methods]
    method_map = {method.name: method for method in config.methods}
    resolved_method_map: dict[str, Any] = {}
    raw_records: dict[
        str,
        dict[str, dict[int, dict[int, dict[int, list[dict[str, Any]]]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    mechanism_seed_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in names:
        method = resolve_effective_method(config, lock, method_map[name])
        resolved_method_map[name] = method
        for action_selection in ("unmasked", "corrected_v1"):
            for backbone_seed in summary_backbones:
                for planner_seed in planner_seed_values(config, method):
                    for search_seed in config.protocol.search_seeds:
                        path = result_path(
                            config,
                            method=name,
                            backbone_seed=backbone_seed,
                            planner_seed=planner_seed,
                            search_seed=search_seed,
                            split_role=args.split_role,
                            action_selection=action_selection,
                        )
                        if not path.exists():
                            missing.append(str(path))
                            continue
                        result = load_result(
                            path,
                            analysis_hash=analysis_hash,
                            method=name,
                            backbone_seed=backbone_seed,
                            planner_seed=planner_seed,
                            search_seed=search_seed,
                            split_role=args.split_role,
                            action_selection=action_selection,
                            expected_count=expected_count,
                            expected_manifest_sha256=expected_manifest_sha256,
                            expected_code_fingerprint=lock["code_fingerprint"],
                            expected_method_sha256=effective_method_sha256(method),
                        )
                        raw_records[name][action_selection][backbone_seed][
                            planner_seed
                        ][search_seed] = result["tasks"]
                        mechanism_seed_rows.append(
                            candidate_mechanism_summary(
                                result,
                                method=name,
                                backbone_seed=int(backbone_seed),
                                planner_seed=int(planner_seed),
                                search_seed=int(search_seed),
                                action_selection=action_selection,
                            )
                        )
    if missing and not args.diagnostic_allow_missing:
        raise FileNotFoundError(
            f"formal summary is missing {len(missing)} result files"
        )
    nested_complete_names: list[str] = []
    for name in names:
        method = resolved_method_map[name]
        planners = set(planner_seed_values(config, method))
        by_action = raw_records.get(name, {})
        if all(
            set(by_action.get(action_selection, {})) == set(summary_backbones)
            and all(
                set(by_action[action_selection][backbone_seed]) == planners
                and all(
                    set(by_action[action_selection][backbone_seed][planner_seed])
                    == set(config.protocol.search_seeds)
                    for planner_seed in planners
                )
                for backbone_seed in summary_backbones
            )
            for action_selection in ("unmasked", "corrected_v1")
        ):
            nested_complete_names.append(name)
    raw_for_collapse = {name: raw_records[name] for name in nested_complete_names}
    records = collapse_nested_records(
        raw_for_collapse,
        config=config,
        methods=resolved_method_map,
        backbone_seeds=summary_backbones,
    )
    complete = {
        name: by_action
        for name, by_action in records.items()
        if all(
            len(by_action.get(action, [])) == len(summary_backbones)
            for action in ("unmasked", "corrected_v1")
        )
    }
    complete_raw = {name: raw_records[name] for name in complete}
    complete_methods = {name: resolved_method_map[name] for name in complete}
    primary = primary_rows(complete)
    per_size = per_size_rows(complete)
    structural_strata = structural_strata_rows(complete)
    candidates = [name for name in names if name in complete]
    if args.split_role == "confirmatory":
        effects = confirmatory_primary_effects(
            complete,
            complete_raw,
            config,
            complete_methods,
            summary_backbones,
            list(confirmation["primary_family"]),
        )
    else:
        effects = (
            paired_effects(
                complete_raw,
                candidates=candidates,
                config=config,
                methods=complete_methods,
                backbone_seeds=summary_backbones,
            )
            if "b0_legacy_l2_cem" in complete
            else []
        )
    assistance = assistance_effects(
        complete_raw,
        config,
        methods=complete_methods,
        backbone_seeds=summary_backbones,
    )
    paired_outcomes = paired_outcome_tables(
        complete_raw,
        config=config,
        methods=complete_methods,
        backbone_seeds=summary_backbones,
    )
    factorial = factorial_effects(
        complete_raw,
        config,
        methods=complete_methods,
        backbone_seeds=summary_backbones,
    )
    mechanism = aggregate_mechanism_rows(
        [row for row in mechanism_seed_rows if row["method"] in complete]
    )
    variance = nested_variance_rows({name: raw_records[name] for name in complete})
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"summary output directory is immutable: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": analysis_hash,
        "split_role": args.split_role,
        "missing_files": missing,
        "primary": primary,
        "paired_effects": effects,
        "assistance_effects": assistance,
        "paired_outcome_tables": paired_outcomes,
        "factorial_effects": factorial,
        "candidate_mechanisms": mechanism,
        "nested_seed_variance": variance,
        "per_size": per_size,
        "structural_strata": structural_strata,
        "confirmation_unblinding": confirmation_marker,
    }
    atomic_json_dump(output_dir / "summary.json", payload)
    atomic_text_dump(output_dir / "primary_results.csv", csv_text(primary))
    atomic_text_dump(output_dir / "paired_effects.csv", csv_text(effects))
    atomic_text_dump(output_dir / "assistance_effects.csv", csv_text(assistance))
    atomic_text_dump(
        output_dir / "paired_outcome_tables.csv", csv_text(paired_outcomes)
    )
    atomic_text_dump(output_dir / "factorial_effects.csv", csv_text(factorial))
    atomic_text_dump(output_dir / "mechanism_results.csv", csv_text(mechanism))
    atomic_text_dump(output_dir / "nested_seed_variance.csv", csv_text(variance))
    atomic_text_dump(output_dir / "per_size.csv", csv_text(per_size))
    atomic_text_dump(output_dir / "structural_strata.csv", csv_text(structural_strata))
    atomic_text_dump(
        output_dir / "REPORT.md",
        markdown_report(args.split_role, primary, effects),
    )


if __name__ == "__main__":
    main()
