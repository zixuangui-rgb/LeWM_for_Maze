"""Produce the final nested-seed full-900 report and permanent closure artifact."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import method_by_name
from vector_jepa_planner_full900_screen.analysis import load_result
from vector_jepa_planner_full900_screen.common import (
    atomic_json_dump,
    atomic_text_dump,
    load_config,
    load_json,
    resolve_path,
    result_path,
    validate_lock,
)
from vector_jepa_planner_full900_screen.methods import (
    component_parity_audits,
    direct_control_name,
    effective_method,
    validate_final_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="vector_jepa_planner_full900_screen/configs/default.json",
    )
    return parser.parse_args()


def _planner_seed_values(config: Any, method: Any) -> tuple[int, ...]:
    return (
        config.replication.final_planner_seeds
        if method.component_checkpoint_required
        else (0,)
    )


def _task_metric(row: dict[str, Any], name: str) -> float:
    if name == "success":
        return float(row["success"])
    if name == "spl":
        return float(row["spl"])
    if name == "loop_or_cycle":
        return float(row["loop_or_cycle"])
    if name == "invalid_actions":
        return float(row["invalid_actions"])
    if name == "path_length":
        return float(row["path_length"])
    if name == "assistance_rate":
        return float(row["assistance_rate"])
    raise ValueError(f"unknown task metric: {name}")


def _nested_tasks(
    config: Any,
    lock: dict[str, Any],
    method: Any,
    action: str,
) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for backbone_seed in config.replication.final_backbone_seeds:
        runs = [
            load_result(
                config,
                lock,
                method=method,
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
                action_selection=action,
            )
            for planner_seed in _planner_seed_values(config, method)
        ]
        task_ids = [[row["task_id"] for row in run["tasks"]] for run in runs]
        if any(ids != task_ids[0] for ids in task_ids[1:]):
            raise ValueError("planner seeds do not share identical task order")
        averaged = []
        for rows in zip(*(run["tasks"] for run in runs), strict=True):
            base = rows[0]
            auxiliary_names = sorted(
                set().union(*(set(row.get("auxiliary", {})) for row in rows))
            )
            averaged.append(
                {
                    "task_id": base["task_id"],
                    "maze_size": int(base["maze_size"]),
                    "success": float(
                        np.mean([_task_metric(row, "success") for row in rows])
                    ),
                    "spl": float(np.mean([_task_metric(row, "spl") for row in rows])),
                    "loop_or_cycle": float(
                        np.mean([_task_metric(row, "loop_or_cycle") for row in rows])
                    ),
                    "invalid_actions": float(
                        np.mean([_task_metric(row, "invalid_actions") for row in rows])
                    ),
                    "path_length": float(
                        np.mean([_task_metric(row, "path_length") for row in rows])
                    ),
                    "decision_count": float(
                        np.mean([float(row["decision_count"]) for row in rows])
                    ),
                    "assistance_rate": float(
                        np.mean([_task_metric(row, "assistance_rate") for row in rows])
                    ),
                    "auxiliary": {
                        name: float(
                            np.mean(
                                [
                                    float(row.get("auxiliary", {}).get(name, 0.0))
                                    for row in rows
                                ]
                            )
                        )
                        for name in auxiliary_names
                    },
                    "episode_seconds": float(
                        np.mean([float(row["episode_seconds"]) for row in rows])
                    ),
                }
            )
        output[int(backbone_seed)] = averaged
    return output


def _split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "overall":
        return rows
    if split == "seen":
        return [row for row in rows if int(row["maze_size"]) <= 21]
    if split == "ood":
        return [row for row in rows if int(row["maze_size"]) > 21]
    raise ValueError(f"unknown split: {split}")


def _aggregate(tasks: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    compute_names = (
        "plan_transitions",
        "assist_transitions",
        "total_transitions",
        "planner_forward_calls",
        "assist_forward_calls",
        "node_expansions",
        "candidate_sequences",
        "duplicate_candidates",
        "verifier_forward_calls",
        "reachability_forward_calls",
        "ranker_forward_calls",
        "proposal_forward_calls",
        "join_forward_calls",
        "dts_forward_calls",
    )

    def summarize_groups(groups: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
        per_backbone = []
        for seed, rows in groups.items():
            path_sum = max(sum(float(row["path_length"]) for row in rows), 1.0)
            decision_sum = max(sum(float(row["decision_count"]) for row in rows), 1.0)
            summary = {
                "backbone_seed": seed,
                "sr": float(np.mean([row["success"] for row in rows])),
                "spl": float(np.mean([row["spl"] for row in rows])),
                "loop_or_cycle_rate": float(
                    np.mean([row["loop_or_cycle"] for row in rows])
                ),
                "invalid_rate": float(
                    sum(float(row["invalid_actions"]) for row in rows) / path_sum
                ),
                "assistance_rate": float(
                    np.mean([row["assistance_rate"] for row in rows])
                ),
                "episode_seconds_per_task": float(
                    np.mean([row["episode_seconds"] for row in rows])
                ),
            }
            summary.update(
                {
                    f"{name}_per_decision": float(
                        sum(float(row["auxiliary"].get(name, 0.0)) for row in rows)
                        / decision_sum
                    )
                    for name in compute_names
                }
            )
            per_backbone.append(summary)
        metric_names = (
            "sr",
            "spl",
            "loop_or_cycle_rate",
            "invalid_rate",
            "assistance_rate",
            "episode_seconds_per_task",
            *(f"{name}_per_decision" for name in compute_names),
        )
        output = {
            name: {
                "mean": float(np.mean([row[name] for row in per_backbone])),
                "sd": float(
                    np.std(
                        [row[name] for row in per_backbone],
                        ddof=1 if len(per_backbone) > 1 else 0,
                    )
                ),
            }
            for name in metric_names
        }
        output["per_backbone"] = per_backbone
        output["task_count_per_backbone"] = len(next(iter(groups.values())))
        return output

    output = {
        split: summarize_groups(
            {seed: _split(rows, split) for seed, rows in tasks.items()}
        )
        for split in ("overall", "seen", "ood")
    }
    output["by_size"] = {
        str(size): summarize_groups(
            {
                seed: [row for row in rows if int(row["maze_size"]) == size]
                for seed, rows in tasks.items()
            }
        )
        for size in sorted(
            {int(row["maze_size"]) for rows in tasks.values() for row in rows}
        )
    }
    return output


def _paired_arrays(
    candidate: dict[int, list[dict[str, Any]]],
    control: dict[int, list[dict[str, Any]]],
    split: str,
) -> tuple[list[int], dict[int, np.ndarray], dict[int, np.ndarray]]:
    seeds = sorted(candidate)
    if seeds != sorted(control):
        raise ValueError("paired methods do not share backbone seeds")
    deltas: dict[int, np.ndarray] = {}
    sizes: dict[int, np.ndarray] = {}
    reference_task_ids: list[str] | None = None
    reference_sizes: np.ndarray | None = None
    for seed in seeds:
        left = _split(candidate[seed], split)
        right = _split(control[seed], split)
        task_ids = [str(row["task_id"]) for row in left]
        if task_ids != [str(row["task_id"]) for row in right]:
            raise ValueError("paired methods do not share task order")
        labels = np.asarray([int(row["maze_size"]) for row in left])
        if reference_task_ids is None:
            reference_task_ids = task_ids
            reference_sizes = labels
        elif task_ids != reference_task_ids or not np.array_equal(
            labels, reference_sizes
        ):
            raise ValueError("backbones do not share the same crossed task panel")
        deltas[seed] = np.asarray(
            [a["success"] - b["success"] for a, b in zip(left, right, strict=True)],
            dtype=np.float64,
        )
        sizes[seed] = labels
    return seeds, deltas, sizes


def _nested_bootstrap(
    candidate: dict[int, list[dict[str, Any]]],
    control: dict[int, list[dict[str, Any]]],
    *,
    split: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    seeds, deltas, sizes = _paired_arrays(candidate, control, split)
    delta_matrix = np.stack([deltas[item] for item in seeds], axis=0)
    labels = sizes[seeds[0]]
    observed = float(delta_matrix.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    chunk_size = min(200, samples)
    backbone_count = len(seeds)
    strata = [np.flatnonzero(labels == size) for size in np.unique(labels)]
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        chunk = stop - start
        sampled_backbones = rng.integers(
            0, backbone_count, size=(chunk, backbone_count)
        )
        numerator = np.zeros(chunk, dtype=np.float64)
        denominator = 0
        for task_indices in strata:
            sampled_slots = rng.integers(
                0,
                len(task_indices),
                size=(chunk, len(task_indices)),
            )
            sampled_tasks = task_indices[sampled_slots]
            sampled_values = delta_matrix[
                sampled_backbones[:, :, None],
                sampled_tasks[:, None, :],
            ]
            numerator += sampled_values.sum(axis=(1, 2))
            denominator += backbone_count * len(task_indices)
        draws[start:stop] = numerator / denominator
    per_backbone = {str(item): float(deltas[item].mean()) for item in seeds}
    return {
        "delta": observed,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "per_backbone_delta": per_backbone,
        "positive_backbones": sum(value > 0.0 for value in per_backbone.values()),
        "backbone_count": len(seeds),
        "planner_seeds_averaged_within_backbone": True,
        "resampling_design": "crossed_backbone_by_task_stratified",
        "task_resampling_stratified_by_maze_size": True,
        "task_resampling_shared_across_backbones": True,
        "confidence_level": 0.95,
        "post_selection_descriptive_interval": True,
    }


def _markdown(payload: dict[str, Any]) -> str:
    def metric_cell(metric: dict[str, float], digits: int = 4) -> str:
        return f"{metric['mean']:.{digits}f} ({metric['sd']:.{digits}f})"

    lines = [
        "# Vector-JEPA Planner Full-900 Screen Results",
        "",
        (
            "> This is an exploratory paired development-set comparison, "
            "not a fresh confirmatory test."
        ),
        "",
        f"Final winner: `{payload['winner']}`",
        f"Direct control: `{payload['direct_control']}`",
        "",
        (
            "| Protocol | Method | SR mean (SD) | Seen SR | OOD SR | SPL | "
            "Loop | Invalid | Assist | Plan trans/decision |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for action, methods in payload["results"].items():
        for name, result in methods.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        action,
                        name,
                        metric_cell(result["overall"]["sr"]),
                        metric_cell(result["seen"]["sr"]),
                        metric_cell(result["ood"]["sr"]),
                        metric_cell(result["overall"]["spl"]),
                        metric_cell(result["overall"]["loop_or_cycle_rate"]),
                        metric_cell(result["overall"]["invalid_rate"]),
                        metric_cell(result["overall"]["assistance_rate"]),
                        metric_cell(
                            result["overall"]["plan_transitions_per_decision"], 2
                        ),
                    ]
                )
                + " |"
            )
    sizes = [str(size) for size in range(9, 26, 2)]
    lines.extend(
        [
            "",
            "## SR By Maze Size",
            "",
            "| Protocol | Method | " + " | ".join(sizes) + " |",
            "|---|---|" + "---:|" * len(sizes),
        ]
    )
    for action, methods in payload["results"].items():
        for name, result in methods.items():
            cells = [metric_cell(result["by_size"][size]["sr"]) for size in sizes]
            lines.append(f"| {action} | {name} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Compute Per Decision",
            "",
            (
                "| Protocol | Method | Plan | Assist | Planner calls | Nodes | "
                "Proposal | Verifier | Reach | Join | DTS | Ranker |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    compute_columns = (
        "plan_transitions_per_decision",
        "assist_transitions_per_decision",
        "planner_forward_calls_per_decision",
        "node_expansions_per_decision",
        "proposal_forward_calls_per_decision",
        "verifier_forward_calls_per_decision",
        "reachability_forward_calls_per_decision",
        "join_forward_calls_per_decision",
        "dts_forward_calls_per_decision",
        "ranker_forward_calls_per_decision",
    )
    for action, methods in payload["results"].items():
        for name, result in methods.items():
            cells = [metric_cell(result["overall"][key], 2) for key in compute_columns]
            lines.append(f"| {action} | {name} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Paired Effects", ""])
    for action, comparisons in payload["paired_effects"].items():
        for comparison, splits in comparisons.items():
            overall = splits["overall"]
            lines.append(
                f"- `{action}` {comparison}: delta SR "
                f"{overall['delta']:.4f} "
                f"[{overall['ci_low']:.4f}, {overall['ci_high']:.4f}]"
            )
    lines.extend(
        [
            "",
            (
                "Intervals are pointwise 95% post-selection descriptive crossed "
                "bootstraps. Planner-head seeds are averaged within each backbone "
                "before all comparisons."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_lock(config, lock)
    final_record = validate_final_selection(config, lock)
    winner_name = final_record.get("winner")
    output_json = resolve_path(config.paths.run_root) / "summary.json"
    output_md = resolve_path(config.paths.run_root) / "REPORT.md"
    closure_path = resolve_path(config.paths.p8_selection)
    if closure_path.exists():
        raise FileExistsError("the completed final summary is immutable")
    if winner_name is None:
        payload = {
            "schema": "vector-jepa-full900-screen-summary-v1",
            "status": "closed_without_winner",
            "protocol_id": config.protocol_id,
            "quick_spec_sha256": lock["quick_spec_sha256"],
            "claim_status": "exploratory_development_not_fresh_confirmation",
            "winner": None,
            "direct_control": None,
            "results": {},
            "paired_effects": {},
            "component_parity_audits": [],
            "selection_records": {
                "q1_parent_sha256": sha256_file(
                    resolve_path(config.paths.p2_selection)
                ),
                "shortlist_sha256": sha256_file(
                    resolve_path(config.paths.p5_advancement)
                ),
                "final_winner_sha256": sha256_file(
                    resolve_path(config.paths.p7_selection)
                ),
            },
        }
    else:
        winner = effective_method(config, lock, str(winner_name))
        direct = effective_method(
            config,
            lock,
            direct_control_name(config, lock, winner.name),
        )
        baseline = method_by_name(config, "b0_legacy_l2_cem")
        methods = list(
            {method.name: method for method in (baseline, direct, winner)}.values()
        )
        component_parity = component_parity_audits(
            config,
            candidates=(winner.name,),
            backbone_seeds=config.replication.final_backbone_seeds,
            planner_seeds=_planner_seed_values(config, winner),
        )
        results: dict[str, Any] = defaultdict(dict)
        task_cache: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = {}
        input_result_sha256s: dict[str, str] = {}
        for action in config.replication.action_selections:
            for method in methods:
                tasks = _nested_tasks(config, lock, method, action)
                task_cache[(method.name, action)] = tasks
                results[action][method.name] = _aggregate(tasks)
                for backbone_seed in config.replication.final_backbone_seeds:
                    for planner_seed in _planner_seed_values(config, method):
                        path = result_path(
                            config,
                            method=method.name,
                            backbone_seed=backbone_seed,
                            planner_seed=planner_seed,
                            action_selection=action,
                        )
                        key = f"{method.name}:b{backbone_seed}:p{planner_seed}:{action}"
                        input_result_sha256s[key] = sha256_file(path)
        comparisons = {
            "winner_minus_b0": baseline.name,
            "winner_minus_direct_control": direct.name,
        }
        paired: dict[str, Any] = defaultdict(dict)
        for action_index, action in enumerate(config.replication.action_selections):
            winner_tasks = task_cache[(winner.name, action)]
            for comparison_index, (label, control_name) in enumerate(
                comparisons.items()
            ):
                control_tasks = task_cache[(control_name, action)]
                paired[action][label] = {
                    split: _nested_bootstrap(
                        winner_tasks,
                        control_tasks,
                        split=split,
                        samples=config.analysis.bootstrap_samples,
                        seed=(
                            config.analysis.bootstrap_seed
                            + action_index * 10
                            + comparison_index
                        ),
                    )
                    for split in ("overall", "seen", "ood")
                }
        payload = {
            "schema": "vector-jepa-full900-screen-summary-v1",
            "status": "complete",
            "protocol_id": config.protocol_id,
            "quick_spec_sha256": lock["quick_spec_sha256"],
            "claim_status": "exploratory_development_not_fresh_confirmation",
            "winner": winner.name,
            "direct_control": direct.name,
            "backbone_seeds": list(config.replication.final_backbone_seeds),
            "planner_seeds": (
                list(config.replication.final_planner_seeds)
                if winner.component_checkpoint_required
                else []
            ),
            "results": dict(results),
            "paired_effects": dict(paired),
            "component_parity_audits": component_parity,
            "input_result_sha256s": dict(sorted(input_result_sha256s.items())),
            "selection_records": {
                "q1_parent_sha256": sha256_file(
                    resolve_path(config.paths.p2_selection)
                ),
                "shortlist_sha256": sha256_file(
                    resolve_path(config.paths.p5_advancement)
                ),
                "final_winner_sha256": sha256_file(
                    resolve_path(config.paths.p7_selection)
                ),
            },
        }
    report = _markdown(payload)
    if output_json.exists():
        if load_json(output_json) != payload:
            raise ValueError("partial summary JSON does not match recomputation")
    else:
        atomic_json_dump(output_json, payload)
    if output_md.exists():
        if output_md.read_text(encoding="utf-8") != report:
            raise ValueError("partial summary Markdown does not match recomputation")
    else:
        atomic_text_dump(output_md, report)
    closure = {
        "schema": "vector-jepa-full900-screen-closure-v1",
        "status": "permanently_closed",
        "protocol_id": config.protocol_id,
        "quick_spec_sha256": lock["quick_spec_sha256"],
        "summary_sha256": sha256_file(output_json),
        "report_sha256": sha256_file(output_md),
        "score_triggered_reruns_allowed": False,
        "fresh_confirmatory_claim": False,
    }
    atomic_json_dump(closure_path, closure)
    print(f"summary written: {output_json}")


if __name__ == "__main__":
    main()
