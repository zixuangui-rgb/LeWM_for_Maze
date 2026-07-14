"""Pure P8 frontier rules and immutable input-artifact fingerprints."""

from __future__ import annotations

from typing import Any

import numpy as np

from final_closure.common import sha256_file
from spatial_jepa_planning.common import canonical_json_sha256
from vector_jepa_planner_frontier.common import (
    component_checkpoint_path,
    method_by_name,
    planner_seed_values,
    resolve_path,
)
from vector_jepa_planner_frontier.effective_methods import (
    effective_method_sha256,
    resolve_effective_method,
)
from vector_jepa_planner_frontier.validation_results import (
    load_validation_seed_rows,
)

FRONTIER_BASES = (
    "p5_track_f_all_hard_memory",
    "p6_track_f_counterexample_ranked",
)
FRONTIER_BUDGETS = (0.5, 1.0, 4.0, 16.0)
NEAR_OPTIMAL_SR_TOLERANCE = 0.01


def frontier_families(
    config: Any,
    *,
    selected_track_j: str | None,
) -> dict[str, dict[float, Any]]:
    """Return each finalist's exact four-point budget family."""

    by_name = {method.name: method for method in config.methods}
    output: dict[str, dict[float, Any]] = {}
    for base_name in FRONTIER_BASES:
        base = by_name[base_name]
        family = {
            float(method.planner.budget.multiplier): method
            for method in config.methods
            if method.stage == "P8" and method.reuse_component_from == base_name
        }
        if float(base.planner.budget.multiplier) in family:
            raise ValueError(f"P8 duplicates the base budget for {base_name}")
        family[float(base.planner.budget.multiplier)] = base
        if set(family) != set(FRONTIER_BUDGETS):
            raise ValueError(f"incomplete P8 budget frontier for {base_name}")
        output[base_name] = family
    if selected_track_j is not None:
        base = by_name[selected_track_j]
        family = {
            float(method.planner.budget.multiplier): method
            for method in config.methods
            if method.stage == "P8" and method.name.startswith("p8_p7_")
        }
        family[4.0] = base
        if set(family) != set(FRONTIER_BUDGETS):
            raise ValueError("incomplete P8 Track J budget frontier")
        output[selected_track_j] = family
    aliases = [method for method in config.methods if method.stage == "P8"]
    if len(aliases) != 3 * (len(FRONTIER_BUDGETS) - 1):
        raise ValueError("P8 must contain exactly three aliases per finalist")
    return output


def compute_frontier_method_rows(
    config: Any, lock: dict[str, Any]
) -> list[dict[str, Any]]:
    """Recompute all P5/P6/P7 budget-frontier metrics from formal results."""

    from vector_jepa_planner_frontier.stage_gates import validate_p7_selection

    p7 = validate_p7_selection(config, lock)
    selected_track_j = p7.get("selected_track_j")
    families = frontier_families(config, selected_track_j=selected_track_j)
    rows: list[dict[str, Any]] = []
    for base_name, family in families.items():
        for budget in FRONTIER_BUDGETS:
            method = resolve_effective_method(config, lock, family[budget])
            seed_rows = [
                load_validation_seed_rows(
                    config,
                    lock,
                    method=method.name,
                    backbone_seed=int(backbone_seed),
                )
                for backbone_seed in config.protocol.training_seeds
            ]
            rows.append(
                {
                    "method": method.name,
                    "family_base": base_name,
                    "track": method.track,
                    "budget_multiplier": float(budget),
                    "method_spec_sha256": effective_method_sha256(method),
                    **aggregate_frontier_metrics(
                        seed_rows,
                        transition_limit=method.planner.budget.transition_limit,
                    ),
                }
            )
    return rows


def aggregate_frontier_metrics(
    seed_rows: list[list[dict[str, Any]]], *, transition_limit: int
) -> dict[str, float]:
    """Aggregate paired nested runs without treating tasks as seed replicates."""

    if not seed_rows or any(not rows for rows in seed_rows):
        raise ValueError("P8 selection requires every configured backbone seed")
    seed_success: list[float] = []
    seed_large_success: list[float] = []
    transitions = 0.0
    decisions = 0.0
    for rows in seed_rows:
        large = [row for row in rows if int(row["maze_size"]) in (19, 21)]
        if not large:
            raise ValueError("P8 selection requires size-19/21 validation tasks")
        seed_success.append(float(np.mean([float(row["success"]) for row in rows])))
        seed_large_success.append(
            float(np.mean([float(row["success"]) for row in large]))
        )
        for row in rows:
            decision_count = float(row.get("decision_count", -1.0))
            plan_transitions = float(
                row.get("auxiliary", {}).get("plan_transitions", -1.0)
            )
            if decision_count < 0.0 or plan_transitions < 0.0:
                raise ValueError("P8 inputs omit decision or transition accounting")
            if plan_transitions > transition_limit * decision_count + 1e-6:
                raise ValueError("a P8 run exceeded its hard per-decision budget")
            decisions += decision_count
            transitions += plan_transitions
    return {
        "corrected_macro_sr": float(np.mean(seed_success)),
        "corrected_size19_21_sr": float(np.mean(seed_large_success)),
        "plan_transitions_per_decision": float(transitions / max(decisions, 1.0)),
    }


def select_near_optimal_budget(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the smallest budget within 0.01 SR of the observed maximum."""

    if {float(row["budget_multiplier"]) for row in rows} != set(FRONTIER_BUDGETS):
        raise ValueError("budget selection requires the complete four-point frontier")
    best_sr = max(float(row["corrected_macro_sr"]) for row in rows)
    near_optimal = [
        row
        for row in rows
        if best_sr - float(row["corrected_macro_sr"])
        <= NEAR_OPTIMAL_SR_TOLERANCE + 1e-12
    ]
    return min(
        near_optimal,
        key=lambda row: (
            float(row["budget_multiplier"]),
            float(row["plan_transitions_per_decision"]),
            str(row["method"]),
        ),
    )


def select_track_f_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose P5 versus P6 at the common 4x budget before budget selection."""

    expected = set(FRONTIER_BASES[:2])
    if {str(row["method"]) for row in rows} != expected:
        raise ValueError("Track F family selection requires the P5 and P6 4x cells")
    if any(float(row["budget_multiplier"]) != 4.0 for row in rows):
        raise ValueError("Track F family selection must be compute matched at 4x")
    best_sr = max(float(row["corrected_macro_sr"]) for row in rows)
    near_optimal = [
        row
        for row in rows
        if best_sr - float(row["corrected_macro_sr"])
        <= NEAR_OPTIMAL_SR_TOLERANCE + 1e-12
    ]
    return min(
        near_optimal,
        key=lambda row: (
            float(row["plan_transitions_per_decision"]),
            str(row["method"]),
        ),
    )


def validation_artifact_digest(
    config: Any, *, method_names: tuple[str, ...]
) -> dict[str, Any]:
    """Fingerprint every corrected validation result and candidate trace used."""

    by_name = {method.name: method for method in config.methods}
    records: list[dict[str, Any]] = []
    for method_name in sorted(method_names):
        method = by_name[method_name]
        for backbone_seed in config.protocol.training_seeds:
            for planner_seed in planner_seed_values(config, method):
                for search_seed in config.protocol.search_seeds:
                    result = resolve_path(
                        config.paths.result_template.format(
                            method=method_name,
                            backbone_seed=backbone_seed,
                            planner_seed=planner_seed,
                            search_seed=search_seed,
                            split="validation",
                            action_selection=config.protocol.primary_action_selection,
                        )
                    )
                    candidate = result.with_name(
                        f"{result.stem}.candidate_traces.jsonl"
                    )
                    if not result.is_file() or not candidate.is_file():
                        raise FileNotFoundError(
                            result if not result.is_file() else candidate
                        )
                    records.append(
                        {
                            "method": method_name,
                            "backbone_seed": int(backbone_seed),
                            "planner_seed": int(planner_seed),
                            "search_seed": int(search_seed),
                            "result_sha256": sha256_file(result),
                            "candidate_trace_sha256": sha256_file(candidate),
                        }
                    )
    return {
        "result_count": len(records),
        "sha256": canonical_json_sha256(records),
    }


def track_j_checkpoint_digest(
    config: Any,
    lock: dict[str, Any],
    *,
    method_name: str,
) -> dict[str, Any]:
    """Fingerprint every checkpoint of the frozen P7 grid winner."""

    method = resolve_effective_method(config, lock, method_by_name(config, method_name))
    records: list[dict[str, Any]] = []
    for backbone_seed in config.protocol.training_seeds:
        for planner_seed in planner_seed_values(config, method):
            path = component_checkpoint_path(
                config,
                method,
                backbone_seed=int(backbone_seed),
                planner_seed=int(planner_seed),
            )
            if path is None:
                raise ValueError("Track J unexpectedly has no component checkpoint")
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append(
                {
                    "backbone_seed": int(backbone_seed),
                    "planner_seed": int(planner_seed),
                    "sha256": sha256_file(path),
                }
            )
    return {
        "checkpoint_count": len(records),
        "sha256": canonical_json_sha256(records),
    }


def track_j_stability_records(
    config: Any,
    lock: dict[str, Any],
    *,
    method_name: str,
) -> list[dict[str, Any]]:
    """Revalidate every checkpoint of the frozen P7 grid winner."""

    from vector_jepa_planner_frontier.freeze_p7_selection import (
        checkpoint_stability_records,
    )

    return checkpoint_stability_records(
        config,
        lock,
        method_name=method_name,
    )


__all__ = [
    "FRONTIER_BASES",
    "FRONTIER_BUDGETS",
    "NEAR_OPTIMAL_SR_TOLERANCE",
    "aggregate_frontier_metrics",
    "compute_frontier_method_rows",
    "frontier_families",
    "select_near_optimal_budget",
    "select_track_f_family",
    "track_j_checkpoint_digest",
    "track_j_stability_records",
    "validation_artifact_digest",
]
