"""Validation for immutable, result-dependent development-stage decisions."""

from __future__ import annotations

import math
from typing import Any

from final_closure.common import sha256_file
from spatial_jepa_planning.common import canonical_json_sha256
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    load_json,
    method_by_name,
    resolve_path,
)
from vector_jepa_planner_frontier.effective_methods import (
    RADICAL_METHODS,
    effective_method_sha256,
    p3_cell_for_components,
    resolve_effective_method,
)
from vector_jepa_planner_frontier.freeze_p2_selection import (
    compute_selection_metrics,
    select_winner,
)
from vector_jepa_planner_frontier.freeze_p5_advancement import (
    _validate_evidence,
    _validate_summary,
    select_radical,
)
from vector_jepa_planner_frontier.freeze_p7_selection import (
    joint_method_rows,
    select_joint_winner,
)
from vector_jepa_planner_frontier.frontier_selection import (
    FRONTIER_BASES,
    FRONTIER_BUDGETS,
    NEAR_OPTIMAL_SR_TOLERANCE,
    compute_frontier_method_rows,
    frontier_families,
    select_near_optimal_budget,
    select_track_f_family,
    track_j_checkpoint_digest,
    track_j_stability_records,
    validation_artifact_digest,
)


def validate_p2_selection(config: Any, lock: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(config.paths.p2_selection)
    if not path.is_file():
        raise RuntimeError("P3 is locked until the P2 selection record is frozen")
    value = load_json(path)
    if value.get("schema") != "vector-jepa-p2-selection-v1":
        raise ValueError("unknown P2 selection schema")
    if value.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("P2 selection belongs to another analysis")
    metrics = compute_selection_metrics(config, lock)
    if value.get("methods") != metrics:
        raise ValueError("P2 selection metrics do not reproduce from formal results")
    winner = select_winner(metrics)
    if value.get("winner") != winner:
        raise ValueError("P2 winner does not reproduce from formal results")
    four_x = [
        method
        for method in config.methods
        if method.stage == "P2"
        and method.planner.budget.multiplier == 4.0
        and method.planner.kind.value == winner["planner_kind"]
    ]
    if len(four_x) != 1:
        raise ValueError("P2 winner does not map to one locked 4x planner")
    selected = four_x[0]
    planner = selected.planner.model_dump(mode="json")
    if (
        value.get("selected_planner_kind") != winner["planner_kind"]
        or value.get("selected_4x_method") != selected.name
        or value.get("selected_4x_planner") != planner
        or value.get("selected_4x_planner_sha256") != canonical_json_sha256(planner)
    ):
        raise ValueError("P2-derived P3 planner does not reproduce")
    return value


def validate_p5_advancement(config: Any, lock: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(config.paths.p5_advancement)
    if not path.is_file():
        raise RuntimeError("P5 is locked until its component advancement is frozen")
    value = load_json(path)
    if value.get("schema") != "vector-jepa-p5-advancement-v1":
        raise ValueError("unknown P5 advancement schema")
    if value.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("P5 advancement belongs to another analysis")
    artifacts: dict[str, Any] = {}
    for key in ("p3_summary", "p4_summary", "evidence"):
        record = value.get(key, {})
        artifact = resolve_path(record.get("path", ""))
        if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
            raise ValueError(f"P5 advancement artifact hash mismatch: {key}")
        artifacts[key] = load_json(artifact)
    analysis_hash = analysis_spec_sha256(config, lock)
    _validate_summary(
        resolve_path(value["p3_summary"]["path"]),
        analysis_hash=analysis_hash,
        required_methods={
            method.name for method in config.methods if method.stage == "P3"
        },
    )
    _validate_summary(
        resolve_path(value["p4_summary"]["path"]),
        analysis_hash=analysis_hash,
        required_methods={
            method.name for method in config.methods if method.stage == "P4"
        },
    )
    _validate_evidence(artifacts["evidence"])
    if value.get("evidence", {}).get("content") != artifacts["evidence"]:
        raise ValueError("P5 embedded evidence differs from its frozen source")
    evidence = value.get("evidence", {}).get("content", {})
    selected_components = list(evidence.get("selected_components", []))
    selected_radical, radical_metrics = select_radical(
        evidence,
        artifacts["p4_summary"],
        action_selection=config.protocol.primary_action_selection,
    )
    if (
        value.get("selected_components") != selected_components
        or value.get("selected_p3_cell") != p3_cell_for_components(selected_components)
        or value.get("selected_radical") != selected_radical
        or value.get("radical_source_method")
        != (RADICAL_METHODS[selected_radical] if selected_radical is not None else None)
        or value.get("radical_selection_metrics") != radical_metrics
    ):
        raise ValueError("P5 effective architecture does not reproduce from evidence")
    if evidence.get("selected_radical") != selected_radical:
        raise ValueError("P5 evidence does not follow the frozen radical rule")
    if not selected_components and selected_radical is None:
        raise RuntimeError("P5 is closed because no component passed advancement")
    if evidence.get("confirmatory_results_viewed") is not False:
        raise ValueError("P5 advancement was not frozen before confirmation")
    return value


def validate_p8_selection(config: Any, lock: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(config.paths.p8_selection)
    if not path.is_file():
        raise RuntimeError("confirmation is locked until the P8 frontier is frozen")
    value = load_json(path)
    if value.get("schema") != "vector-jepa-p8-selection-v1":
        raise ValueError("unknown P8 selection schema")
    if value.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("P8 selection belongs to another analysis")
    if (
        value.get("validation_results_viewed") is not True
        or value.get("confirmatory_results_viewed") is not False
        or value.get("action_selection") != config.protocol.primary_action_selection
        or float(value.get("near_optimal_sr_tolerance", -1.0))
        != NEAR_OPTIMAL_SR_TOLERANCE
    ):
        raise ValueError("P8 selection boundary or endpoint drifted")

    p7_selection = validate_p7_selection(config, lock)
    p7_winner = p7_selection.get("selected_track_j")
    if value.get("p7_track_j_winner") != p7_winner or value.get(
        "p7_selection_sha256"
    ) != sha256_file(resolve_path(config.paths.p7_selection)):
        raise ValueError("P8 does not reproduce the frozen P7 decision")
    families = frontier_families(config, selected_track_j=p7_winner)
    rows = value.get("method_metrics", [])
    if not isinstance(rows, list):
        raise ValueError("P8 method metrics must be a list")
    if rows != compute_frontier_method_rows(config, lock):
        raise ValueError("P8 metrics do not reproduce from formal validation results")
    by_method = {str(row.get("method")): row for row in rows}
    expected_methods = {
        method.name for family in families.values() for method in family.values()
    }
    if set(by_method) != expected_methods or len(rows) != len(expected_methods):
        raise ValueError("P8 selection does not cover the exact frontier")
    for name, row in by_method.items():
        method = resolve_effective_method(config, lock, method_by_name(config, name))
        if row.get("method_spec_sha256") != effective_method_sha256(method):
            raise ValueError(f"P8 method changed after selection: {name}")
        if float(row.get("budget_multiplier", -1.0)) != float(
            method.planner.budget.multiplier
        ):
            raise ValueError(f"P8 method budget label mismatch: {name}")
        metrics = (
            row.get("corrected_macro_sr"),
            row.get("corrected_size19_21_sr"),
            row.get("plan_transitions_per_decision"),
        )
        if any(not math.isfinite(float(metric)) for metric in metrics):
            raise ValueError(f"P8 method has a non-finite metric: {name}")

    track_f_family = select_track_f_family(
        [by_method[FRONTIER_BASES[0]], by_method[FRONTIER_BASES[1]]]
    )
    track_f_base = str(track_f_family["method"])
    if value.get("track_f_family_winner") != track_f_base:
        raise ValueError("P8 Track F family winner does not reproduce")
    selected_track_f = select_near_optimal_budget(
        [by_method[families[track_f_base][budget].name] for budget in FRONTIER_BUDGETS]
    )
    if value.get("selected_track_f") != selected_track_f["method"]:
        raise ValueError("P8 Track F budget choice does not reproduce")
    selected_track_j: dict[str, Any] | None = None
    stability_records: list[dict[str, Any]] = []
    stability_passed = False
    sr_noninferior = False
    expected_checkpoint_digest: dict[str, Any] | None = None
    if p7_winner is not None:
        selected_track_j = select_near_optimal_budget(
            [by_method[families[p7_winner][budget].name] for budget in FRONTIER_BUDGETS]
        )
        stability_records = track_j_stability_records(
            config,
            lock,
            method_name=p7_winner,
        )
        stability_passed = bool(stability_records) and all(
            record["jepa_stability_gate_passed"] for record in stability_records
        )
        sr_noninferior = (
            float(selected_track_f["corrected_macro_sr"])
            - float(selected_track_j["corrected_macro_sr"])
            <= NEAR_OPTIMAL_SR_TOLERANCE + 1e-12
        )
        expected_checkpoint_digest = track_j_checkpoint_digest(
            config,
            lock,
            method_name=p7_winner,
        )
    expected_frontier_choice = (
        str(selected_track_j["method"]) if selected_track_j is not None else None
    )
    if value.get("track_j_frontier_choice") != expected_frontier_choice:
        raise ValueError("P8 Track J budget choice does not reproduce")
    expected_stability_sha = (
        canonical_json_sha256(stability_records) if stability_records else None
    )
    if value.get("track_j_stability_records_sha256") != expected_stability_sha:
        raise ValueError("P8 Track J stability evidence changed after selection")
    if value.get("track_j_stability_passed") is not stability_passed:
        raise ValueError("P8 Track J stability decision does not reproduce")
    admitted = bool(p7_winner is not None and stability_passed and sr_noninferior)
    if (
        value.get("track_j_sr_noninferior") is not sr_noninferior
        or value.get("track_j_admitted") is not admitted
    ):
        raise ValueError("P8 Track J admission decision does not reproduce")
    expected_track_j = (
        str(selected_track_j["method"])
        if admitted and selected_track_j is not None
        else None
    )
    if value.get("selected_track_j") != expected_track_j:
        raise ValueError("P8 Track J selected method is inconsistent")

    expected_confirmation = ["b0_legacy_l2_cem", str(selected_track_f["method"])]
    if expected_track_j is not None:
        expected_confirmation.append(expected_track_j)
    if value.get("confirmation_methods") != expected_confirmation:
        raise ValueError("P8 confirmation method family is inconsistent")
    if int(value.get("comparison_count", -1)) != (4 if admitted else 2):
        raise ValueError("P8 comparison count is inconsistent")
    if any(
        not method_by_name(config, name).confirmatory_eligible
        for name in expected_confirmation
    ):
        raise ValueError("a P8 winner is not confirmation eligible")
    if value.get("validation_artifacts") != validation_artifact_digest(
        config, method_names=tuple(sorted(expected_methods))
    ):
        raise ValueError("P8 validation artifacts changed after selection")
    if value.get("track_j_checkpoints") != expected_checkpoint_digest:
        raise ValueError("P8 Track J checkpoints changed after selection")
    return value


def validate_p7_selection(config: Any, lock: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(config.paths.p7_selection)
    if not path.is_file():
        raise RuntimeError("P8 is locked until the P7 grid decision is frozen")
    value = load_json(path)
    if value.get("schema") != "vector-jepa-p7-selection-v1":
        raise ValueError("unknown P7 selection schema")
    if value.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("P7 selection belongs to another analysis")
    if (
        value.get("validation_results_viewed") is not True
        or value.get("confirmatory_results_viewed") is not False
        or value.get("action_selection") != config.protocol.primary_action_selection
        or float(value.get("near_optimal_sr_tolerance", -1.0))
        != NEAR_OPTIMAL_SR_TOLERANCE
    ):
        raise ValueError("P7 selection boundary drifted")
    rows = joint_method_rows(config, lock)
    if value.get("method_metrics") != rows:
        raise ValueError("P7 metrics do not reproduce from formal results/checkpoints")
    winner = select_joint_winner(rows)
    selected = str(winner["method"]) if winner is not None else None
    if (
        value.get("track_j_failed") is not (winner is None)
        or value.get("selected_track_j") != selected
    ):
        raise ValueError("P7 winner or stability failure does not reproduce")
    method_names = tuple(sorted(str(row["method"]) for row in rows))
    if value.get("validation_artifacts") != validation_artifact_digest(
        config, method_names=method_names
    ):
        raise ValueError("P7 validation artifacts changed after selection")
    return value


__all__ = [
    "validate_p2_selection",
    "validate_p5_advancement",
    "validate_p7_selection",
    "validate_p8_selection",
]
