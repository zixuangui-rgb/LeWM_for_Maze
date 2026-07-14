"""Freeze the reviewed component and radical gates that define effective P5."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_json_dump,
    load_json,
    load_study_config,
    require_clean_worktree,
    resolve_path,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.effective_methods import (
    RADICAL_METHODS,
    p3_cell_for_components,
)

COMPONENTS = ("verifier", "reachability", "proposal", "memory")
GATES = (
    "mechanism_improved",
    "overall_noninferiority",
    "large_maze_noninferiority",
    "equal_compute",
    "negative_control_passed",
    "direction_consistency_passed",
)
RADICAL_TIE_TOLERANCE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--p3-summary", required=True)
    parser.add_argument("--p4-summary", required=True)
    parser.add_argument("--evidence", required=True)
    return parser.parse_args()


def _validate_summary(
    path: Path,
    *,
    analysis_hash: str,
    required_methods: set[str],
) -> dict[str, Any]:
    value = load_json(path)
    if value.get("analysis_spec_sha256") != analysis_hash:
        raise ValueError(f"stage summary belongs to another analysis: {path}")
    if value.get("split_role") != "validation" or value.get("missing_files"):
        raise ValueError(f"advancement requires a complete validation summary: {path}")
    reported = {str(row["method"]) for row in value.get("primary", [])}
    if not required_methods <= reported:
        raise ValueError(f"stage summary omits configured methods: {path}")
    return value


def _passes(row: dict[str, Any]) -> bool:
    return all(row.get(gate) is True for gate in GATES)


def _validate_evidence(value: dict[str, Any]) -> None:
    if value.get("schema") != "vector-jepa-p5-evidence-v1":
        raise ValueError("unknown P5 evidence schema")
    if not str(value.get("reviewer", "")).strip():
        raise ValueError("P5 gate requires a named reviewer")
    if value.get("validation_results_viewed") is not True:
        raise ValueError("P5 evidence must acknowledge validation review")
    if value.get("confirmatory_results_viewed") is not False:
        raise ValueError("P5 evidence cannot follow confirmatory inspection")
    gates = value.get("component_gates", {})
    if set(gates) != set(COMPONENTS):
        raise ValueError("P5 evidence must cover exactly four components")
    for component in COMPONENTS:
        row = gates[component]
        for gate in GATES:
            if not isinstance(row.get(gate), bool):
                raise ValueError(f"P5 gate is not boolean: {component}.{gate}")
        if not str(row.get("evidence", "")).strip():
            raise ValueError(f"P5 gate lacks an evidence reference: {component}")
    selected_components = tuple(
        component for component in COMPONENTS if _passes(gates[component])
    )
    if tuple(value.get("selected_components", ())) != selected_components:
        raise ValueError(
            "selected components do not equal the components passing all gates"
        )
    radical_gates = value.get("radical_gates", {})
    if set(radical_gates) != set(RADICAL_METHODS):
        raise ValueError("P5 evidence must cover every preregistered radical")
    for radical, row in radical_gates.items():
        for gate in GATES:
            if not isinstance(row.get(gate), bool):
                raise ValueError(f"radical gate is not boolean: {radical}.{gate}")
        if not str(row.get("evidence", "")).strip():
            raise ValueError(f"radical gate lacks an evidence reference: {radical}")
    selected_radical = value.get("selected_radical")
    if selected_radical is not None:
        if selected_radical not in RADICAL_METHODS:
            raise ValueError("P5 selected an unknown radical method")
        if not _passes(radical_gates[selected_radical]):
            raise RuntimeError("selected radical did not pass every advancement gate")
    if not str(value.get("radical_decision_reason", "")).strip():
        raise ValueError("P5 evidence must explain the radical selection decision")


def radical_selection_metrics(
    evidence: dict[str, Any],
    p4_summary: dict[str, Any],
    *,
    action_selection: str,
) -> list[dict[str, Any]]:
    """Extract the locked SR tie-break metrics for gate-passing radicals."""

    output: list[dict[str, Any]] = []
    for radical, method_name in RADICAL_METHODS.items():
        if not _passes(evidence["radical_gates"][radical]):
            continue
        primary = [
            row
            for row in p4_summary.get("primary", [])
            if row.get("method") == method_name
            and row.get("action_selection") == action_selection
        ]
        large = [
            row
            for row in p4_summary.get("per_size", [])
            if row.get("method") == method_name
            and row.get("action_selection") == action_selection
            and int(row.get("maze_size", -1)) in (19, 21)
        ]
        if len(primary) != 1 or {int(row["maze_size"]) for row in large} != {19, 21}:
            raise ValueError(
                f"P4 summary lacks unique radical selection metrics: {radical}"
            )
        overall_sr = float(primary[0]["sr"])
        large_sr = sum(float(row["sr"]) for row in large) / 2.0
        if not math.isfinite(overall_sr) or not math.isfinite(large_sr):
            raise ValueError(f"P4 radical metrics are non-finite: {radical}")
        output.append(
            {
                "radical": radical,
                "method": method_name,
                "corrected_macro_sr": overall_sr,
                "corrected_size19_21_sr": large_sr,
            }
        )
    return sorted(output, key=lambda row: str(row["radical"]))


def select_radical(
    evidence: dict[str, Any],
    p4_summary: dict[str, Any],
    *,
    action_selection: str,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Apply the pre-run deterministic at-most-one-radical decision rule."""

    metrics = radical_selection_metrics(
        evidence,
        p4_summary,
        action_selection=action_selection,
    )
    if not metrics:
        return None, metrics
    best_overall = max(float(row["corrected_macro_sr"]) for row in metrics)
    finalists = [
        row
        for row in metrics
        if best_overall - float(row["corrected_macro_sr"])
        <= RADICAL_TIE_TOLERANCE + 1e-12
    ]
    best_large = max(float(row["corrected_size19_21_sr"]) for row in finalists)
    finalists = [
        row
        for row in finalists
        if best_large - float(row["corrected_size19_21_sr"])
        <= RADICAL_TIE_TOLERANCE + 1e-12
    ]
    winner = min(finalists, key=lambda row: str(row["radical"]))
    return str(winner["radical"]), metrics


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    require_clean_worktree(allow_dirty=False)
    if resolve_path(config.paths.confirmation_opened).exists():
        raise RuntimeError("P5 advancement cannot change after confirmation opens")
    output = resolve_path(config.paths.p5_advancement)
    if output.exists():
        raise FileExistsError("P5 advancement decision is immutable")
    analysis_hash = analysis_spec_sha256(config, lock)
    p3_path = resolve_path(args.p3_summary)
    p4_path = resolve_path(args.p4_summary)
    p3_methods = {method.name for method in config.methods if method.stage == "P3"}
    p4_methods = {method.name for method in config.methods if method.stage == "P4"}
    _validate_summary(p3_path, analysis_hash=analysis_hash, required_methods=p3_methods)
    p4_summary = _validate_summary(
        p4_path,
        analysis_hash=analysis_hash,
        required_methods=p4_methods,
    )
    evidence_path = resolve_path(args.evidence)
    evidence = load_json(evidence_path)
    _validate_evidence(evidence)
    selected_components = list(evidence["selected_components"])
    selected_radical, radical_metrics = select_radical(
        evidence,
        p4_summary,
        action_selection=config.protocol.primary_action_selection,
    )
    if evidence.get("selected_radical") != selected_radical:
        raise ValueError("selected radical does not reproduce from the locked rule")
    if not selected_components and selected_radical is None:
        raise RuntimeError(
            "all P3/P4 components failed; protocol rule 13.4 closes the experiment"
        )
    payload = {
        "schema": "vector-jepa-p5-advancement-v1",
        "analysis_spec_sha256": analysis_hash,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "p3_summary": {"path": str(p3_path), "sha256": sha256_file(p3_path)},
        "p4_summary": {"path": str(p4_path), "sha256": sha256_file(p4_path)},
        "evidence": {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
            "content": evidence,
        },
        "selected_components": selected_components,
        "selected_p3_cell": p3_cell_for_components(selected_components),
        "selected_radical": selected_radical,
        "radical_selection_metrics": radical_metrics,
        "radical_selection_rule": [
            "retain only radicals passing all six advancement gates",
            "retain corrected macro-SR cells within 0.01 of the passing maximum",
            "retain size-19/21 SR cells within 0.01 of the remaining maximum",
            "resolve an exact residual tie lexicographically by radical name",
        ],
        "radical_source_method": (
            RADICAL_METHODS[selected_radical] if selected_radical is not None else None
        ),
    }
    atomic_json_dump(output, payload)


if __name__ == "__main__":
    main()
