"""Freeze the final Track F/Track J budget choices before confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from final_closure.common import sha256_file
from spatial_jepa_planning.common import canonical_json_sha256
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_json_dump,
    load_json,
    load_study_config,
    require_clean_worktree,
    resolve_path,
    validate_locked_artifacts,
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
from vector_jepa_planner_frontier.stage_gates import (
    validate_p2_selection,
    validate_p5_advancement,
    validate_p7_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    validate_p2_selection(config, lock)
    validate_p5_advancement(config, lock)
    p7_selection = validate_p7_selection(config, lock)
    require_clean_worktree(allow_dirty=False)
    if resolve_path(config.paths.confirmation_opened).exists():
        raise RuntimeError("P8 selection cannot change after confirmation opens")
    output = resolve_path(config.paths.p8_selection)
    if output.exists():
        raise FileExistsError("P8 selection is immutable")

    p7_winner = p7_selection.get("selected_track_j")
    families = frontier_families(config, selected_track_j=p7_winner)
    method_rows = compute_frontier_method_rows(config, lock)
    by_method: dict[str, dict[str, Any]] = {
        str(row["method"]): row for row in method_rows
    }

    track_f_family = select_track_f_family(
        [by_method[FRONTIER_BASES[0]], by_method[FRONTIER_BASES[1]]]
    )
    track_f_base = str(track_f_family["method"])
    selected_track_f = select_near_optimal_budget(
        [by_method[families[track_f_base][budget].name] for budget in FRONTIER_BUDGETS]
    )
    selected_track_j_frontier: dict[str, Any] | None = None
    track_j_records: list[dict[str, Any]] = []
    stability_passed = False
    sr_noninferior = False
    track_j_checkpoints: dict[str, Any] | None = None
    if p7_winner is not None:
        selected_track_j_frontier = select_near_optimal_budget(
            [by_method[families[p7_winner][budget].name] for budget in FRONTIER_BUDGETS]
        )
        track_j_records = track_j_stability_records(
            config,
            lock,
            method_name=p7_winner,
        )
        stability_passed = bool(track_j_records) and all(
            record["jepa_stability_gate_passed"] for record in track_j_records
        )
        sr_noninferior = (
            float(selected_track_f["corrected_macro_sr"])
            - float(selected_track_j_frontier["corrected_macro_sr"])
            <= NEAR_OPTIMAL_SR_TOLERANCE + 1e-12
        )
        track_j_checkpoints = track_j_checkpoint_digest(
            config,
            lock,
            method_name=p7_winner,
        )
    track_j_admitted = bool(
        p7_winner is not None and stability_passed and sr_noninferior
    )
    selected_track_j = (
        str(selected_track_j_frontier["method"])
        if track_j_admitted and selected_track_j_frontier is not None
        else None
    )
    selected_track_f_name = str(selected_track_f["method"])
    confirmation_methods = ["b0_legacy_l2_cem", selected_track_f_name]
    if selected_track_j is not None:
        confirmation_methods.append(selected_track_j)

    frontier_method_names = tuple(sorted(by_method))
    payload = {
        "schema": "vector-jepa-p8-selection-v1",
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_results_viewed": True,
        "confirmatory_results_viewed": False,
        "action_selection": config.protocol.primary_action_selection,
        "near_optimal_sr_tolerance": NEAR_OPTIMAL_SR_TOLERANCE,
        "selection_rule": [
            "choose P5 versus P6 at their common 4x budget",
            "within 0.01 corrected validation SR choose fewer actual plan transitions",
            (
                "within the winning family choose the smallest budget within 0.01 "
                "of max SR"
            ),
            (
                "admit Track J only if every JEPA stability gate passes and its "
                "selected budget is within 0.01 SR of Track F"
            ),
            "resolve any remaining tie lexicographically by method name",
        ],
        "method_metrics": method_rows,
        "validation_artifacts": validation_artifact_digest(
            config, method_names=frontier_method_names
        ),
        "track_f_family_winner": track_f_base,
        "selected_track_f": selected_track_f_name,
        "p7_track_j_winner": p7_winner,
        "p7_selection_sha256": sha256_file(resolve_path(config.paths.p7_selection)),
        "track_j_frontier_choice": (
            str(selected_track_j_frontier["method"])
            if selected_track_j_frontier is not None
            else None
        ),
        "track_j_stability_passed": stability_passed,
        "track_j_sr_noninferior": sr_noninferior,
        "track_j_admitted": track_j_admitted,
        "selected_track_j": selected_track_j,
        "track_j_checkpoints": track_j_checkpoints,
        "track_j_stability_records_sha256": (
            canonical_json_sha256(track_j_records) if track_j_records else None
        ),
        "confirmation_methods": confirmation_methods,
        "comparison_count": 4 if track_j_admitted else 2,
    }
    atomic_json_dump(output, payload)


if __name__ == "__main__":
    main()
