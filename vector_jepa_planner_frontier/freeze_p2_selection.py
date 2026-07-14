"""Apply the preregistered P2 winner rule and freeze the P3 search backbone."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np

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
from vector_jepa_planner_frontier.schemas import PlannerKind
from vector_jepa_planner_frontier.validation_results import (
    load_validation_seed_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    return parser.parse_args()


def method_metrics(seed_rows: list[list[dict[str, Any]]]) -> dict[str, float]:
    rows = [row for backbone in seed_rows for row in backbone]
    large = [row for row in rows if int(row["maze_size"]) in (19, 21)]
    if not rows or not large:
        raise ValueError("P2 selection requires complete overall and size-19/21 rows")
    total_decisions = sum(float(row["decision_count"]) for row in rows)
    total_serial_calls = sum(
        float(row["auxiliary"].get("planner_forward_calls", 0.0)) for row in rows
    )
    if total_decisions <= 0.0:
        raise ValueError("P2 selection requires at least one planner decision")
    return {
        "assisted_macro_sr": float(np.mean([float(row["success"]) for row in rows])),
        "assisted_size19_21_sr": float(
            np.mean([float(row["success"]) for row in large])
        ),
        "predictor_serial_calls_per_decision": float(
            total_serial_calls / total_decisions
        ),
    }


def select_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_overall = max(float(row["assisted_macro_sr"]) for row in rows)
    pool = [
        row for row in rows if best_overall - float(row["assisted_macro_sr"]) < 0.01
    ]
    best_large = max(float(row["assisted_size19_21_sr"]) for row in pool)
    pool = [
        row for row in pool if best_large - float(row["assisted_size19_21_sr"]) < 0.01
    ]
    return min(
        pool,
        key=lambda row: (
            float(row["predictor_serial_calls_per_decision"]),
            str(row["method"]),
        ),
    )


def compute_selection_metrics(
    config: Any, lock: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = [
        method
        for method in config.methods
        if method.stage == "P2" and method.planner.budget.multiplier in (1.0, 4.0)
    ]
    metrics: list[dict[str, Any]] = []
    for method in candidates:
        seed_rows = [
            load_validation_seed_rows(
                config,
                lock,
                method=method.name,
                backbone_seed=int(backbone_seed),
            )
            for backbone_seed in config.protocol.training_seeds
        ]
        kind = (
            PlannerKind.CATEGORICAL_CEM
            if method.planner.kind == PlannerKind.LEGACY_CEM
            else method.planner.kind
        )
        metrics.append(
            {
                "method": method.name,
                "planner_kind": kind.value,
                "budget_multiplier": method.planner.budget.multiplier,
                **method_metrics(seed_rows),
            }
        )
    return metrics


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    require_clean_worktree(allow_dirty=False)
    output = resolve_path(config.paths.p2_selection)
    if output.exists():
        raise FileExistsError("P2 selection is immutable")
    metrics = compute_selection_metrics(config, lock)
    winner = select_winner(metrics)
    four_x = [
        method
        for method in config.methods
        if method.stage == "P2"
        and method.planner.budget.multiplier == 4.0
        and method.planner.kind.value == winner["planner_kind"]
    ]
    if len(four_x) != 1:
        raise ValueError("P2 winner does not map to exactly one preregistered 4x cell")
    selected_four_x = four_x[0]
    payload = {
        "schema": "vector-jepa-p2-selection-v1",
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": [
            "max assisted macro-SR among 1x/4x",
            "within 0.01 choose max size-19/21 SR",
            "within 0.01 choose fewer predictor serial calls",
            "final deterministic tie-break by method name",
        ],
        "methods": metrics,
        "winner": winner,
        "selected_planner_kind": winner["planner_kind"],
        "selected_4x_method": selected_four_x.name,
        "selected_4x_planner": selected_four_x.planner.model_dump(mode="json"),
        "selected_4x_planner_sha256": canonical_json_sha256(
            selected_four_x.planner.model_dump(mode="json")
        ),
    }
    atomic_json_dump(output, payload)


if __name__ == "__main__":
    main()
