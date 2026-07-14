"""Shared, hash-validated loading of nested validation task records."""

from __future__ import annotations

from typing import Any

from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    method_by_name,
    planner_seed_values,
)
from vector_jepa_planner_frontier.effective_methods import (
    effective_method_sha256,
    resolve_effective_method,
)
from vector_jepa_planner_frontier.summarize import (
    average_nested_task_rows,
    load_result,
    result_path,
)


def load_validation_seed_rows(
    config: Any,
    lock: dict[str, Any],
    *,
    method: str,
    backbone_seed: int,
) -> list[dict[str, Any]]:
    """Average search then planner seeds for one independent backbone."""

    method_config = resolve_effective_method(
        config, lock, method_by_name(config, method)
    )
    planner_averages: list[list[dict[str, Any]]] = []
    for planner_seed in planner_seed_values(config, method_config):
        search_rows: list[list[dict[str, Any]]] = []
        for search_seed in config.protocol.search_seeds:
            path = result_path(
                config,
                method=method,
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
                search_seed=search_seed,
                split_role="validation",
                action_selection=config.protocol.primary_action_selection,
            )
            result = load_result(
                path,
                analysis_hash=analysis_spec_sha256(config, lock),
                method=method,
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
                search_seed=search_seed,
                split_role="validation",
                action_selection=config.protocol.primary_action_selection,
                expected_count=int(lock["validation_manifest"]["count"]),
                expected_manifest_sha256=lock["validation_manifest"]["sha256"],
                expected_code_fingerprint=lock["code_fingerprint"],
                expected_method_sha256=effective_method_sha256(method_config),
            )
            search_rows.append(result["tasks"])
        planner_averages.append(average_nested_task_rows(search_rows))
    return average_nested_task_rows(planner_averages)


__all__ = ["load_validation_seed_rows"]
