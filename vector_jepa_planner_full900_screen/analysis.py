"""Paired task-level analysis helpers used by all frozen stage decisions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from final_closure.common import read_jsonl, sha256_file
from spatial_jepa_planning.common import canonical_json_sha256, task_id
from vector_jepa_planner_full900_screen.common import (
    load_json,
    resolve_path,
    result_path,
)


def screen_planner_seed(method: Any) -> int:
    return 104_729 if method.component_checkpoint_required else 0


def load_result(
    config: Any,
    lock: dict[str, Any],
    *,
    method: Any,
    backbone_seed: int,
    planner_seed: int,
    action_selection: str,
) -> dict[str, Any]:
    path = result_path(
        config,
        method=method.name,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
        action_selection=action_selection,
    )
    if not path.exists():
        raise FileNotFoundError(f"missing full-900 result: {path}")
    value = load_json(path)
    metadata = value.get("metadata", {})
    if metadata.get("protocol_id") != config.protocol_id:
        raise ValueError(f"result protocol mismatch: {path}")
    if metadata.get("quick_spec_sha256") != lock["quick_spec_sha256"]:
        raise ValueError(f"result quick-spec mismatch: {path}")
    expected_method = method.model_dump(mode="json")
    if metadata.get("method") != expected_method:
        raise ValueError(f"result effective-method mismatch: {path}")
    if metadata.get("method_sha256") != canonical_json_sha256(expected_method):
        raise ValueError(f"result method hash mismatch: {path}")
    if metadata.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]:
        raise ValueError(f"result analysis-spec mismatch: {path}")
    if metadata.get("code_fingerprint") != lock["code_fingerprint"]:
        raise ValueError(f"result code fingerprint mismatch: {path}")
    if int(metadata.get("backbone_seed", -1)) != backbone_seed:
        raise ValueError(f"result backbone mismatch: {path}")
    expected_planner = planner_seed if method.component_checkpoint_required else None
    if metadata.get("planner_seed") != expected_planner:
        raise ValueError(f"result planner-seed mismatch: {path}")
    if value.get("action_selection") != action_selection:
        raise ValueError(f"result action-protocol mismatch: {path}")
    if value.get("stage") != "full900_planner_evaluation":
        raise ValueError(f"result stage mismatch: {path}")
    if value.get("split_role") != config.replication.evaluation_manifest_role:
        raise ValueError(f"result split-role mismatch: {path}")
    manifest_path = resolve_path(config.paths.development_manifest)
    manifest = value.get("manifest", {})
    if (
        manifest.get("sha256") != lock["development_manifest"]["sha256"]
        or int(manifest.get("count", -1)) != config.replication.task_count
        or sha256_file(manifest_path) != manifest.get("sha256")
    ):
        raise ValueError(f"result manifest provenance mismatch: {path}")
    tasks = value.get("tasks", [])
    if len(tasks) != config.replication.task_count:
        raise ValueError(f"result is not a complete full-900 evaluation: {path}")
    task_ids = [str(row["task_id"]) for row in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"duplicate task IDs in result: {path}")
    expected_task_ids = [task_id(row) for row in read_jsonl(manifest_path)]
    if task_ids != expected_task_ids:
        raise ValueError(f"result task order/hash mismatch: {path}")
    return value


def split_rows(result: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = list(result["tasks"])
    if split == "overall":
        return rows
    if split == "seen":
        return [row for row in rows if int(row["maze_size"]) <= 21]
    if split == "ood":
        return [row for row in rows if int(row["maze_size"]) > 21]
    raise ValueError(f"unknown analysis split: {split}")


def sr(result: dict[str, Any], split: str = "overall") -> float:
    rows = split_rows(result, split)
    return float(np.mean([float(row["success"]) for row in rows]))


def spl(result: dict[str, Any], split: str = "overall") -> float:
    rows = split_rows(result, split)
    return float(np.mean([float(row["spl"]) for row in rows]))


def _paired_rows(
    candidate: dict[str, Any], control: dict[str, Any], split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left = split_rows(candidate, split)
    right = split_rows(control, split)
    if [row["task_id"] for row in left] != [row["task_id"] for row in right]:
        raise ValueError("paired comparison task order/hash mismatch")
    return left, right


def delta_sr(
    candidate: dict[str, Any], control: dict[str, Any], split: str = "overall"
) -> float:
    left, right = _paired_rows(candidate, control, split)
    return float(
        np.mean(
            [
                float(a["success"]) - float(b["success"])
                for a, b in zip(left, right, strict=True)
            ]
        )
    )


def stratified_paired_bootstrap(
    candidate: dict[str, Any],
    control: dict[str, Any],
    *,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    if samples < 100 or not 0.0 < alpha < 1.0:
        raise ValueError("bootstrap requires samples>=100 and alpha in (0, 1)")
    left, right = _paired_rows(candidate, control, "overall")
    by_size: dict[int, np.ndarray] = {}
    for size in sorted({int(row["maze_size"]) for row in left}):
        values = np.asarray(
            [
                float(a["success"]) - float(b["success"])
                for a, b in zip(left, right, strict=True)
                if int(a["maze_size"]) == size
            ],
            dtype=np.float64,
        )
        by_size[size] = values
    rng = np.random.default_rng(seed)
    draws = np.zeros(samples, dtype=np.float64)
    total = sum(len(values) for values in by_size.values())
    chunk_size = min(2_000, samples)
    for values in by_size.values():
        for start in range(0, samples, chunk_size):
            stop = min(start + chunk_size, samples)
            sampled = rng.integers(
                0,
                len(values),
                size=(stop - start, len(values)),
            )
            draws[start:stop] += values[sampled].sum(axis=1)
    draws /= total
    return {
        "delta": delta_sr(candidate, control),
        "ci_low": float(np.quantile(draws, alpha / 2.0)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "alpha": float(alpha),
        "confidence_level": float(1.0 - alpha),
    }


def compute_per_decision(result: dict[str, Any]) -> dict[str, float]:
    decisions = max(sum(int(row["decision_count"]) for row in result["tasks"]), 1)
    names = ("plan_transitions", "planner_forward_calls", "node_expansions")
    return {
        name: float(
            sum(float(row["auxiliary"].get(name, 0.0)) for row in result["tasks"])
            / decisions
        )
        for name in names
    }


def mean(values: Iterable[float]) -> float:
    parsed = list(values)
    if not parsed:
        raise ValueError("cannot average an empty sequence")
    return float(np.mean(parsed))


__all__ = [
    "compute_per_decision",
    "delta_sr",
    "load_result",
    "mean",
    "screen_planner_seed",
    "spl",
    "sr",
    "stratified_paired_bootstrap",
]
