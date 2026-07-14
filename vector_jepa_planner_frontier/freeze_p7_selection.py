"""Freeze the preregistered 54-cell Track-J grid winner or stability failure."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from final_closure.common import sha256_file
from spatial_jepa_planning.common import canonical_json_sha256
from vector_jepa_planner_frontier import EXPERIMENT_FAMILY, FORMAT_VERSION
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_json_dump,
    component_checkpoint_path,
    load_json,
    load_study_config,
    planner_seed_values,
    require_clean_worktree,
    resolve_path,
    training_spec_sha256,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.compat import checkpoint_path
from vector_jepa_planner_frontier.effective_methods import (
    effective_method_sha256,
    resolve_effective_method,
)
from vector_jepa_planner_frontier.frontier_selection import (
    NEAR_OPTIMAL_SR_TOLERANCE,
    validation_artifact_digest,
)
from vector_jepa_planner_frontier.validation_results import (
    load_validation_seed_rows,
)


@cache
def _sha256_for_immutable_file(
    path_string: str,
    size: int,
    modified_ns: int,
) -> str:
    del size, modified_ns
    return sha256_file(Path(path_string))


def cached_sha256(path: Path) -> str:
    """Hash immutable artifacts once while invalidating on ordinary file changes."""

    resolved = path.resolve()
    stat = resolved.stat()
    return _sha256_for_immutable_file(
        str(resolved),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    return parser.parse_args()


def joint_candidates(config: Any) -> list[Any]:
    candidates = [
        method
        for method in config.methods
        if method.stage == "P7" and method.track == "J"
    ]
    if len(candidates) != 54:
        raise ValueError("Track J selection requires the complete 54-cell grid")
    return candidates


def validate_joint_training_metadata(
    value: dict[str, Any],
    config: Any,
) -> None:
    training = value.get("training_summary", {})
    metrics = value.get("validation_metrics", {})
    module_limits = training.get("module_step_limits", {})
    if (
        int(training.get("steps", -1)) != config.training.joint_steps
        or int(training.get("locked_steps", -1)) != config.training.joint_steps
        or not isinstance(module_limits, dict)
        or not module_limits
        or any(
            int(limit) != config.training.joint_steps
            for limit in module_limits.values()
        )
        or training.get("schedule") != "5pct_linear_warmup_cosine_to_10pct"
        or training.get("random_untrained_control") is not False
        or training.get("inherited_frozen_control") is not False
        or int(training.get("joint_jepa_sequence_length", -1))
        != config.training.sequence_length
        or training.get("joint_jepa_size_schedule") != "round_robin_over_train_sizes"
        or int(metrics.get("jepa_validation_sequence_length", -1))
        != config.training.sequence_length
        or metrics.get("jepa_validation_size_schedule")
        != "round_robin_over_validation_sizes"
    ):
        raise ValueError("Track J did not use the locked joint-training protocol")
    if not isinstance(value.get("model_state_dict"), dict):
        raise ValueError("Track J checkpoint lacks jointly adapted backbone parameters")
    parent = value.get("initialization_parent", {})
    if (
        not isinstance(parent, dict)
        or parent.get("method") != "p6_track_f_counterexample_ranked"
        or parent.get("stage") != "counterexample_training_round"
    ):
        raise ValueError("Track J did not inherit the locked hard-negative parent")
    backbone_seed = int(value.get("backbone_seed", -1))
    planner_seed = int(value.get("planner_seed", -1))
    provenance = value.get("joint_counterexample_provenance")
    if not isinstance(provenance, list) or len(provenance) != (
        config.training.counterexample_rounds
    ):
        raise ValueError("Track J lacks the complete hard-negative provenance chain")
    total_records = 0
    for round_index, record in enumerate(provenance, start=1):
        dataset_path = resolve_path(
            config.paths.counterexample_dataset_template.format(
                method="p6_track_f_counterexample_ranked",
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
                round=round_index,
            )
        )
        checkpoint_path = resolve_path(
            config.paths.counterexample_round_template.format(
                method="p6_track_f_counterexample_ranked",
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
                round=round_index,
            )
        )
        source_path = (
            resolve_path(
                config.paths.component_checkpoint_template.format(
                    method="p6_track_f_counterexample_ranked",
                    backbone_seed=backbone_seed,
                    planner_seed=planner_seed,
                )
            )
            if round_index == 1
            else resolve_path(
                config.paths.counterexample_round_template.format(
                    method="p6_track_f_counterexample_ranked",
                    backbone_seed=backbone_seed,
                    planner_seed=planner_seed,
                    round=round_index - 1,
                )
            )
        )
        if (
            int(record.get("round", -1)) != round_index
            or Path(str(record.get("dataset_path", ""))).resolve()
            != dataset_path.resolve()
            or Path(str(record.get("checkpoint_path", ""))).resolve()
            != checkpoint_path.resolve()
            or Path(str(record.get("source_checkpoint_path", ""))).resolve()
            != source_path.resolve()
        ):
            raise ValueError("Track J hard-negative provenance path mismatch")
        for path, key in (
            (dataset_path, "dataset_sha256"),
            (checkpoint_path, "checkpoint_sha256"),
            (source_path, "source_checkpoint_sha256"),
        ):
            if not path.is_file() or record.get(key) != cached_sha256(path):
                raise ValueError(f"Track J hard-negative artifact mismatch: {path}")
        record_count = int(record.get("record_count", -1))
        if record_count < 0:
            raise ValueError("Track J hard-negative record count is invalid")
        total_records += record_count
    if total_records == 0:
        raise ValueError("Track J cannot train its ranker without hard negatives")
    final_parent = resolve_path(
        config.paths.counterexample_round_template.format(
            method="p6_track_f_counterexample_ranked",
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
            round=config.training.counterexample_rounds,
        )
    )
    if Path(
        str(parent.get("path", ""))
    ).resolve() != final_parent.resolve() or parent.get("sha256") != cached_sha256(
        final_parent
    ):
        raise ValueError("Track J initialization is not the authenticated P6 round-3")


def checkpoint_stability_records(
    config: Any,
    lock: dict[str, Any],
    *,
    method_name: str,
) -> list[dict[str, Any]]:
    base = next(
        method for method in joint_candidates(config) if method.name == method_name
    )
    method = resolve_effective_method(config, lock, base)
    records: list[dict[str, Any]] = []
    for backbone_seed in config.protocol.training_seeds:
        source = checkpoint_path(config, seed=int(backbone_seed))
        source_sha = sha256_file(source)
        for planner_seed in planner_seed_values(config, method):
            path = component_checkpoint_path(
                config,
                method,
                backbone_seed=int(backbone_seed),
                planner_seed=int(planner_seed),
            )
            if path is None or not path.is_file():
                raise FileNotFoundError(path or method.name)
            value = torch.load(path, map_location="cpu", weights_only=False)
            if (
                value.get("experiment_family") != EXPERIMENT_FAMILY
                or int(value.get("format_version", -1)) != FORMAT_VERSION
                or value.get("stage") != "component_calibration"
            ):
                raise ValueError(f"invalid Track J checkpoint: {path}")
            if (
                value.get("method_name") != method.name
                or int(value.get("backbone_seed", -1)) != int(backbone_seed)
                or int(value.get("planner_seed", -1)) != int(planner_seed)
            ):
                raise ValueError(f"Track J checkpoint label mismatch: {path}")
            if value.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
                raise ValueError(f"Track J checkpoint analysis mismatch: {path}")
            if value.get("training_spec_sha256") != training_spec_sha256(
                config,
                lock,
                method=method,
                backbone_seed=int(backbone_seed),
                planner_seed=int(planner_seed),
            ):
                raise ValueError(f"Track J checkpoint training spec mismatch: {path}")
            if value.get("source_checkpoint_sha256") != source_sha:
                raise ValueError(f"Track J checkpoint source mismatch: {path}")
            protocol = value.get("protocol", {})
            if (
                protocol.get("git_dirty") is not False
                or protocol.get("code_fingerprint") != lock["code_fingerprint"]
            ):
                raise ValueError(f"Track J checkpoint provenance mismatch: {path}")
            metrics = value.get("validation_metrics", {})
            validate_joint_training_metadata(value, config)
            relative_change = float(
                metrics.get("jepa_validation_relative_change", math.nan)
            )
            threshold = float(metrics.get("jepa_stability_threshold", math.nan))
            gate = metrics.get("jepa_stability_gate_passed") is True
            if (
                not math.isfinite(relative_change)
                or threshold != 0.10
                or gate != (relative_change <= threshold)
            ):
                raise ValueError(f"Track J stability record is inconsistent: {path}")
            records.append(
                {
                    "backbone_seed": int(backbone_seed),
                    "planner_seed": int(planner_seed),
                    "checkpoint_sha256": sha256_file(path),
                    "jepa_validation_relative_change": relative_change,
                    "jepa_stability_gate_passed": gate,
                }
            )
    return records


def joint_method_rows(config: Any, lock: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in joint_candidates(config):
        method = resolve_effective_method(config, lock, base)
        seed_rows = [
            load_validation_seed_rows(
                config,
                lock,
                method=method.name,
                backbone_seed=int(backbone_seed),
            )
            for backbone_seed in config.protocol.training_seeds
        ]
        seed_sr = [
            float(np.mean([float(row["success"]) for row in backbone]))
            for backbone in seed_rows
        ]
        large_sr = [
            float(
                np.mean(
                    [
                        float(row["success"])
                        for row in backbone
                        if int(row["maze_size"]) in (19, 21)
                    ]
                )
            )
            for backbone in seed_rows
        ]
        stability = checkpoint_stability_records(config, lock, method_name=method.name)
        changes = [float(row["jepa_validation_relative_change"]) for row in stability]
        rows.append(
            {
                "method": method.name,
                "effective_method_sha256": effective_method_sha256(method),
                "joint_hyperparameters": method.joint_hyperparameters.model_dump(
                    mode="json"
                ),
                "corrected_macro_sr": float(np.mean(seed_sr)),
                "corrected_size19_21_sr": float(np.mean(large_sr)),
                "stable_checkpoint_count": int(
                    sum(bool(row["jepa_stability_gate_passed"]) for row in stability)
                ),
                "checkpoint_count": len(stability),
                "all_checkpoints_stable": all(
                    bool(row["jepa_stability_gate_passed"]) for row in stability
                ),
                "max_jepa_relative_change": float(max(changes)),
                "mean_jepa_relative_change": float(np.mean(changes)),
                "checkpoint_evidence_sha256": canonical_json_sha256(stability),
            }
        )
    return rows


def select_joint_winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    stable = [row for row in rows if row["all_checkpoints_stable"] is True]
    if not stable:
        return None
    best_sr = max(float(row["corrected_macro_sr"]) for row in stable)
    near = [
        row
        for row in stable
        if best_sr - float(row["corrected_macro_sr"])
        <= NEAR_OPTIMAL_SR_TOLERANCE + 1e-12
    ]
    return min(
        near,
        key=lambda row: (
            float(row["max_jepa_relative_change"]),
            float(row["mean_jepa_relative_change"]),
            -float(row["corrected_size19_21_sr"]),
            str(row["method"]),
        ),
    )


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    require_clean_worktree(allow_dirty=False)
    output = resolve_path(config.paths.p7_selection)
    if output.exists():
        raise FileExistsError("P7 selection is immutable")
    if resolve_path(config.paths.confirmation_opened).exists():
        raise RuntimeError("P7 selection cannot change after confirmation opens")
    method_rows = joint_method_rows(config, lock)
    winner = select_joint_winner(method_rows)
    method_names = tuple(sorted(str(row["method"]) for row in method_rows))
    payload = {
        "schema": "vector-jepa-p7-selection-v1",
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_results_viewed": True,
        "confirmatory_results_viewed": False,
        "action_selection": config.protocol.primary_action_selection,
        "near_optimal_sr_tolerance": NEAR_OPTIMAL_SR_TOLERANCE,
        "selection_rule": [
            "require all 40 nested checkpoints to pass the 10pct JEPA stability gate",
            "retain cells within 0.01 corrected validation SR of the stable maximum",
            "minimize maximum then mean JEPA relative degradation",
            "prefer higher size-19/21 SR, then lexicographic method name",
        ],
        "method_metrics": method_rows,
        "validation_artifacts": validation_artifact_digest(
            config, method_names=method_names
        ),
        "track_j_failed": winner is None,
        "selected_track_j": str(winner["method"]) if winner is not None else None,
    }
    atomic_json_dump(output, payload)


if __name__ == "__main__":
    main()
