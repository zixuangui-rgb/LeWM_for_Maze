"""Fail-closed authorization and provenance checks for the one-shot test set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    load_json,
    method_by_name,
    planner_seed_values,
    resolve_path,
)


def load_confirmation_artifacts(
    config: Any, lock: dict[str, Any], *, require_opened: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    confirmation_path = resolve_path(config.paths.confirmation_lock)
    mapping_path = resolve_path(config.paths.confirmation_mapping)
    schedule_path = resolve_path(config.paths.confirmation_schedule)
    for path in (confirmation_path, mapping_path, schedule_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    confirmation = load_json(confirmation_path)
    mapping = load_json(mapping_path)
    schedule = load_json(schedule_path)
    if confirmation.get("schema") != "vector-jepa-confirmation-lock-v1":
        raise ValueError("unknown confirmation-lock schema")
    if confirmation.get("status") != "frozen_unopened":
        raise RuntimeError("confirmation lock is not in the frozen state")
    analysis_hash = analysis_spec_sha256(config, lock)
    if confirmation.get("analysis_spec_sha256") != analysis_hash:
        raise ValueError("confirmation lock belongs to another analysis")
    if confirmation.get("base_protocol_lock_sha256") != sha256_file(
        resolve_path(config.paths.protocol_lock)
    ):
        raise ValueError("confirmation lock references another base lock")
    for key, configured_path in (
        ("power_record", config.paths.confirmation_power),
        ("p8_selection", config.paths.p8_selection),
    ):
        record = confirmation.get(key, {})
        path = resolve_path(configured_path)
        if (
            Path(str(record.get("path", ""))) != path
            or not path.is_file()
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"confirmation lock has a stale {key} artifact")
    if confirmation.get("mapping_sha256") != sha256_file(mapping_path):
        raise ValueError("private confirmation mapping hash mismatch")
    if confirmation.get("schedule_sha256") != sha256_file(schedule_path):
        raise ValueError("public confirmation schedule hash mismatch")
    if mapping.get("analysis_spec_sha256") != analysis_hash:
        raise ValueError("private confirmation mapping analysis mismatch")
    if schedule.get("analysis_spec_sha256") != analysis_hash:
        raise ValueError("public confirmation schedule analysis mismatch")
    rows = mapping.get("runs")
    public_rows = schedule.get("runs")
    if not isinstance(rows, list) or not isinstance(public_rows, list):
        raise ValueError("confirmation schedules must contain run lists")
    if [row.get("run_id") for row in rows] != [
        row.get("run_id") for row in public_rows
    ]:
        raise ValueError("private and public confirmation schedules disagree")
    if len(rows) != int(confirmation.get("run_count", -1)):
        raise ValueError("confirmation schedule is incomplete")
    p8_selection = load_json(config.paths.p8_selection)
    power = load_json(config.paths.confirmation_power)
    if confirmation.get("methods") != p8_selection.get("confirmation_methods"):
        raise ValueError("confirmation methods differ from the frozen P8 decision")
    if confirmation.get("track_f_winner") != p8_selection.get("selected_track_f"):
        raise ValueError("confirmation Track F differs from the frozen P8 winner")
    if confirmation.get("track_j_winner") != p8_selection.get("selected_track_j"):
        raise ValueError("confirmation Track J differs from the frozen P8 decision")
    comparison_count = int(p8_selection.get("comparison_count", -1))
    if comparison_count != int(power.get("comparison_count", -2)):
        raise ValueError("confirmation multiplicity differs from P8 or power")
    if confirmation.get("track_j_included") is not (comparison_count == 4):
        raise ValueError("confirmation Track J inclusion flag is inconsistent")
    primary_family = confirmation.get("primary_family")
    if not isinstance(primary_family, list) or len(primary_family) not in (2, 4):
        raise ValueError("confirmation lock has an invalid primary family")
    if len(primary_family) != int(
        confirmation.get("power_record", {}).get("comparison_count", -1)
    ):
        raise ValueError("confirmation family and power multiplicity disagree")
    expected_family = [
        {
            "hypothesis": "H1",
            "candidate": p8_selection["selected_track_f"],
            "baseline": "b0_legacy_l2_cem",
            "subset": subset,
        }
        for subset in ("overall", "ood")
    ]
    if comparison_count == 4:
        expected_family.extend(
            {
                "hypothesis": "H2",
                "candidate": p8_selection["selected_track_j"],
                "baseline": p8_selection["selected_track_f"],
                "subset": subset,
            }
            for subset in ("overall", "ood")
        )
    if primary_family != expected_family:
        raise ValueError("confirmation primary family differs from P8")

    methods = [
        method_by_name(config, name) for name in p8_selection["confirmation_methods"]
    ]
    backbones = tuple(int(seed) for seed in confirmation.get("backbone_seeds", []))
    required_backbones = int(power.get("required_backbones", -1))
    if (
        required_backbones < 1
        or required_backbones > len(config.protocol.training_seeds)
        or backbones != tuple(config.protocol.training_seeds[:required_backbones])
        or confirmation.get("planner_seeds") != list(config.protocol.planner_seeds)
        or confirmation.get("search_seeds") != list(config.protocol.search_seeds)
    ):
        raise ValueError("confirmation nested seed matrix drifted")
    expected_runs = {
        (
            method.name,
            int(backbone_seed),
            int(planner_seed),
            int(search_seed),
            action_selection,
        )
        for method in methods
        for backbone_seed in backbones
        for planner_seed in planner_seed_values(config, method)
        for search_seed in config.protocol.search_seeds
        for action_selection in ("unmasked", "corrected_v1")
    }
    actual_runs = [
        (
            str(row.get("method")),
            int(row.get("backbone_seed", -1)),
            int(row.get("planner_seed", -1)),
            int(row.get("search_seed", -1)),
            str(row.get("action_selection")),
        )
        for row in rows
    ]
    if len(actual_runs) != len(set(actual_runs)) or set(actual_runs) != expected_runs:
        raise ValueError("opaque confirmation mapping is not the exact run matrix")
    for key in ("run_id", "opaque_output", "formal_output"):
        values = [str(row.get(key, "")) for row in rows]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError(f"opaque confirmation mapping has duplicate {key}")
    if resolve_path(config.paths.confirmation_unblinded).exists():
        raise RuntimeError("confirmatory family is already unblinded and closed")
    if require_opened:
        opened_path = resolve_path(config.paths.confirmation_opened)
        if not opened_path.is_file():
            raise RuntimeError("confirmatory execution has not been opened")
        opened = load_json(opened_path)
        if opened.get("confirmation_lock_sha256") != sha256_file(confirmation_path):
            raise ValueError("confirmation-opened marker references another lock")
    return confirmation, mapping, schedule


def confirmation_row(
    config: Any,
    lock: dict[str, Any],
    *,
    run_id: str,
    require_opened: bool,
) -> dict[str, Any]:
    _, mapping, _ = load_confirmation_artifacts(
        config, lock, require_opened=require_opened
    )
    rows = [row for row in mapping["runs"] if row.get("run_id") == run_id]
    if len(rows) != 1:
        raise ValueError(f"unknown or duplicate opaque run id: {run_id}")
    return dict(rows[0])


def authorize_confirmatory_evaluation(
    config: Any,
    lock: dict[str, Any],
    *,
    method: Any,
    backbone_seed: int,
    planner_seed: int,
    search_seed: int,
    action_selection: str,
    output: Path,
    component_checkpoint: Path | None,
) -> str:
    confirmation, mapping, _ = load_confirmation_artifacts(
        config, lock, require_opened=True
    )
    expected = {
        "method": method.name,
        "backbone_seed": int(backbone_seed),
        "planner_seed": int(planner_seed),
        "search_seed": int(search_seed),
        "action_selection": action_selection,
        "opaque_output": str(output),
    }
    rows = [
        row
        for row in mapping["runs"]
        if all(row.get(key) == value for key, value in expected.items())
    ]
    if len(rows) != 1:
        raise PermissionError(
            "confirmatory evaluation is not an exact opaque-schedule entry"
        )
    row = rows[0]
    if Path(str(row["formal_output"])) == output:
        raise PermissionError("confirmatory evaluation cannot write a named result")
    source = confirmation.get("source_checkpoints", {}).get(str(backbone_seed), {})
    source_path = Path(str(source.get("path", "")))
    if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
        raise ValueError("frozen source checkpoint hash mismatch")
    key = f"{method.name}:{backbone_seed}:{planner_seed}"
    frozen_components = confirmation.get("component_checkpoints", {})
    if component_checkpoint is None:
        if key in frozen_components:
            raise ValueError("headless method unexpectedly has a frozen component")
    else:
        record = frozen_components.get(key, {})
        if Path(str(record.get("path", ""))) != component_checkpoint:
            raise ValueError("confirmatory component path differs from the freeze")
        if not component_checkpoint.is_file() or sha256_file(
            component_checkpoint
        ) != record.get("sha256"):
            raise ValueError("frozen component checkpoint hash mismatch")
        retrieval = record.get("retrieval_bank")
        if retrieval is not None:
            retrieval_path = Path(str(retrieval.get("path", "")))
            if not retrieval_path.is_file() or sha256_file(
                retrieval_path
            ) != retrieval.get("sha256"):
                raise ValueError("frozen retrieval bank hash mismatch")
    return str(row["run_id"])


__all__ = [
    "authorize_confirmatory_evaluation",
    "confirmation_row",
    "load_confirmation_artifacts",
]
