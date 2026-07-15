"""Resolve result-dependent Q2 templates without changing their scientific role."""

from __future__ import annotations

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import method_by_name
from vector_jepa_planner_frontier.schemas import MethodConfig
from vector_jepa_planner_full900_screen.common import (
    load_json,
    resolve_path,
    result_path,
    role_by_name,
)
from vector_jepa_planner_full900_screen.parity import validate_q0_gate
from vector_jepa_planner_full900_screen.schemas import QuickStudyConfig


def _validate_input_result_hashes(
    config: QuickStudyConfig, inputs: dict[object, object]
) -> None:
    for raw_key, expected in inputs.items():
        parts = str(raw_key).split(":")
        if len(parts) == 2:
            method, action = parts
            backbone_seed, planner_seed = 42, 0
        elif len(parts) == 4 and parts[1].startswith("b") and parts[2].startswith("p"):
            method, action = parts[0], parts[3]
            backbone_seed = int(parts[1][1:])
            planner_seed = int(parts[2][1:])
        else:
            raise ValueError(f"invalid decision input key: {raw_key}")
        if action not in config.replication.action_selections:
            raise ValueError(f"invalid action protocol in decision input: {raw_key}")
        configured_method = method_by_name(config, method)
        expected_planner_seed = (
            config.replication.screen_planner_seeds[0]
            if configured_method.component_checkpoint_required
            else 0
        )
        if (
            backbone_seed not in config.replication.final_backbone_seeds
            or planner_seed != expected_planner_seed
        ):
            raise ValueError(f"invalid seed labels in decision input: {raw_key}")
        path = result_path(
            config,
            method=method,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
            action_selection=action,
        )
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"decision input hash mismatch: {path}")


def validate_q1_selection(
    config: QuickStudyConfig, lock: dict[str, object]
) -> dict[str, object]:
    path = resolve_path(config.paths.p2_selection)
    if not path.exists():
        raise FileNotFoundError("Q1 parent is not frozen; run freeze_q1 first")
    value = load_json(path)
    expected = {
        "schema": "vector-jepa-full900-q1-parent-v1",
        "protocol_id": config.protocol_id,
        "quick_spec_sha256": lock["quick_spec_sha256"],
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError("Q1 parent record belongs to another protocol")
    if value.get("q0_parity_sha256s") != validate_q0_gate(config, lock):
        raise ValueError("Q1 parent record does not bind the current Q0 parity")
    selected = str(value.get("selected_parent"))
    allowed = {
        "q1_control_categorical_cem_1x",
        "q1_icem_1x",
        "q1_beam_1x",
        "q1_best_first_1x",
        "q1_mcts_1x",
    }
    if selected not in allowed:
        raise ValueError("Q1 selected an invalid scorer-compatible parent")
    ranked = value.get("ranked_candidates")
    if (
        not isinstance(ranked, list)
        or {str(row.get("method")) for row in ranked if isinstance(row, dict)}
        != allowed
    ):
        raise ValueError("Q1 decision does not contain the complete parent ranking")
    if value.get("categorical_bridge_exact_task_parity") is not True:
        raise ValueError("Q1 decision lacks categorical-CEM bridge parity")
    inputs = value.get("input_sha256s")
    if not isinstance(inputs, dict) or len(inputs) != 12:
        raise ValueError("Q1 decision does not hash all twelve full-900 inputs")
    _validate_input_result_hashes(config, inputs)
    return value


def validate_shortlist(
    config: QuickStudyConfig, lock: dict[str, object]
) -> dict[str, object]:
    path = resolve_path(config.paths.p5_advancement)
    if not path.exists():
        raise FileNotFoundError("shortlist is not frozen; run freeze_shortlist first")
    value = load_json(path)
    expected = {
        "schema": "vector-jepa-full900-shortlist-v1",
        "protocol_id": config.protocol_id,
        "quick_spec_sha256": lock["quick_spec_sha256"],
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError("shortlist record belongs to another protocol")
    if value.get("q1_parent_sha256") != sha256_file(
        resolve_path(config.paths.p2_selection)
    ):
        raise ValueError("shortlist record does not match the frozen Q1 parent")
    eligible = {role.name for role in config.method_roles if role.advancement_eligible}
    shortlist = value.get("shortlist")
    if not isinstance(shortlist, list):
        raise ValueError("shortlist must be a list")
    names = [str(name) for name in shortlist]
    if (
        len(names) != len(set(names))
        or len(names) > config.gates.max_shortlist_size
        or not set(names) <= eligible
    ):
        raise ValueError("shortlist contains an ineligible or duplicate method")
    audits = value.get("candidate_audits")
    if (
        not isinstance(audits, list)
        or {str(row.get("method")) for row in audits if isinstance(row, dict)}
        != eligible
    ):
        raise ValueError("shortlist record lacks the complete candidate audit")
    inputs = value.get("input_sha256s")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("shortlist record lacks input result hashes")
    _validate_input_result_hashes(config, inputs)
    return value


def validate_final_selection(
    config: QuickStudyConfig, lock: dict[str, object]
) -> dict[str, object]:
    shortlist_record = validate_shortlist(config, lock)
    path = resolve_path(config.paths.p7_selection)
    if not path.exists():
        raise FileNotFoundError("final winner is not frozen; run freeze_final first")
    value = load_json(path)
    expected = {
        "schema": "vector-jepa-full900-final-winner-v1",
        "protocol_id": config.protocol_id,
        "quick_spec_sha256": lock["quick_spec_sha256"],
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError("final-winner record belongs to another protocol")
    if value.get("shortlist_sha256") != sha256_file(
        resolve_path(config.paths.p5_advancement)
    ):
        raise ValueError("final record does not match the frozen shortlist")
    shortlist = {str(name) for name in shortlist_record["shortlist"]}
    winner = value.get("winner")
    if winner is not None and str(winner) not in shortlist:
        raise ValueError("final winner was not in the frozen shortlist")
    audits = value.get("candidate_audits")
    if (
        not isinstance(audits, list)
        or {str(row.get("method")) for row in audits if isinstance(row, dict)}
        != shortlist
    ):
        raise ValueError("final record lacks the complete shortlist audit")
    inputs = value.get("input_sha256s")
    if not isinstance(inputs, dict) or (shortlist and not inputs):
        raise ValueError("final record lacks input result hashes")
    _validate_input_result_hashes(config, inputs)
    return value


def selected_q1_parent(config: QuickStudyConfig, lock: dict[str, object]) -> str:
    return str(validate_q1_selection(config, lock)["selected_parent"])


def effective_method(
    config: QuickStudyConfig,
    lock: dict[str, object],
    method: MethodConfig | str,
) -> MethodConfig:
    base = method_by_name(config, method) if isinstance(method, str) else method
    role = role_by_name(config, base.name)
    if role.parent == "fixed":
        return base
    parent_name = (
        selected_q1_parent(config, lock)
        if role.parent == "q1_winner"
        else "q1_best_first_1x"
    )
    parent = method_by_name(config, parent_name)
    selection_digest = (
        sha256_file(resolve_path(config.paths.p2_selection))
        if role.parent == "q1_winner"
        else "fixed-q1-best-first-parent"
    )
    decisions = tuple(
        dict.fromkeys((*base.effective_decision_sha256s, selection_digest))
    )
    return base.model_copy(
        update={
            "planner": parent.planner,
            "effective_decision_sha256s": decisions,
        }
    )


def direct_control_name(
    config: QuickStudyConfig, lock: dict[str, object], method_name: str
) -> str:
    role = role_by_name(config, method_name)
    return (
        selected_q1_parent(config, lock)
        if role.direct_control == "__q1_winner__"
        else role.direct_control
    )


__all__ = [
    "direct_control_name",
    "effective_method",
    "selected_q1_parent",
    "validate_final_selection",
    "validate_q1_selection",
    "validate_shortlist",
]
