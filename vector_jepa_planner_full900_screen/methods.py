"""Resolve result-dependent Q2 templates without changing their scientific role."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import method_by_name
from vector_jepa_planner_frontier.schemas import MethodConfig
from vector_jepa_planner_full900_screen.analysis import screen_planner_seed
from vector_jepa_planner_full900_screen.common import (
    component_checkpoint_path,
    load_json,
    resolve_path,
    result_path,
    role_by_name,
)
from vector_jepa_planner_full900_screen.parity import validate_q0_gate
from vector_jepa_planner_full900_screen.schemas import QuickStudyConfig

COMPONENT_PARITY_GROUPS: dict[str, tuple[str, ...]] = {
    "q2b_vector_dts": (
        "q2b_vector_dts",
        "q2b_control_dts_uniform_expansion",
        "q2b_control_dts_direct",
    ),
    "q2b_bidirectional": (
        "q2b_bidirectional",
        "q2b_control_bidirectional_forward",
    ),
}


def _hash_tree(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value):
            _hash_tree(digest, str(key))
            _hash_tree(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_tree(digest, item)
        return
    digest.update(b"scalar\0")
    digest.update(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _scientific_component_sha256(checkpoint: dict[str, Any]) -> str:
    training_summary = dict(checkpoint.get("training_summary", {}))
    training_summary.pop("elapsed_seconds", None)
    payload = {
        "source_checkpoint_sha256": checkpoint.get("source_checkpoint_sha256"),
        "train_manifest_sha256": checkpoint.get("train_manifest_sha256"),
        "validation_manifest_sha256": checkpoint.get("validation_manifest_sha256"),
        "head_config": checkpoint.get("head_config"),
        "head_state_dicts": checkpoint.get("head_state_dicts"),
        "training_summary": training_summary,
        "validation_metrics": checkpoint.get("validation_metrics"),
        "initialization_parent": checkpoint.get("initialization_parent"),
        "joint_counterexample_provenance": checkpoint.get(
            "joint_counterexample_provenance"
        ),
    }
    digest = hashlib.sha256()
    _hash_tree(digest, payload)
    return digest.hexdigest()


def validate_component_parity(
    config: QuickStudyConfig,
    *,
    candidate: str,
    backbone_seed: int,
    planner_seed: int,
    include_secondary_controls: bool = False,
) -> dict[str, Any]:
    group = COMPONENT_PARITY_GROUPS.get(candidate)
    if group is None:
        return {}
    names = group if include_secondary_controls else group[:2]
    digests: dict[str, str] = {}
    checkpoint_sha256s: dict[str, str] = {}
    for name in names:
        method = method_by_name(config, name)
        path = component_checkpoint_path(
            config,
            method,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
        )
        if not path.exists():
            raise FileNotFoundError(f"missing component parity checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("stage") != "component_calibration"
            or checkpoint.get("method_name") != name
            or int(checkpoint.get("backbone_seed", -1)) != backbone_seed
            or int(checkpoint.get("planner_seed", -1)) != planner_seed
        ):
            raise ValueError(f"component parity checkpoint metadata mismatch: {path}")
        digests[name] = _scientific_component_sha256(checkpoint)
        checkpoint_sha256s[name] = sha256_file(path)
    if len(set(digests.values())) != 1:
        raise ValueError(
            f"shared learned components diverged inside {candidate}: {digests}"
        )
    return {
        "candidate": candidate,
        "methods": list(names),
        "backbone_seed": backbone_seed,
        "planner_seed": planner_seed,
        "scientific_component_sha256": next(iter(digests.values())),
        "checkpoint_sha256s": dict(sorted(checkpoint_sha256s.items())),
        "status": "exact_match",
    }


def component_parity_audits(
    config: QuickStudyConfig,
    *,
    candidates: list[str] | tuple[str, ...] | set[str],
    backbone_seeds: tuple[int, ...],
    planner_seeds: tuple[int, ...],
    include_dts_secondary_control: bool = False,
) -> list[dict[str, Any]]:
    audits = []
    for candidate in sorted(set(candidates).intersection(COMPONENT_PARITY_GROUPS)):
        for backbone_seed in backbone_seeds:
            for planner_seed in planner_seeds:
                audits.append(
                    validate_component_parity(
                        config,
                        candidate=candidate,
                        backbone_seed=backbone_seed,
                        planner_seed=planner_seed,
                        include_secondary_controls=(
                            include_dts_secondary_control
                            and candidate == "q2b_vector_dts"
                        ),
                    )
                )
    return audits


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
    expected_input_keys = {
        f"{name}:{action}"
        for name in {"b0_legacy_l2_cem", *allowed}
        for action in config.replication.action_selections
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_input_keys:
        raise ValueError("Q1 decision does not hash all twelve full-900 inputs")
    expected_ranking = sorted(
        ranked,
        key=lambda row: (
            -float(row["corrected_sr"]),
            -float(row["corrected_ood_sr"]),
            -float(row["unmasked_sr"]),
            float(row["planner_forward_calls_per_decision"]),
            str(row["method"]),
        ),
    )
    if ranked != expected_ranking or selected != str(ranked[0]["method"]):
        raise ValueError("Q1 decision violates its locked deterministic ranking")
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
    expected_component_parity = component_parity_audits(
        config,
        candidates=("q2b_vector_dts", "q2b_bidirectional"),
        backbone_seeds=(42,),
        planner_seeds=(config.replication.screen_planner_seeds[0],),
        include_dts_secondary_control=True,
    )
    if value.get("component_parity_audits") != expected_component_parity:
        raise ValueError("shortlist shared-component parity evidence mismatch")
    inputs = value.get("input_sha256s")
    expected_methods = {"b0_legacy_l2_cem"}
    for role in config.method_roles:
        if not role.advancement_eligible:
            continue
        method = effective_method(config, lock, role.name)
        expected_methods.add(method.name)
        expected_methods.add(direct_control_name(config, lock, method.name))
    expected_input_keys = {
        (
            f"{name}:b42:p"
            f"{screen_planner_seed(effective_method(config, lock, name))}:"
            f"{action}"
        )
        for name in expected_methods
        for action in config.replication.action_selections
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_input_keys:
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
    expected_component_parity = component_parity_audits(
        config,
        candidates=shortlist,
        backbone_seeds=config.replication.expansion_backbone_seeds,
        planner_seeds=(config.replication.screen_planner_seeds[0],),
    )
    if value.get("component_parity_audits") != expected_component_parity:
        raise ValueError("final shared-component parity evidence mismatch")
    inputs = value.get("input_sha256s")
    expected_input_keys: set[str] = set()
    if shortlist:
        expected_methods = {"b0_legacy_l2_cem"}
        for name in shortlist:
            expected_methods.add(name)
            expected_methods.add(direct_control_name(config, lock, name))
        expected_input_keys = {
            (
                f"{name}:b{seed}:p"
                f"{screen_planner_seed(effective_method(config, lock, name))}:"
                f"{action}"
            )
            for name in expected_methods
            for seed in config.replication.expansion_backbone_seeds
            for action in config.replication.action_selections
        }
    if not isinstance(inputs, dict) or set(inputs) != expected_input_keys:
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
    "COMPONENT_PARITY_GROUPS",
    "component_parity_audits",
    "direct_control_name",
    "effective_method",
    "selected_q1_parent",
    "validate_component_parity",
    "validate_final_selection",
    "validate_q1_selection",
    "validate_shortlist",
]
