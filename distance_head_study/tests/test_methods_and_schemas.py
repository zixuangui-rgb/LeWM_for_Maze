from __future__ import annotations

import json
from pathlib import Path

import pytest

from distance_head_study import ACTION_IDS, MODEL_ACTION_VOCAB_SIZE
from distance_head_study.common import (
    canonical_json_sha256,
    load_study_config,
    sha256_file,
)
from distance_head_study.lock_confirmation_n import _confirmation_selection
from distance_head_study.make_decision import (
    _expand_with_locked_incumbent,
    _validate_decision_request,
)
from distance_head_study.methods import resolve_method, validate_static_catalog
from distance_head_study.schemas import CostKind, TrainingScope
from distance_head_study.taxonomy import (
    MAIN_CANDIDATES,
    NEGATIVE_CLOSURE_CANDIDATES,
    negative_route_group,
    strongest_negative_pair,
)


def test_study_config_locks_split_seed_and_budget_contracts() -> None:
    config = load_study_config("distance_head_study/configs/default.json")
    assert config.splits.train_sizes == (9, 11, 13, 15, 17, 19, 21)
    assert config.seeds.screen_backbones == (42,)
    assert config.seeds.select_backbones == (42, 43, 44)
    assert config.seeds.ordered_confirmation_backbones[0] == 1001
    assert not set(config.seeds.historical_backbones) & set(
        config.seeds.ordered_confirmation_backbones
    )
    assert config.planner.horizon == 12
    assert config.planner.num_candidates == 64
    assert config.planner.cem_iters == 1
    assert config.planner.action_ids == ACTION_IDS == (1, 2, 3, 4)
    assert config.planner.model_action_vocab_size == MODEL_ACTION_VOCAB_SIZE == 5
    assert MODEL_ACTION_VOCAB_SIZE == max(ACTION_IDS) + 1
    assert config.training.checkpoint_selection == "final_step"


def test_every_method_resolves_and_declared_diffs_are_exact(
    method_catalog, decision_root
) -> None:
    audit = validate_static_catalog(method_catalog)
    assert audit == {
        "method_count": 45,
        "root_count": 10,
        "static_derived_count": 5,
        "dynamic_count": 30,
    }
    resolved = {}
    for template in method_catalog.methods:
        method, decision_hashes = resolve_method(
            method_catalog,
            template.name,
            decision_root=decision_root,
        )
        resolved[method.name] = method
        if template.parent and template.parent.startswith("@"):
            assert decision_hashes
    assert resolved["c2_dual_calibration"].head.domain_adapter
    assert resolved["c1_predicted_listwise"].head.domain_adapter is False
    assert resolved["p_path_integrated"].reuse_parent_checkpoint
    assert resolved["p_path_integrated"].planner.cost == CostKind.PATH_INTEGRATED
    assert resolved["j1_dist_projector"].training_scope == (
        TrainingScope.PROJECTOR_PREDICTOR
    )
    for control, treatment in (
        ("j0_cont_predictor", "j0_dist_predictor"),
        ("j1_cont_projector", "j1_dist_projector"),
        ("j2_cont_full", "j2_dist_full"),
    ):
        assert resolved[control].update_head
        assert resolved[control].objectives == resolved[treatment].objectives
        assert not resolved[control].distance_gradients_to_backbone
        assert resolved[treatment].distance_gradients_to_backbone


def test_dynamic_parent_fails_if_evidence_changes(
    method_catalog, decision_root
) -> None:
    evidence = decision_root / "immutable_evidence.txt"
    evidence.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency changed"):
        resolve_method(
            method_catalog,
            "a2_distance_balanced",
            decision_root=decision_root,
        )


def test_dynamic_parent_must_belong_to_current_protocol(
    method_catalog, decision_root
) -> None:
    with pytest.raises(ValueError, match="another protocol lock"):
        resolve_method(
            method_catalog,
            "a2_distance_balanced",
            decision_root=decision_root,
            protocol_lock={
                "analysis_spec_sha256": "analysis",
                "protocol_lock_sha256": "protocol",
            },
        )


def test_dynamic_parent_fails_if_decision_name_is_spoofed(
    method_catalog, decision_root
) -> None:
    path = decision_root / "a_target_parent.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decision_name"] = "wrong_name"
    from distance_head_study.common import canonical_json_sha256

    payload["decision_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "decision_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="name/path mismatch"):
        resolve_method(
            method_catalog,
            "a2_distance_balanced",
            decision_root=decision_root,
        )


def test_negative_closure_taxonomy_is_complete_and_stratified() -> None:
    assert set(MAIN_CANDIDATES) < set(NEGATIVE_CLOSURE_CANDIDATES)
    assert {negative_route_group(method) for method in NEGATIVE_CLOSURE_CANDIDATES} == {
        "frozen_scorer",
        "system_or_planner",
    }
    pair = strongest_negative_pair(
        ["a2_distance_balanced", "a3_full_horizon", "p_beam", "j1_dist_projector"]
    )
    assert pair == ("a2_distance_balanced", "p_beam")


def test_screen_selection_requires_every_preregistered_main_candidate() -> None:
    _validate_decision_request(
        None,
        {},
        name="screen_selection",
        criterion="screen",
        eligible=MAIN_CANDIDATES,
        baseline="b_dh_cem",
        split_role="screen",
        backbones=(42,),
        heads=(0, 1, 2),
    )
    with pytest.raises(ValueError, match="complete preregistered main set"):
        _validate_decision_request(
            None,
            {},
            name="screen_selection",
            criterion="screen",
            eligible=MAIN_CANDIDATES[:-1],
            baseline="b_dh_cem",
            split_role="screen",
            backbones=(42,),
            heads=(0, 1, 2),
        )


def test_sequential_decision_keeps_signed_upstream_incumbent(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("locked\n", encoding="utf-8")
    payload = {
        "protocol_id": "procgen-maze-distance-head-staged-v1",
        "decision_name": "a_target_parent",
        "analysis_spec_sha256": "analysis",
        "protocol_lock_sha256": "protocol",
        "selected_method": "b_dh_cem",
        "eligible_methods": ["b_dh_cem", "a1_log"],
        "input_hashes": {evidence.as_posix(): sha256_file(evidence)},
    }
    payload["decision_sha256"] = canonical_json_sha256(payload)
    decision = tmp_path / "a_target_parent.json"
    decision.write_text(json.dumps(payload), encoding="utf-8")
    config = load_study_config("distance_head_study/configs/default.json")
    config = config.model_copy(
        update={"paths": config.paths.model_copy(update={"decision_root": tmp_path})}
    )
    eligible, hashes, incumbent = _expand_with_locked_incumbent(
        config,
        {
            "analysis_spec_sha256": "analysis",
            "protocol_lock_sha256": "protocol",
        },
        decision_name="a_sampling_parent",
        candidates=("a2_distance_balanced", "a3_full_horizon"),
    )
    assert eligible == (
        "b_dh_cem",
        "a2_distance_balanced",
        "a3_full_horizon",
    )
    assert hashes == {
        decision.as_posix(): sha256_file(decision),
        evidence.as_posix(): sha256_file(evidence),
    }
    assert incumbent is not None and incumbent["selected_method"] == "b_dh_cem"


def test_failed_seed3_resolves_to_separate_negative_fallback(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("locked\n", encoding="utf-8")
    lock = {
        "analysis_spec_sha256": "analysis",
        "protocol_lock_sha256": "protocol",
    }

    def write_signed(name: str, payload: dict, signature: str) -> Path:
        path = tmp_path / name
        value = {
            "protocol_id": "procgen-maze-distance-head-staged-v1",
            **lock,
            "input_hashes": {evidence.as_posix(): sha256_file(evidence)},
            **payload,
        }
        value[signature] = canonical_json_sha256(value)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    primary_shortlist = write_signed(
        "shortlist.json",
        {
            "selected_methods": ["d2_trm_full", "d4_reachability"],
            "negative_closure_sha256": None,
        },
        "shortlist_sha256",
    )
    primary_finalist = write_signed(
        "finalist_lock.json",
        {
            "decision_name": "finalist_lock",
            "eligible_methods": ["d2_trm_full", "d4_reachability"],
            "selected_method": "d2_trm_full",
            "ranked_methods": ["d2_trm_full", "d4_reachability"],
            "metrics": {"d2_trm_full": {"seed10_expansion_pass": False}},
        },
        "decision_sha256",
    )
    closure_decision = write_signed(
        "closure_selection.json",
        {
            "decision_name": "closure_selection",
            "eligible_methods": ["a2_distance_balanced", "p_beam"],
            "selected_method": "a2_distance_balanced",
            "ranked_methods": ["a2_distance_balanced", "p_beam"],
        },
        "decision_sha256",
    )
    closure_payload = json.loads(closure_decision.read_text(encoding="utf-8"))
    primary_shortlist_payload = json.loads(
        primary_shortlist.read_text(encoding="utf-8")
    )
    primary_finalist_payload = json.loads(primary_finalist.read_text(encoding="utf-8"))
    fallback_shortlist = write_signed(
        "negative_shortlist.json",
        {
            "selected_methods": ["a2_distance_balanced", "p_beam"],
            "negative_closure_sha256": "closure",
            "negative_fallback_after_seed3": True,
            "selection_decision_path": closure_decision.as_posix(),
            "screen_decision_sha256": closure_payload["decision_sha256"],
            "prior_shortlist_sha256": primary_shortlist_payload["shortlist_sha256"],
            "prior_finalist_decision_sha256": primary_finalist_payload[
                "decision_sha256"
            ],
        },
        "shortlist_sha256",
    )
    config = load_study_config("distance_head_study/configs/default.json")
    config = config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={
                    "decision_root": tmp_path,
                    "shortlist_lock": primary_shortlist,
                    "negative_shortlist_lock": fallback_shortlist,
                }
            )
        }
    )
    finalist_path, _, shortlist_path, _, source = _confirmation_selection(
        config, lock, "negative"
    )
    assert finalist_path == closure_decision
    assert shortlist_path == fallback_shortlist
    assert source == "screen_closure_fallback"
