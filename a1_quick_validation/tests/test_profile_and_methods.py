from __future__ import annotations

import copy

import pytest

from a1_quick_validation import ALL_METHODS, NEW_METHODS, REFERENCE_METHODS
from a1_quick_validation.audit import run_audit
from a1_quick_validation.common import DEFAULT_PROFILE, load_json
from a1_quick_validation.profile import load_profile, verify_reproduction_contract
from a1_quick_validation.release_seed_tier import expected_seed_matrix
from a1_quick_validation.schemas import QuickProfile
from distance_head_study.common import (
    load_method_catalog,
    load_study_config,
)
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.protocol import verify_protocol_lock


def test_profile_locks_exact_phase_matrices() -> None:
    profile = load_profile()
    assert profile.q1.methods == ALL_METHODS
    assert profile.q1.backbone_seeds == (42,)
    assert profile.q1.head_seeds == (0,)
    assert profile.q2.head_seeds == (0, 1)
    assert profile.q2.action_protocols == ("corrected_v1", "unmasked")
    assert profile.q3.split_role == "legacy"
    assert profile.q1.dynamic_method_source == "none"
    assert profile.q2.dynamic_method_source == "q1_shortlist"
    assert profile.q3.dynamic_method_source == "q2_winner"


def test_quick_seed_release_scope_is_narrower_than_parent_study() -> None:
    profile = load_profile()
    source = load_study_config(profile.paths.source_config)
    assert expected_seed_matrix(profile, "seed1") == ((42,), (0,))
    assert expected_seed_matrix(profile, "seed3") == ((42,), (0, 1))
    assert source.seeds.screen_head_seeds == (0, 1, 2)
    assert source.seeds.select_backbones == (42, 43, 44)


def test_reproduction_contract_binds_historical_and_repaired_layers() -> None:
    profile = load_profile()
    contract = verify_reproduction_contract(profile)
    assert contract["lineage_layers"] == {
        "historical_raw_reproduction": "traceability_only_not_formal_comparator",
        "corrected_vector_anchor": "final_closure",
        "immediate_distance_head_comparator": "distance_head_study",
        "quick_treatments": "a1_quick_validation",
    }
    assert contract["quick_comparison_contract"]["bootstrap_task_resampling"] == (
        "paired_by_task_id_within_maze_size"
    )
    assert (
        contract["deliberate_nonidentity_with_raw_history"]["checkpoint_selection"][
            "formal"
        ]
        == "final_step"
    )


def test_quick_audit_explicitly_covers_every_heldout_role() -> None:
    audit = run_audit()
    assert audit["calibration_layouts_are_train_subset"] is True
    assert audit["heldout_train_overlap"] == {
        "screen": {"layout": 0, "task": 0},
        "select": {"layout": 0, "task": 0},
        "legacy": {"layout": 0, "task": 0},
        "confirm": {"layout": 0, "task": 0},
        "stress": {"layout": 0, "task": 0},
    }
    assert {
        value
        for overlap in audit["heldout_pair_overlap"].values()
        for value in overlap.values()
    } == {0}


def test_profile_rejects_quick_budget_drift() -> None:
    payload = copy.deepcopy(load_json(DEFAULT_PROFILE))
    payload["q1"]["head_seeds"] = [0, 1]
    with pytest.raises(ValueError, match="q1 execution matrix"):
        QuickProfile.model_validate(payload)


def test_quick_config_matches_every_locked_source_section() -> None:
    profile = load_profile()
    source = load_study_config(profile.paths.source_config)
    quick = load_study_config(profile.paths.quick_config)
    for name in ("splits", "seeds", "planner", "training", "analysis"):
        assert getattr(source, name) == getattr(quick, name)


def test_catalog_has_exact_methods_and_direct_parents() -> None:
    profile = load_profile()
    catalog = load_method_catalog("a1_quick_validation/configs/methods.json")
    assert tuple(method.name for method in catalog.methods) == ALL_METHODS
    parents = {method.name: method.parent for method in catalog.methods}
    assert parents["a1_bellman"] == "a1_log"
    assert parents["a1_predicted"] == "a1_log"
    assert parents["a1_hcond"] == "a1_log"
    assert parents["a1_reach"] == "a1_hcond"
    quick = load_study_config(profile.paths.quick_config)
    lock = verify_protocol_lock(quick)
    for method_name in NEW_METHODS:
        method, _, decisions = load_and_resolve_method(
            quick.paths.method_catalog,
            method_name,
            decision_root=quick.paths.decision_root,
            protocol_lock=lock,
        )
        assert not decisions
        assert method.training_scope.value == "frozen"
        assert not method.uses_test_bfs


@pytest.mark.parametrize("method_name", REFERENCE_METHODS)
def test_reference_method_object_and_hash_match_source(method_name: str) -> None:
    profile = load_profile()
    source = load_study_config(profile.paths.source_config)
    quick = load_study_config(profile.paths.quick_config)
    source_lock = verify_protocol_lock(source)
    quick_lock = verify_protocol_lock(quick)
    source_method, source_hash, source_decisions = load_and_resolve_method(
        source.paths.method_catalog,
        method_name,
        decision_root=source.paths.decision_root,
        protocol_lock=source_lock,
    )
    quick_method, quick_hash, quick_decisions = load_and_resolve_method(
        quick.paths.method_catalog,
        method_name,
        decision_root=quick.paths.decision_root,
        protocol_lock=quick_lock,
    )
    assert source_method == quick_method
    assert source_hash == quick_hash
    assert source_decisions == quick_decisions == ()


def test_inner_protocol_lock_regenerates() -> None:
    profile = load_profile()
    quick = load_study_config(profile.paths.quick_config)
    lock = verify_protocol_lock(quick, regenerate=True)
    assert lock["catalog_audit"] == {
        "method_count": 7,
        "root_count": 2,
        "static_derived_count": 5,
        "dynamic_count": 0,
    }
