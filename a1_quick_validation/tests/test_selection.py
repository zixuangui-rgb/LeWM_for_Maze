from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from a1_quick_validation import ALL_METHODS, DECISION_SCHEMA, NEW_METHODS, PROFILE_ID
from a1_quick_validation.common import (
    atomic_json_dump,
    canonical_json_sha256,
    sha256_file,
)
from a1_quick_validation.profile import load_profile
from a1_quick_validation.selection import (
    _bootstrap_ci,
    load_q1_shortlist,
    load_q2_winner,
    paired_metrics,
    q1_mechanism_gate,
    q2_promotion_gate,
    q3_success_gate,
)


def _rows(successes: list[int]) -> list[dict[str, object]]:
    return [
        {
            "task_id": f"task-{index}",
            "success": success,
            "spl": float(success) * (0.8 + index * 0.01),
            "maze_size": 19 if index < len(successes) // 2 else 23,
        }
        for index, success in enumerate(successes)
    ]


def _diagnostic() -> dict[str, object]:
    return {
        "true_latent_local": {
            "top1": {"mean": 0.50},
            "regret_steps": {"mean": 1.0},
        },
        "predicted_latent_local": {
            "top1": {"mean": 0.50},
            "regret_steps": {"mean": 1.0},
        },
        "reachability": {"available": False},
    }


def test_paired_metrics_aligns_task_ids_and_is_deterministic() -> None:
    reference = _rows([0, 0, 1, 1])
    treatment = list(reversed(_rows([1, 0, 1, 1])))
    first = paired_metrics(treatment, reference, [1, 2, 3, 4])
    second = paired_metrics(treatment, reference, [1, 2, 3, 4])
    assert first == second
    assert first["success"]["delta"] == pytest.approx(0.25)
    assert first["success"]["seen"]["n"] == 2
    assert first["success"]["ood"]["n"] == 2
    assert first["task_resampling"] == "paired_by_task_id_within_maze_size"
    assert first["bootstrap_samples"] == 4


def test_paired_bootstrap_resamples_within_each_maze_size() -> None:
    differences = np.asarray([0.0, 10.0, 100.0])
    strata = np.asarray([9, 9, 25])
    seed = 17
    rng = np.random.default_rng(seed)
    selected = np.concatenate(
        (
            rng.choice([0, 1], size=2, replace=True),
            rng.choice([2], size=1, replace=True),
        )
    )
    expected = float(differences[selected].mean())
    assert _bootstrap_ci(differences, strata, [seed]) == pytest.approx(
        [expected, expected]
    )


def test_paired_metrics_rejects_different_task_sets() -> None:
    treatment = _rows([1, 0])
    treatment[0]["task_id"] = "other"
    with pytest.raises(ValueError, match="different task IDs"):
        paired_metrics(treatment, _rows([0, 0]), [1])


def test_paired_metrics_rejects_duplicate_ids_and_stratum_drift() -> None:
    duplicated = _rows([1, 0])
    duplicated[1]["task_id"] = duplicated[0]["task_id"]
    with pytest.raises(ValueError, match="duplicate task IDs"):
        paired_metrics(duplicated, _rows([0, 0]), [1])
    treatment = _rows([1, 0])
    treatment[0]["maze_size"] = 25
    with pytest.raises(ValueError, match="strata differ"):
        paired_metrics(treatment, _rows([0, 0]), [1])


def test_bellman_gate_accepts_top1_or_regret_but_enforces_sr_safety() -> None:
    thresholds = load_profile().q1_thresholds
    baseline = _diagnostic()
    treatment = copy.deepcopy(baseline)
    treatment["true_latent_local"]["top1"]["mean"] = 0.54
    passed = q1_mechanism_gate(
        "a1_bellman",
        treatment,
        baseline,
        baseline,
        sr_delta_vs_a1=-0.005,
        sr_delta_vs_hcond=0.0,
        thresholds=thresholds,
    )
    assert passed["q1_gate_pass"]
    failed = q1_mechanism_gate(
        "a1_bellman",
        treatment,
        baseline,
        baseline,
        sr_delta_vs_a1=-0.011,
        sr_delta_vs_hcond=0.0,
        thresholds=thresholds,
    )
    assert not failed["q1_gate_pass"]


def test_predicted_gate_requires_both_top1_and_regret() -> None:
    thresholds = load_profile().q1_thresholds
    baseline = _diagnostic()
    treatment = copy.deepcopy(baseline)
    treatment["predicted_latent_local"]["top1"]["mean"] = 0.54
    only_top1 = q1_mechanism_gate(
        "a1_predicted",
        treatment,
        baseline,
        baseline,
        sr_delta_vs_a1=0.0,
        sr_delta_vs_hcond=0.0,
        thresholds=thresholds,
    )
    assert not only_top1["q1_gate_pass"]
    treatment["predicted_latent_local"]["regret_steps"]["mean"] = 0.80
    both = q1_mechanism_gate(
        "a1_predicted",
        treatment,
        baseline,
        baseline,
        sr_delta_vs_a1=0.0,
        sr_delta_vs_hcond=0.0,
        thresholds=thresholds,
    )
    assert both["q1_gate_pass"]


def test_reachability_gate_and_horizon_control_boundary() -> None:
    thresholds = load_profile().q1_thresholds
    baseline = _diagnostic()
    treatment = copy.deepcopy(baseline)
    treatment["reachability"] = {
        "available": True,
        "macro_auroc": 0.70,
        "macro_brier": 0.20,
        "macro_ece10": 0.10,
        "monotonic_violation_rate": 0.02,
    }
    reach = q1_mechanism_gate(
        "a1_reach",
        treatment,
        baseline,
        baseline,
        sr_delta_vs_a1=0.0,
        sr_delta_vs_hcond=0.0,
        thresholds=thresholds,
    )
    assert reach["q1_gate_pass"]
    diagnostic_only = q1_mechanism_gate(
        "a1_reach",
        treatment,
        baseline,
        baseline,
        sr_delta_vs_a1=0.0,
        sr_delta_vs_hcond=0.0,
        thresholds=thresholds,
        enforce_sr_safety=False,
    )
    assert diagnostic_only["intended_mechanism_pass"]
    assert diagnostic_only["sr_safety_pass"] is None
    assert diagnostic_only["sr_delta_vs_horizon_control"] is None
    control = q1_mechanism_gate(
        "a1_hcond",
        baseline,
        baseline,
        baseline,
        sr_delta_vs_a1=1.0,
        sr_delta_vs_hcond=1.0,
        thresholds=thresholds,
    )
    assert not control["q1_gate_pass"]


def test_q2_requires_mean_gain_each_head_direction_and_safety() -> None:
    thresholds = load_profile().q2_thresholds
    passed = q2_promotion_gate(
        corrected_sr=[0.03, 0.02],
        corrected_spl=[0.0, -0.01],
        unmasked_sr=[0.0, -0.01],
        intended_mechanism_passes=[True, True],
        thresholds=thresholds,
    )
    assert passed["q2_gate_pass"]
    negative_head = q2_promotion_gate(
        corrected_sr=[0.06, -0.01],
        corrected_spl=[0.0, 0.0],
        unmasked_sr=[0.0, 0.0],
        intended_mechanism_passes=[True, True],
        thresholds=thresholds,
    )
    assert not negative_head["q2_gate_pass"]


def test_q3_requires_seen_and_ood_nonnegative() -> None:
    thresholds = load_profile().q3_thresholds
    corrected = {
        "success": {
            "delta": 0.03,
            "seen": {"delta": 0.04},
            "ood": {"delta": -0.01},
        },
        "spl": {"delta": 0.0},
    }
    unmasked = {"success": {"delta": 0.0}}
    assert not q3_success_gate(
        corrected=corrected, unmasked=unmasked, thresholds=thresholds
    )
    corrected["success"]["ood"]["delta"] = 0.0
    assert q3_success_gate(
        corrected=corrected, unmasked=unmasked, thresholds=thresholds
    )
    del corrected["success"]["ood"]
    assert not q3_success_gate(
        corrected=corrected, unmasked=unmasked, thresholds=thresholds
    )


def test_q1_shortlist_loader_reproduces_the_signed_gate_decision(tmp_path) -> None:
    dependency = tmp_path / "evidence.json"
    dependency.write_text("evidence", encoding="utf-8")
    lock = {
        "analysis_spec_sha256": "a" * 64,
        "protocol_lock_sha256": "b" * 64,
    }
    metrics = {}
    for method in ALL_METHODS:
        metrics[method] = {"paired_vs_a1": {"success": {"delta": 0.03}}}
        if method in NEW_METHODS:
            metrics[method]["mechanism_gate"] = {"q1_gate_pass": method == "a1_bellman"}
    decision = {
        "schema": DECISION_SCHEMA,
        "profile_id": PROFILE_ID,
        "protocol_id": "protocol",
        "decision_name": "q1_screen_selection",
        **lock,
        "package_lock_sha256": "c" * 64,
        "split_role": "screen",
        "backbone_seeds": [42],
        "head_seeds": [0],
        "action_protocols": ["corrected_v1"],
        "ranked_passing_methods": ["a1_bellman"],
        "selected_methods": ["a1_bellman"],
        "metrics": metrics,
        "stopped_for_no_candidate": False,
        "input_hashes": {dependency.as_posix(): sha256_file(dependency)},
    }
    decision["decision_sha256"] = canonical_json_sha256(decision)
    decision_path = tmp_path / "q1_decision.json"
    atomic_json_dump(decision_path, decision)
    shortlist = {
        "schema": "distance-head-shortlist-lock-v1",
        "quick_profile_id": PROFILE_ID,
        "protocol_id": "protocol",
        **lock,
        "package_lock_sha256": "c" * 64,
        "selected_methods": ["a1_log", "a1_bellman"],
        "new_methods": ["a1_bellman"],
        "anchors": ["b_l2_cem", "b_dh_cem", "a1_log"],
        "route": "quick_positive_mechanism_screen",
        "selection_decision_path": decision_path.as_posix(),
        "screen_decision_sha256": decision["decision_sha256"],
        "d_select_may_not_add_methods": True,
        "max_new_methods": 2,
        "input_hashes": {
            decision_path.as_posix(): sha256_file(decision_path),
            dependency.as_posix(): sha256_file(dependency),
        },
    }
    shortlist["shortlist_sha256"] = canonical_json_sha256(shortlist)
    shortlist_path = tmp_path / "shortlist.json"
    atomic_json_dump(shortlist_path, shortlist)
    config = SimpleNamespace(
        protocol_id="protocol",
        paths=SimpleNamespace(
            decision_root=tmp_path.as_posix(),
            shortlist_lock=shortlist_path.as_posix(),
        ),
    )
    assert load_q1_shortlist(config, lock, package_lock_sha256="c" * 64)[
        "new_methods"
    ] == ["a1_bellman"]
    shortlist["new_methods"] = ["a1_predicted"]
    shortlist["selected_methods"] = ["a1_log", "a1_predicted"]
    shortlist.pop("shortlist_sha256")
    shortlist["shortlist_sha256"] = canonical_json_sha256(shortlist)
    atomic_json_dump(shortlist_path, shortlist)
    with pytest.raises(ValueError, match="selection decision"):
        load_q1_shortlist(config, lock, package_lock_sha256="c" * 64)


def test_q2_winner_loader_rejects_inconsistent_or_control_winner(
    tmp_path, monkeypatch
) -> None:
    dependency = tmp_path / "evidence.json"
    dependency.write_text("evidence", encoding="utf-8")
    shortlist = {
        "shortlist_sha256": "d" * 64,
        "new_methods": ["a1_bellman"],
    }
    monkeypatch.setattr(
        "a1_quick_validation.selection.load_q1_shortlist",
        lambda *args, **kwargs: shortlist,
    )
    config = SimpleNamespace(
        protocol_id="protocol",
        paths=SimpleNamespace(decision_root=tmp_path.as_posix()),
    )
    lock = {
        "analysis_spec_sha256": "a" * 64,
        "protocol_lock_sha256": "b" * 64,
    }

    def gate_metrics(method: str) -> dict[str, object]:
        cells = {
            f"head{seed}_corrected_v1": {
                "success": {"delta": 0.03},
                "spl": {"delta": 0.0},
            }
            for seed in (0, 1)
        }
        cells.update(
            {
                f"head{seed}_unmasked": {
                    "success": {"delta": 0.01},
                    "spl": {"delta": 0.0},
                }
                for seed in (0, 1)
            }
        )
        mechanisms = {
            f"head{seed}": {
                "sr_safety_evaluated": False,
                "sr_safety_pass": None,
                "intended_mechanism_pass": True,
            }
            for seed in (0, 1)
        }
        return {
            method: {
                "cells": cells,
                "mechanism_rechecks": mechanisms,
                "promotion_gate": q2_promotion_gate(
                    corrected_sr=[0.03, 0.03],
                    corrected_spl=[0.0, 0.0],
                    unmasked_sr=[0.01, 0.01],
                    intended_mechanism_passes=[True, True],
                    thresholds=load_profile().q2_thresholds,
                ),
            }
        }

    payload = {
        "schema": DECISION_SCHEMA,
        "profile_id": PROFILE_ID,
        "protocol_id": "protocol",
        "decision_name": "q2_winner_selection",
        **lock,
        "package_lock_sha256": "c" * 64,
        "shortlist_sha256": shortlist["shortlist_sha256"],
        "split_role": "select",
        "backbone_seeds": [42],
        "head_seeds": [0, 1],
        "action_protocols": ["corrected_v1", "unmasked"],
        "eligible_methods": ["a1_bellman"],
        "ranked_passing_methods": ["a1_bellman"],
        "selected_method": "a1_bellman",
        "stopped_for_no_winner": False,
        "metrics": gate_metrics("a1_bellman"),
        "input_hashes": {dependency.as_posix(): sha256_file(dependency)},
    }
    payload["decision_sha256"] = canonical_json_sha256(payload)
    path = tmp_path / "q2_winner.json"
    atomic_json_dump(path, payload)
    assert (
        load_q2_winner(config, lock, package_lock_sha256="c" * 64)["selected_method"]
        == "a1_bellman"
    )

    payload["eligible_methods"] = ["a1_predicted"]
    payload["ranked_passing_methods"] = ["a1_predicted"]
    payload["selected_method"] = "a1_predicted"
    payload["metrics"] = gate_metrics("a1_predicted")
    payload.pop("decision_sha256")
    payload["decision_sha256"] = canonical_json_sha256(payload)
    atomic_json_dump(path, payload)
    with pytest.raises(ValueError, match="locked Q1 shortlist"):
        load_q2_winner(config, lock, package_lock_sha256="c" * 64)

    payload["eligible_methods"] = ["a1_hcond"]
    payload["ranked_passing_methods"] = ["a1_hcond"]
    payload["selected_method"] = "a1_hcond"
    payload["metrics"] = gate_metrics("a1_hcond")
    payload.pop("decision_sha256")
    payload["decision_sha256"] = canonical_json_sha256(payload)
    atomic_json_dump(path, payload)
    with pytest.raises(ValueError, match="promotion set"):
        load_q2_winner(config, lock, package_lock_sha256="c" * 64)
