from __future__ import annotations

import copy

import pytest

from a1_quick_validation.profile import load_profile
from a1_quick_validation.selection import (
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


def test_paired_metrics_rejects_different_task_sets() -> None:
    treatment = _rows([1, 0])
    treatment[0]["task_id"] = "other"
    with pytest.raises(ValueError, match="different task IDs"):
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
