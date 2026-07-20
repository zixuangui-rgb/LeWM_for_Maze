"""Deterministic Q1/Q2 promotion and locked Q3 exploratory assessment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from a1_quick_validation import (
    ALL_METHODS,
    DECISION_SCHEMA,
    NEW_METHODS,
    PROFILE_ID,
    PROMOTABLE_METHODS,
)
from a1_quick_validation.common import (
    DEFAULT_PROFILE,
    atomic_json_dump,
    canonical_json_sha256,
    load_json,
    prepare_immutable,
    resolve_path,
    sha256_file,
)
from a1_quick_validation.evidence import (
    load_validated_diagnostic,
    quick_cache_evidence_hashes,
    validate_candidate_bank,
    validate_quick_checkpoint,
    validate_result_cell,
)
from a1_quick_validation.profile import verify_package_lock
from a1_quick_validation.schemas import Q2Thresholds
from distance_head_study.common import load_study_config
from distance_head_study.gates import load_signed_artifact
from distance_head_study.protocol import verify_protocol_lock


def diagnostic_path(
    run_root: str | Path,
    *,
    split_role: str,
    method: str,
    backbone_seed: int,
    head_seed: int,
) -> Path:
    return resolve_path(run_root) / (
        f"diagnostics/{split_role}/{method}/"
        f"backbone{backbone_seed}_head{head_seed}.json"
    )


def _load_rows(
    config: Any,
    quick_lock: dict[str, Any],
    *,
    split_role: str,
    method: str,
    backbone_seed: int,
    head_seed: int,
    action_protocol: str,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    return validate_result_cell(
        config,
        quick_lock,
        split_role=split_role,
        method_name=method,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
        action_protocol=action_protocol,
    )


def _aligned_arrays(
    treatment: list[dict[str, Any]], reference: list[dict[str, Any]], field: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    left = {str(row["task_id"]): row for row in treatment}
    right = {str(row["task_id"]): row for row in reference}
    if len(left) != len(treatment) or len(right) != len(reference):
        raise ValueError("paired result table contains duplicate task IDs")
    if set(left) != set(right):
        raise ValueError("paired result tables contain different task IDs")
    for identifier in left:
        if int(left[identifier]["maze_size"]) != int(right[identifier]["maze_size"]):
            raise ValueError("paired task maze-size strata differ")
    ordered = [right[key] for key in sorted(right)]
    return (
        np.asarray([float(left[str(row["task_id"])][field]) for row in ordered]),
        np.asarray([float(row[field]) for row in ordered]),
        ordered,
    )


def _bootstrap_ci(
    differences: np.ndarray,
    strata: np.ndarray,
    replicate_seeds: list[int],
) -> list[float]:
    if differences.ndim != 1 or not len(differences):
        raise ValueError("paired bootstrap needs a nonempty one-dimensional vector")
    if strata.shape != differences.shape:
        raise ValueError("paired bootstrap strata do not align with task differences")
    if not replicate_seeds:
        raise ValueError("paired bootstrap needs a nonempty replicate schedule")
    stratum_indices = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    estimates = np.empty(len(replicate_seeds), dtype=np.float64)
    for index, seed in enumerate(replicate_seeds):
        rng = np.random.default_rng(seed)
        selected = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in stratum_indices
            ]
        )
        estimates[index] = differences[selected].mean()
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def paired_metrics(
    treatment: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    replicate_seeds: list[int],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "pairing": "exact_task_id",
        "task_resampling": "paired_by_task_id_within_maze_size",
        "bootstrap_samples": len(replicate_seeds),
    }
    for field in ("success", "spl"):
        left, right, ordered = _aligned_arrays(treatment, reference, field)
        differences = left - right
        strata = np.asarray([int(row["maze_size"]) for row in ordered])
        output[field] = {
            "treatment_mean": float(left.mean()),
            "reference_mean": float(right.mean()),
            "delta": float(differences.mean()),
            "paired_bootstrap_95ci": _bootstrap_ci(
                differences, strata, replicate_seeds
            ),
            "n": int(len(differences)),
        }
        if field == "success":
            for label, predicate in (
                ("seen", lambda row: int(row["maze_size"]) <= 21),
                ("ood", lambda row: int(row["maze_size"]) > 21),
            ):
                mask = np.asarray([predicate(row) for row in ordered], dtype=bool)
                if bool(mask.any()):
                    delta = differences[mask]
                    subgroup_strata = strata[mask]
                    output[field][label] = {
                        "delta": float(delta.mean()),
                        "paired_bootstrap_95ci": _bootstrap_ci(
                            delta, subgroup_strata, replicate_seeds
                        ),
                        "n": int(mask.sum()),
                    }
    return output


def _mean_metric(diagnostic: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = diagnostic
    for key in path:
        value = value[key]
    return float(value)


def _relative_reduction(reference: float, treatment: float) -> float:
    if reference <= 0:
        return 0.0
    return float((reference - treatment) / reference)


def q1_mechanism_gate(
    method: str,
    diagnostic: dict[str, Any],
    a1_diagnostic: dict[str, Any],
    hcond_diagnostic: dict[str, Any],
    *,
    sr_delta_vs_a1: float,
    sr_delta_vs_hcond: float,
    thresholds: Any,
    enforce_sr_safety: bool = True,
) -> dict[str, Any]:
    safety = sr_delta_vs_a1 >= thresholds.sr_safety_delta if enforce_sr_safety else True
    if method == "a1_bellman":
        top1_delta = _mean_metric(
            diagnostic, ("true_latent_local", "top1", "mean")
        ) - _mean_metric(a1_diagnostic, ("true_latent_local", "top1", "mean"))
        regret_reduction = _relative_reduction(
            _mean_metric(a1_diagnostic, ("true_latent_local", "regret_steps", "mean")),
            _mean_metric(diagnostic, ("true_latent_local", "regret_steps", "mean")),
        )
        intended = (
            top1_delta >= thresholds.local_top1_delta
            or regret_reduction >= thresholds.regret_relative_reduction
        )
        details = {
            "true_local_top1_delta": top1_delta,
            "true_local_regret_relative_reduction": regret_reduction,
        }
    elif method == "a1_predicted":
        top1_delta = _mean_metric(
            diagnostic, ("predicted_latent_local", "top1", "mean")
        ) - _mean_metric(a1_diagnostic, ("predicted_latent_local", "top1", "mean"))
        regret_reduction = _relative_reduction(
            _mean_metric(
                a1_diagnostic,
                ("predicted_latent_local", "regret_steps", "mean"),
            ),
            _mean_metric(
                diagnostic,
                ("predicted_latent_local", "regret_steps", "mean"),
            ),
        )
        intended = (
            top1_delta >= thresholds.local_top1_delta
            and regret_reduction >= thresholds.regret_relative_reduction
        )
        details = {
            "predicted_local_top1_delta": top1_delta,
            "predicted_local_regret_relative_reduction": regret_reduction,
        }
    elif method == "a1_reach":
        reach = diagnostic["reachability"]
        requirements = [
            reach.get("macro_auroc") is not None
            and float(reach["macro_auroc"]) >= thresholds.reachability_min_auroc,
            float(reach.get("macro_brier", 1.0)) <= thresholds.reachability_max_brier,
            float(reach.get("macro_ece10", 1.0)) <= thresholds.reachability_max_ece,
            float(reach.get("monotonic_violation_rate", 1.0))
            <= thresholds.reachability_max_monotonic_violation,
        ]
        if enforce_sr_safety:
            requirements.append(sr_delta_vs_hcond >= thresholds.sr_safety_delta)
        intended = bool(reach.get("available")) and all(requirements)
        details = {
            "macro_auroc": reach.get("macro_auroc"),
            "macro_brier": reach.get("macro_brier"),
            "macro_ece10": reach.get("macro_ece10"),
            "monotonic_violation_rate": reach.get("monotonic_violation_rate"),
            "sr_delta_vs_horizon_control": (
                sr_delta_vs_hcond if enforce_sr_safety else None
            ),
        }
    elif method == "a1_hcond":
        intended = False
        details = {"matched_control_only": True}
    else:
        raise ValueError(f"unknown Q1 mechanism: {method}")
    return {
        "sr_safety_evaluated": enforce_sr_safety,
        "sr_safety_pass": safety if enforce_sr_safety else None,
        "intended_mechanism_pass": bool(intended),
        "q1_gate_pass": bool(safety and intended and method != "a1_hcond"),
        **details,
    }


def _input_hashes(paths: set[Path]) -> dict[str, str]:
    return {path.as_posix(): sha256_file(path) for path in sorted(paths)}


def _add_evidence_paths(paths: set[Path], hashes: dict[str, str]) -> None:
    for value, expected in hashes.items():
        path = resolve_path(value)
        if sha256_file(path) != expected:
            raise ValueError(f"evidence changed while constructing a decision: {path}")
        paths.add(path)


def _write_signed(output: Path, payload: dict[str, Any], signature_field: str) -> Path:
    prepare_immutable(output)
    payload[signature_field] = canonical_json_sha256(payload)
    atomic_json_dump(output, payload)
    return output


def select_q1(profile_path: str | Path = DEFAULT_PROFILE) -> tuple[Path, Path | None]:
    profile, package_lock, quick_lock = verify_package_lock(profile_path)
    config = load_study_config(profile.paths.quick_config)
    verify_protocol_lock(config)
    schedule_path = resolve_path(config.paths.bootstrap_schedule)
    replicate_seeds = [
        int(value) for value in load_json(schedule_path)["replicate_seeds"]
    ]
    paths: set[Path] = {resolve_path(profile.paths.package_lock), schedule_path}
    for split_role in ("train", "cal", "screen"):
        _add_evidence_paths(
            paths,
            quick_cache_evidence_hashes(
                profile_path,
                split_role=split_role,
                backbone_seed=42,
            ),
        )
    _, _, bank_hashes = validate_candidate_bank(config, quick_lock, backbone_seed=42)
    _add_evidence_paths(paths, bank_hashes)
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for method in profile.q1.methods:
        _add_evidence_paths(
            paths,
            validate_quick_checkpoint(
                profile_path,
                method_name=method,
                backbone_seed=42,
                head_seed=0,
            ),
        )
        result, _, rows, result_hashes = _load_rows(
            config,
            quick_lock,
            split_role="screen",
            method=method,
            backbone_seed=42,
            head_seed=0,
            action_protocol="corrected_v1",
        )
        rows_by_method[method] = rows
        _add_evidence_paths(paths, result_hashes)
        diag = diagnostic_path(
            profile.paths.run_root,
            split_role="screen",
            method=method,
            backbone_seed=42,
            head_seed=0,
        )
        diagnostics[method], diagnostic_hashes = load_validated_diagnostic(
            config,
            quick_lock,
            diag,
            split_role="screen",
            method_name=method,
            backbone_seed=42,
            head_seed=0,
        )
        _add_evidence_paths(paths, diagnostic_hashes)
    a1_rows = rows_by_method["a1_log"]
    hcond_rows = rows_by_method["a1_hcond"]
    metrics: dict[str, Any] = {}
    for method in profile.q1.methods:
        paired = paired_metrics(rows_by_method[method], a1_rows, replicate_seeds)
        metrics[method] = {"paired_vs_a1": paired}
        if method in NEW_METHODS:
            hcond_delta = paired_metrics(
                rows_by_method[method], hcond_rows, replicate_seeds
            )["success"]["delta"]
            metrics[method]["mechanism_gate"] = q1_mechanism_gate(
                method,
                diagnostics[method],
                diagnostics["a1_log"],
                diagnostics["a1_hcond"],
                sr_delta_vs_a1=paired["success"]["delta"],
                sr_delta_vs_hcond=hcond_delta,
                thresholds=profile.q1_thresholds,
            )
    passing = [
        method
        for method in PROMOTABLE_METHODS
        if metrics[method]["mechanism_gate"]["q1_gate_pass"]
    ]
    passing.sort(
        key=lambda method: (
            -metrics[method]["paired_vs_a1"]["success"]["delta"],
            NEW_METHODS.index(method),
        )
    )
    selected = passing[: profile.q1_thresholds.max_promoted_new_methods]
    input_hashes = _input_hashes(paths)
    decision = {
        "schema": DECISION_SCHEMA,
        "profile_id": profile.profile_id,
        "decision_name": "q1_screen_selection",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": quick_lock["analysis_spec_sha256"],
        "protocol_lock_sha256": quick_lock["protocol_lock_sha256"],
        "package_lock_sha256": package_lock["package_lock_sha256"],
        "split_role": "screen",
        "backbone_seeds": [42],
        "head_seeds": [0],
        "action_protocols": ["corrected_v1"],
        "ranked_passing_methods": passing,
        "selected_methods": selected,
        "metrics": metrics,
        "input_hashes": input_hashes,
        "stopped_for_no_candidate": not bool(selected),
        "evidence_status": profile.evidence_status,
        "claim_boundary": profile.claim_boundary,
    }
    decision_path = resolve_path(config.paths.decision_root) / "q1_decision.json"
    _write_signed(decision_path, decision, "decision_sha256")
    if not selected:
        return decision_path, None
    shortlist = {
        "schema": "distance-head-shortlist-lock-v1",
        "quick_profile_id": profile.profile_id,
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": quick_lock["analysis_spec_sha256"],
        "protocol_lock_sha256": quick_lock["protocol_lock_sha256"],
        "selected_methods": ["a1_log", *selected],
        "new_methods": selected,
        "anchors": ["b_l2_cem", "b_dh_cem", "a1_log"],
        "route": "quick_positive_mechanism_screen",
        "selection_decision_path": decision_path.as_posix(),
        "screen_decision_sha256": decision["decision_sha256"],
        "package_lock_sha256": package_lock["package_lock_sha256"],
        "d_select_may_not_add_methods": True,
        "max_new_methods": profile.q1_thresholds.max_promoted_new_methods,
        "input_hashes": {
            decision_path.as_posix(): sha256_file(decision_path),
            **input_hashes,
        },
    }
    shortlist_path = resolve_path(config.paths.shortlist_lock)
    _write_signed(shortlist_path, shortlist, "shortlist_sha256")
    return decision_path, shortlist_path


def load_q1_shortlist(
    config: Any,
    quick_lock: dict[str, Any],
    *,
    package_lock_sha256: str | None = None,
) -> dict[str, Any]:
    shortlist = load_signed_artifact(
        config.paths.shortlist_lock,
        signature_field="shortlist_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    if (
        shortlist.get("analysis_spec_sha256") != quick_lock["analysis_spec_sha256"]
        or shortlist.get("protocol_lock_sha256") != quick_lock["protocol_lock_sha256"]
    ):
        raise ValueError("Q1 shortlist uses another protocol lock")
    if (
        package_lock_sha256 is not None
        and shortlist.get("package_lock_sha256") != package_lock_sha256
    ):
        raise ValueError("Q1 shortlist uses another package lock")
    new_methods = tuple(str(value) for value in shortlist.get("new_methods", ()))
    if (
        shortlist.get("schema") != "distance-head-shortlist-lock-v1"
        or shortlist.get("quick_profile_id") != PROFILE_ID
        or not new_methods
        or len(new_methods) > 2
        or len(new_methods) != len(set(new_methods))
        or not set(new_methods) <= set(PROMOTABLE_METHODS)
        or shortlist.get("selected_methods") != ["a1_log", *new_methods]
        or shortlist.get("anchors") != ["b_l2_cem", "b_dh_cem", "a1_log"]
        or shortlist.get("route") != "quick_positive_mechanism_screen"
        or shortlist.get("d_select_may_not_add_methods") is not True
        or int(shortlist.get("max_new_methods", -1)) != 2
    ):
        raise ValueError("Q1 shortlist contains an invalid method set")
    decision_path = resolve_path(config.paths.decision_root) / "q1_decision.json"
    if resolve_path(str(shortlist.get("selection_decision_path", ""))) != decision_path:
        raise ValueError("Q1 shortlist points at a noncanonical selection decision")
    decision = load_signed_artifact(
        decision_path,
        signature_field="decision_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    metrics = decision.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(ALL_METHODS):
        raise ValueError("Q1 decision has an incomplete metric matrix")
    expected_passing = [
        method
        for method in PROMOTABLE_METHODS
        if metrics[method].get("mechanism_gate", {}).get("q1_gate_pass") is True
    ]
    expected_passing.sort(
        key=lambda method: (
            -float(metrics[method]["paired_vs_a1"]["success"]["delta"]),
            NEW_METHODS.index(method),
        )
    )
    decision_valid = (
        decision.get("schema") == DECISION_SCHEMA
        and decision.get("profile_id") == PROFILE_ID
        and decision.get("decision_name") == "q1_screen_selection"
        and decision.get("analysis_spec_sha256") == quick_lock["analysis_spec_sha256"]
        and decision.get("protocol_lock_sha256") == quick_lock["protocol_lock_sha256"]
        and decision.get("package_lock_sha256") == shortlist.get("package_lock_sha256")
        and decision.get("split_role") == "screen"
        and decision.get("backbone_seeds") == [42]
        and decision.get("head_seeds") == [0]
        and decision.get("action_protocols") == ["corrected_v1"]
        and decision.get("ranked_passing_methods") == expected_passing
        and decision.get("selected_methods") == expected_passing[:2]
        and tuple(decision.get("selected_methods", ())) == new_methods
        and decision.get("stopped_for_no_candidate") is False
        and shortlist.get("screen_decision_sha256") == decision["decision_sha256"]
    )
    if not decision_valid:
        raise ValueError("Q1 shortlist does not reproduce its selection decision")
    expected_inputs = {
        decision_path.as_posix(): sha256_file(decision_path),
        **decision["input_hashes"],
    }
    if shortlist.get("input_hashes") != expected_inputs:
        raise ValueError("Q1 shortlist does not contain the complete decision lineage")
    return shortlist


def load_q2_winner(
    config: Any,
    quick_lock: dict[str, Any],
    *,
    package_lock_sha256: str | None = None,
) -> dict[str, Any]:
    path = resolve_path(config.paths.decision_root) / "q2_winner.json"
    payload = load_signed_artifact(
        path,
        signature_field="decision_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    if (
        payload.get("schema") != DECISION_SCHEMA
        or payload.get("profile_id") != PROFILE_ID
        or payload.get("decision_name") != "q2_winner_selection"
        or payload.get("analysis_spec_sha256") != quick_lock["analysis_spec_sha256"]
        or payload.get("protocol_lock_sha256") != quick_lock["protocol_lock_sha256"]
        or payload.get("split_role") != "select"
        or payload.get("backbone_seeds") != [42]
        or payload.get("head_seeds") != [0, 1]
        or payload.get("action_protocols") != ["corrected_v1", "unmasked"]
    ):
        raise ValueError("Q2 winner uses another quick protocol")
    if (
        package_lock_sha256 is not None
        and payload.get("package_lock_sha256") != package_lock_sha256
    ):
        raise ValueError("Q2 winner uses another package lock")
    winner = payload.get("selected_method")
    eligible = tuple(str(value) for value in payload.get("eligible_methods", ()))
    passing = tuple(str(value) for value in payload.get("ranked_passing_methods", ()))
    metrics = payload.get("metrics")
    expected_passing: list[str] = []
    if isinstance(metrics, dict) and set(metrics) == set(eligible):
        for method in eligible:
            metric = metrics[method]
            cells = metric.get("cells", {})
            mechanisms = metric.get("mechanism_rechecks", {})
            expected_cells = {
                f"head{head_seed}_{action_protocol}"
                for head_seed in (0, 1)
                for action_protocol in ("corrected_v1", "unmasked")
            }
            if set(cells) != expected_cells or set(mechanisms) != {
                "head0",
                "head1",
            }:
                raise ValueError("Q2 winner has an incomplete method-cell matrix")
            if any(
                gate.get("sr_safety_evaluated") is not False
                or gate.get("sr_safety_pass") is not None
                for gate in mechanisms.values()
            ):
                raise ValueError("Q2 mechanism recheck contains a synthetic SR gate")
            recomputed = q2_promotion_gate(
                corrected_sr=[
                    float(cells[f"head{seed}_corrected_v1"]["success"]["delta"])
                    for seed in (0, 1)
                ],
                corrected_spl=[
                    float(cells[f"head{seed}_corrected_v1"]["spl"]["delta"])
                    for seed in (0, 1)
                ],
                unmasked_sr=[
                    float(cells[f"head{seed}_unmasked"]["success"]["delta"])
                    for seed in (0, 1)
                ],
                intended_mechanism_passes=[
                    mechanisms[f"head{seed}"].get("intended_mechanism_pass") is True
                    for seed in (0, 1)
                ],
                thresholds=Q2Thresholds(),
            )
            if metric.get("promotion_gate") != recomputed:
                raise ValueError("Q2 promotion gate does not reproduce from its cells")
        expected_passing = [
            method
            for method in eligible
            if metrics[method].get("promotion_gate", {}).get("q2_gate_pass") is True
        ]
        expected_passing.sort(
            key=lambda method: (
                -float(metrics[method]["promotion_gate"]["mean_corrected_sr_delta"]),
                -float(metrics[method]["promotion_gate"]["mean_unmasked_sr_delta"]),
                NEW_METHODS.index(method),
            )
        )
    if (
        not eligible
        or len(eligible) > 2
        or len(eligible) != len(set(eligible))
        or not set(eligible) <= set(PROMOTABLE_METHODS)
        or not isinstance(metrics, dict)
        or set(metrics) != set(eligible)
        or not set(passing) <= set(eligible)
        or len(passing) != len(set(passing))
        or list(passing) != expected_passing
        or winner != (passing[0] if passing else None)
        or bool(payload.get("stopped_for_no_winner")) != (winner is None)
    ):
        raise ValueError("Q2 winner decision has an invalid promotion set")
    if winner is not None and winner not in PROMOTABLE_METHODS:
        raise ValueError("Q2 winner contains an invalid selected method")
    shortlist = load_q1_shortlist(
        config,
        quick_lock,
        package_lock_sha256=package_lock_sha256,
    )
    if (
        payload.get("shortlist_sha256") != shortlist["shortlist_sha256"]
        or list(eligible) != shortlist["new_methods"]
    ):
        raise ValueError("Q2 winner does not descend from the locked Q1 shortlist")
    return payload


def _average(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric list")
    return float(np.mean(values))


def q2_promotion_gate(
    *,
    corrected_sr: list[float],
    corrected_spl: list[float],
    unmasked_sr: list[float],
    intended_mechanism_passes: list[bool],
    thresholds: Any,
) -> dict[str, Any]:
    gate = {
        "mean_corrected_sr_delta": _average(corrected_sr),
        "minimum_head_corrected_sr_delta": min(corrected_sr),
        "mean_corrected_spl_delta": _average(corrected_spl),
        "mean_unmasked_sr_delta": _average(unmasked_sr),
        "all_head_mechanism_pass": all(intended_mechanism_passes),
    }
    gate["q2_gate_pass"] = bool(
        gate["mean_corrected_sr_delta"] >= thresholds.minimum_mean_sr_delta
        and gate["minimum_head_corrected_sr_delta"]
        >= thresholds.minimum_each_head_sr_delta
        and gate["mean_corrected_spl_delta"] >= -thresholds.maximum_secondary_drop
        and gate["mean_unmasked_sr_delta"] >= -thresholds.maximum_secondary_drop
        and gate["all_head_mechanism_pass"]
    )
    return gate


def q3_success_gate(
    *, corrected: dict[str, Any], unmasked: dict[str, Any], thresholds: Any
) -> bool:
    success = corrected["success"]
    has_required_subgroups = all(
        isinstance(success.get(label), dict) and "delta" in success[label]
        for label in ("seen", "ood")
    )
    return bool(
        success["delta"] >= thresholds.minimum_overall_sr_delta
        and corrected["spl"]["delta"] >= -thresholds.maximum_secondary_drop
        and unmasked["success"]["delta"] >= -thresholds.maximum_secondary_drop
        and (
            not thresholds.require_seen_and_ood_nonnegative
            or (
                has_required_subgroups
                and success["seen"]["delta"] >= 0.0
                and success["ood"]["delta"] >= 0.0
            )
        )
    )


def select_q2(profile_path: str | Path = DEFAULT_PROFILE) -> Path:
    profile, package_lock, quick_lock = verify_package_lock(profile_path)
    config = load_study_config(profile.paths.quick_config)
    shortlist = load_q1_shortlist(
        config,
        quick_lock,
        package_lock_sha256=package_lock["package_lock_sha256"],
    )
    candidates = list(shortlist["new_methods"])
    methods = [*profile.q2.methods, *candidates]
    schedule_path = resolve_path(config.paths.bootstrap_schedule)
    replicate_seeds = [
        int(value) for value in load_json(schedule_path)["replicate_seeds"]
    ]
    paths: set[Path] = {
        resolve_path(profile.paths.package_lock),
        resolve_path(config.paths.shortlist_lock),
        schedule_path,
    }
    _add_evidence_paths(paths, shortlist["input_hashes"])
    for split_role in ("train", "cal", "select"):
        _add_evidence_paths(
            paths,
            quick_cache_evidence_hashes(
                profile_path,
                split_role=split_role,
                backbone_seed=42,
            ),
        )
    _, _, bank_hashes = validate_candidate_bank(config, quick_lock, backbone_seed=42)
    _add_evidence_paths(paths, bank_hashes)
    rows: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    diagnostics: dict[tuple[str, int], dict[str, Any]] = {}
    for method in methods:
        for head_seed in profile.q2.head_seeds:
            _add_evidence_paths(
                paths,
                validate_quick_checkpoint(
                    profile_path,
                    method_name=method,
                    backbone_seed=42,
                    head_seed=head_seed,
                ),
            )
            for action_protocol in profile.q2.action_protocols:
                result, _, table, result_hashes = _load_rows(
                    config,
                    quick_lock,
                    split_role="select",
                    method=method,
                    backbone_seed=42,
                    head_seed=head_seed,
                    action_protocol=action_protocol,
                )
                rows[(method, head_seed, action_protocol)] = table
                _add_evidence_paths(paths, result_hashes)
            diag = diagnostic_path(
                profile.paths.run_root,
                split_role="select",
                method=method,
                backbone_seed=42,
                head_seed=head_seed,
            )
            diagnostics[(method, head_seed)], diagnostic_hashes = (
                load_validated_diagnostic(
                    config,
                    quick_lock,
                    diag,
                    split_role="select",
                    method_name=method,
                    backbone_seed=42,
                    head_seed=head_seed,
                )
            )
            _add_evidence_paths(paths, diagnostic_hashes)
    metrics: dict[str, Any] = {}
    passing: list[str] = []
    for method in candidates:
        cells: dict[str, Any] = {}
        mechanism_rechecks: dict[str, Any] = {}
        corrected_sr = []
        corrected_spl = []
        unmasked_sr = []
        intended_passes = []
        for head_seed in profile.q2.head_seeds:
            for action_protocol in profile.q2.action_protocols:
                paired = paired_metrics(
                    rows[(method, head_seed, action_protocol)],
                    rows[("a1_log", head_seed, action_protocol)],
                    replicate_seeds,
                )
                cells[f"head{head_seed}_{action_protocol}"] = paired
                if action_protocol == "corrected_v1":
                    corrected_sr.append(paired["success"]["delta"])
                    corrected_spl.append(paired["spl"]["delta"])
                else:
                    unmasked_sr.append(paired["success"]["delta"])
            gate = q1_mechanism_gate(
                method,
                diagnostics[(method, head_seed)],
                diagnostics[("a1_log", head_seed)],
                diagnostics.get(
                    ("a1_hcond", head_seed), diagnostics[("a1_log", head_seed)]
                ),
                sr_delta_vs_a1=0.0,
                sr_delta_vs_hcond=0.0,
                thresholds=profile.q1_thresholds,
                enforce_sr_safety=False,
            )
            mechanism_rechecks[f"head{head_seed}"] = gate
            intended_passes.append(gate["intended_mechanism_pass"])
        gate = q2_promotion_gate(
            corrected_sr=corrected_sr,
            corrected_spl=corrected_spl,
            unmasked_sr=unmasked_sr,
            intended_mechanism_passes=intended_passes,
            thresholds=profile.q2_thresholds,
        )
        metrics[method] = {
            "cells": cells,
            "mechanism_rechecks": mechanism_rechecks,
            "promotion_gate": gate,
        }
        if gate["q2_gate_pass"]:
            passing.append(method)
    passing.sort(
        key=lambda method: (
            -metrics[method]["promotion_gate"]["mean_corrected_sr_delta"],
            -metrics[method]["promotion_gate"]["mean_unmasked_sr_delta"],
            NEW_METHODS.index(method),
        )
    )
    winner = passing[0] if passing else None
    payload = {
        "schema": DECISION_SCHEMA,
        "profile_id": profile.profile_id,
        "decision_name": "q2_winner_selection",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": quick_lock["analysis_spec_sha256"],
        "protocol_lock_sha256": quick_lock["protocol_lock_sha256"],
        "package_lock_sha256": package_lock["package_lock_sha256"],
        "shortlist_sha256": shortlist["shortlist_sha256"],
        "split_role": profile.q2.split_role,
        "backbone_seeds": list(profile.q2.backbone_seeds),
        "head_seeds": list(profile.q2.head_seeds),
        "action_protocols": list(profile.q2.action_protocols),
        "eligible_methods": candidates,
        "ranked_passing_methods": passing,
        "selected_method": winner,
        "metrics": metrics,
        "input_hashes": _input_hashes(paths),
        "stopped_for_no_winner": winner is None,
        "evidence_status": profile.evidence_status,
        "claim_boundary": profile.claim_boundary,
    }
    output = resolve_path(config.paths.decision_root) / "q2_winner.json"
    return _write_signed(output, payload, "decision_sha256")


def assess_q3(profile_path: str | Path = DEFAULT_PROFILE) -> Path:
    profile, package_lock, quick_lock = verify_package_lock(profile_path)
    config = load_study_config(profile.paths.quick_config)
    winner_path = resolve_path(config.paths.decision_root) / "q2_winner.json"
    winner_decision = load_q2_winner(
        config,
        quick_lock,
        package_lock_sha256=package_lock["package_lock_sha256"],
    )
    winner = winner_decision.get("selected_method")
    if winner not in PROMOTABLE_METHODS:
        raise RuntimeError("Q3 cannot run without a locked new-method winner")
    schedule_path = resolve_path(config.paths.bootstrap_schedule)
    replicate_seeds = [
        int(value) for value in load_json(schedule_path)["replicate_seeds"]
    ]
    methods = [*profile.q3.methods, winner]
    paths: set[Path] = {
        resolve_path(profile.paths.package_lock),
        winner_path,
        schedule_path,
    }
    _add_evidence_paths(paths, winner_decision["input_hashes"])
    for split_role in ("train", "cal"):
        _add_evidence_paths(
            paths,
            quick_cache_evidence_hashes(
                profile_path,
                split_role=split_role,
                backbone_seed=42,
            ),
        )
    _, _, bank_hashes = validate_candidate_bank(config, quick_lock, backbone_seed=42)
    _add_evidence_paths(paths, bank_hashes)
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for method in methods:
        effective_head_seed = 0
        _add_evidence_paths(
            paths,
            validate_quick_checkpoint(
                profile_path,
                method_name=method,
                backbone_seed=42,
                head_seed=effective_head_seed,
            ),
        )
        for action_protocol in profile.q3.action_protocols:
            result, _, table, result_hashes = _load_rows(
                config,
                quick_lock,
                split_role="legacy",
                method=method,
                backbone_seed=42,
                head_seed=effective_head_seed,
                action_protocol=action_protocol,
            )
            rows[(method, action_protocol)] = table
            _add_evidence_paths(paths, result_hashes)
    metrics = {}
    for method in methods:
        metrics[method] = {
            protocol: paired_metrics(
                rows[(method, protocol)],
                rows[("a1_log", protocol)],
                replicate_seeds,
            )
            for protocol in profile.q3.action_protocols
        }
    winner_metrics = metrics[winner]
    corrected = winner_metrics["corrected_v1"]
    unmasked = winner_metrics["unmasked"]
    q3_pass = q3_success_gate(
        corrected=corrected,
        unmasked=unmasked,
        thresholds=profile.q3_thresholds,
    )
    payload = {
        "schema": DECISION_SCHEMA,
        "profile_id": profile.profile_id,
        "decision_name": "q3_full900_exploratory_assessment",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": quick_lock["analysis_spec_sha256"],
        "protocol_lock_sha256": quick_lock["protocol_lock_sha256"],
        "package_lock_sha256": package_lock["package_lock_sha256"],
        "winner_decision_sha256": winner_decision["decision_sha256"],
        "split_role": profile.q3.split_role,
        "backbone_seeds": list(profile.q3.backbone_seeds),
        "head_seeds": list(profile.q3.head_seeds),
        "action_protocols": list(profile.q3.action_protocols),
        "locked_winner": winner,
        "metrics": metrics,
        "q3_gate_pass": q3_pass,
        "input_hashes": _input_hashes(paths),
        "evidence_status": "exploratory_single_backbone_full900",
        "confirmatory": False,
        "claim_boundary": profile.claim_boundary,
    }
    output = resolve_path(config.paths.decision_root) / "q3_assessment.json"
    return _write_signed(output, payload, "decision_sha256")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("q1", "q2", "q3"))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    args = parser.parse_args()
    if args.command == "q1":
        decision, shortlist = select_q1(args.profile)
        print(decision)
        if shortlist is not None:
            print(shortlist)
    elif args.command == "q2":
        print(select_q2(args.profile))
    else:
        print(assess_q3(args.profile))


if __name__ == "__main__":
    main()


__all__ = [
    "assess_q3",
    "diagnostic_path",
    "load_q1_shortlist",
    "load_q2_winner",
    "paired_metrics",
    "q1_mechanism_gate",
    "q2_promotion_gate",
    "q3_success_gate",
    "select_q1",
    "select_q2",
]
