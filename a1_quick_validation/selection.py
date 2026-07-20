"""Deterministic Q1/Q2 promotion and locked Q3 exploratory assessment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from a1_quick_validation import DECISION_SCHEMA, NEW_METHODS
from a1_quick_validation.common import (
    DEFAULT_PROFILE,
    atomic_json_dump,
    canonical_json_sha256,
    load_json,
    prepare_immutable,
    resolve_path,
    sha256_file,
)
from a1_quick_validation.profile import verify_package_lock
from distance_head_study.common import load_study_config
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.results import load_complete_rows, result_directory


def _result_files(directory: Path) -> tuple[Path, ...]:
    return tuple(
        directory / name for name in ("metadata.json", "rows.jsonl", "summary.json")
    )


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
    *,
    split_role: str,
    method: str,
    backbone_seed: int,
    head_seed: int,
    action_protocol: str,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    path = result_directory(
        config,
        split_role=split_role,
        method=method,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
        action_protocol=action_protocol,
    )
    metadata, rows = load_complete_rows(path)
    return path, metadata, rows


def _aligned_arrays(
    treatment: list[dict[str, Any]], reference: list[dict[str, Any]], field: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    left = {str(row["task_id"]): row for row in treatment}
    right = {str(row["task_id"]): row for row in reference}
    if set(left) != set(right):
        raise ValueError("paired result tables contain different task IDs")
    ordered = [right[key] for key in sorted(right)]
    return (
        np.asarray([float(left[str(row["task_id"])][field]) for row in ordered]),
        np.asarray([float(row[field]) for row in ordered]),
        ordered,
    )


def _bootstrap_ci(differences: np.ndarray, replicate_seeds: list[int]) -> list[float]:
    if differences.ndim != 1 or not len(differences):
        raise ValueError("paired bootstrap needs a nonempty one-dimensional vector")
    estimates = np.empty(len(replicate_seeds), dtype=np.float64)
    count = len(differences)
    for index, seed in enumerate(replicate_seeds):
        rng = np.random.default_rng(seed)
        estimates[index] = differences[rng.integers(0, count, count)].mean()
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def paired_metrics(
    treatment: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    replicate_seeds: list[int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("success", "spl"):
        left, right, ordered = _aligned_arrays(treatment, reference, field)
        differences = left - right
        output[field] = {
            "treatment_mean": float(left.mean()),
            "reference_mean": float(right.mean()),
            "delta": float(differences.mean()),
            "paired_bootstrap_95ci": _bootstrap_ci(differences, replicate_seeds),
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
                    output[field][label] = {
                        "delta": float(delta.mean()),
                        "paired_bootstrap_95ci": _bootstrap_ci(delta, replicate_seeds),
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
) -> dict[str, Any]:
    safety = sr_delta_vs_a1 >= thresholds.sr_safety_delta
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
        intended = bool(reach.get("available")) and all(
            (
                reach.get("macro_auroc") is not None
                and float(reach["macro_auroc"]) >= thresholds.reachability_min_auroc,
                float(reach.get("macro_brier", 1.0))
                <= thresholds.reachability_max_brier,
                float(reach.get("macro_ece10", 1.0)) <= thresholds.reachability_max_ece,
                float(reach.get("monotonic_violation_rate", 1.0))
                <= thresholds.reachability_max_monotonic_violation,
                sr_delta_vs_hcond >= thresholds.sr_safety_delta,
            )
        )
        details = {
            "macro_auroc": reach.get("macro_auroc"),
            "macro_brier": reach.get("macro_brier"),
            "macro_ece10": reach.get("macro_ece10"),
            "monotonic_violation_rate": reach.get("monotonic_violation_rate"),
            "sr_delta_vs_horizon_control": sr_delta_vs_hcond,
        }
    elif method == "a1_hcond":
        intended = False
        details = {"matched_control_only": True}
    else:
        raise ValueError(f"unknown Q1 mechanism: {method}")
    return {
        "sr_safety_pass": safety,
        "intended_mechanism_pass": bool(intended),
        "q1_gate_pass": bool(safety and intended and method != "a1_hcond"),
        **details,
    }


def _input_hashes(paths: set[Path]) -> dict[str, str]:
    return {path.as_posix(): sha256_file(path) for path in sorted(paths)}


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
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for method in profile.q1.methods:
        result, _, rows = _load_rows(
            config,
            split_role="screen",
            method=method,
            backbone_seed=42,
            head_seed=0,
            action_protocol="corrected_v1",
        )
        rows_by_method[method] = rows
        paths.update(_result_files(result))
        diag = diagnostic_path(
            profile.paths.run_root,
            split_role="screen",
            method=method,
            backbone_seed=42,
            head_seed=0,
        )
        diagnostics[method] = load_json(diag)
        paths.add(diag)
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
        for method in NEW_METHODS
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


def _load_shortlist(config: Any, quick_lock: dict[str, Any]) -> dict[str, Any]:
    shortlist = load_json(config.paths.shortlist_lock)
    signature = shortlist.get("shortlist_sha256")
    unsigned = {
        key: value for key, value in shortlist.items() if key != "shortlist_sha256"
    }
    if signature != canonical_json_sha256(unsigned):
        raise ValueError("Q1 shortlist signature mismatch")
    if shortlist.get("protocol_lock_sha256") != quick_lock["protocol_lock_sha256"]:
        raise ValueError("Q1 shortlist uses another protocol lock")
    for path, expected in shortlist["input_hashes"].items():
        if sha256_file(path) != expected:
            raise ValueError(f"Q1 shortlist dependency changed: {path}")
    return shortlist


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
    return bool(
        corrected["success"]["delta"] >= thresholds.minimum_overall_sr_delta
        and corrected["spl"]["delta"] >= -thresholds.maximum_secondary_drop
        and unmasked["success"]["delta"] >= -thresholds.maximum_secondary_drop
        and (
            not thresholds.require_seen_and_ood_nonnegative
            or (
                corrected["success"].get("seen", {}).get("delta", 0.0) >= 0.0
                and corrected["success"].get("ood", {}).get("delta", 0.0) >= 0.0
            )
        )
    )


def select_q2(profile_path: str | Path = DEFAULT_PROFILE) -> Path:
    profile, package_lock, quick_lock = verify_package_lock(profile_path)
    config = load_study_config(profile.paths.quick_config)
    shortlist = _load_shortlist(config, quick_lock)
    candidates = list(shortlist["new_methods"])
    methods = ["b_dh_cem", "a1_log", *candidates]
    schedule_path = resolve_path(config.paths.bootstrap_schedule)
    replicate_seeds = [
        int(value) for value in load_json(schedule_path)["replicate_seeds"]
    ]
    paths: set[Path] = {
        resolve_path(profile.paths.package_lock),
        resolve_path(config.paths.shortlist_lock),
        schedule_path,
    }
    rows: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    diagnostics: dict[tuple[str, int], dict[str, Any]] = {}
    for method in methods:
        for head_seed in profile.q2.head_seeds:
            for action_protocol in profile.q2.action_protocols:
                result, _, table = _load_rows(
                    config,
                    split_role="select",
                    method=method,
                    backbone_seed=42,
                    head_seed=head_seed,
                    action_protocol=action_protocol,
                )
                rows[(method, head_seed, action_protocol)] = table
                paths.update(_result_files(result))
            diag = diagnostic_path(
                profile.paths.run_root,
                split_role="select",
                method=method,
                backbone_seed=42,
                head_seed=head_seed,
            )
            diagnostics[(method, head_seed)] = load_json(diag)
            paths.add(diag)
    metrics: dict[str, Any] = {}
    passing: list[str] = []
    for method in candidates:
        cells: dict[str, Any] = {}
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
            )
            intended_passes.append(gate["intended_mechanism_pass"])
        gate = q2_promotion_gate(
            corrected_sr=corrected_sr,
            corrected_spl=corrected_spl,
            unmasked_sr=unmasked_sr,
            intended_mechanism_passes=intended_passes,
            thresholds=profile.q2_thresholds,
        )
        metrics[method] = {"cells": cells, "promotion_gate": gate}
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
    winner_decision = load_json(winner_path)
    signature = winner_decision.get("decision_sha256")
    unsigned = {
        key: value for key, value in winner_decision.items() if key != "decision_sha256"
    }
    if signature != canonical_json_sha256(unsigned):
        raise ValueError("Q2 winner signature mismatch")
    if (
        winner_decision.get("protocol_lock_sha256")
        != quick_lock["protocol_lock_sha256"]
    ):
        raise ValueError("Q2 winner uses another protocol lock")
    for path, expected in winner_decision.get("input_hashes", {}).items():
        if sha256_file(path) != expected:
            raise ValueError(f"Q2 winner dependency changed: {path}")
    winner = winner_decision.get("selected_method")
    if winner not in NEW_METHODS:
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
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for method in methods:
        effective_head_seed = 0
        for action_protocol in profile.q3.action_protocols:
            result, _, table = _load_rows(
                config,
                split_role="legacy",
                method=method,
                backbone_seed=42,
                head_seed=effective_head_seed,
                action_protocol=action_protocol,
            )
            rows[(method, action_protocol)] = table
            paths.update(_result_files(result))
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
    "paired_metrics",
    "q1_mechanism_gate",
    "q2_promotion_gate",
    "q3_success_gate",
    "select_q1",
    "select_q2",
]
