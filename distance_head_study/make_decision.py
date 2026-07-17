"""Create a signed, deterministic stage decision from immutable evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from distance_head_study.common import (
    atomic_json_dump,
    canonical_json_sha256,
    load_study_config,
    merge_hash_bindings,
    require_clean_worktree,
    resolve_path,
    sha256_file,
)
from distance_head_study.evidence import diagnostic_evidence_hashes
from distance_head_study.gates import load_signed_artifact
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.results import (
    load_complete_rows,
    result_directory,
    result_evidence_hashes,
)
from distance_head_study.taxonomy import (
    MAIN_CANDIDATES,
    NEGATIVE_CLOSURE_CANDIDATES,
)


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("seed list must be nonempty and unique")
    return parsed


def _complete_directory(path: Path) -> Path:
    if (path / "rows.jsonl").exists():
        return path
    if (path / "merged" / "rows.jsonl").exists():
        return path / "merged"
    raise FileNotFoundError(path)


def _diagnostic_metrics(
    config: Any,
    analysis_spec_sha256: str,
    protocol_lock_sha256: str,
    *,
    split_role: str,
    method: str,
    backbones: tuple[int, ...],
    heads: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, str]]:
    _, method_hash, decision_hashes = load_and_resolve_method(
        config.paths.method_catalog,
        method,
        decision_root=config.paths.decision_root,
        protocol_lock={
            "analysis_spec_sha256": analysis_spec_sha256,
            "protocol_lock_sha256": protocol_lock_sha256,
        },
    )
    rows = []
    hashes: dict[str, str] = {}
    for backbone in backbones:
        for head in (0,) if method == "b_l2_cem" else heads:
            path = resolve_path(
                f"distance_head_study_runs/diagnostics/{split_role}/{method}/"
                f"backbone{backbone}_head{head}.json"
            )
            payload = load_signed_artifact(
                path,
                signature_field="diagnostic_sha256",
                expected_protocol_id=config.protocol_id,
            )
            expected = (
                payload.get("analysis_spec_sha256") == analysis_spec_sha256
                and payload.get("protocol_lock_sha256") == protocol_lock_sha256
                and payload.get("split_role") == split_role
                and payload.get("method") == method
                and payload.get("method_sha256") == method_hash
                and payload.get("decision_sha256s") == list(decision_hashes)
                and int(payload.get("backbone_seed", -1)) == backbone
                and int(payload.get("head_seed", -1)) == head
                and int(payload.get("sample_count", -1))
                == config.analysis.diagnostic_batches
                * config.training.effective_batch_size
                and int(payload.get("cache_binding", {}).get("diagnostic_limit", -1))
                == 0
            )
            if not expected:
                raise ValueError(f"diagnostic artifact differs from protocol: {path}")
            rows.append(payload)
            hashes = merge_hash_bindings(
                hashes,
                diagnostic_evidence_hashes(
                    path,
                    payload,
                    split_role=split_role,
                    backbone_seed=backbone,
                    protocol_lock={
                        "analysis_spec_sha256": analysis_spec_sha256,
                        "protocol_lock_sha256": protocol_lock_sha256,
                    },
                ),
            )
    absolute_values = [
        float(row["absolute_distance"]["mae_steps"])
        for row in rows
        if row["absolute_distance"].get("mae_steps") is not None
    ]
    reachability_rows = [
        row["reachability"] for row in rows if row["reachability"].get("available")
    ]
    return (
        {
            "predicted_local_top1": float(
                np.mean([row["predicted_latent_local"]["top1"]["mean"] for row in rows])
            ),
            "candidate_regret": float(
                np.mean(
                    [
                        row["candidate_order"]["predicted_dynamics_regret_steps"][
                            "mean"
                        ]
                        for row in rows
                    ]
                )
            ),
            "candidate_spearman": float(
                np.mean(
                    [
                        row["candidate_order"]["predicted_dynamics_spearman"]["mean"]
                        for row in rows
                    ]
                )
            ),
            "absolute_mae": float(np.mean(absolute_values))
            if absolute_values
            else float("inf"),
            "reachability_macro_brier": (
                float(np.mean([row["macro_brier"] for row in reachability_rows]))
                if reachability_rows
                else None
            ),
            "reachability_macro_ece10": (
                float(np.mean([row["macro_ece10"] for row in reachability_rows]))
                if reachability_rows
                else None
            ),
            "reachability_macro_auroc": (
                float(
                    np.mean(
                        [
                            row["macro_auroc"]
                            for row in reachability_rows
                            if row["macro_auroc"] is not None
                        ]
                    )
                )
                if any(row["macro_auroc"] is not None for row in reachability_rows)
                else None
            ),
            "reachability_monotonic_violation": (
                float(
                    np.mean(
                        [row["monotonic_violation_rate"] for row in reachability_rows]
                    )
                )
                if reachability_rows
                else None
            ),
        },
        hashes,
    )


def _closed_loop_metrics(
    config: Any,
    analysis_spec_sha256: str,
    protocol_lock_sha256: str,
    *,
    split_role: str,
    method: str,
    backbones: tuple[int, ...],
    heads: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, str]]:
    resolved_method, method_hash, decision_hashes = load_and_resolve_method(
        config.paths.method_catalog,
        method,
        decision_root=config.paths.decision_root,
        protocol_lock={
            "analysis_spec_sha256": analysis_spec_sha256,
            "protocol_lock_sha256": protocol_lock_sha256,
        },
    )
    by_backbone: dict[str, dict[str, float]] = {}
    input_hashes: dict[str, str] = {}
    all_head_corrected: list[float] = []
    for backbone in backbones:
        per_protocol: dict[str, list[dict[str, float]]] = {
            "corrected_v1": [],
            "unmasked": [],
        }
        selected_heads = (0,) if method == "b_l2_cem" else heads
        for head in selected_heads:
            for protocol in per_protocol:
                directory = _complete_directory(
                    result_directory(
                        config,
                        split_role=split_role,
                        method=method,
                        backbone_seed=backbone,
                        head_seed=head,
                        action_protocol=protocol,
                    )
                )
                metadata, rows = load_complete_rows(directory)
                expected = (
                    metadata.get("analysis_spec_sha256") == analysis_spec_sha256
                    and metadata.get("protocol_lock_sha256") == protocol_lock_sha256
                    and metadata.get("split_role") == split_role
                    and metadata.get("method", {}).get("name") == method
                    and metadata.get("method")
                    == resolved_method.model_dump(mode="json")
                    and metadata.get("method_sha256") == method_hash
                    and metadata.get("decision_sha256s") == list(decision_hashes)
                    and int(metadata.get("backbone_seed", -1)) == backbone
                    and int(metadata.get("head_seed", -1)) == head
                    and metadata.get("action_protocol") == protocol
                    and int(metadata.get("diagnostic_limit", -1)) == 0
                )
                if not expected:
                    raise ValueError(
                        f"closed-loop result differs from protocol: {directory}"
                    )
                decisions = max(sum(int(row["path_length"]) for row in rows), 1)
                metrics = {
                    "sr": float(np.mean([float(row["success"]) for row in rows])),
                    "spl": float(np.mean([float(row["spl"]) for row in rows])),
                    "loop": float(
                        np.mean([float(row["loop_or_cycle"]) for row in rows])
                    ),
                    "assistance": float(
                        np.mean([float(row["assistance_rate"]) for row in rows])
                    ),
                    "compute": float(
                        sum(
                            int(row["plan_transitions"])
                            + int(row["fallback_transitions"])
                            for row in rows
                        )
                        / decisions
                    ),
                    "seconds_per_decision": float(
                        sum(float(row["episode_seconds"]) for row in rows) / decisions
                    ),
                }
                per_protocol[protocol].append(metrics)
                input_hashes = merge_hash_bindings(
                    input_hashes, result_evidence_hashes(directory, metadata)
                )
                if protocol == "corrected_v1":
                    all_head_corrected.append(metrics["sr"])
        by_backbone[str(backbone)] = {
            f"{protocol}_{name}": float(np.mean([row[name] for row in protocol_rows]))
            for protocol, protocol_rows in per_protocol.items()
            for name in (
                "sr",
                "spl",
                "loop",
                "assistance",
                "compute",
                "seconds_per_decision",
            )
        }
    aggregate = {
        name: float(np.mean([values[name] for values in by_backbone.values()]))
        for name in next(iter(by_backbone.values()))
    }
    aggregate["per_backbone"] = by_backbone
    aggregate["per_head_corrected_sr"] = all_head_corrected
    return aggregate, input_hashes


_FIXED_DECISIONS: dict[str, dict[str, Any]] = {
    "a_target_parent": {
        "criterion": "diagnostic",
        "eligible": ("b_dh_cem", "a1_log"),
        "split_role": "screen",
        "backbones": (42,),
        "heads": (0, 1, 2),
    },
    "a_sampling_parent": {
        "criterion": "diagnostic",
        "eligible": ("a2_distance_balanced", "a3_full_horizon"),
        "split_role": "screen",
        "backbones": (42,),
        "heads": (0, 1, 2),
    },
    "b_structural_winner": {
        "criterion": "diagnostic",
        "eligible": ("b2_bellman", "b3_multistep"),
        "split_role": "screen",
        "backbones": (42,),
        "heads": (0, 1, 2),
    },
    "b_parent": {
        "criterion": "screen",
        "eligible": (
            "b1_listwise",
            "b2_bellman",
            "b3_multistep",
            "b5_local_structural",
        ),
        "split_role": "screen",
        "backbones": (42,),
        "heads": (0, 1, 2),
    },
    "c_parent": {
        "criterion": "screen",
        "eligible": ("c1_predicted_listwise", "c2_dual_calibration"),
        "split_role": "screen",
        "backbones": (42,),
        "heads": (0, 1, 2),
    },
}

_INCUMBENT_DECISIONS = {
    "a_sampling_parent": "a_target_parent",
    "b_structural_winner": "a_sampling_parent",
    "b_parent": "a_sampling_parent",
    "c_parent": "b_parent",
}


def _expand_with_locked_incumbent(
    config: Any,
    protocol_lock: dict[str, Any],
    *,
    decision_name: str,
    candidates: tuple[str, ...],
) -> tuple[tuple[str, ...], dict[str, str], dict[str, Any] | None]:
    """Keep the upstream no-change method eligible in sequential decisions."""

    upstream_name = _INCUMBENT_DECISIONS.get(decision_name)
    if upstream_name is None:
        return candidates, {}, None
    path = resolve_path(config.paths.decision_root) / f"{upstream_name}.json"
    artifact = load_signed_artifact(
        path,
        signature_field="decision_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    if (
        artifact.get("decision_name") != upstream_name
        or artifact.get("analysis_spec_sha256") != protocol_lock["analysis_spec_sha256"]
        or artifact.get("protocol_lock_sha256") != protocol_lock["protocol_lock_sha256"]
    ):
        raise ValueError("upstream incumbent decision uses another locked protocol")
    incumbent = artifact.get("selected_method")
    if not isinstance(incumbent, str) or not incumbent:
        raise ValueError("upstream incumbent decision has no selected method")
    expanded = candidates if incumbent in candidates else (incumbent, *candidates)
    return (
        expanded,
        merge_hash_bindings(
            {path.as_posix(): sha256_file(path)}, artifact["input_hashes"]
        ),
        artifact,
    )


def _validate_decision_request(
    config: Any,
    protocol_lock: dict[str, Any],
    *,
    name: str,
    criterion: str,
    eligible: tuple[str, ...],
    baseline: str,
    split_role: str,
    backbones: tuple[int, ...],
    heads: tuple[int, ...],
) -> None:
    if baseline != "b_dh_cem":
        raise ValueError("stage decisions are locked to b_dh_cem as reference")
    requested = {
        "criterion": criterion,
        "eligible": eligible,
        "split_role": split_role,
        "backbones": backbones,
        "heads": heads,
    }
    if name in _FIXED_DECISIONS:
        if requested != _FIXED_DECISIONS[name]:
            raise ValueError(f"decision request differs from locked policy: {name}")
        return
    if name == "screen_selection":
        if (
            criterion != "screen"
            or split_role != "screen"
            or backbones != (42,)
            or heads != (0, 1, 2)
        ):
            raise ValueError("screen selection must use the locked Seed-1 protocol")
        if eligible != MAIN_CANDIDATES:
            raise ValueError(
                "screen selection must rank the complete preregistered main set"
            )
        return
    if name == "closure_selection":
        if (
            criterion != "screen"
            or split_role != "screen"
            or backbones != (42,)
            or heads != (0, 1, 2)
            or eligible != NEGATIVE_CLOSURE_CANDIDATES
        ):
            raise ValueError(
                "closure selection must rank the complete preregistered candidate set"
            )
        return
    if name == "finalist_lock":
        shortlist = load_signed_artifact(
            config.paths.shortlist_lock,
            signature_field="shortlist_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        if (
            shortlist.get("analysis_spec_sha256")
            != protocol_lock["analysis_spec_sha256"]
            or shortlist.get("protocol_lock_sha256")
            != protocol_lock["protocol_lock_sha256"]
        ):
            raise ValueError("shortlist uses another protocol lock")
        if (
            criterion != "select"
            or split_role != "select"
            or backbones != (42, 43, 44)
            or heads != (0, 1)
            or eligible != tuple(shortlist["selected_methods"])
        ):
            raise ValueError(
                "finalist decision differs from the locked shortlist protocol"
            )
        return
    raise ValueError(f"unknown preregistered decision name: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--criterion", choices=("diagnostic", "screen", "select"), required=True
    )
    parser.add_argument("--eligible", required=True)
    parser.add_argument("--baseline", default="b_dh_cem")
    parser.add_argument("--split-role", default="screen")
    parser.add_argument("--backbone-seeds", default="42")
    parser.add_argument("--head-seeds", default="0,1,2")
    args = parser.parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    protocol_lock = verify_protocol_lock(config)
    eligible_arg = args.eligible.strip()
    if eligible_arg == "@main":
        requested_eligible = MAIN_CANDIDATES
    elif eligible_arg == "@negative_closure":
        requested_eligible = NEGATIVE_CLOSURE_CANDIDATES
    else:
        requested_eligible = tuple(
            item.strip() for item in args.eligible.split(",") if item.strip()
        )
    if not requested_eligible or len(requested_eligible) != len(
        set(requested_eligible)
    ):
        raise ValueError("eligible method list must be nonempty and unique")
    backbones = _parse_ints(args.backbone_seeds)
    heads = _parse_ints(args.head_seeds)
    _validate_decision_request(
        config,
        protocol_lock,
        name=args.name,
        criterion=args.criterion,
        eligible=requested_eligible,
        baseline=args.baseline,
        split_role=args.split_role,
        backbones=backbones,
        heads=heads,
    )
    eligible, incumbent_hashes, incumbent_decision = _expand_with_locked_incumbent(
        config,
        protocol_lock,
        decision_name=args.name,
        candidates=requested_eligible,
    )
    metrics: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = dict(incumbent_hashes)
    for method in set((*eligible, args.baseline)):
        diagnostic, hashes = _diagnostic_metrics(
            config,
            protocol_lock["analysis_spec_sha256"],
            protocol_lock["protocol_lock_sha256"],
            split_role=args.split_role,
            method=method,
            backbones=backbones,
            heads=heads,
        )
        metrics[method] = dict(diagnostic)
        input_hashes = merge_hash_bindings(input_hashes, hashes)
        if args.criterion != "diagnostic":
            closed, hashes = _closed_loop_metrics(
                config,
                protocol_lock["analysis_spec_sha256"],
                protocol_lock["protocol_lock_sha256"],
                split_role=args.split_role,
                method=method,
                backbones=backbones,
                heads=heads,
            )
            metrics[method].update(closed)
            input_hashes = merge_hash_bindings(input_hashes, hashes)
    baseline = metrics[args.baseline]
    for method in eligible:
        value = metrics[method]
        value["predicted_local_gain"] = (
            value["predicted_local_top1"] - baseline["predicted_local_top1"]
        )
        value["candidate_regret_reduction"] = (
            baseline["candidate_regret"] - value["candidate_regret"]
        ) / max(abs(baseline["candidate_regret"]), 1e-8)
        if args.criterion != "diagnostic":
            value["corrected_sr_gain"] = (
                value["corrected_v1_sr"] - baseline["corrected_v1_sr"]
            )
            value["unmasked_sr_gain"] = value["unmasked_sr"] - baseline["unmasked_sr"]
            value["spl_gain"] = value["corrected_v1_spl"] - baseline["corrected_v1_spl"]
            baseline_heads = baseline["per_head_corrected_sr"]
            candidate_heads = value["per_head_corrected_sr"]
            if len(baseline_heads) == 1 and len(candidate_heads) > 1:
                baseline_heads = baseline_heads * len(candidate_heads)
            value["all_head_directions_positive"] = all(
                candidate > reference
                for candidate, reference in zip(
                    candidate_heads, baseline_heads, strict=True
                )
            )
            ranking_gate = (
                value["predicted_local_gain"] >= 0.05
                or value["candidate_regret_reduction"] >= 0.20
            )
            sr_gate = value["corrected_sr_gain"] >= config.analysis.screen_regular_delta
            secondary_gate = (
                value["unmasked_sr_gain"] >= -config.analysis.max_secondary_drop
                and value["spl_gain"] >= -config.analysis.max_secondary_drop
            )
            reachability_gate = True
            if method in {
                "d4_reachability",
                "j3_rcaux_reach",
                "p_reachability",
            }:
                reachability_gate = bool(
                    value["reachability_macro_auroc"] is not None
                    and value["reachability_macro_auroc"]
                    >= config.analysis.reachability_min_auroc
                    and value["reachability_macro_brier"]
                    <= config.analysis.reachability_max_brier
                    and value["reachability_macro_ece10"]
                    <= config.analysis.reachability_max_ece
                    and value["reachability_monotonic_violation"]
                    <= config.analysis.reachability_max_monotonic_violation
                )
            value["reachability_gate_pass"] = reachability_gate
            value["ordinary_gate_pass"] = bool(
                ranking_gate and sr_gate and secondary_gate and reachability_gate
            )
            value["strong_gate_pass"] = bool(
                value["all_head_directions_positive"]
                and value["corrected_sr_gain"] >= config.analysis.screen_strong_delta
                and ranking_gate
                and secondary_gate
                and reachability_gate
            )
            if args.criterion == "select":
                per_backbone = value["per_backbone"]
                baseline_by_backbone = baseline["per_backbone"]
                deltas = [
                    per_backbone[str(seed)]["corrected_v1_sr"]
                    - baseline_by_backbone[str(seed)]["corrected_v1_sr"]
                    for seed in backbones
                ]
                value["per_backbone_corrected_sr_gain"] = deltas
                value["seed10_expansion_pass"] = bool(
                    float(np.mean(deltas)) >= config.analysis.minimum_overall_delta
                    and sum(delta > 0 for delta in deltas) >= 2
                    and min(deltas) >= -config.analysis.max_secondary_drop
                    and ranking_gate
                    and secondary_gate
                )
    if args.criterion == "diagnostic":
        ranked = sorted(
            eligible,
            key=lambda method: (
                -metrics[method]["predicted_local_top1"],
                metrics[method]["candidate_regret"],
                metrics[method]["absolute_mae"],
                eligible.index(method),
            ),
        )
    else:
        ranked = sorted(
            eligible,
            key=lambda method: (
                -metrics[method]["corrected_v1_sr"],
                metrics[method]["candidate_regret"],
                -metrics[method]["unmasked_sr"],
                -metrics[method]["corrected_v1_spl"],
                metrics[method]["corrected_v1_compute"],
                eligible.index(method),
            ),
        )
    selected_method = ranked[0]
    selection_basis = "preregistered_lexicographic_rank"
    if args.criterion == "select":
        expansion_passing = [
            method
            for method in ranked
            if metrics[method].get("seed10_expansion_pass", False)
        ]
        if expansion_passing:
            selected_method = expansion_passing[0]
            selection_basis = "first_ranked_seed10_expansion_pass"
    payload: dict[str, Any] = {
        "schema": "distance-head-stage-decision-v1",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": protocol_lock["analysis_spec_sha256"],
        "protocol_lock_sha256": protocol_lock["protocol_lock_sha256"],
        "decision_name": args.name,
        "criterion": args.criterion,
        "split_role": args.split_role,
        "eligible_methods": list(eligible),
        "requested_candidate_methods": list(requested_eligible),
        "incumbent_method": (
            incumbent_decision["selected_method"]
            if incumbent_decision is not None
            else None
        ),
        "upstream_decision_sha256": (
            incumbent_decision["decision_sha256"]
            if incumbent_decision is not None
            else None
        ),
        "baseline": args.baseline,
        "backbone_seeds": list(backbones),
        "head_seeds_nested": list(heads),
        "metrics": metrics,
        "ranked_methods": ranked,
        "selected_method": selected_method,
        "selection_basis": selection_basis,
        "input_hashes": input_hashes,
        "evidence_status": (
            "exploratory_single_backbone"
            if len(backbones) == 1
            else "replicated_development"
        ),
    }
    payload["decision_sha256"] = canonical_json_sha256(payload)
    output = resolve_path(config.paths.decision_root) / f"{args.name}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite stage decision: {output}")
    atomic_json_dump(output, payload)
    print(Path(output))


if __name__ == "__main__":
    main()
