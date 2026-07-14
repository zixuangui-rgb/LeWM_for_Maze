"""Fail-closed audit of splits, source parity, methods, and the analysis lock."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from final_closure.common import next_state, read_jsonl, sha256_file
from spatial_jepa_planning.common import validate_manifest_entry
from vector_jepa_planner_frontier import ACTION_IDS, INVERSE_ACTION
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_json_dump,
    load_json,
    load_study_config,
    require_clean_worktree,
    resolve_path,
    validate_manifest_isolation,
)
from vector_jepa_planner_frontier.compat import validate_source_contract
from vector_jepa_planner_frontier.data import planner_chunk_eligibility
from vector_jepa_planner_frontier.frontier_selection import frontier_families
from vector_jepa_planner_frontier.generate_manifests import generate_entries, serialized
from vector_jepa_planner_frontier.lock_protocol import build_lock, canonical
from vector_jepa_planner_frontier.oracle_ladder import ORACLES
from vector_jepa_planner_frontier.schemas import PlannerKind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--output")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


def _check_counts(
    entries: list[dict[str, Any]], expected_count: int, role: str
) -> dict[str, int]:
    if len(entries) != expected_count:
        raise ValueError(f"{role} count mismatch: {len(entries)} != {expected_count}")
    counts = Counter(int(entry["maze_size"]) for entry in entries)
    if len(set(counts.values())) != 1:
        raise ValueError(f"{role} is not balanced across maze sizes")
    return {str(size): int(count) for size, count in sorted(counts.items())}


def _check_inverse_actions(entries: list[dict[str, Any]]) -> int:
    checked = 0
    for entry in entries:
        env = validate_manifest_entry(entry)
        free = (~env._maze_mask).reshape(-1).nonzero()[0]
        for raw_state in free.tolist():
            state = int(raw_state)
            for action in ACTION_IDS:
                successor = next_state(env, state, action)
                if successor == state:
                    continue
                restored = next_state(env, successor, INVERSE_ACTION[action])
                if restored != state:
                    raise ValueError(
                        "inverse-action mapping failed on a legal maze edge"
                    )
                checked += 1
    return checked


def audit_config(config_path: str) -> dict[str, Any]:
    config = load_study_config(config_path)
    lock = load_json(config.paths.protocol_lock)
    if lock.get("status") != "locked" or lock.get("confirmation_opened") is not False:
        raise ValueError(
            "protocol lock is missing or confirmatory data is marked opened"
        )
    if canonical(lock) != canonical(build_lock(config_path)):
        raise ValueError("protocol lock no longer reproduces from current artifacts")
    actual_analysis = analysis_spec_sha256(config, lock)
    if lock.get("analysis_spec_sha256") != actual_analysis:
        raise ValueError("analysis specification hash mismatch")
    amendments = read_jsonl(resolve_path(config.paths.amendments))
    if len(amendments) != 1:
        raise ValueError("this implementation requires exactly one pre-run amendment")
    amendment = amendments[0]
    expected_amendment = {
        "schema": "vector-jepa-protocol-amendment-v1",
        "amendment_id": "001-pre-run-implementation-clarifications",
        "validation_results_viewed": False,
        "confirmatory_results_viewed": False,
        "protocol_document_sha256": sha256_file(
            resolve_path("vector_jepa_planner_frontier/EXPERIMENT_PROTOCOL.md")
        ),
        "amendment_document_sha256": sha256_file(
            resolve_path(config.paths.amendment_document)
        ),
        "before_clause_sha256": sha256_file(
            resolve_path(config.paths.amendment_before)
        ),
        "after_clause_sha256": sha256_file(resolve_path(config.paths.amendment_after)),
        "config_sha256_after": sha256_file(resolve_path(config_path)),
    }
    for key, expected in expected_amendment.items():
        if amendment.get(key) != expected:
            raise ValueError(f"protocol amendment field mismatch: {key}")
    effective_clause = load_json(config.paths.amendment_after)
    if (
        effective_clause.get("unique_state_ratio")
        != "newly_discovered_states_excluding_initial_divided_by_max_executed_steps_1"
        or effective_clause.get("two_cycle_rate")
        != "lag_two_matches_divided_by_max_eligible_lag_two_positions_1"
    ):
        raise ValueError("secondary trajectory metric amendment changed")
    baseline = validate_source_contract(config, lock)
    overlaps = validate_manifest_isolation(config)
    roles = {
        "train": (config.paths.train_manifest, 2800),
        "development": (config.paths.development_manifest, 900),
        "validation": (
            config.paths.validation_manifest,
            config.protocol.validation_count,
        ),
        "confirmatory": (
            config.paths.confirmatory_manifest,
            config.protocol.confirmatory_count,
        ),
    }
    counts: dict[str, dict[str, int]] = {}
    all_entries: list[dict[str, Any]] = []
    entries_by_role: dict[str, list[dict[str, Any]]] = {}
    for role, (path, expected) in roles.items():
        resolved = resolve_path(path)
        entries = read_jsonl(resolved)
        record = lock[f"{role}_manifest"]
        if sha256_file(resolved) != record["sha256"]:
            raise ValueError(f"{role} manifest hash mismatch")
        counts[role] = _check_counts(entries, int(expected), role)
        entries_by_role[role] = entries
        all_entries.extend(entries)
    for role in ("validation", "confirmatory"):
        path = resolve_path(getattr(config.paths, f"{role}_manifest"))
        if path.read_text(encoding="utf-8") != serialized(generate_entries(role)):
            raise ValueError(f"{role} manifest is not deterministically reproducible")
    b0 = [
        method
        for method in config.methods
        if method.planner.kind == PlannerKind.LEGACY_CEM
    ]
    if len(b0) != 1 or b0[0].name != "b0_legacy_l2_cem":
        raise ValueError("method matrix does not contain exactly one named B0")
    p2_matrix = {
        (
            PlannerKind.CATEGORICAL_CEM
            if method.planner.kind == PlannerKind.LEGACY_CEM
            else method.planner.kind,
            method.planner.budget.multiplier,
        )
        for method in config.methods
        if method.stage == "P2"
    }
    expected_p2 = {
        (kind, budget)
        for kind in (
            PlannerKind.CATEGORICAL_CEM,
            PlannerKind.ICEM,
            PlannerKind.BEAM,
            PlannerKind.BEST_FIRST,
            PlannerKind.MCTS,
        )
        for budget in (0.5, 1.0, 4.0, 16.0)
    }
    if p2_matrix != expected_p2:
        raise ValueError("P2 must contain the complete 5-by-4 compute matrix")
    if len(config.methods) != 118:
        raise ValueError("the locked method matrix must contain exactly 118 methods")
    factorial = [
        method for method in config.methods if method.name.startswith("p3_factorial_")
    ]
    factor_codes = {method.name.rsplit("_", 1)[-1] for method in factorial}
    expected_codes = {
        f"v{v}r{r}p{p}m{m}"
        for v in (0, 1)
        for r in (0, 1)
        for p in (0, 1)
        for m in (0, 1)
    }
    if factor_codes != expected_codes:
        raise ValueError("P3 is not the complete preregistered 2^4 factorial")
    if any(method.planner.budget.multiplier != 4.0 for method in factorial):
        raise ValueError("all P3 factorial cells must use the locked 4x budget")
    expected_p4 = {
        "p4_vector_dts",
        "p4_control_dts_direct",
        "p4_control_dts_random_expansion",
        "p4_control_dts_fixed_breadth",
        "p4_bidirectional",
        "p4_control_bidirectional_forward_only",
        "p4_denoising_icem",
        "p4_control_denoising_uniform",
        "p4_control_denoising_retrieval",
        "p4_control_denoising_proposal_only",
    }
    if {
        method.name for method in config.methods if method.stage == "P4"
    } != expected_p4:
        raise ValueError("P4 does not contain every matched non-oracle control")
    joint_methods = [
        method
        for method in config.methods
        if method.stage == "P7" and method.track == "J"
    ]
    joint_grid = {
        (
            method.joint_hyperparameters.planner_learning_rate,
            method.joint_hyperparameters.backbone_lr_multiplier,
            method.joint_hyperparameters.planner_loss_weight,
            method.joint_hyperparameters.sigreg_multiplier,
        )
        for method in joint_methods
    }
    expected_joint_grid = {
        (planner_lr, backbone_lr, planner_weight, sigreg)
        for planner_lr in (0.0001, 0.0003)
        for backbone_lr in (0.01, 0.03, 0.1)
        for planner_weight in (0.1, 0.3, 1.0)
        for sigreg in (0.5, 1.0, 2.0)
    }
    if joint_grid != expected_joint_grid or len(joint_methods) != 54:
        raise ValueError("P7 does not contain the complete 2x3x3x3 joint grid")
    families = frontier_families(
        config,
        selected_track_j="p7_track_j_joint_all",
    )
    p8_names = {method.name for method in config.methods if method.stage == "P8"}
    expected_confirmation = {
        "b0_legacy_l2_cem",
        "p5_track_f_all_hard_memory",
        "p6_track_f_counterexample_ranked",
        *(method.name for method in joint_methods),
        *p8_names,
    }
    if {
        method.name for method in config.methods if method.confirmatory_eligible
    } != expected_confirmation:
        raise ValueError("confirmatory candidate pool changed")
    if len(families) != 3 or len(p8_names) != 9:
        raise ValueError("P8 does not contain three complete compute frontiers")
    by_name = {method.name: method for method in config.methods}
    p6 = by_name["p6_track_f_counterexample_ranked"]
    aligned_control = by_name["p7_control_action_aligned_frozen"]
    if (
        aligned_control.reuse_component_from != p6.name
        or aligned_control.initialization_parent is not None
        or aligned_control.trainable_components is not None
        or aligned_control.track != "F"
        or aligned_control.planner.rollout_semantics.value != "action_aligned_v2"
        or aligned_control.scorer != p6.scorer
        or aligned_control.proposal != p6.proposal
        or aligned_control.memory != p6.memory
        or aligned_control.control != p6.control
    ):
        raise ValueError(
            "P7 action-alignment control no longer isolates rollout semantics"
        )
    if config.protocol.primary_action_selection != "corrected_v1":
        raise ValueError("Corrected-v1 must remain the locked primary endpoint")
    if config.protocol.allow_confirmatory_model_selection:
        raise ValueError("confirmatory model selection must remain disabled")
    if config.training.checkpoint_selection != "final_step":
        raise ValueError("amendment 001 requires uniform final-step checkpoints")
    if ORACLES != (
        "O0",
        "O1_PROP",
        "O2_SELECT",
        "O3_DYN",
        "O4_VALUE",
        "O5_JOIN",
        "O6_VALID_FUTURE",
    ):
        raise ValueError("oracle ladder no longer matches the locked protocol")
    inverse_edges = _check_inverse_actions(all_entries)
    chunk_eligibility = planner_chunk_eligibility(entries_by_role["train"], horizon=12)
    if chunk_eligibility["eligible_count"] < 0.99 * len(entries_by_role["train"]):
        raise ValueError(
            "fewer than 99% of train topologies support a full action chunk"
        )
    return {
        "status": "passed",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": actual_analysis,
        "source_baseline": baseline["name"],
        "method_count": len(config.methods),
        "factorial_method_count": len(factorial),
        "frontier_alias_count": len(p8_names),
        "oracle_rung_count": len(ORACLES),
        "confirmatory_eligible": [
            method.name for method in config.methods if method.confirmatory_eligible
        ],
        "counts_by_size": counts,
        "holdout_checks": overlaps,
        "inverse_edges_checked": inverse_edges,
        "planner_chunk_eligibility": chunk_eligibility,
    }


def main() -> None:
    args = parse_args()
    if args.formal:
        require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    report = audit_config(args.config)
    if args.output:
        atomic_json_dump(args.output, report)


if __name__ == "__main__":
    main()
