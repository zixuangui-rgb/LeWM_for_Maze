"""Fail-closed scientific and compatibility audit for the quick package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from a1_quick_validation import ALL_METHODS, NEW_METHODS
from a1_quick_validation.common import (
    DEFAULT_PROFILE,
    atomic_json_dump,
    canonical_json_sha256,
    load_json,
    resolve_path,
)
from a1_quick_validation.profile import (
    verify_package_lock,
    verify_reproduction_contract,
)
from distance_head_study.common import (
    load_study_config,
    read_jsonl,
    sha256_file,
)
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.protocol import verify_protocol_lock


def _manifest_ids(path: str | Path) -> tuple[set[str], set[str]]:
    rows = read_jsonl(path)
    return (
        {str(row["layout_hash"]) for row in rows},
        {str(row["task_hash"]) for row in rows},
    )


def _overlap_counts(
    left: tuple[set[str], set[str]], right: tuple[set[str], set[str]]
) -> dict[str, int]:
    return {
        "layout": len(left[0] & right[0]),
        "task": len(left[1] & right[1]),
    }


def run_audit(profile_path: str | Path = DEFAULT_PROFILE) -> dict[str, Any]:
    profile, package_lock, quick_lock = verify_package_lock(profile_path)
    reproduction = verify_reproduction_contract(profile)
    quick_config = load_study_config(profile.paths.quick_config)
    source_config = load_study_config(profile.paths.source_config)
    source_lock = verify_protocol_lock(source_config, regenerate=True)
    verify_protocol_lock(quick_config, regenerate=True)
    locked_sections = ("splits", "seeds", "planner", "training", "analysis")
    if reproduction["quick_comparison_contract"]["exact_inherited_sections"] != list(
        locked_sections
    ):
        raise ValueError("reproduction contract changes the inherited section set")
    parity = {}
    for section in locked_sections:
        source_value = getattr(source_config, section).model_dump(mode="json")
        quick_value = getattr(quick_config, section).model_dump(mode="json")
        if source_value != quick_value:
            raise ValueError(f"quick protocol changed locked {section}")
        parity[section] = canonical_json_sha256(source_value)
    manifest_roles = ("train", "legacy", "cal", "screen", "select", "confirm", "stress")
    manifests = {}
    for role in manifest_roles:
        source_path = resolve_path(getattr(source_config.paths, f"{role}_manifest"))
        quick_path = resolve_path(getattr(quick_config.paths, f"{role}_manifest"))
        if source_path != quick_path or sha256_file(source_path) != sha256_file(
            quick_path
        ):
            raise ValueError(f"quick protocol changed the {role} manifest")
        manifests[role] = {
            "path": source_path.as_posix(),
            "sha256": sha256_file(source_path),
            "count": len(read_jsonl(source_path)),
        }
    train_ids = _manifest_ids(quick_config.paths.train_manifest)
    cal_ids = _manifest_ids(quick_config.paths.cal_manifest)
    if not cal_ids[0] < train_ids[0]:
        raise ValueError("calibration layouts are not a strict training-data subset")
    heldout_overlap = {}
    heldout_roles = ("screen", "select", "legacy", "confirm", "stress")
    heldout_ids = {
        role: _manifest_ids(getattr(quick_config.paths, f"{role}_manifest"))
        for role in heldout_roles
    }
    for role, identifiers in heldout_ids.items():
        overlap = _overlap_counts(train_ids, identifiers)
        heldout_overlap[role] = overlap
        if any(overlap.values()):
            raise ValueError(f"train/{role} topology or task leakage")
    heldout_pair_overlap = {}
    for index, left in enumerate(heldout_roles):
        for right in heldout_roles[index + 1 :]:
            overlap = _overlap_counts(heldout_ids[left], heldout_ids[right])
            heldout_pair_overlap[f"{left}__{right}"] = overlap
            if any(overlap.values()):
                raise ValueError(f"{left}/{right} topology or task leakage")
    resolved = {}
    for method_name in ALL_METHODS:
        method, method_hash, decisions = load_and_resolve_method(
            quick_config.paths.method_catalog,
            method_name,
            decision_root=quick_config.paths.decision_root,
            protocol_lock=quick_lock,
        )
        if decisions:
            raise ValueError("quick methods must not have dynamic parents")
        if method.uses_test_bfs:
            raise ValueError("quick ranking method uses test-time BFS")
        if method_name in NEW_METHODS and method.training_scope.value != "frozen":
            raise ValueError("quick method unexpectedly updates JEPA parameters")
        resolved[method_name] = {
            "sha256": method_hash,
            "role": method.role,
            "head": method.head.model_dump(mode="json") if method.head else None,
            "objectives": (
                method.objectives.model_dump(mode="json") if method.objectives else None
            ),
            "planner": method.planner.model_dump(mode="json"),
        }
    for method_name in ("b_l2_cem", "b_dh_cem", "a1_log"):
        source_method, source_hash, source_decisions = load_and_resolve_method(
            source_config.paths.method_catalog,
            method_name,
            decision_root=source_config.paths.decision_root,
            protocol_lock=source_lock,
        )
        quick_method, quick_hash, _ = load_and_resolve_method(
            quick_config.paths.method_catalog,
            method_name,
            decision_root=quick_config.paths.decision_root,
            protocol_lock=quick_lock,
        )
        if (
            source_decisions
            or source_method != quick_method
            or source_hash != quick_hash
        ):
            raise ValueError(f"reference method parity failed: {method_name}")
    if tuple(reproduction["quick_comparison_contract"]["exact_reference_methods"]) != (
        "b_l2_cem",
        "b_dh_cem",
        "a1_log",
    ):
        raise ValueError("reproduction contract changes the reference method set")

    historical = reproduction["historical_distance_head"]
    b_dh = resolved["b_dh_cem"]["head"]
    expected_head = {
        "architecture": historical["architecture"],
        "hidden_dims": historical["hidden_dims"],
        "target": historical["resolved_target_name"],
        "regression_loss": historical["resolved_loss_name"],
    }
    if any(b_dh.get(key) != value for key, value in expected_head.items()):
        raise ValueError("DistanceHead baseline differs from historical core fields")
    baseline_provenance = load_json(quick_config.paths.baseline_provenance)
    provenance_head = baseline_provenance.get("head", {})
    for key in (
        "architecture",
        "hidden_dims",
        "steps",
        "effective_batch_size",
        "pairs_per_topology",
        "learning_rate",
        "weight_decay",
    ):
        if provenance_head.get(key) != historical.get(key):
            raise ValueError(f"baseline provenance differs for {key}")
    if (
        baseline_provenance.get("protocol_repair", {}).get(
            "historical_checkpoint_selection"
        )
        != reproduction["deliberate_nonidentity_with_raw_history"][
            "checkpoint_selection"
        ]["historical"]
        or baseline_provenance.get("protocol_repair", {}).get(
            "new_checkpoint_selection"
        )
        != reproduction["deliberate_nonidentity_with_raw_history"][
            "checkpoint_selection"
        ]["formal"]
    ):
        raise ValueError("checkpoint-selection repair differs from its provenance")

    source_closure = load_json(quick_config.paths.source_config)
    vector_baseline = next(
        item
        for item in source_closure["baselines"]
        if item["name"] == "lewm_l2_cem_seqlen2"
    )
    vector_contract = reproduction["historical_vector_jepa"]
    vector_expected = {
        "sequence_length": vector_contract["sequence_length"],
        "latent_dim": vector_contract["latent_dim"],
        "steps": vector_contract["training_steps"],
        "batch_size": vector_contract["training_batch_size"],
        "checkpoint_selection": vector_contract["checkpoint_selection"],
    }
    if any(
        vector_baseline["train"].get(key) != value
        for key, value in vector_expected.items()
    ):
        raise ValueError("corrected Vector-JEPA anchor differs from its lineage lock")
    planner_contract = reproduction["corrected_planner"]
    planner_expected = {
        "history_size": quick_config.planner.history_size,
        "horizon": quick_config.planner.horizon,
        "num_candidates": quick_config.planner.num_candidates,
        "num_elites": quick_config.planner.num_elites,
        "cem_iters": quick_config.planner.cem_iters,
        "momentum": quick_config.planner.momentum,
        "max_steps": quick_config.planner.max_steps,
        "allowed_actions": list(quick_config.planner.action_ids),
        "rollout_semantics": quick_config.planner.rollout_semantics,
        "formal_action_protocols": list(quick_config.planner.action_protocols),
    }
    if any(
        planner_contract.get(key) != value for key, value in planner_expected.items()
    ):
        raise ValueError("quick planner differs from the corrected lineage contract")
    if set(profile.q1.methods) != set(ALL_METHODS):
        raise ValueError("Q1 does not cover the complete locked method set")
    if (
        resolved["a1_reach"]["planner"] != resolved["a1_hcond"]["planner"]
        or resolved["a1_reach"]["planner"] != resolved["a1_log"]["planner"]
        or resolved["a1_reach"]["planner"]["cost"] != "terminal_distance"
        or resolved["a1_reach"]["planner"]["reachability_weight"] != 0.0
    ):
        raise ValueError(
            "A1 reachability auxiliary unexpectedly changes the planner cost"
        )
    if profile.q2.split_role != "select" or profile.q3.split_role != "legacy":
        raise ValueError("quick stage split roles changed")
    if (
        profile.q1.dynamic_method_source != "none"
        or profile.q2.dynamic_method_source != "q1_shortlist"
        or profile.q3.dynamic_method_source != "q2_winner"
    ):
        raise ValueError("quick dynamic method sources changed")
    if (
        reproduction["quick_comparison_contract"]["q1_head_seeds"]
        != list(profile.q1.head_seeds)
        or reproduction["quick_comparison_contract"]["q2_head_seeds"]
        != list(profile.q2.head_seeds)
        or reproduction["quick_comparison_contract"]["backbone_seed"] != 42
    ):
        raise ValueError("quick seed matrix differs from the reproduction contract")
    output = {
        "schema": "a1-quick-validation-audit-v1",
        "profile_id": profile.profile_id,
        "package_lock_sha256": package_lock["package_lock_sha256"],
        "quick_analysis_spec_sha256": quick_lock["analysis_spec_sha256"],
        "quick_protocol_lock_sha256": quick_lock["protocol_lock_sha256"],
        "source_analysis_spec_sha256": source_lock["analysis_spec_sha256"],
        "source_protocol_lock_sha256": source_lock["protocol_lock_sha256"],
        "locked_section_parity": parity,
        "manifests": manifests,
        "calibration_layouts_are_train_subset": True,
        "heldout_train_overlap": heldout_overlap,
        "heldout_pair_overlap": heldout_pair_overlap,
        "methods": resolved,
        "reproduction_contract": {
            "path": resolve_path(profile.paths.reproduction_contract).as_posix(),
            "sha256": sha256_file(profile.paths.reproduction_contract),
            "lineage_layers": reproduction["lineage_layers"],
            "deliberate_nonidentity_with_raw_history": reproduction[
                "deliberate_nonidentity_with_raw_history"
            ],
        },
        "planner_parity": True,
        "reachability_auxiliary_only": True,
        "training_parity": True,
        "topology_holdout_preserved": True,
        "test_bfs_enters_action_selection": False,
        "confirmatory_claim_allowed": False,
        "claim_boundary": profile.claim_boundary,
    }
    output["audit_sha256"] = canonical_json_sha256(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = run_audit(args.profile)
    if args.output:
        path = atomic_json_dump(args.output, result)
        print(path)
    else:
        print(result["audit_sha256"])


if __name__ == "__main__":
    main()
