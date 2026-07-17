"""Lock confirmation n from baseline-only power before fresh training."""

from __future__ import annotations

import argparse

from distance_head_study.common import (
    atomic_json_dump,
    canonical_json_sha256,
    load_study_config,
    merge_hash_bindings,
    require_clean_worktree,
    resolve_path,
    sha256_file,
)
from distance_head_study.gates import load_signed_artifact
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.taxonomy import negative_route_group


def _load_current_artifact(
    path,
    *,
    signature_field: str,
    config,
    lock: dict,
) -> dict:
    artifact = load_signed_artifact(
        path,
        signature_field=signature_field,
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    if (
        artifact.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
        or artifact.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
    ):
        raise ValueError(f"selection artifact uses another protocol lock: {path}")
    return artifact


def _confirmation_selection(config, lock: dict, claim_route: str):
    """Resolve the immutable positive path or post-Seed-3 negative fallback."""

    primary_finalist_path = (
        resolve_path(config.paths.decision_root) / "finalist_lock.json"
    )
    primary_shortlist_path = resolve_path(config.paths.shortlist_lock)
    primary_finalist = _load_current_artifact(
        primary_finalist_path,
        signature_field="decision_sha256",
        config=config,
        lock=lock,
    )
    primary_shortlist = _load_current_artifact(
        primary_shortlist_path,
        signature_field="shortlist_sha256",
        config=config,
        lock=lock,
    )
    if tuple(primary_finalist["eligible_methods"]) != tuple(
        primary_shortlist["selected_methods"]
    ):
        raise ValueError("primary finalist decision does not match its shortlist")
    selected = primary_finalist["selected_method"]
    positive_evidence = bool(
        primary_finalist["metrics"][selected].get("seed10_expansion_pass", False)
    )
    expected_route = "positive" if positive_evidence else "negative"
    if claim_route != expected_route:
        raise ValueError(
            f"claim route must follow the locked Seed-3 evidence: {expected_route}"
        )
    if claim_route == "positive" or primary_shortlist.get("negative_closure_sha256"):
        return (
            primary_finalist_path,
            primary_finalist,
            primary_shortlist_path,
            primary_shortlist,
            "d_select_finalist",
        )

    fallback_shortlist_path = resolve_path(config.paths.negative_shortlist_lock)
    fallback_shortlist = _load_current_artifact(
        fallback_shortlist_path,
        signature_field="shortlist_sha256",
        config=config,
        lock=lock,
    )
    if (
        not fallback_shortlist.get("negative_fallback_after_seed3")
        or fallback_shortlist.get("prior_shortlist_sha256")
        != primary_shortlist["shortlist_sha256"]
        or fallback_shortlist.get("prior_finalist_decision_sha256")
        != primary_finalist["decision_sha256"]
    ):
        raise ValueError("negative fallback is not bound to the failed Seed-3 path")
    fallback_decision_path = resolve_path(
        str(fallback_shortlist.get("selection_decision_path", ""))
    )
    fallback_decision = _load_current_artifact(
        fallback_decision_path,
        signature_field="decision_sha256",
        config=config,
        lock=lock,
    )
    selected_methods = list(fallback_shortlist["selected_methods"])
    ranked_selected = [
        method
        for method in fallback_decision["ranked_methods"]
        if method in selected_methods
    ]
    if (
        fallback_decision.get("decision_name") != "closure_selection"
        or fallback_shortlist.get("screen_decision_sha256")
        != fallback_decision["decision_sha256"]
        or ranked_selected != selected_methods
    ):
        raise ValueError("negative fallback shortlist and closure ranking differ")
    return (
        fallback_decision_path,
        fallback_decision,
        fallback_shortlist_path,
        fallback_shortlist,
        "screen_closure_fallback",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--power-artifact", required=True)
    parser.add_argument(
        "--claim-route", choices=("positive", "negative"), required=True
    )
    args = parser.parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    lock = verify_protocol_lock(config)
    power = load_signed_artifact(
        args.power_artifact,
        signature_field="power_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    if not power.get("baseline_only") or not power.get("does_not_use_candidate_effect"):
        raise ValueError("confirmation n must come from baseline-only inputs")
    if (
        power.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
        or power.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
    ):
        raise ValueError("power artifact uses another protocol lock")
    (
        finalist_path,
        finalist,
        shortlist_path,
        shortlist,
        selection_source,
    ) = _confirmation_selection(config, lock, args.claim_route)
    negative_closure_hash = shortlist.get("negative_closure_sha256")
    closure = None
    if args.claim_route == "negative":
        if not negative_closure_hash or len(shortlist["selected_methods"]) != 2:
            raise ValueError(
                "negative route lacks a complete two-method closure shortlist"
            )
        if {
            negative_route_group(method) for method in shortlist["selected_methods"]
        } != {"frozen_scorer", "system_or_planner"}:
            raise ValueError("negative shortlist does not span both method strata")
        closure = load_signed_artifact(
            config.paths.negative_closure_lock,
            signature_field="negative_closure_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        if closure["negative_closure_sha256"] != negative_closure_hash:
            raise ValueError("shortlist and negative closure hashes differ")
        if (
            closure.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
            or closure.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
        ):
            raise ValueError("negative closure uses another protocol lock")
    recommendation_key = f"recommended_confirmation_n_{args.claim_route}"
    count = max(
        config.analysis.minimum_confirmation_backbones,
        int(power[recommendation_key]),
    )
    seeds = config.seeds.ordered_confirmation_backbones[:count]
    if len(seeds) != count:
        raise ValueError("ordered confirmation seed list is too short")
    payload = {
        "schema": "distance-head-confirmation-n-lock-v1",
        "protocol_id": config.protocol_id,
        "claim_route": args.claim_route,
        "confirmation_backbone_n": count,
        "ordered_backbone_seeds": list(seeds),
        "head_seed": config.seeds.confirmation_head_seed,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "power_artifact_sha256": power["power_sha256"],
        "power_artifact_file_sha256": sha256_file(args.power_artifact),
        "candidate_effect_used": False,
        "selection_source": selection_source,
        "finalist_decision_path": finalist_path.as_posix(),
        "finalist_decision_sha256": finalist["decision_sha256"],
        "shortlist_path": shortlist_path.as_posix(),
        "shortlist_sha256": shortlist["shortlist_sha256"],
        "negative_closure_sha256": negative_closure_hash,
        "input_hashes": merge_hash_bindings(
            {
                resolve_path(args.power_artifact).as_posix(): sha256_file(
                    args.power_artifact
                ),
                finalist_path.as_posix(): sha256_file(finalist_path),
                shortlist_path.as_posix(): sha256_file(shortlist_path),
            },
            power["input_hashes"],
            finalist["input_hashes"],
            shortlist["input_hashes"],
            (
                {
                    resolve_path(config.paths.negative_closure_lock).as_posix(): (
                        sha256_file(config.paths.negative_closure_lock)
                    )
                }
                if closure is not None
                else {}
            ),
            closure["input_hashes"] if closure is not None else {},
        ),
    }
    payload["confirmation_n_sha256"] = canonical_json_sha256(payload)
    output = resolve_path(config.paths.confirmation_n_lock)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite confirmation n lock: {output}")
    atomic_json_dump(output, payload)
    print(output)


if __name__ == "__main__":
    main()
