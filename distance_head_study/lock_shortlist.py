"""Lock at most two D_screen candidates before opening D_select."""

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
from distance_head_study.taxonomy import (
    mechanism_family,
    negative_route_group,
    strongest_negative_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--screen-decision", required=True)
    parser.add_argument("--negative-closure-artifact", default="")
    parser.add_argument(
        "--negative-fallback",
        action="store_true",
        help="Write the preregistered post-Seed-3 negative fallback lock.",
    )
    args = parser.parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    protocol_lock = verify_protocol_lock(config)
    decision = load_signed_artifact(
        args.screen_decision,
        signature_field="decision_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    if (
        decision.get("analysis_spec_sha256") != protocol_lock["analysis_spec_sha256"]
        or decision.get("protocol_lock_sha256") != protocol_lock["protocol_lock_sha256"]
    ):
        raise ValueError("screen decision uses another protocol lock")
    fallback_hashes: dict[str, str] = {}
    prior_shortlist = None
    prior_finalist = None
    if args.negative_fallback:
        if decision.get("decision_name") != "closure_selection":
            raise ValueError("negative fallback requires closure_selection evidence")
        prior_shortlist = load_signed_artifact(
            config.paths.shortlist_lock,
            signature_field="shortlist_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        prior_finalist_path = (
            resolve_path(config.paths.decision_root) / "finalist_lock.json"
        )
        prior_finalist = load_signed_artifact(
            prior_finalist_path,
            signature_field="decision_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        for name, artifact in (
            ("prior shortlist", prior_shortlist),
            ("prior finalist", prior_finalist),
        ):
            if (
                artifact.get("analysis_spec_sha256")
                != protocol_lock["analysis_spec_sha256"]
                or artifact.get("protocol_lock_sha256")
                != protocol_lock["protocol_lock_sha256"]
            ):
                raise ValueError(f"{name} uses another protocol lock")
        if prior_shortlist.get("negative_closure_sha256") is not None:
            raise ValueError("the original shortlist already carries negative closure")
        if tuple(prior_finalist.get("eligible_methods", ())) != tuple(
            prior_shortlist["selected_methods"]
        ):
            raise ValueError("prior finalist and shortlist differ")
        prior_selected = prior_finalist["selected_method"]
        if prior_finalist["metrics"][prior_selected].get(
            "seed10_expansion_pass", False
        ):
            raise ValueError(
                "negative fallback is forbidden after a positive Seed-3 gate"
            )
        fallback_hashes = merge_hash_bindings(
            {
                resolve_path(config.paths.shortlist_lock).as_posix(): sha256_file(
                    config.paths.shortlist_lock
                ),
                prior_finalist_path.as_posix(): sha256_file(prior_finalist_path),
            },
            prior_shortlist["input_hashes"],
            prior_finalist["input_hashes"],
        )
    ranked = list(decision["ranked_methods"])
    passing = [
        method
        for method in ranked
        if decision["metrics"][method].get("ordinary_gate_pass", False)
    ]
    closure = None
    closure_selection = decision.get("decision_name") == "closure_selection"
    if closure_selection:
        if not args.negative_closure_artifact:
            raise RuntimeError(
                "closure selection requires its complete negative-closure artifact"
            )
        closure = load_signed_artifact(
            args.negative_closure_artifact,
            signature_field="negative_closure_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        if not closure.get("complete"):
            raise ValueError("negative-closure artifact is not complete")
        if (
            closure.get("analysis_spec_sha256") != protocol_lock["analysis_spec_sha256"]
            or closure.get("protocol_lock_sha256")
            != protocol_lock["protocol_lock_sha256"]
        ):
            raise ValueError("negative closure uses another protocol lock")
        if closure.get("screen_decision_sha256") != decision["decision_sha256"]:
            raise ValueError("negative closure was created from another ranking")
    elif args.negative_closure_artifact:
        raise ValueError("negative closure can only accompany closure_selection")
    if passing:
        if closure_selection:
            primary = passing[0]
            opposite_group = (
                "system_or_planner"
                if negative_route_group(primary) == "frozen_scorer"
                else "frozen_scorer"
            )
            secondary = next(
                (
                    method
                    for method in ranked
                    if negative_route_group(method) == opposite_group
                ),
                None,
            )
            if secondary is None:
                raise ValueError("closure ranking lacks the opposite finalist stratum")
            selected = sorted((primary, secondary), key=ranked.index)
            route = "closure_ranked"
        else:
            selected = passing[:2]
            route = "positive_or_borderline"
    else:
        if not closure_selection:
            raise ValueError(
                "negative route must rank the completed main and reserve candidate set"
            )
        selected = list(strongest_negative_pair(ranked))
        route = "negative_claim_closure"
    payload = {
        "schema": "distance-head-shortlist-lock-v1",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": protocol_lock["analysis_spec_sha256"],
        "protocol_lock_sha256": protocol_lock["protocol_lock_sha256"],
        "selected_methods": selected,
        "mechanism_families": {method: mechanism_family(method) for method in selected},
        "negative_route_groups": {
            method: negative_route_group(method) for method in selected
        },
        "route": route,
        "negative_fallback_after_seed3": bool(args.negative_fallback),
        "selection_decision_path": resolve_path(args.screen_decision).as_posix(),
        "screen_decision_sha256": decision["decision_sha256"],
        "prior_shortlist_sha256": (
            prior_shortlist["shortlist_sha256"] if prior_shortlist is not None else None
        ),
        "prior_finalist_decision_sha256": (
            prior_finalist["decision_sha256"] if prior_finalist is not None else None
        ),
        "d_select_may_not_add_methods": True,
        "negative_closure_sha256": (
            closure["negative_closure_sha256"] if closure is not None else None
        ),
        "input_hashes": merge_hash_bindings(
            {
                resolve_path(args.screen_decision).as_posix(): sha256_file(
                    args.screen_decision
                )
            },
            decision["input_hashes"],
            (
                {
                    resolve_path(args.negative_closure_artifact).as_posix(): (
                        sha256_file(args.negative_closure_artifact)
                    )
                }
                if args.negative_closure_artifact
                else {}
            ),
            closure["input_hashes"] if closure is not None else {},
            fallback_hashes,
        ),
    }
    payload["shortlist_sha256"] = canonical_json_sha256(payload)
    output = resolve_path(
        config.paths.negative_shortlist_lock
        if args.negative_fallback
        else config.paths.shortlist_lock
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite shortlist lock: {output}")
    atomic_json_dump(output, payload)
    print(output)


if __name__ == "__main__":
    main()
