"""Close the study and issue a bounded positive, negative, or null conclusion."""

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
from distance_head_study.taxonomy import mechanism_family, negative_route_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument(
        "--analyses",
        required=True,
        help="Comma-separated confirmation analysis JSON files",
    )
    args = parser.parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    lock = verify_protocol_lock(config)
    opened = load_signed_artifact(
        config.paths.confirm_opened,
        signature_field="confirm_open_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    if (
        opened.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
        or opened.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
    ):
        raise ValueError("confirmation opening uses another protocol lock")
    for checkpoint_path, expected_hash in opened["locked_checkpoint_hashes"].items():
        if sha256_file(checkpoint_path) != expected_hash:
            raise ValueError("a sealed confirmation checkpoint changed")
    paths = [
        resolve_path(item.strip()) for item in args.analyses.split(",") if item.strip()
    ]
    analyses = [
        load_signed_artifact(
            path,
            signature_field="analysis_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_evidence_hashes",),
        )
        for path in paths
    ]
    if not analyses or any(
        analysis["split_role"] != "confirm" for analysis in analyses
    ):
        raise ValueError("closure requires confirmation analyses only")
    candidates = [analysis["candidate"] for analysis in analyses]
    if len(candidates) != len(set(candidates)):
        raise ValueError("closure analyses contain a duplicate candidate")
    for analysis in analyses:
        if (
            analysis.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
            or analysis.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
        ):
            raise ValueError("closure analysis uses another protocol lock")
        if analysis.get("baselines") != ["b_dh_cem", "b_l2_cem"]:
            raise ValueError("closure analysis does not use both locked baselines")
        if analysis.get("backbone_seeds") != opened["backbone_seeds"]:
            raise ValueError("closure analysis backbone seeds differ from confirmation")
        if analysis.get("head_seeds_nested_within_backbone") != [opened["head_seed"]]:
            raise ValueError("closure analysis head seed differs from confirmation")
        for input_path, expected_hash in analysis["input_evidence_hashes"].items():
            if sha256_file(input_path) != expected_hash:
                raise ValueError("a result file changed after confirmatory analysis")
    expected = [
        method
        for method in opened["allowed_methods"]
        if method not in {"b_l2_cem", "b_dh_cem"}
    ]
    if set(candidates) != set(expected):
        raise ValueError("closure analyses do not match the opened confirmation matrix")
    negative_closure = None
    if opened["claim_route"] == "positive":
        if len(analyses) != 1:
            raise ValueError("positive route has exactly one finalist")
        if analyses[0].get("new_vector_jepa_frontier_pass"):
            status = "positive_frontier"
        elif analyses[0].get("distance_head_treatment_pass"):
            status = "positive_distance_head_only"
        else:
            status = "null"
        checks = analyses[0]["positive_claim_checks"]
        if int(analyses[0].get("familywise_primary_count", 0)) < 4:
            raise ValueError("positive analysis under-corrects the primary family")
    else:
        if len(analyses) != 2:
            raise ValueError("negative route requires two mechanism-distinct finalists")
        if len({mechanism_family(candidate) for candidate in candidates}) != 2:
            raise ValueError("negative finalists are not mechanism-distinct")
        if {negative_route_group(candidate) for candidate in candidates} != {
            "frozen_scorer",
            "system_or_planner",
        }:
            raise ValueError("negative finalists do not span both required strata")
        if any(
            int(analysis.get("familywise_primary_count", 0)) < 8
            for analysis in analyses
        ):
            raise ValueError("negative analyses must correct over all eight contrasts")
        checks = {}
        passes = []
        for analysis in analyses:
            candidate = analysis["candidate"]
            candidate_checks = {}
            for baseline in ("b_dh_cem", "b_l2_cem"):
                for endpoint, threshold in (
                    ("overall", config.analysis.minimum_overall_delta),
                    ("ood", config.analysis.minimum_ood_delta),
                ):
                    result = analysis["primary_endpoints"][
                        f"{candidate}__vs__{baseline}__corrected_{endpoint}_sr"
                    ]
                    candidate_checks[f"{baseline}_{endpoint}_excludes_mei"] = (
                        result["one_sided_upper_familywise"] < threshold
                    )
            candidate_pass = all(candidate_checks.values())
            checks[candidate] = candidate_checks
            passes.append(candidate_pass)
        negative_closure = load_signed_artifact(
            config.paths.negative_closure_lock,
            signature_field="negative_closure_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        if (
            negative_closure.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
            or negative_closure.get("protocol_lock_sha256")
            != lock["protocol_lock_sha256"]
        ):
            raise ValueError("negative closure uses another protocol lock")
        checks["negative_method_family_closure"] = bool(negative_closure["complete"])
        status = (
            "bounded_negative"
            if all(passes) and negative_closure["complete"]
            else "null"
        )
    payload = {
        "schema": "distance-head-study-closure-v1",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "claim_route": opened["claim_route"],
        "conclusion_status": status,
        "checks": checks,
        "allowed_stress_methods": opened["allowed_methods"],
        "backbone_seeds": opened["backbone_seeds"],
        "head_seed": opened["head_seed"],
        "analysis_file_hashes": {path.as_posix(): sha256_file(path) for path in paths},
        "input_hashes": merge_hash_bindings(
            {
                resolve_path(config.paths.confirm_opened).as_posix(): sha256_file(
                    config.paths.confirm_opened
                ),
                **{path.as_posix(): sha256_file(path) for path in paths},
            },
            opened["input_hashes"],
            *[analysis["input_evidence_hashes"] for analysis in analyses],
            (
                {
                    resolve_path(config.paths.negative_closure_lock).as_posix(): (
                        sha256_file(config.paths.negative_closure_lock)
                    )
                }
                if negative_closure is not None
                else {}
            ),
            (negative_closure["input_hashes"] if negative_closure is not None else {}),
        ),
        "claim_boundary": (
            "Corrected/assisted pooled Vector-JEPA under this preregistered "
            "method family"
        ),
        "future_stress_results_cannot_change_primary_conclusion": True,
    }
    payload["closure_sha256"] = canonical_json_sha256(payload)
    output = resolve_path(config.paths.closure_gate)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite study closure: {output}")
    atomic_json_dump(output, payload)
    print(output)


if __name__ == "__main__":
    main()
