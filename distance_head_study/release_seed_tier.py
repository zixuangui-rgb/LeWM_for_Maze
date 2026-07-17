"""Release exactly one preregistered backbone/head seed tier."""

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
from distance_head_study.gates import (
    load_confirmation_selection,
    load_signed_artifact,
    seed_release_path,
)
from distance_head_study.protocol import verify_protocol_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--tier", choices=("seed1", "seed3", "seed10"), required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    protocol = verify_protocol_lock(config, regenerate=False)
    prerequisite_hashes: dict[str, str] = {}
    if args.tier == "seed1":
        backbones = config.seeds.screen_backbones
        heads = config.seeds.screen_head_seeds
        evidence_status = "exploratory_single_backbone"
    elif args.tier == "seed3":
        shortlist = load_signed_artifact(
            config.paths.shortlist_lock,
            signature_field="shortlist_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        if (
            shortlist.get("analysis_spec_sha256") != protocol["analysis_spec_sha256"]
            or shortlist.get("protocol_lock_sha256") != protocol["protocol_lock_sha256"]
        ):
            raise ValueError("shortlist uses another protocol")
        prerequisite_hashes = merge_hash_bindings(
            prerequisite_hashes,
            {
                resolve_path(config.paths.shortlist_lock).as_posix(): sha256_file(
                    config.paths.shortlist_lock
                )
            },
        )
        prerequisite_hashes = merge_hash_bindings(
            prerequisite_hashes, shortlist["input_hashes"]
        )
        backbones = config.seeds.select_backbones
        heads = config.seeds.select_head_seeds
        evidence_status = "replicated_development"
    else:
        n_lock = load_signed_artifact(
            config.paths.confirmation_n_lock,
            signature_field="confirmation_n_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        finalist_path, finalist, shortlist_path, shortlist = (
            load_confirmation_selection(config, n_lock)
        )
        for name, artifact in (
            ("finalist decision", finalist),
            ("confirmation n lock", n_lock),
        ):
            if (
                artifact.get("analysis_spec_sha256") != protocol["analysis_spec_sha256"]
                or artifact.get("protocol_lock_sha256")
                != protocol["protocol_lock_sha256"]
            ):
                raise ValueError(f"{name} uses another protocol")
        if n_lock.get("finalist_decision_sha256") != finalist["decision_sha256"]:
            raise ValueError("confirmation n lock and finalist decision differ")
        if (
            not finalist["metrics"][finalist["selected_method"]].get(
                "seed10_expansion_pass", False
            )
            and n_lock.get("claim_route") != "negative"
        ):
            raise RuntimeError("positive route did not pass the Seed-3 expansion gate")
        count = int(n_lock["confirmation_backbone_n"])
        backbones = config.seeds.ordered_confirmation_backbones[:count]
        heads = (config.seeds.confirmation_head_seed,)
        evidence_status = "confirmatory"
        direct_prerequisites = {}
        for path in (
            finalist_path,
            shortlist_path,
            resolve_path(config.paths.confirmation_n_lock),
        ):
            direct_prerequisites[path.as_posix()] = sha256_file(path)
        prerequisite_hashes = merge_hash_bindings(
            prerequisite_hashes,
            direct_prerequisites,
            finalist["input_hashes"],
            shortlist["input_hashes"],
            n_lock["input_hashes"],
        )
    output = seed_release_path(config, args.tier)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite seed release: {output}")
    payload = {
        "schema": "distance-head-seed-release-v1",
        "protocol_id": config.protocol_id,
        "tier": args.tier,
        "backbone_seeds": list(backbones),
        "head_seeds": list(heads),
        "evidence_status": evidence_status,
        "analysis_spec_sha256": protocol["analysis_spec_sha256"],
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "prerequisite_hashes": prerequisite_hashes,
        "no_performance_based_seed_skipping": True,
    }
    payload["release_sha256"] = canonical_json_sha256(payload)
    atomic_json_dump(output, payload)
    print(output)


if __name__ == "__main__":
    main()
