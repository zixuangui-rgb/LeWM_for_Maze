"""Open sealed D_confirm only after finalist, n, and seed-release locks."""

from __future__ import annotations

import argparse

import torch

from distance_head_study.common import (
    atomic_json_dump,
    canonical_json_sha256,
    head_checkpoint_path,
    load_study_config,
    merge_hash_bindings,
    require_clean_worktree,
    resolve_path,
    sha256_file,
    source_backbone_path,
    validate_backbone_protocol_binding,
)
from distance_head_study.evaluate import _checkpoint_owner
from distance_head_study.gates import (
    load_confirmation_selection,
    load_signed_artifact,
    seed_release_path,
)
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.plan_jobs import _existing_head
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.taxonomy import mechanism_family, negative_route_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    args = parser.parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    lock = verify_protocol_lock(config)
    n_lock = load_signed_artifact(
        config.paths.confirmation_n_lock,
        signature_field="confirmation_n_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("input_hashes",),
    )
    finalist_path, finalist, shortlist_path, shortlist = load_confirmation_selection(
        config, n_lock
    )
    release = load_signed_artifact(
        seed_release_path(config, "seed10"),
        signature_field="release_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("prerequisite_hashes",),
    )
    for name, artifact in (
        ("finalist decision", finalist),
        ("shortlist", shortlist),
        ("confirmation n lock", n_lock),
    ):
        if (
            artifact.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
            or artifact.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
        ):
            raise ValueError(f"{name} uses another protocol lock")
    if (
        release.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
        or release.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
    ):
        raise ValueError("Seed-10 release uses another protocol lock")
    if n_lock.get("finalist_decision_sha256") != finalist["decision_sha256"]:
        raise ValueError("confirmation n lock and finalist decision differ")
    if n_lock.get("shortlist_sha256") != shortlist["shortlist_sha256"]:
        raise ValueError("confirmation n lock and shortlist differ")
    if release["backbone_seeds"] != n_lock["ordered_backbone_seeds"]:
        raise ValueError("Seed-10 release and confirmation n lock differ")
    if n_lock["claim_route"] == "positive":
        finalists = [finalist["selected_method"]]
        if not finalist["metrics"][finalists[0]].get("seed10_expansion_pass", False):
            raise RuntimeError(
                "positive finalist did not pass the Seed-3 expansion gate"
            )
    else:
        finalists = [
            method
            for method in finalist["ranked_methods"]
            if method in shortlist["selected_methods"]
        ][:2]
        if len(finalists) != 2:
            raise ValueError("negative confirmation requires two locked finalists")
        families = {mechanism_family(method) for method in finalists}
        if len(families) != 2:
            raise ValueError(
                "negative confirmation finalists are not mechanism-distinct"
            )
        if {negative_route_group(method) for method in finalists} != {
            "frozen_scorer",
            "system_or_planner",
        }:
            raise ValueError(
                "negative confirmation must span frozen and system/planner strata"
            )
    allowed = ["b_l2_cem", "b_dh_cem", *finalists]
    checkpoint_hashes = {}
    for backbone_seed in n_lock["ordered_backbone_seeds"]:
        backbone = source_backbone_path(config, int(backbone_seed))
        if not backbone.exists():
            raise FileNotFoundError(backbone)
        backbone_hash = sha256_file(backbone)
        checkpoint_hashes = merge_hash_bindings(
            checkpoint_hashes, {backbone.as_posix(): backbone_hash}
        )
        backbone_payload = torch.load(backbone, map_location="cpu", weights_only=False)
        validate_backbone_protocol_binding(
            config,
            backbone_payload,
            backbone_seed=int(backbone_seed),
            protocol_lock=lock,
        )
        for method_name in ["b_dh_cem", *finalists]:
            method, _, _ = load_and_resolve_method(
                config.paths.method_catalog,
                method_name,
                decision_root=config.paths.decision_root,
                protocol_lock=lock,
            )
            owner, _ = _checkpoint_owner(config, method, protocol_lock=lock)
            if owner is None:
                continue
            if not _existing_head(
                config,
                lock,
                owner=owner,
                backbone_seed=int(backbone_seed),
                head_seed=int(n_lock["head_seed"]),
            ):
                raise FileNotFoundError("confirmation head checkpoint is missing")
            checkpoint = head_checkpoint_path(
                config,
                method=owner,
                backbone_seed=int(backbone_seed),
                head_seed=int(n_lock["head_seed"]),
            )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            if (
                checkpoint_payload.get("protocol_id") != config.protocol_id
                or not checkpoint_payload.get("formal_run")
                or checkpoint_payload.get("checkpoint_selection") != "final_step"
                or int(checkpoint_payload.get("final_step", -1))
                != config.training.steps
                or checkpoint_payload.get("analysis_spec_sha256")
                != lock["analysis_spec_sha256"]
                or checkpoint_payload.get("protocol_lock_sha256")
                != lock["protocol_lock_sha256"]
                or checkpoint_payload.get("backbone_sha256") != backbone_hash
                or checkpoint_payload.get("method", {}).get("name") != owner
            ):
                raise ValueError("confirmation head checkpoint provenance mismatch")
            checkpoint_hashes = merge_hash_bindings(
                checkpoint_hashes,
                {checkpoint.as_posix(): sha256_file(checkpoint)},
            )
    payload = {
        "schema": "distance-head-confirm-open-v1",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "claim_route": n_lock["claim_route"],
        "allowed_methods": allowed,
        "backbone_seeds": n_lock["ordered_backbone_seeds"],
        "head_seed": n_lock["head_seed"],
        "finalist_decision_sha256": finalist["decision_sha256"],
        "shortlist_sha256": shortlist["shortlist_sha256"],
        "confirmation_n_sha256": n_lock["confirmation_n_sha256"],
        "seed10_release_sha256": release["release_sha256"],
        "confirm_manifest_sha256": sha256_file(config.paths.confirm_manifest),
        "locked_checkpoint_hashes": checkpoint_hashes,
        "input_hashes": merge_hash_bindings(
            {
                finalist_path.as_posix(): sha256_file(finalist_path),
                shortlist_path.as_posix(): sha256_file(shortlist_path),
                resolve_path(config.paths.confirmation_n_lock).as_posix(): sha256_file(
                    config.paths.confirmation_n_lock
                ),
                seed_release_path(config, "seed10").as_posix(): sha256_file(
                    seed_release_path(config, "seed10")
                ),
            },
            finalist["input_hashes"],
            shortlist["input_hashes"],
            n_lock["input_hashes"],
            release["prerequisite_hashes"],
            checkpoint_hashes,
        ),
        "no_further_model_selection": True,
    }
    payload["confirm_open_sha256"] = canonical_json_sha256(payload)
    output = resolve_path(config.paths.confirm_opened)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite confirmation-open artifact: {output}"
        )
    atomic_json_dump(output, payload)
    print(output)


if __name__ == "__main__":
    main()
