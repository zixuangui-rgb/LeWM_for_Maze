#!/usr/bin/env python3
"""Fail-fast AIR0 protocol, hardware, pairing, and sealed-role audit."""

from __future__ import annotations

import argparse
import hashlib
from typing import Any

import torch

from air_jepa.stage0_workspace import AIR_METHODS, PAIRING_AUDIT_BATCHES
from air_jepa.stage0_workspace.checkpoints import verify_source_lock
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    canonical_json_sha256,
    code_fingerprint,
    git_commit,
    git_worktree_dirty,
    load_config,
    prepare_new_output,
    relative_path,
    require_clean_worktree,
    resolve_path,
    runtime_metadata,
    set_seed,
    signed_payload,
    state_dict_sha256,
)
from air_jepa.stage0_workspace.data import (
    make_rng_streams,
    paired_stream_record,
    require_balanced_training_manifest,
    sample_training_batch,
    select_progressive_iterations,
)
from air_jepa.stage0_workspace.models import AIRWorkspaceModel
from air_jepa.stage0_workspace.protocol import (
    expected_matrix,
    verify_package_lock,
    verify_protocol_lock,
)
from diagnostics.common import read_jsonl
from spatial_jepa_planning.common import ManifestSampler

FORMAL_PAIRING_BATCHES = PAIRING_AUDIT_BATCHES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    parser.add_argument("--pairing-batches", type=int, default=128)
    parser.add_argument("--skip-hardware", action="store_true")
    return parser.parse_args()


def hardware_audit(*, skip: bool) -> dict[str, Any]:
    if skip:
        return {"skipped": True, "formal_eligible": False}
    count = torch.cuda.device_count()
    if count < 4:
        raise RuntimeError(f"AIR0 formal protocol requires four GPUs; found {count}")
    devices = []
    for index in range(4):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory": int(properties.total_memory),
                "capability": [properties.major, properties.minor],
            }
        )
    signatures = {
        (item["name"], item["total_memory"], tuple(item["capability"]))
        for item in devices
    }
    if len(signatures) != 1:
        raise RuntimeError("the four formal GPU workers are not hardware-identical")
    if any("H800" not in str(item["name"]).upper() for item in devices):
        raise RuntimeError("AIR0-v1 hardware lock requires four NVIDIA H800 GPUs")
    return {"skipped": False, "formal_eligible": True, "devices": devices}


def paired_stream_signature(
    entries: list[dict[str, Any]],
    *,
    seed: int,
    batches: int,
    batch_size: int,
    phase_steps: int,
    k_train: tuple[int, ...],
) -> str:
    streams = make_rng_streams(seed)
    sampler = ManifestSampler(entries)
    digest = hashlib.sha256()
    for step in range(1, batches + 1):
        batch = sample_training_batch(
            sampler,
            entry_rng=streams.entries,
            state_rng=streams.states,
            batch_size=batch_size,
            device=torch.device("cpu"),
        )
        iterations = select_progressive_iterations(
            step=step,
            phase_steps=phase_steps,
            k_train=k_train,
            rng=streams.iterations,
        )
        digest.update(
            canonical_json_sha256(
                paired_stream_record(batch, iterations=iterations)
            ).encode("ascii")
        )
    return digest.hexdigest()


def pairing_audit(config: Any, batches: int) -> dict[str, Any]:
    if batches <= 0 or batches > config.training.phase_steps:
        raise ValueError("pairing-batches must be in the first locked training phase")
    entries = read_jsonl(resolve_path(config.paths.train_manifest))
    require_balanced_training_manifest(entries)
    output: dict[str, Any] = {}
    for seed in config.seeds:
        torch.manual_seed(seed + 70_000)
        first = AIRWorkspaceModel(config.model)
        first_hash = state_dict_sha256(first.state_dict())
        torch.manual_seed(seed + 70_000)
        second = AIRWorkspaceModel(config.model)
        second_hash = state_dict_sha256(second.state_dict())
        if first_hash != second_hash:
            raise RuntimeError("paired AIR methods do not initialize identically")
        signatures = {
            method: paired_stream_signature(
                entries,
                seed=seed,
                batches=batches,
                batch_size=config.training.batch_size,
                phase_steps=config.training.phase_steps,
                k_train=config.training.k_train,
            )
            for method in AIR_METHODS
        }
        if len(set(signatures.values())) != 1:
            raise RuntimeError(
                "paired AIR methods do not receive identical sample/K streams"
            )
        output[str(seed)] = {
            "initial_model_state_sha256": first_hash,
            "sample_stream_sha256": signatures["air0_direct"],
            "checked_batches": batches,
        }
    return output


def sealed_role_audit(config: Any) -> dict[str, Any]:
    run_root = resolve_path(config.paths.run_root)
    forbidden = (
        run_root / "results" / "air_select",
        run_root / "results" / "air_final",
        run_root / "releases" / "air_select",
        run_root / "releases" / "air_final",
    )
    violations = [relative_path(path) for path in forbidden if path.exists()]
    if violations:
        raise PermissionError(f"sealed AIR roles were accessed: {violations}")
    return {"forbidden_paths_absent": [relative_path(path) for path in forbidden]}


def main() -> None:
    args = parse_args()
    if (args.skip_hardware or args.pairing_batches != FORMAL_PAIRING_BATCHES) and not (
        args.output
    ):
        raise ValueError("non-formal protocol audit requires an explicit --output")
    if not args.skip_hardware and args.pairing_batches != FORMAL_PAIRING_BATCHES:
        raise ValueError(
            f"formal L0 audit requires exactly {FORMAL_PAIRING_BATCHES} pairing batches"
        )
    require_clean_worktree(allow_dirty=False)
    config = load_config(args.config)
    set_seed(0, deterministic=True)
    protocol = verify_protocol_lock(config)
    package = verify_package_lock(config)
    source = verify_source_lock(config, deep=True)
    if expected_matrix(config) != protocol["matrix"]:
        raise ValueError("protocol lock no longer matches the executable matrix")
    payload = signed_payload(
        {
            "schema": "air-jepa-stage0-protocol-audit-v1",
            "experiment_id": config.experiment_id,
            "passed": True,
            "protocol_sha256": protocol["protocol_sha256"],
            "package_sha256": package["package_sha256"],
            "source_lock_sha256": source["source_lock_sha256"],
            "git_commit": git_commit(),
            "git_dirty": git_worktree_dirty(),
            "code_fingerprint": code_fingerprint(),
            "runtime": runtime_metadata(),
            "hardware": hardware_audit(skip=args.skip_hardware),
            "pairing": pairing_audit(config, args.pairing_batches),
            "sealed_roles": sealed_role_audit(config),
            "matrix_counts": {
                key: len(value) for key, value in protocol["matrix"].items()
            },
        },
        "protocol_audit_sha256",
    )
    if args.skip_hardware:
        unsigned = {
            key: value
            for key, value in payload.items()
            if key != "protocol_audit_sha256"
        }
        unsigned["passed"] = False
        unsigned["label"] = "NONFORMAL_HARDWARE_SKIPPED"
        payload = signed_payload(unsigned, "protocol_audit_sha256")
    output = args.output or config.paths.audit_output
    prepare_new_output(output)
    atomic_json_dump(output, payload)
    print(f"saved={relative_path(output)} passed={payload['passed']}")


if __name__ == "__main__":
    main()
