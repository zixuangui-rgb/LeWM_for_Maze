#!/usr/bin/env python3
"""Performance-blind L0 K128 throughput and numerical preflight benchmark."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import torch

from air_jepa.stage0_workspace.checkpoints import (
    load_frozen_representation,
    load_source_planner,
    verify_source_lock,
)
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    code_fingerprint,
    git_commit,
    git_worktree_dirty,
    load_config,
    prepare_new_output,
    relative_path,
    require_clean_worktree,
    require_h800_device,
    resolve_device,
    resolve_path,
    runtime_metadata,
    set_seed,
    signed_payload,
)
from air_jepa.stage0_workspace.data import make_rng_streams, sample_training_batch
from air_jepa.stage0_workspace.losses import air_loss
from air_jepa.stage0_workspace.models import AIRWorkspaceModel, require_finite_output
from air_jepa.stage0_workspace.protocol import (
    verify_package_lock,
    verify_protocol_lock,
)
from diagnostics.common import observe_state, read_jsonl
from spatial_jepa_planning.common import ManifestSampler, validate_manifest_entry

FORMAL_BENCHMARK_TASKS = 50
FORMAL_BACKWARD_REPEATS = 5
FORMAL_BACKWARD_K = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--backward-repeats", type=int, default=5)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if (
        args.tasks != FORMAL_BENCHMARK_TASKS
        or args.backward_repeats != FORMAL_BACKWARD_REPEATS
    ):
        raise ValueError(
            "formal L0 benchmark requires exactly 50 tasks and 5 backward repeats"
        )
    require_clean_worktree(allow_dirty=False)
    config = load_config(args.config)
    output = args.output or config.paths.benchmark_output
    prepare_new_output(output)
    protocol = verify_protocol_lock(config)
    package = verify_package_lock(config)
    source = verify_source_lock(config)
    device = resolve_device(args.device)
    require_h800_device(device)
    set_seed(42 + 70_000, deterministic=True)
    representation, _ = load_frozen_representation(
        config, seed=42, device=device, source_lock=source
    )
    _, _, j1_checkpoint = load_source_planner(
        config,
        seed=42,
        method="j1_receding",
        device=device,
        source_lock=source,
    )
    model = AIRWorkspaceModel(config.model).to(device)
    preflight_entries = read_jsonl(resolve_path(config.paths.preflight_manifest))
    selected = preflight_entries[: args.tasks]
    if len(selected) != args.tasks:
        raise ValueError("preflight manifest does not contain 50 benchmark tasks")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    forward_times: list[float] = []
    with torch.no_grad():
        for entry in selected:
            env = validate_manifest_entry(entry, check_bfs=False)
            observation = torch.as_tensor(
                observe_state(env, int(entry["start_cell"])),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            latent = representation.planning_latent(observation)
            mask = torch.ones(
                (1, latent.shape[-2], latent.shape[-1]),
                dtype=torch.bool,
                device=device,
            )
            synchronize(device)
            started = time.perf_counter()
            output = model(latent, iterations=128, valid_mask=mask)[-1]
            synchronize(device)
            forward_times.append(time.perf_counter() - started)
            require_finite_output(output)

    train_entries = read_jsonl(resolve_path(config.paths.train_manifest))
    sampler = ManifestSampler(train_entries)
    streams = make_rng_streams(42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    backward_times: list[float] = []
    model.train()
    for _ in range(args.backward_repeats):
        batch = sample_training_batch(
            sampler,
            entry_rng=streams.entries,
            state_rng=streams.states,
            batch_size=config.training.batch_size,
            device=device,
        )
        batch_size, actions, height, width, channels = (
            batch.successor_observations.shape
        )
        with torch.no_grad():
            source_latent = representation.planning_latent(batch.current_observation)
            successor_latent = representation.planning_latent(
                batch.successor_observations.reshape(
                    batch_size * actions, height, width, channels
                )
            ).reshape(batch_size, actions, -1, height, width)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        synchronize(device)
        started = time.perf_counter()
        outputs = model(
            source_latent,
            iterations=FORMAL_BACKWARD_K,
            deep_supervision_every=config.training.deep_supervision_every,
            valid_mask=mask,
        )
        result = air_loss(
            outputs,
            successor_latent=successor_latent,
            source_latent=source_latent,
            candidate_distances=batch.candidate_distances,
            optimal_action_mask=batch.optimal_action_mask,
            valid_mask=mask,
            weights=config.training.methods["air0_jepa"],
            max_distance=config.model.max_distance,
            target_variance_epsilon=config.training.target_variance_epsilon,
        )
        optimizer.zero_grad(set_to_none=True)
        result.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        synchronize(device)
        backward_times.append(time.perf_counter() - started)

    workspace_macs = {
        str(size): {
            str(k): model.analytical_macs(size, k) for k in config.evaluation.k_values
        }
        for size in (21, 25)
    }
    workspace_macs_by_component = {
        str(size): {
            str(k): model.analytical_mac_breakdown(size, k)
            for k in config.evaluation.k_values
        }
        for size in (21, 25)
    }
    if any(
        sum(workspace_macs_by_component[str(size)][str(k)].values())
        != workspace_macs[str(size)][str(k)]
        for size in (21, 25)
        for k in config.evaluation.k_values
    ):
        raise RuntimeError("AIR component MAC accounting is incomplete")
    representation_macs = {
        str(size): int(j1_checkpoint["representation_inference_conv_macs"][str(size)])
        for size in (21, 25)
    }
    air_total_macs = {
        str(size): {
            str(k): representation_macs[str(size)] + workspace_macs[str(size)][str(k)]
            for k in config.evaluation.k_values
        }
        for size in (21, 25)
    }
    j1_total_macs = {
        str(size): representation_macs[str(size)]
        + int(j1_checkpoint["planner_inference_conv_macs"][str(size)]["128"])
        for size in (21, 25)
    }
    if any(
        value <= 0 for value in (*representation_macs.values(), *j1_total_macs.values())
    ):
        raise ValueError("source checkpoint has invalid MAC accounting")
    compute_match_by_size: dict[str, int] = {}
    for size in (21, 25):
        eligible = [
            k
            for k in config.evaluation.k_values
            if air_total_macs[str(size)][str(k)] <= 1.05 * j1_total_macs[str(size)]
        ]
        if not eligible:
            raise ValueError(f"no AIR compute-matched K exists at size {size}")
        compute_match_by_size[str(size)] = max(eligible)
    compute_match_joint = min(compute_match_by_size.values())
    representation_parameters = sum(
        parameter.numel()
        for module in (representation.encoder, representation.planning_projector)
        for parameter in module.parameters()
    )
    j1_representation_parameters = int(
        j1_checkpoint["representation_planning_parameter_count"]
    )
    j1_planner_parameters = int(j1_checkpoint["planner_parameter_count"])
    j1_total_parameters = int(j1_checkpoint["total_inference_parameter_count"])
    if (
        representation_parameters != j1_representation_parameters
        or j1_total_parameters != j1_representation_parameters + j1_planner_parameters
    ):
        raise ValueError("source J1 parameter accounting does not match representation")
    payload: dict[str, Any] = signed_payload(
        {
            "schema": "air-jepa-stage0-benchmark-v1",
            "experiment_id": config.experiment_id,
            "performance_blind": True,
            "protocol_sha256": protocol["protocol_sha256"],
            "package_sha256": package["package_sha256"],
            "source_lock_sha256": source["source_lock_sha256"],
            "git_commit": git_commit(),
            "git_dirty": git_worktree_dirty(),
            "code_fingerprint": code_fingerprint(),
            "runtime": runtime_metadata(device),
            "k128_forward": {
                "tasks": len(forward_times),
                "seconds_total": float(sum(forward_times)),
                "seconds_mean": float(np.mean(forward_times)),
                "tasks_per_second": len(forward_times) / sum(forward_times),
            },
            "k128_forward_backward": {
                "iterations": FORMAL_BACKWARD_K,
                "repeats": len(backward_times),
                "seconds_mean": float(np.mean(backward_times)),
            },
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "parameter_counts": {
                "frozen_representation": representation_parameters,
                "air_workspace": sum(p.numel() for p in model.parameters()),
                "air_total_inference": representation_parameters
                + sum(p.numel() for p in model.parameters()),
                "j1_total_inference": j1_total_parameters,
            },
            "workspace_analytical_macs": workspace_macs,
            "workspace_analytical_macs_by_component": workspace_macs_by_component,
            "representation_planning_conv_macs": representation_macs,
            "air_total_inference_macs": air_total_macs,
            "j1_k128_total_inference_macs": j1_total_macs,
            "compute_match": {
                "rule": "AIR <= 1.05 * seed42 J1-receding@K128 at size 21/25",
                "k_by_size": compute_match_by_size,
                "joint_k": compute_match_joint,
                "performance_used": False,
            },
        },
        "benchmark_sha256",
    )
    atomic_json_dump(output, payload)
    print(f"saved={relative_path(output)}")


if __name__ == "__main__":
    main()
