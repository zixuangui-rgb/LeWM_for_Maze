#!/usr/bin/env python3
"""Train the paired AIR0-direct/AIR0-jepa workspace reasoners."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from air_jepa.stage0_workspace import (
    AIR_METHODS,
    EXPERIMENT_ID,
    FORMAT_VERSION,
    PAIRING_AUDIT_BATCHES,
)
from air_jepa.stage0_workspace.checkpoints import (
    load_frozen_representation,
    save_air_checkpoint,
    verify_source_lock,
)
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    canonical_json_sha256,
    code_fingerprint,
    format_template,
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
    sha256_file,
    state_dict_sha256,
)
from air_jepa.stage0_workspace.data import (
    AIRBatch,
    make_rng_streams,
    paired_stream_record,
    progressive_iteration_signature,
    require_balanced_training_manifest,
    sample_training_batch,
    select_progressive_iterations,
)
from air_jepa.stage0_workspace.losses import AIRLossResult, air_loss
from air_jepa.stage0_workspace.models import AIRWorkspaceModel, require_finite_output
from air_jepa.stage0_workspace.protocol import (
    verify_package_lock,
    verify_protocol_lock,
)
from diagnostics.common import read_jsonl
from spatial_jepa_planning.common import (
    ManifestSampler,
    estimate_representation_planning_conv_macs,
)


class ChannelMoments:
    """Float64 sufficient statistics for the future-zero intervention."""

    def __init__(self, channels: int) -> None:
        self.count = 0
        self.channels = int(channels)
        self.sum: torch.Tensor | None = None
        self.sum_square: torch.Tensor | None = None

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.ndim != 5 or values.shape[2] != self.channels:
            raise ValueError("target moments require [B,4,C,H,W]")
        detached = values.detach().to(dtype=torch.float64)
        if self.sum is None or self.sum_square is None:
            self.sum = torch.zeros(
                self.channels,
                dtype=torch.float64,
                device=detached.device,
            )
            self.sum_square = torch.zeros_like(self.sum)
        if detached.device != self.sum.device:
            raise ValueError("target moments cannot mix devices")
        self.count += int(
            detached.shape[0]
            * detached.shape[1]
            * detached.shape[3]
            * detached.shape[4]
        )
        self.sum += detached.sum(dim=(0, 1, 3, 4))
        self.sum_square += detached.square().sum(dim=(0, 1, 3, 4))

    def summary(self) -> dict[str, Any]:
        if self.count <= 0 or self.sum is None or self.sum_square is None:
            raise ValueError("target moments have no observations")
        mean = self.sum / float(self.count)
        variance = self.sum_square / float(self.count) - mean.square()
        return {
            "count_per_channel": self.count,
            "mean": mean.cpu().tolist(),
            "variance": variance.clamp_min(0.0).cpu().tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--method", choices=AIR_METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--smoke-k", type=int, default=4)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


def _branch_gradient_stats(
    result: AIRLossResult,
    parameters: list[torch.nn.Parameter],
) -> dict[str, float | None]:
    losses = {
        "action": result.action,
        "future": result.future,
        "cost": result.cost,
    }
    gradients: dict[str, list[torch.Tensor | None]] = {}
    norms: dict[str, torch.Tensor] = {}
    for name, loss in losses.items():
        current = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        gradients[name] = list(current)
        norm_square = loss.new_tensor(0.0)
        for gradient in current:
            if gradient is not None:
                norm_square = norm_square + gradient.square().sum()
        norms[name] = norm_square.sqrt()
    output: dict[str, float | None] = {
        f"{name}_norm": float(value.detach().cpu()) for name, value in norms.items()
    }
    for left, right in (("action", "future"), ("action", "cost"), ("future", "cost")):
        dot = result.total.new_tensor(0.0)
        for first, second in zip(gradients[left], gradients[right], strict=True):
            if first is not None and second is not None:
                dot = dot + (first * second).sum()
        denominator = norms[left] * norms[right]
        key = f"{left}_{right}_cosine"
        output[key] = (
            float((dot / denominator).detach().cpu())
            if float(denominator.detach().cpu()) > 0.0
            else None
        )
    return output


def _stream_update(
    digest: Any,
    *,
    batch: AIRBatch,
    iterations: int,
) -> None:
    record = paired_stream_record(batch, iterations=iterations)
    digest.update(canonical_json_sha256(record).encode("ascii"))


def _mean_rows(rows: list[dict[str, float]], count: int) -> dict[str, float]:
    selected = rows[-min(count, len(rows)) :]
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in selected:
        for key, value in row.items():
            if math.isfinite(value):
                grouped[key].append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _encode_batch(
    representation: torch.nn.Module,
    current: torch.Tensor,
    successors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, actions, height, width, channels = successors.shape
    if actions != 4:
        raise ValueError("AIR0 requires exactly four action successors")
    with torch.no_grad():
        source = representation.planning_latent(current)
        future = representation.planning_latent(
            successors.reshape(batch * actions, height, width, channels)
        )
    future = future.reshape(batch, actions, *future.shape[1:])
    if source.requires_grad or future.requires_grad:
        raise RuntimeError(
            "frozen representation unexpectedly created a gradient graph"
        )
    return source, future


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.seed not in config.seeds:
        raise ValueError(f"seed must be one of the locked seeds {config.seeds}")
    formal = args.mode == "formal"
    if formal and args.allow_dirty_worktree:
        raise ValueError("formal training cannot enable dirty-worktree override")
    require_clean_worktree(allow_dirty=not formal and args.allow_dirty_worktree)
    protocol_lock = verify_protocol_lock(config)
    package_lock = verify_package_lock(config)
    source_lock = verify_source_lock(config)

    device = resolve_device(args.device)
    if formal:
        require_h800_device(device)
    set_seed(args.seed, deterministic=True)
    representation, _ = load_frozen_representation(
        config,
        seed=args.seed,
        device=device,
        source_lock=source_lock,
    )
    model_seed = args.seed + 70_000
    set_seed(model_seed, deterministic=True)
    model = AIRWorkspaceModel(config.model).to(device)
    initial_state_sha256 = state_dict_sha256(model.state_dict())
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=config.training.betas,
        eps=config.training.epsilon,
        weight_decay=config.training.weight_decay,
    )
    steps = config.training.steps if formal else int(args.smoke_steps)
    if steps <= 0:
        raise ValueError("training step count must be positive")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    entries = read_jsonl(resolve_path(config.paths.train_manifest))
    require_balanced_training_manifest(entries)
    sampler = ManifestSampler(entries)
    streams = make_rng_streams(args.seed)
    stream_digest = hashlib.sha256()
    iteration_digest = hashlib.sha256()
    stream_prefix_sha256: str | None = None
    stream_prefix_batches = min(steps, PAIRING_AUDIT_BATCHES)
    moments = ChannelMoments(config.model.input_dim)
    weights = config.training.methods[args.method]
    history: list[dict[str, float]] = []
    training_log: list[dict[str, Any]] = []
    gradient_history: list[dict[str, Any]] = []
    k_counts: Counter[int] = Counter()
    window_k_counts: Counter[int] = Counter()
    started = time.perf_counter()
    log_started = started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, steps + 1):
        batch = sample_training_batch(
            sampler,
            entry_rng=streams.entries,
            state_rng=streams.states,
            batch_size=config.training.batch_size,
            device=device,
        )
        if formal:
            iterations = select_progressive_iterations(
                step=step,
                phase_steps=config.training.phase_steps,
                k_train=config.training.k_train,
                rng=streams.iterations,
            )
        else:
            iterations = int(args.smoke_k)
            if iterations <= 0:
                raise ValueError("smoke-k must be positive")
        k_counts[iterations] += 1
        window_k_counts[iterations] += 1
        iteration_digest.update(iterations.to_bytes(4, "big", signed=False))
        _stream_update(
            stream_digest,
            batch=batch,
            iterations=iterations,
        )
        if step == stream_prefix_batches:
            stream_prefix_sha256 = stream_digest.copy().hexdigest()
        source_latent, successor_latent = _encode_batch(
            representation,
            batch.current_observation,
            batch.successor_observations,
        )
        moments.update(successor_latent)
        spatial_mask = torch.ones(
            (batch.batch_size, batch.maze_size, batch.maze_size),
            dtype=torch.bool,
            device=device,
        )
        outputs = model(
            source_latent,
            iterations=iterations,
            deep_supervision_every=config.training.deep_supervision_every,
            valid_mask=spatial_mask,
        )
        for output in outputs:
            require_finite_output(output)
        result = air_loss(
            outputs,
            successor_latent=successor_latent,
            source_latent=source_latent,
            candidate_distances=batch.candidate_distances,
            optimal_action_mask=batch.optimal_action_mask,
            valid_mask=spatial_mask,
            weights=weights,
            max_distance=config.model.max_distance,
            target_variance_epsilon=config.training.target_variance_epsilon,
        )
        if step == 1 or step % config.training.gradient_audit_every == 0:
            shared = [parameter for parameter in model.reasoner.parameters()]
            gradient_history.append(
                {
                    "step": step,
                    "iterations": iterations,
                    **_branch_gradient_stats(result, shared),
                }
            )
        optimizer.zero_grad(set_to_none=True)
        result.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        scheduler.step()
        row = {
            "total": float(result.total.detach().cpu()),
            "action": float(result.action.detach().cpu()),
            "future": float(result.future.detach().cpu()),
            "cost": float(result.cost.detach().cpu()),
            "future_field_normalized": float(
                result.future_metrics.normalized_field.detach().cpu()
            ),
            "future_delta_normalized": float(
                result.future_metrics.normalized_delta.detach().cpu()
            ),
            "future_field_raw_mse": float(
                result.future_metrics.raw_field_mse.detach().cpu()
            ),
            "future_delta_raw_mse": float(
                result.future_metrics.raw_delta_mse.detach().cpu()
            ),
            "copy_delta_normalized": float(
                result.future_metrics.copy_delta_normalized.detach().cpu()
            ),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "iterations": float(iterations),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        if any(not math.isfinite(value) for value in row.values()):
            raise FloatingPointError(f"non-finite training metric at step {step}")
        history.append(row)
        if step % config.training.log_every == 0 or step == steps:
            elapsed = max(time.perf_counter() - log_started, 1e-9)
            averages = _mean_rows(history, config.training.log_every)
            window_steps = sum(window_k_counts.values())
            if window_steps <= 0:
                raise RuntimeError(
                    "training log window unexpectedly contains zero steps"
                )
            steps_per_second = window_steps / elapsed
            current_peak_memory = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            training_log.append(
                {
                    "step": step,
                    **averages,
                    "window_steps": window_steps,
                    "window_elapsed_seconds": elapsed,
                    "steps_per_second": steps_per_second,
                    "window_k_counts": {
                        str(key): value
                        for key, value in sorted(window_k_counts.items())
                    },
                    "cumulative_k_counts": {
                        str(key): value for key, value in sorted(k_counts.items())
                    },
                    "peak_cuda_memory_bytes": current_peak_memory,
                }
            )
            print(
                f"method={args.method} seed={args.seed} step={step}/{steps} "
                f"K={iterations} loss={averages['total']:.5f} "
                f"action={averages['action']:.5f} future={averages['future']:.5f} "
                f"cost={averages['cost']:.5f} steps/s={steps_per_second:.3f}"
            )
            window_k_counts.clear()
            log_started = time.perf_counter()

    elapsed = time.perf_counter() - started
    if stream_prefix_sha256 is None:
        raise RuntimeError("training did not produce the paired stream prefix hash")
    if formal:
        expected_iterations = progressive_iteration_signature(
            seed=args.seed,
            steps=config.training.steps,
            phase_steps=config.training.phase_steps,
            k_train=config.training.k_train,
        )
        actual_counts = {str(key): value for key, value in sorted(k_counts.items())}
        if (
            iteration_digest.hexdigest() != expected_iterations["sha256"]
            or actual_counts != expected_iterations["counts"]
        ):
            raise RuntimeError("formal progressive-K stream differs from its seed lock")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    representation_component_parameter_counts = {
        "encoder": sum(
            parameter.numel() for parameter in representation.encoder.parameters()
        ),
        "planning_projector": sum(
            parameter.numel()
            for parameter in representation.planning_projector.parameters()
        ),
    }
    representation_parameter_count = sum(
        representation_component_parameter_counts.values()
    )
    component_parameter_counts = {
        "adapter_and_workspace_init": (
            sum(
                parameter.numel()
                for module in (
                    model.input_adapter,
                    model.initial_state,
                    model.goal_attention,
                )
                for parameter in module.parameters()
            )
            + model.goal_query.numel()
            + model.action_embeddings.numel()
            + model.future_embeddings.numel()
        ),
        "shared_reasoner": sum(
            parameter.numel() for parameter in model.reasoner.parameters()
        ),
        "future_decoder": sum(
            parameter.numel() for parameter in model.future_decoder.parameters()
        ),
        "energy_head": sum(
            parameter.numel() for parameter in model.energy_head.parameters()
        ),
    }
    model_parameter_count = sum(p.numel() for p in model.parameters())
    if sum(component_parameter_counts.values()) != model_parameter_count:
        raise RuntimeError("AIR component parameter accounting is incomplete")
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
        str(size): estimate_representation_planning_conv_macs(
            representation,
            maze_size=size,
            device=device,
        )
        for size in (21, 25)
    }
    total_macs = {
        str(size): {
            str(k): representation_macs[str(size)] + workspace_macs[str(size)][str(k)]
            for k in config.evaluation.k_values
        }
        for size in (21, 25)
    }
    payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "format_version": FORMAT_VERSION,
        "method": args.method,
        "seed": args.seed,
        "formal": formal,
        "optimizer_steps": steps,
        "checkpoint_role": "final_step" if formal else "smoke_only",
        "config": config.model_dump(mode="json", by_alias=True),
        "config_sha256": sha256_file(args.config),
        "protocol_sha256": protocol_lock["protocol_sha256"],
        "package_sha256": package_lock["package_sha256"],
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "source_representation": source_lock["records"][str(args.seed)][
            "representation"
        ],
        "git_commit": git_commit(),
        "git_dirty": git_worktree_dirty(),
        "code_fingerprint": code_fingerprint(),
        "runtime": runtime_metadata(device),
        "model_seed": model_seed,
        "rng_stream_seeds": streams.stream_seeds,
        "paired_sample_stream_sha256": stream_digest.hexdigest(),
        "progressive_iteration_stream_sha256": iteration_digest.hexdigest(),
        "paired_sample_stream_prefix_batches": stream_prefix_batches,
        "paired_sample_stream_prefix_sha256": stream_prefix_sha256,
        "k_counts": {str(key): value for key, value in sorted(k_counts.items())},
        "initial_model_state_sha256": initial_state_sha256,
        "model_parameter_count": model_parameter_count,
        "component_parameter_counts": component_parameter_counts,
        "representation_component_parameter_counts": (
            representation_component_parameter_counts
        ),
        "representation_planning_parameter_count": representation_parameter_count,
        "total_inference_parameter_count": (
            model_parameter_count + representation_parameter_count
        ),
        "model_trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "representation_trainable_parameter_count": sum(
            p.numel() for p in representation.parameters() if p.requires_grad
        ),
        "training_accounting": {
            "elapsed_seconds": elapsed,
            "map_state_examples": steps * config.training.batch_size,
            "successor_examples": steps * config.training.batch_size * 4,
            "peak_cuda_memory_bytes": peak_memory,
        },
        "training_summary_last_window": _mean_rows(history, config.training.log_every),
        "training_log": training_log,
        "gradient_history": gradient_history,
        "future_target_channel_moments": moments.summary(),
        "workspace_analytical_macs": workspace_macs,
        "workspace_analytical_macs_by_component": workspace_macs_by_component,
        "representation_planning_conv_macs": representation_macs,
        "total_inference_macs": total_macs,
        "model_state_dict": model.state_dict(),
    }
    payload["model_state_sha256"] = state_dict_sha256(payload["model_state_dict"])
    return payload


def main() -> None:
    args = parse_args()
    if args.mode == "smoke" and not args.output:
        raise ValueError("smoke training requires an explicit non-formal --output")
    config = load_config(args.config)
    output = (
        Path(args.output)
        if args.output
        else format_template(
            config.paths.air_checkpoint_template,
            method=args.method,
            seed=args.seed,
        )
    )
    prepare_new_output(output)
    payload = train(args)
    save_air_checkpoint(output, payload)
    print(f"saved={relative_path(output)} sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
