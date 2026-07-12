#!/usr/bin/env python3
"""Train either fixed final-closure baseline under the locked protocol."""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from final_closure import EXPERIMENT_FAMILY, FORMAT_VERSION
from final_closure.common import (
    analysis_spec_sha256,
    atomic_torch_save,
    baseline_config,
    load_config,
    protocol_metadata,
    read_jsonl,
    require_clean_worktree,
    require_new_output,
    require_study_open,
    resolve_device,
    set_seed,
    sha256_file,
    training_spec_sha256,
)
from final_closure.data import (
    build_bc_dataset,
    epoch_batches,
    materialize_bc_dataset,
    sample_lewm_sequence,
)
from final_closure.models import (
    BCPolicyConfig,
    DeepCNNPolicy,
    build_lewm,
    serialize_lewm_config,
)
from hdwm.losses import SIGReg
from scripts.train.train_canonical_lewm import compute_position_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="final_closure/configs/default.json")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--diagnostic-train-limit", type=int, default=0)
    parser.add_argument("--diagnostic-epochs", type=int, default=0)
    parser.add_argument("--diagnostic-steps", type=int, default=0)
    parser.add_argument("--diagnostic-sigreg-num-proj", type=int, default=0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def verify_manifest_files(config: dict[str, Any], lock: dict[str, Any]) -> None:
    role_paths = {
        "train_manifest": config["paths"]["train_manifest"],
        "development_manifest": config["paths"]["development_manifest"],
        "confirmatory_manifest": config["paths"]["confirmatory_manifest"],
    }
    for role, path in role_paths.items():
        actual = sha256_file(path)
        expected = str(lock[role]["sha256"])
        if actual != expected:
            raise ValueError(f"{role} hash mismatch: {actual} != {expected}")


def train_bc(
    entries: list[dict[str, Any]],
    train_config: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    epochs: int,
) -> tuple[DeepCNNPolicy, dict[str, Any]]:
    if train_config["architecture"] != (
        "historical_deepcnn_res2_down_res1_pool_mlp512_256"
    ):
        raise ValueError("unsupported locked BC architecture")
    if train_config["target_population"] != "all_non_goal_free_states":
        raise ValueError("unsupported locked BC target population")
    if train_config["epoch_order"] != "seeded_global_permutation":
        raise ValueError("unsupported locked BC epoch order")
    dataset = build_bc_dataset(entries)
    policy_config = BCPolicyConfig(
        dropout=float(train_config["dropout"]),
        action_count=int(train_config["class_count"]),
    )
    model = DeepCNNPolicy(policy_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    batch_size = int(train_config["batch_size"])
    canvas_size = int(train_config["train_canvas_size"])
    if train_config["observation_cache"] != "uint8_cpu_full_state":
        raise ValueError("unsupported locked BC observation cache")
    cached_observations, cached_labels = materialize_bc_dataset(
        dataset, canvas_size=canvas_size
    )
    grad_clip = float(train_config["grad_clip"])
    log_every = int(train_config["log_every_epochs"])
    started = time.perf_counter()
    final_metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for indices in epoch_batches(
            dataset.sample_count,
            batch_size,
            seed=seed,
            epoch=epoch,
            namespace=int(train_config["epoch_permutation_namespace"]),
        ):
            index_tensor = torch.from_numpy(indices.astype(np.int64, copy=False))
            observations = cached_observations.index_select(0, index_tensor).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            labels = cached_labels.index_select(0, index_tensor).to(
                device, non_blocking=True
            )
            logits = model(observations)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite BC loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite BC gradient norm at epoch {epoch}"
                )
            optimizer.step()
            count = int(labels.numel())
            loss_sum += float(loss.detach()) * count
            correct += int((logits.argmax(dim=-1) == labels).sum())
            seen += count
        if seen != dataset.sample_count:
            raise RuntimeError(
                f"BC epoch coverage mismatch: {seen} != {dataset.sample_count}"
            )
        scheduler.step()
        final_metrics = {
            "epoch": float(epoch),
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(
                f"BC epoch {epoch:>3d}/{epochs}: "
                f"loss={final_metrics['train_loss']:.6f} "
                f"acc={final_metrics['train_accuracy']:.4f}",
                flush=True,
            )
    label_counts = Counter()
    for pool in dataset.pools.values():
        label_counts.update(int(value) for value in pool.labels.tolist())
    metadata = {
        "model_config": policy_config.to_dict(),
        "sample_count": dataset.sample_count,
        "sample_count_by_size": {
            str(size): pool.sample_count for size, pool in sorted(dataset.pools.items())
        },
        "target_count_by_action_slot": {
            str(slot): int(label_counts[slot])
            for slot in range(policy_config.action_count)
        },
        "optimizer_steps": int(epochs * math.ceil(dataset.sample_count / batch_size)),
        "examples_seen": int(epochs * dataset.sample_count),
        "observation_cache_bytes": int(cached_observations.numel()),
        "final_train_metrics": final_metrics,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return model, metadata


def lewm_loss(
    model: torch.nn.Module,
    sigreg: SIGReg,
    batch: Any,
    *,
    maze_size: int,
    device: torch.device,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    observations = batch.observations.to(device=device, dtype=torch.float32)
    actions = batch.actions.to(device=device, dtype=torch.long)
    output = model(observations, actions, maze_size)
    prediction = F.mse_loss(output["prediction"], output["target"])
    sigreg_loss = sigreg(output["sigreg_embedding"].transpose(0, 1))
    x_pos, y_pos, dx, dy = compute_position_labels(
        batch.states,
        observations[..., 3],
        maze_size,
    )
    absolute_target = torch.stack([x_pos, y_pos], dim=-1).to(device)
    relative_target = torch.stack([dx, dy], dim=-1).to(device)
    absolute = F.mse_loss(output["abs_pos_pred"], absolute_target)
    relative = F.mse_loss(output["rel_pos_pred"], relative_target)
    batch_size, sequence_length = observations.shape[:2]
    goal_state = (
        observations[..., 3].reshape(batch_size, sequence_length, -1).argmax(-1)
    )
    denominator = max(maze_size - 1, 1)
    goal_target = torch.stack(
        [
            (goal_state % maze_size).float() / denominator,
            (goal_state // maze_size).float() / denominator,
        ],
        dim=-1,
    )
    goal = F.mse_loss(output["goal_pos_pred"], goal_target)
    total = (
        weights["prediction"] * prediction
        + weights["sigreg"] * sigreg_loss
        + weights["absolute"] * absolute
        + weights["relative"] * relative
        + weights["goal"] * goal
    )
    return total, {
        "total": total,
        "prediction": prediction,
        "sigreg": sigreg_loss,
        "absolute": absolute,
        "relative": relative,
        "goal": goal,
    }


def train_lewm(
    entries: list[dict[str, Any]],
    train_config: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    steps: int,
    sigreg_num_proj: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if train_config["architecture"] != (
        "unisize256_sizecond_cnn_projector_transformer"
    ):
        raise ValueError("unsupported locked LeWM architecture")
    if train_config["entry_schedule"] != (
        "historical_step_mod_manifest_length_starting_at_one"
    ):
        raise ValueError("unsupported locked LeWM entry schedule")
    if train_config["environment_seed_stream"] != "numpy_default_rng_training_seed":
        raise ValueError("unsupported locked LeWM environment seed stream")
    model, model_config = build_lewm(train_config)
    model = model.to(device)
    sigreg = SIGReg(
        knots=int(train_config["sigreg_knots"]),
        num_proj=sigreg_num_proj,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    grad_clip = float(train_config["grad_clip"])
    batch_size = int(train_config["batch_size"])
    sequence_length = int(train_config["sequence_length"])
    log_every = int(train_config["log_every_steps"])
    weights = {
        "prediction": float(train_config["lambda_prediction"]),
        "sigreg": float(train_config["lambda_sigreg"]),
        "absolute": float(train_config["lambda_abs_position"]),
        "relative": float(train_config["lambda_relative_position"]),
        "goal": float(train_config["lambda_goal_position"]),
    }
    rng = np.random.default_rng(seed)
    recent = {
        name: deque(maxlen=log_every)
        for name in ("total", "prediction", "sigreg", "absolute", "relative", "goal")
    }
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        entry = entries[step % len(entries)]
        maze_size = int(entry["maze_size"])
        batch = sample_lewm_sequence(
            entry,
            rng=rng,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        total, metrics = lewm_loss(
            model,
            sigreg,
            batch,
            maze_size=maze_size,
            device=device,
            weights=weights,
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite LeWM loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite LeWM gradient norm at step {step}")
        optimizer.step()
        for name, value in metrics.items():
            scalar = float(value.detach())
            if not math.isfinite(scalar):
                raise FloatingPointError(
                    f"non-finite LeWM metric {name} at step {step}"
                )
            recent[name].append(scalar)
        if step == 1 or step == steps or step % log_every == 0:
            means = {name: float(np.mean(values)) for name, values in recent.items()}
            print(
                f"LeWM step {step:>6d}/{steps}: total={means['total']:.6f} "
                f"pred={means['prediction']:.6f} sigreg={means['sigreg']:.6f} "
                f"rel={means['relative']:.6f}",
                flush=True,
            )
    metadata = {
        "model_config": serialize_lewm_config(model_config),
        "entry_schedule": "historical_step_mod_manifest_length_starting_at_one",
        "runtime_environment_seed_stream": "numpy.default_rng(training_seed)",
        "optimizer_steps": int(steps),
        "sequences_seen": int(steps * batch_size),
        "frames_seen": int(steps * batch_size * sequence_length),
        "transitions_seen": int(steps * batch_size * (sequence_length - 1)),
        "final_train_metrics": {
            name: float(np.mean(values)) for name, values in recent.items()
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return model, metadata


def main() -> None:
    args = parse_args()
    config, lock = load_config(args.config)
    baseline = baseline_config(config, args.baseline)
    if args.seed not in [int(value) for value in config["seeds"]]:
        raise ValueError(f"seed {args.seed} is outside the locked seed matrix")
    diagnostic_values = (
        args.diagnostic_train_limit,
        args.diagnostic_epochs,
        args.diagnostic_steps,
        args.diagnostic_sigreg_num_proj,
    )
    if any(value < 0 for value in diagnostic_values):
        raise ValueError("diagnostic overrides must be non-negative")
    if not args.diagnostic and any(value > 0 for value in diagnostic_values):
        raise ValueError("diagnostic overrides require --diagnostic")
    if not args.diagnostic:
        require_study_open(config)
    require_clean_worktree(args.allow_dirty_worktree or args.diagnostic)
    require_new_output(args.output, args.overwrite)
    verify_manifest_files(config, lock)
    set_seed(args.seed, deterministic=True)
    device = resolve_device(args.device or config["device"])
    entries = read_jsonl(config["paths"]["train_manifest"])
    if len(entries) != int(lock["train_manifest"]["count"]):
        raise ValueError("training manifest count differs from protocol lock")
    if args.diagnostic_train_limit > 0:
        entries = entries[: args.diagnostic_train_limit]
    train_config = baseline["train"]
    if baseline["kind"] == "bc":
        epochs = int(args.diagnostic_epochs or train_config["epochs"])
        if epochs <= 0:
            raise ValueError("BC epochs must be positive")
        model, training = train_bc(
            entries,
            train_config,
            seed=args.seed,
            device=device,
            epochs=epochs,
        )
        model_payload = {
            "model_config": training.pop("model_config"),
            "policy_state_dict": cpu_state_dict(model),
        }
    elif baseline["kind"] == "lewm_l2_cem":
        steps = int(args.diagnostic_steps or train_config["steps"])
        sigreg_num_proj = int(
            args.diagnostic_sigreg_num_proj or train_config["sigreg_num_proj"]
        )
        if steps <= 0 or sigreg_num_proj <= 0:
            raise ValueError("LeWM steps and SIGReg projections must be positive")
        model, training = train_lewm(
            entries,
            train_config,
            seed=args.seed,
            device=device,
            steps=steps,
            sigreg_num_proj=sigreg_num_proj,
        )
        model_payload = {
            "model_config": training.pop("model_config"),
            "model_state_dict": cpu_state_dict(model),
        }
    else:
        raise ValueError(f"unsupported baseline kind: {baseline['kind']}")
    payload = {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": FORMAT_VERSION,
        "stage": "baseline_training",
        "baseline_name": baseline["name"],
        "baseline_kind": baseline["kind"],
        "training_seed": int(args.seed),
        "formal_run": not args.diagnostic,
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "training_spec_sha256": training_spec_sha256(
            config, lock, name=baseline["name"], seed=args.seed
        ),
        "protocol": protocol_metadata(config, lock, seed=args.seed, device=device),
        "training_config": train_config,
        "training": training,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        **model_payload,
    }
    atomic_torch_save(args.output, payload)
    print(f"saved {baseline['name']} seed={args.seed} to {Path(args.output)}")


if __name__ == "__main__":
    main()
