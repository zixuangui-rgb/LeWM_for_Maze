"""Train one preregistered DistanceHead cell under a matched sample schedule."""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distance_head_study import MODEL_ACTION_VOCAB_SIZE
from distance_head_study.candidates import candidate_bank_path, load_candidate_bank
from distance_head_study.common import (
    atomic_torch_save,
    canonical_json_sha256,
    head_checkpoint_path,
    hierarchical_seed,
    load_json,
    load_study_config,
    read_jsonl,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    set_seed,
    sha256_file,
    source_backbone_path,
    validate_backbone_protocol_binding,
)
from distance_head_study.data import (
    ShardedGoalDataset,
    TrainingBatch,
    cache_index_path,
    evenly_spaced_indices,
    load_backbone_checkpoint,
    refresh_joint_latents,
    sample_training_batch,
    slice_training_batch,
    true_candidate_distances,
    validate_cache_binding,
)
from distance_head_study.gates import require_seed_released
from distance_head_study.losses import (
    TrajectoryBatch,
    compute_objective_terms,
    gradient_calibrated_weights,
    weighted_total,
)
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.models import DistanceHeadModel, build_distance_head
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.schemas import (
    InitializationMode,
    ResolvedMethod,
    TrainingScope,
)
from final_closure.common import baseline_config
from final_closure.data import sample_lewm_sequence
from final_closure.train import cpu_state_dict, lewm_loss
from hdwm.losses import SIGReg
from vector_jepa_planner_frontier.schemas import RolloutSemantics
from vector_jepa_planner_frontier.world_model import VectorContext, VectorWorldModel

TRAIN_STATE_SCHEMA = "distance-head-training-state-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--method", required=True)
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--head-seed", type=int, required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--diagnostic-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


def _active_head_weights(method: ResolvedMethod) -> dict[str, float]:
    if method.objectives is None:
        return {}
    values = method.objectives.model_dump()
    values.pop("original_jepa")
    return {name: float(value) for name, value in values.items() if value > 0.0}


def _configure_backbone_scope(model: torch.nn.Module, scope: TrainingScope) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if scope == TrainingScope.FROZEN:
        return
    if scope == TrainingScope.PREDICTOR:
        modules = (model.predictor,)
    elif scope == TrainingScope.PROJECTOR_PREDICTOR:
        modules = (model.embedding_projector, model.predictor)
    elif scope == TrainingScope.FULL:
        model.train()
        modules = (model,)
    else:
        raise ValueError(f"unsupported training scope: {scope}")
    for module in modules:
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad = True


def _predict_all_actions(
    model: torch.nn.Module,
    batch: TrainingBatch,
    *,
    gradients: bool,
) -> torch.Tensor:
    batch_size, history, latent_dim = batch.history_latents.shape
    action_count = MODEL_ACTION_VOCAB_SIZE
    corrected_embeddings = torch.cat(
        [batch.history_latents[:, 1:], batch.history_latents[:, -1:]], dim=1
    )
    embeddings = (
        corrected_embeddings[:, None]
        .expand(-1, action_count, -1, -1)
        .reshape(batch_size * action_count, history, latent_dim)
    )
    actions = (
        batch.history_actions[:, None, -1:]
        .expand(-1, action_count, -1)
        .reshape(batch_size * action_count, 1)
    )
    candidate = torch.arange(action_count, device=embeddings.device).repeat(batch_size)
    predictor_actions = torch.cat([actions, candidate[:, None]], dim=1)
    manager = torch.enable_grad() if gradients else torch.no_grad()
    with manager:
        predicted = model.predictor(embeddings, predictor_actions)[:, -1]
    result = predicted.reshape(batch_size, action_count, latent_dim)
    if not torch.isfinite(result).all():
        raise FloatingPointError("all-action predictor output is non-finite")
    return result


def _trajectory_batch(
    model: torch.nn.Module,
    dataset: ShardedGoalDataset,
    batch: TrainingBatch,
    candidate_set: torch.Tensor,
    *,
    contexts: int,
    horizon: int,
    device: torch.device,
    gradients: bool,
) -> TrajectoryBatch:
    if contexts > batch.source.shape[0]:
        raise ValueError("trajectory contexts exceed microbatch")
    rollout_slots = int(horizon) + 1
    if rollout_slots > candidate_set.shape[1]:
        raise ValueError("executed-action horizon exceeds the candidate bank")
    sequences = candidate_set[:, :rollout_slots]
    context_indices = evenly_spaced_indices(batch.source.shape[0], contexts)
    world_model = VectorWorldModel(model, device=device, history_size=3)
    terminals: list[torch.Tensor] = []
    for context_index in context_indices.tolist():
        context = VectorContext(
            embeddings=batch.history_latents[context_index : context_index + 1],
            actions=batch.history_actions[context_index : context_index + 1],
            goal=batch.goal[context_index : context_index + 1, None],
            maze_size=batch.maze_size,
            remaining_steps=128,
        )
        rollout = world_model.rollout(
            context,
            sequences,
            semantics=RolloutSemantics.LEGACY_WARMUP_V1,
            gradients=gradients,
        )
        terminals.append(rollout.terminal)
    labels = true_candidate_distances(
        dataset,
        batch,
        sequences[None].expand(contexts, -1, -1),
        context_indices=context_indices,
        executed_action_count=horizon,
    ).to(device)
    return TrajectoryBatch(
        predicted_terminal=torch.stack(terminals),
        goals=batch.goal.index_select(0, context_indices.to(batch.goal.device)),
        max_distance=batch.max_distance.index_select(
            0, context_indices.to(batch.max_distance.device)
        ),
        true_endpoint_distance=labels,
        horizon=rollout_slots,
    )


def _calibrate_impl(
    head: DistanceHeadModel,
    method: ResolvedMethod,
    model: torch.nn.Module,
    dataset: ShardedGoalDataset,
    candidate_actions: torch.Tensor,
    *,
    config: Any,
    backbone_seed: int,
    device: torch.device,
) -> dict[str, float]:
    active = _active_head_weights(method)
    if not active:
        return {}
    full = sample_training_batch(
        dataset,
        sampler=method.sampler,
        effective_batch_size=config.training.effective_batch_size,
        pairs_per_topology=config.training.pairs_per_topology,
        schedule_seed=config.seeds.sample_schedule_seed,
        backbone_seed=backbone_seed,
        step=0,
    )
    calibration_batch = slice_training_batch(full, 0, config.training.microbatch_size)
    if method.training_scope in (
        TrainingScope.PROJECTOR_PREDICTOR,
        TrainingScope.FULL,
    ):
        calibration_batch = refresh_joint_latents(
            dataset,
            calibration_batch,
            model,
            device=device,
            gradients=False,
        )
    batch = calibration_batch.to(device)
    needs_predicted = bool(
        method.objectives.predicted_listwise or method.objectives.predicted_consistency
    )
    predicted = (
        _predict_all_actions(
            model,
            batch,
            gradients=False,
        )
        if needs_predicted
        else None
    )
    trajectory = None
    if method.objectives.trajectory_listwise > 0:
        contexts = config.training.trajectory_contexts_per_step // (
            config.training.effective_batch_size // config.training.microbatch_size
        )
        trajectory = _trajectory_batch(
            model,
            dataset,
            batch,
            candidate_actions[0],
            contexts=contexts,
            horizon=max(method.trajectory_horizons),
            device=device,
            gradients=False,
        )
    terms = compute_objective_terms(
        head,
        method,
        batch,
        predicted_next=predicted,
        trajectory=trajectory,
    )
    if set(terms) != set(active):
        raise ValueError(
            f"implemented/declared objective mismatch: {set(terms)} != {set(active)}"
        )
    # Calibrate against the shared head path.  Joint treatment/control pairs
    # therefore receive identical initial loss weights; enabling a backbone
    # gradient cannot silently change the intervention's coefficients.
    parameters = [
        parameter for parameter in head.parameters() if parameter.requires_grad
    ]
    return gradient_calibrated_weights(
        terms,
        active,
        parameters,
        target_ratio=config.training.calibration_gradient_ratio,
        clip=config.training.calibration_weight_clip,
    )


def _calibrate(
    head: DistanceHeadModel,
    method: ResolvedMethod,
    model: torch.nn.Module,
    dataset: ShardedGoalDataset,
    candidate_actions: torch.Tensor,
    *,
    config: Any,
    backbone_seed: int,
    device: torch.device,
) -> dict[str, float]:
    modules = list(model.modules()) + list(head.modules())
    training_modes = [module.training for module in modules]
    model.eval()
    head.eval()
    try:
        return _calibrate_impl(
            head,
            method,
            model,
            dataset,
            candidate_actions,
            config=config,
            backbone_seed=backbone_seed,
            device=device,
        )
    finally:
        for module, training in zip(modules, training_modes, strict=True):
            module.training = training


def _scheduler_multiplier(
    step: int, *, total_steps: int, warmup_fraction: float, final_ratio: float
) -> float:
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return float(final_ratio + (1.0 - final_ratio) * cosine)


def _training_state_path(
    method: str,
    backbone_seed: int,
    head_seed: int,
    *,
    diagnostic_steps: int = 0,
) -> Path:
    prefix = "smoke/train_state" if diagnostic_steps else "train_state"
    suffix = f"_steps{diagnostic_steps}" if diagnostic_steps else ""
    return resolve_path(
        f"distance_head_study_runs/{prefix}/{method}/"
        f"backbone{backbone_seed}_head{head_seed}{suffix}.pt"
    )


def _diagnostic_checkpoint_path(
    method: str, backbone_seed: int, head_seed: int, steps: int
) -> Path:
    return resolve_path(
        f"distance_head_study_runs/smoke/checkpoints/heads/{method}/"
        f"backbone{backbone_seed}_head{head_seed}_steps{steps}.pt"
    )


def _initialize_from_parent(
    head: DistanceHeadModel,
    method: ResolvedMethod,
    parent: dict[str, Any],
) -> dict[str, Any]:
    if method.initialization_parent is None:
        return {"mode": "none", "loaded_keys": []}
    assert method.head is not None
    parent_state = parent["head_state_dict"]
    if method.initialization_mode == InitializationMode.STRICT:
        if parent.get("head_spec") != method.head.model_dump(mode="json"):
            raise ValueError(
                "strict initialization parent has a different head specification"
            )
        head.load_state_dict(parent_state, strict=True)
        loaded_keys = sorted(parent_state)
    elif method.initialization_mode == InitializationMode.COMPATIBLE_SHARED:
        current = head.state_dict()
        compatible = {
            key: value
            for key, value in parent_state.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        required = {"primary.weight", "primary.bias"}
        if not required <= set(compatible):
            raise ValueError(
                "compatible initialization did not preserve the scalar scoring head"
            )
        if not any(key.startswith("trunk.") for key in compatible):
            raise ValueError(
                "compatible initialization did not preserve any shared trunk layer"
            )
        head.load_state_dict(compatible, strict=False)
        loaded_keys = sorted(compatible)
    else:
        raise ValueError(
            f"unsupported initialization mode: {method.initialization_mode}"
        )
    return {
        "mode": method.initialization_mode.value,
        "parent_method": method.initialization_parent,
        "loaded_keys": loaded_keys,
        "loaded_key_count": len(loaded_keys),
        "parent_head_spec": parent.get("head_spec"),
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("resume checkpoint has an incomplete RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state cannot be restored without CUDA")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def main() -> None:
    args = parse_args()
    if args.diagnostic_steps < 0:
        raise ValueError("diagnostic steps must be non-negative")
    diagnostic = args.diagnostic_steps > 0
    if diagnostic and not args.allow_dirty_worktree:
        raise ValueError("diagnostic training requires the explicit dirty flag")
    config = load_study_config(args.config)
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    if not diagnostic:
        require_seed_released(
            config,
            backbone_seed=args.backbone_seed,
            head_seed=args.head_seed,
        )
    lock = verify_protocol_lock(config)
    method, method_hash, decision_hashes = load_and_resolve_method(
        config.paths.method_catalog,
        args.method,
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    if not method.trainable:
        raise ValueError(
            f"method {method.name} reuses a checkpoint and is not trainable"
        )
    if method.reuse_parent_checkpoint:
        raise ValueError(
            f"method {method.name} is planner-only and must not be trained"
        )
    assert method.head is not None and method.objectives is not None
    if method.objectives.trajectory_listwise > 0 and not set(
        method.trajectory_horizons
    ) <= set(config.training.horizons):
        raise ValueError(
            "method trajectory horizons differ from the locked action-horizon grid"
        )
    seed = hierarchical_seed("distance-head-init", args.backbone_seed, args.head_seed)
    set_seed(seed, deterministic=config.training.deterministic)
    device = resolve_device(args.device or config.device)
    backbone_path = source_backbone_path(config, args.backbone_seed)
    model, backbone_payload = load_backbone_checkpoint(
        backbone_path, device, freeze=True
    )
    validate_backbone_protocol_binding(
        config,
        backbone_payload,
        backbone_seed=args.backbone_seed,
        protocol_lock=lock,
    )
    _configure_backbone_scope(model, method.training_scope)
    head = build_distance_head(method.head).to(device)
    initialization: dict[str, Any] = {"mode": "none", "loaded_keys": []}
    if method.initialization_parent:
        parent_path = head_checkpoint_path(
            config,
            method=method.initialization_parent,
            backbone_seed=args.backbone_seed,
            head_seed=args.head_seed,
        )
        parent = torch.load(parent_path, map_location=device, weights_only=False)
        parent_method, parent_hash, _ = load_and_resolve_method(
            config.paths.method_catalog,
            method.initialization_parent,
            decision_root=config.paths.decision_root,
            protocol_lock=lock,
        )
        if parent.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]:
            raise ValueError("joint initialization parent uses another analysis lock")
        if parent.get("protocol_id") != config.protocol_id:
            raise ValueError("joint initialization parent uses another protocol ID")
        if parent.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]:
            raise ValueError("joint initialization parent uses another protocol lock")
        if parent.get("backbone_sha256") != sha256_file(backbone_path):
            raise ValueError("joint initialization parent uses another backbone")
        if parent.get("method", {}).get("name") != method.initialization_parent:
            raise ValueError("joint initialization parent metadata mismatch")
        if (
            parent.get("formal_run") is not True
            or parent.get("checkpoint_selection") != "final_step"
            or int(parent.get("final_step", -1)) != config.training.steps
            or parent.get("method_sha256") != parent_hash
            or parent_method.head is None
        ):
            raise ValueError("joint initialization parent is not a locked formal head")
        parent_bank = parent.get("candidate_bank", {})
        parent_bank_path = parent_bank.get("path")
        if not isinstance(parent_bank_path, str) or sha256_file(
            parent_bank_path
        ) != parent_bank.get("sha256"):
            raise ValueError("joint initialization parent candidate bank changed")
        initialization = _initialize_from_parent(head, method, parent)
        initialization.update(
            {
                "parent_checkpoint_path": parent_path.as_posix(),
                "parent_checkpoint_sha256": sha256_file(parent_path),
                "parent_training_spec_sha256": parent.get("training_spec_sha256"),
            }
        )
    for parameter in head.parameters():
        parameter.requires_grad = bool(method.update_head)

    train_dataset = ShardedGoalDataset(
        cache_index_path(config, split_role="train", backbone_seed=args.backbone_seed)
    )
    cal_dataset = ShardedGoalDataset(
        cache_index_path(config, split_role="cal", backbone_seed=args.backbone_seed)
    )
    cache_bindings = {
        "train": validate_cache_binding(
            train_dataset,
            config,
            split_role="train",
            backbone_seed=args.backbone_seed,
            protocol_lock=lock,
        ),
        "cal": validate_cache_binding(
            cal_dataset,
            config,
            split_role="cal",
            backbone_seed=args.backbone_seed,
            protocol_lock=lock,
        ),
    }
    bank_metadata, candidate_actions = load_candidate_bank(
        candidate_bank_path(
            config, split_role="train", backbone_seed=args.backbone_seed
        )
    )
    if int(bank_metadata["backbone_seed"]) != args.backbone_seed:
        raise ValueError("candidate bank backbone seed mismatch")
    if bank_metadata.get("protocol_id") != config.protocol_id:
        raise ValueError("candidate bank protocol mismatch")
    if bank_metadata.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]:
        raise ValueError("candidate bank analysis lock mismatch")
    if bank_metadata.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]:
        raise ValueError("candidate bank protocol-lock mismatch")
    if bank_metadata.get("split_role") != "train":
        raise ValueError("training requires the train-role candidate bank")
    if int(bank_metadata["candidate_count"]) != config.training.trajectory_candidates:
        raise ValueError("candidate bank count differs from training protocol")
    if int(bank_metadata["horizon"]) != config.planner.horizon:
        raise ValueError("candidate bank horizon differs from training protocol")
    calibrated = _calibrate(
        head,
        method,
        model,
        cal_dataset,
        candidate_actions,
        config=config,
        backbone_seed=args.backbone_seed,
        device=device,
    )
    training_rng_seed = hierarchical_seed(
        "distance-head-training-rng", args.backbone_seed, args.head_seed
    )
    set_seed(training_rng_seed, deterministic=config.training.deterministic)
    trainable_head = [
        parameter for parameter in head.parameters() if parameter.requires_grad
    ]
    trainable_backbone = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    groups = []
    if trainable_head:
        groups.append({"params": trainable_head, "lr": config.training.learning_rate})
    if trainable_backbone:
        groups.append(
            {
                "params": trainable_backbone,
                "lr": config.training.joint_backbone_learning_rate,
            }
        )
    if not groups:
        raise ValueError("training method has no trainable parameter")
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=config.training.weight_decay,
    )
    total_steps = int(args.diagnostic_steps or config.training.steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _scheduler_multiplier(
            step,
            total_steps=total_steps,
            warmup_fraction=config.training.warmup_fraction,
            final_ratio=config.training.final_lr_ratio,
        ),
    )
    source_config = load_json(config.paths.source_config)
    source_train = baseline_config(source_config, "lewm_l2_cem_seqlen2")["train"]
    source_entries = read_jsonl(config.paths.train_manifest)
    sigreg = SIGReg(
        knots=int(source_train["sigreg_knots"]),
        num_proj=int(source_train["sigreg_num_proj"]),
    ).to(device)
    jepa_rng = np.random.default_rng(
        hierarchical_seed(
            "distance-head-jepa-continuation", args.backbone_seed, args.head_seed
        )
    )
    start_step = 0
    state_path = _training_state_path(
        method.name,
        args.backbone_seed,
        args.head_seed,
        diagnostic_steps=int(args.diagnostic_steps),
    )
    bank_path = candidate_bank_path(
        config, split_role="train", backbone_seed=args.backbone_seed
    )
    candidate_bank_binding = {
        "path": bank_path.as_posix(),
        "sha256": sha256_file(bank_path),
        "metadata": bank_metadata,
    }
    training_spec = canonical_json_sha256(
        {
            "schema": TRAIN_STATE_SCHEMA,
            "analysis_spec_sha256": lock["analysis_spec_sha256"],
            "protocol_lock_sha256": lock["protocol_lock_sha256"],
            "method_sha256": method_hash,
            "backbone_sha256": sha256_file(backbone_path),
            "backbone_seed": args.backbone_seed,
            "head_seed": args.head_seed,
            "training_rng_seed": training_rng_seed,
            "steps": total_steps,
            "calibrated_weights": calibrated,
            "candidate_bank": candidate_bank_binding,
            "cache_bindings": cache_bindings,
            "initialization": initialization,
        }
    )
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("training_spec_sha256") != training_spec:
            raise ValueError(
                "resume state belongs to a different training specification"
            )
        head.load_state_dict(state["head_state_dict"], strict=True)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        jepa_rng.bit_generator.state = state["jepa_rng_state"]
        _restore_rng_state(state["rng_state"])
        start_step = int(state["step"])
    elif not args.resume and state_path.exists():
        raise FileExistsError(
            f"stale training state exists; use --resume: {state_path}"
        )

    accumulation = (
        config.training.effective_batch_size // config.training.microbatch_size
    )
    trajectory_contexts = config.training.trajectory_contexts_per_step // accumulation
    recent: list[dict[str, float]] = []
    started = time.perf_counter()
    for step in range(start_step + 1, total_steps + 1):
        full_batch = sample_training_batch(
            train_dataset,
            sampler=method.sampler,
            effective_batch_size=config.training.effective_batch_size,
            pairs_per_topology=config.training.pairs_per_topology,
            schedule_seed=config.seeds.sample_schedule_seed,
            backbone_seed=args.backbone_seed,
            step=step,
        )
        optimizer.zero_grad(set_to_none=True)
        step_metrics: dict[str, float] = {}
        for micro_index in range(accumulation):
            start = micro_index * config.training.microbatch_size
            stop = start + config.training.microbatch_size
            cpu_batch = slice_training_batch(full_batch, start, stop)
            if method.training_scope in (
                TrainingScope.PROJECTOR_PREDICTOR,
                TrainingScope.FULL,
            ):
                cpu_batch = refresh_joint_latents(
                    train_dataset,
                    cpu_batch,
                    model,
                    device=device,
                    gradients=method.distance_gradients_to_backbone,
                )
            batch = cpu_batch.to(device)
            needs_predicted = bool(
                method.objectives.predicted_listwise
                or method.objectives.predicted_consistency
            )
            predicted = (
                _predict_all_actions(
                    model,
                    batch,
                    gradients=method.distance_gradients_to_backbone,
                )
                if needs_predicted
                else None
            )
            trajectory = None
            if method.objectives.trajectory_listwise > 0:
                horizon = method.trajectory_horizons[
                    (step - 1) % len(method.trajectory_horizons)
                ]
                bank_index = (step - 1) % candidate_actions.shape[0]
                trajectory = _trajectory_batch(
                    model,
                    train_dataset,
                    batch,
                    candidate_actions[bank_index],
                    contexts=trajectory_contexts,
                    horizon=horizon,
                    device=device,
                    gradients=method.distance_gradients_to_backbone,
                )
            terms = compute_objective_terms(
                head,
                method,
                batch,
                predicted_next=predicted,
                trajectory=trajectory,
            )
            if terms:
                distance_total = weighted_total(terms, calibrated) / accumulation
                distance_total.backward()
                for name, value in terms.items():
                    step_metrics[name] = (
                        step_metrics.get(name, 0.0)
                        + float(value.detach()) / accumulation
                    )
        if method.objectives.original_jepa > 0:
            entry = source_entries[step % len(source_entries)]
            sequence = sample_lewm_sequence(
                entry,
                rng=jepa_rng,
                batch_size=int(source_train["batch_size"]),
                sequence_length=int(source_train["sequence_length"]),
            )
            jepa_total, jepa_terms = lewm_loss(
                model,
                sigreg,
                sequence,
                maze_size=int(entry["maze_size"]),
                device=device,
                weights={
                    "prediction": float(source_train["lambda_prediction"]),
                    "sigreg": float(source_train["lambda_sigreg"]),
                    "absolute": float(source_train["lambda_abs_position"]),
                    "relative": float(source_train["lambda_relative_position"]),
                    "goal": float(source_train["lambda_goal_position"]),
                },
            )
            (method.objectives.original_jepa * jepa_total).backward()
            step_metrics["original_jepa"] = float(jepa_total.detach())
            for name, value in jepa_terms.items():
                step_metrics[f"jepa_{name}"] = float(value.detach())
        parameters = trainable_head + trainable_backbone
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, config.training.grad_clip
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        step_metrics["gradient_norm"] = float(gradient_norm)
        recent.append(step_metrics)
        if len(recent) > 100:
            recent.pop(0)
        if step == 1 or step == total_steps or step % 500 == 0:
            means = {
                name: float(np.mean([row.get(name, 0.0) for row in recent]))
                for name in sorted({key for row in recent for key in row})
            }
            print(f"{method.name} step {step}/{total_steps}: {means}", flush=True)
        if step % 1000 == 0 and step < total_steps:
            atomic_torch_save(
                state_path,
                {
                    "schema": TRAIN_STATE_SCHEMA,
                    "training_spec_sha256": training_spec,
                    "step": step,
                    "head_state_dict": cpu_state_dict(head),
                    "model_state_dict": cpu_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "jepa_rng_state": jepa_rng.bit_generator.state,
                    "rng_state": _rng_state(),
                },
            )

    output = (
        _diagnostic_checkpoint_path(
            method.name,
            args.backbone_seed,
            args.head_seed,
            total_steps,
        )
        if diagnostic
        else head_checkpoint_path(
            config,
            method=method.name,
            backbone_seed=args.backbone_seed,
            head_seed=args.head_seed,
        )
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final-step checkpoint: {output}")
    payload: dict[str, Any] = {
        "experiment_family": "procgen_maze_distance_head_study",
        "format_version": 1,
        "protocol_id": config.protocol_id,
        "stage": "distance_head_training",
        "formal_run": not diagnostic,
        "method": method.model_dump(mode="json"),
        "method_sha256": method_hash,
        "decision_sha256s": list(decision_hashes),
        "training_spec_sha256": training_spec,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "backbone_seed": int(args.backbone_seed),
        "head_seed": int(args.head_seed),
        "training_rng_seed": training_rng_seed,
        "backbone_path": backbone_path.as_posix(),
        "backbone_sha256": sha256_file(backbone_path),
        "head_spec": method.head.model_dump(mode="json"),
        "head_state_dict": cpu_state_dict(head),
        "calibrated_weights": calibrated,
        "checkpoint_selection": "final_step",
        "final_step": total_steps,
        "elapsed_seconds": float(time.perf_counter() - started),
        "recent_metrics": recent,
        "source_backbone_training_spec_sha256": backbone_payload.get(
            "source_training_spec_sha256",
            backbone_payload.get("training_spec_sha256"),
        ),
        "cache_bindings": cache_bindings,
        "candidate_bank": candidate_bank_binding,
        "initialization": initialization,
    }
    if method.training_scope != TrainingScope.FROZEN:
        payload["model_config"] = backbone_payload["model_config"]
        payload["model_state_dict"] = cpu_state_dict(model)
    atomic_torch_save(output, payload)
    state_path.unlink(missing_ok=True)
    print(Path(output))


if __name__ == "__main__":
    main()
