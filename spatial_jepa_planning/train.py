#!/usr/bin/env python3
"""Train Spatial-JEPA representations and aligned maze planners."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdwm.losses import SIGReg
from spatial_jepa_planning import EXPERIMENT_FAMILY, FORMAT_VERSION
from spatial_jepa_planning.common import (
    ManifestSampler,
    configure_representation_training,
    estimate_planner_conv_macs,
    estimate_representation_planning_conv_macs,
    experiment_code_fingerprint,
    git_commit,
    gradient_cosine,
    load_representation_checkpoint,
    make_rng_streams,
    parameter_count,
    parse_int_list,
    planner_features,
    protocol_metadata,
    require_clean_worktree,
    require_new_output,
    resolve_device,
    sample_map_batch,
    sample_sequence_batch,
    save_checkpoint,
    set_seed,
    sha256_file,
    validate_manifest_pair,
)
from spatial_jepa_planning.losses import (
    PlannerLossWeights,
    RepresentationLossWeights,
    covariance_loss,
    map_decoder_loss,
    planner_loss,
    variance_floor_loss,
)
from spatial_jepa_planning.models import (
    PlannerConfig,
    SpatialRepresentation,
    SpatialRepresentationConfig,
    build_planner,
    make_ema_target,
    update_ema_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("representation", "planner", "joint"), required=True
    )
    parser.add_argument("--variant-name", required=True)
    parser.add_argument(
        "--train-manifest", default="data/splits/unisize_train_manifest.jsonl"
    )
    parser.add_argument(
        "--eval-manifest",
        default="data/splits/spatial_jepa_confirm_eval_manifest.jsonl",
    )
    parser.add_argument(
        "--development-manifest", default="data/splits/unisize_eval_manifest.jsonl"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--representation-ckpt", default=None)
    parser.add_argument("--input-mode", choices=("raw", "spatial_jepa"), default="raw")
    parser.add_argument(
        "--encoder-mode", choices=("frozen", "last_block", "all"), default="frozen"
    )
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--map-batch-size", type=int, default=8)
    parser.add_argument("--trajectories-per-map", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=128)

    parser.add_argument("--spatial-dim", type=int, default=64)
    parser.add_argument("--planning-dim", type=int, default=64)
    parser.add_argument("--encoder-blocks", type=int, default=3)
    parser.add_argument("--predictor-blocks", type=int, default=2)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--sigreg-max-tokens", type=int, default=64)
    parser.add_argument("--lambda-prediction", type=float, default=1.0)
    parser.add_argument("--lambda-sigreg", type=float, default=0.09)
    parser.add_argument("--lambda-variance", type=float, default=0.0)
    parser.add_argument("--lambda-covariance", type=float, default=0.0)
    parser.add_argument("--lambda-map-wall", type=float, default=0.5)
    parser.add_argument("--lambda-map-agent", type=float, default=0.25)
    parser.add_argument("--lambda-map-goal", type=float, default=0.25)
    parser.add_argument("--lambda-map-valid", type=float, default=0.5)

    parser.add_argument(
        "--planner-type",
        choices=("feedforward", "feedforward_dilated", "iterative"),
        default="iterative",
    )
    parser.add_argument("--planner-hidden-dim", type=int, default=64)
    parser.add_argument("--planner-depth", type=int, default=8)
    parser.add_argument("--no-recall", dest="recall", action="store_false")
    parser.set_defaults(recall=True)
    parser.add_argument("--train-iterations", default="4,8,16,32,64,128")
    parser.add_argument(
        "--iteration-schedule",
        choices=("fixed", "random", "progressive"),
        default="random",
    )
    parser.add_argument("--deep-supervision-every", type=int, default=0)
    parser.add_argument("--distance-scale", type=float, default=128.0)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--lambda-action", type=float, default=1.0)
    parser.add_argument("--lambda-valid", type=float, default=0.25)
    parser.add_argument("--lambda-bellman", type=float, default=0.5)
    parser.add_argument("--lambda-gap", type=float, default=0.5)
    parser.add_argument("--lambda-convergence", type=float, default=0.0)
    parser.add_argument("--gap-margin", type=float, default=1.0)
    parser.add_argument("--lambda-joint-representation", type=float, default=1.0)
    parser.add_argument("--lambda-planner-map", type=float, default=0.0)
    parser.add_argument("--gradient-audit-every", type=int, default=500)
    parser.add_argument("--experiment-spec-sha256", default="")
    parser.add_argument("--analysis-spec-sha256", default="")
    parser.add_argument("--allow-unlocked-spec", action="store_true")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def representation_weights(args: argparse.Namespace) -> RepresentationLossWeights:
    return RepresentationLossWeights(
        prediction=args.lambda_prediction,
        sigreg=args.lambda_sigreg,
        variance=args.lambda_variance,
        covariance=args.lambda_covariance,
        wall=args.lambda_map_wall,
        agent=args.lambda_map_agent,
        goal=args.lambda_map_goal,
        valid=args.lambda_map_valid,
    )


def planner_weights(args: argparse.Namespace) -> PlannerLossWeights:
    return PlannerLossWeights(
        value=args.lambda_value,
        action=args.lambda_action,
        valid=args.lambda_valid,
        bellman=args.lambda_bellman,
        gap=args.lambda_gap,
        convergence=args.lambda_convergence,
        gap_margin=args.gap_margin,
    )


def build_representation(args: argparse.Namespace) -> SpatialRepresentation:
    config = SpatialRepresentationConfig(
        spatial_dim=args.spatial_dim,
        planning_dim=args.planning_dim,
        encoder_blocks=args.encoder_blocks,
        predictor_blocks=args.predictor_blocks,
    )
    return SpatialRepresentation(config)


def load_or_build_representation(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SpatialRepresentation, dict[str, Any] | None]:
    if args.representation_ckpt:
        representation, data = load_representation_checkpoint(
            args.representation_ckpt, device
        )
        return representation, data
    if args.stage != "representation":
        raise ValueError(
            "planner/joint stage with spatial_jepa requires --representation-ckpt"
        )
    return build_representation(args).to(device), None


def prepare_sigreg_tokens(latent: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if latent.ndim != 5:
        raise ValueError("SIGReg spatial latent must be [B,T,C,H,W]")
    batch, time, channels, height, width = latent.shape
    tokens = latent.permute(1, 3, 4, 0, 2).reshape(
        time * height * width, batch, channels
    )
    if max_tokens > 0 and tokens.shape[0] > max_tokens:
        indices = torch.linspace(
            0,
            tokens.shape[0] - 1,
            max_tokens,
            device=tokens.device,
        ).long()
        tokens = tokens[indices]
    return tokens


def compute_representation_loss(
    online: SpatialRepresentation,
    target: SpatialRepresentation,
    observations: torch.Tensor,
    actions: torch.Tensor,
    valid_fields: torch.Tensor,
    sigreg: SIGReg,
    weights: RepresentationLossWeights,
    sigreg_max_tokens: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dynamics = online.dynamics_latent(observations)
    with torch.no_grad():
        target_dynamics = target.dynamics_latent(observations)
    batch, time, channels, height, width = dynamics.shape
    current = dynamics[:, :-1].reshape(-1, channels, height, width)
    action_flat = actions.reshape(-1)
    prediction = online.predictor(current, action_flat)
    prediction_target = target_dynamics[:, 1:].reshape_as(prediction)
    prediction_loss = F.smooth_l1_loss(prediction, prediction_target)
    sigreg_loss = sigreg(prepare_sigreg_tokens(dynamics, sigreg_max_tokens))
    variance = variance_floor_loss(dynamics.reshape(-1, channels, height, width))
    covariance = covariance_loss(dynamics.reshape(-1, channels, height, width))

    planning = online.planning_latent(observations)
    planning_flat = planning.reshape(-1, planning.shape[2], height, width)
    decoded = online.map_decoder(planning_flat)
    observations_flat = observations.reshape(-1, height, width, observations.shape[-1])
    valid_flat = valid_fields.reshape(-1, 4, height, width)
    map_losses = map_decoder_loss(decoded, observations_flat, valid_flat)
    total = (
        weights.prediction * prediction_loss
        + weights.sigreg * sigreg_loss
        + weights.variance * variance
        + weights.covariance * covariance
        + weights.wall * map_losses["wall"]
        + weights.agent * map_losses["agent"]
        + weights.goal * map_losses["goal"]
        + weights.valid * map_losses["valid"]
    )
    metrics = {
        "representation_total": total,
        "prediction": prediction_loss,
        "sigreg": sigreg_loss,
        "variance": variance,
        "covariance": covariance,
        "map_wall": map_losses["wall"],
        "map_agent": map_losses["agent"],
        "map_goal": map_losses["goal"],
        "map_valid": map_losses["valid"],
    }
    return total, metrics


def choose_iterations(
    values: tuple[int, ...],
    schedule: str,
    step: int,
    total_steps: int,
    rng: np.random.Generator,
) -> int:
    if schedule == "fixed":
        return int(values[-1])
    if schedule == "random":
        return int(rng.choice(values))
    if schedule == "progressive":
        fraction = step / max(total_steps, 1)
        available = max(1, min(len(values), int(math.ceil(fraction * len(values)))))
        return int(rng.choice(values[:available]))
    raise ValueError(f"unknown iteration schedule: {schedule}")


def enable_joint_representation_parameters(
    representation: SpatialRepresentation,
    mode: str,
) -> list[torch.nn.Parameter]:
    parameters = configure_representation_training(representation, mode)
    if mode == "frozen":
        raise ValueError("joint stage cannot use encoder-mode=frozen")
    modules = [representation.dynamics_projector, representation.predictor]
    for module in modules:
        for parameter in module.parameters():
            if not parameter.requires_grad:
                parameter.requires_grad = True
                parameters.append(parameter)
    return list(dict.fromkeys(parameters))


def mean_window(history: list[dict[str, float]], size: int) -> dict[str, float]:
    names = sorted({name for row in history[-size:] for name in row})
    return {
        name: float(np.mean([row[name] for row in history[-size:] if name in row]))
        for name in names
    }


def main() -> None:
    args = parse_args()
    require_clean_worktree(args.allow_dirty_worktree)
    require_new_output(args.output, args.overwrite)
    if not args.experiment_spec_sha256 and not args.allow_unlocked_spec:
        raise ValueError(
            "formal training requires --experiment-spec-sha256 from run_plan.py"
        )
    if not args.analysis_spec_sha256 and not args.allow_unlocked_spec:
        raise ValueError(
            "formal training requires --analysis-spec-sha256 from run_plan.py"
        )
    if not args.deterministic and not args.allow_unlocked_spec:
        raise ValueError("formal training requires --deterministic")
    if args.steps <= 0 or args.map_batch_size <= 0:
        raise ValueError("steps and map-batch-size must be positive")
    if args.stage == "representation" and args.input_mode != "raw":
        raise ValueError("representation stage must use input-mode=raw")
    if args.stage in {"planner", "joint"} and args.input_mode == "spatial_jepa":
        if not args.representation_ckpt:
            raise ValueError("spatial_jepa planner requires --representation-ckpt")
    if args.stage == "joint" and args.input_mode != "spatial_jepa":
        raise ValueError("joint stage is defined only for spatial_jepa input")
    if not 0.0 <= args.ema_momentum < 1.0:
        raise ValueError("ema-momentum must be in [0,1)")
    if not 0.0 < args.encoder_lr_multiplier <= 1.0:
        raise ValueError("encoder-lr-multiplier must be in (0,1]")
    if args.lambda_planner_map > 0.0 and args.encoder_mode == "frozen":
        raise ValueError("planner map preservation requires a trainable representation")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")

    device = resolve_device(args.device)
    args.device = str(device)
    set_seed(args.seed, deterministic=args.deterministic)
    streams = make_rng_streams(args.seed)
    train_entries, _, overlap = validate_manifest_pair(
        args.train_manifest, args.eval_manifest
    )
    _, _, development_overlap = validate_manifest_pair(
        args.train_manifest, args.development_manifest
    )
    _, _, eval_overlap = validate_manifest_pair(
        args.development_manifest, args.eval_manifest
    )
    sampler = ManifestSampler(train_entries)
    iterations = parse_int_list(args.train_iterations)
    rep_weights = representation_weights(args)
    plan_weights = planner_weights(args)

    representation: SpatialRepresentation | None = None
    target: SpatialRepresentation | None = None
    source_checkpoint: dict[str, Any] | None = None
    planner: torch.nn.Module | None = None
    base_parameters: list[torch.nn.Parameter] = []
    representation_parameters: list[torch.nn.Parameter] = []

    if args.stage == "representation" or args.input_mode == "spatial_jepa":
        representation, source_checkpoint = load_or_build_representation(args, device)
        if source_checkpoint is not None:
            source_protocol = source_checkpoint.get("protocol", {})
            if int(source_protocol.get("seed", -1)) != args.seed:
                raise ValueError("planner and source representation seeds must match")
            if source_protocol.get("train_manifest_sha256") != sha256_file(
                args.train_manifest
            ):
                raise ValueError("source representation train manifest does not match")
            if source_protocol.get("eval_manifest_sha256") != sha256_file(
                args.eval_manifest
            ):
                raise ValueError("source representation eval manifest does not match")
            if source_protocol.get("development_manifest_sha256") != sha256_file(
                args.development_manifest
            ):
                raise ValueError(
                    "source representation development manifest does not match"
                )
            if source_protocol.get("code_fingerprint") != experiment_code_fingerprint():
                raise ValueError(
                    "source representation was produced by different experiment code"
                )
            if (
                source_protocol.get("git_dirty") is not False
                and not args.allow_dirty_worktree
            ):
                raise ValueError(
                    "formal planner training rejects a representation checkpoint "
                    "created from a dirty worktree"
                )
            if not source_protocol.get("git_commit") and not args.allow_dirty_worktree:
                raise ValueError("source representation is missing its Git commit")
            if (
                source_protocol.get("git_commit") != git_commit()
                and not args.allow_dirty_worktree
            ):
                raise ValueError(
                    "source representation and planner training must use the same "
                    "Git commit"
                )
            if (
                source_checkpoint.get("analysis_spec_sha256")
                != args.analysis_spec_sha256
            ):
                raise ValueError(
                    "source representation analysis spec does not match "
                    "planner training"
                )
        if source_checkpoint and "target_state_dict" in source_checkpoint:
            target = make_ema_target(representation)
            target.load_state_dict(source_checkpoint["target_state_dict"], strict=True)
        else:
            target = make_ema_target(representation)

    if args.stage == "representation":
        assert representation is not None
        for parameter in representation.parameters():
            parameter.requires_grad = True
        base_parameters.extend(representation.parameters())
    else:
        input_channels = (
            5 if args.input_mode == "raw" else int(representation.config.planning_dim)
        )
        planner_config = PlannerConfig(
            input_channels=input_channels,
            hidden_dim=args.planner_hidden_dim,
            planner_type=args.planner_type,
            depth=args.planner_depth,
            recall=args.recall,
        )
        planner = build_planner(planner_config).to(device)
        base_parameters.extend(planner.parameters())
        if representation is not None:
            if args.stage == "joint":
                representation_parameters.extend(
                    enable_joint_representation_parameters(
                        representation, args.encoder_mode
                    )
                )
            else:
                representation_parameters.extend(
                    configure_representation_training(representation, args.encoder_mode)
                )

    base_parameters = list(dict.fromkeys(base_parameters))
    representation_parameters = list(dict.fromkeys(representation_parameters))
    optimizer_parameters = [*base_parameters, *representation_parameters]
    if not optimizer_parameters:
        raise ValueError("no trainable parameters were selected")
    optimizer_groups: list[dict[str, Any]] = [
        {"params": base_parameters, "lr": args.lr}
    ]
    if representation_parameters:
        optimizer_groups.append(
            {
                "params": representation_parameters,
                "lr": args.lr * args.encoder_lr_multiplier,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
        if args.scheduler == "cosine"
        else None
    )
    sigreg = SIGReg(knots=17, num_proj=args.sigreg_num_proj).to(device)

    print("=" * 88)
    print("SPATIAL-JEPA ITERATIVE PLANNING TRAIN")
    print("=" * 88)
    print(
        f"variant={args.variant_name} stage={args.stage} input={args.input_mode} "
        f"seed={args.seed} device={device}"
    )
    print(
        f"train_entries={len(train_entries)} sizes={sampler.sizes} holdout={overlap} "
        f"trainable_params={sum(p.numel() for p in optimizer_parameters):,}"
    )

    history: list[dict[str, float]] = []
    gradient_history: list[dict[str, float]] = []
    run_started = time.time()
    started = time.time()
    for step in range(1, args.steps + 1):
        selected = sampler.sample(streams.entries, args.map_batch_size)
        metrics: dict[str, torch.Tensor] = {}
        representation_total: torch.Tensor | None = None
        planning_total: torch.Tensor | None = None

        if args.stage in {"representation", "joint"}:
            assert representation is not None and target is not None
            sequence_obs, sequence_actions, sequence_valid = sample_sequence_batch(
                selected,
                rng=streams.sequences,
                device=device,
                sequence_length=args.seq_len,
                trajectories_per_map=args.trajectories_per_map,
            )
            representation_total, representation_metrics = compute_representation_loss(
                representation,
                target,
                sequence_obs,
                sequence_actions,
                sequence_valid,
                sigreg,
                rep_weights,
                args.sigreg_max_tokens,
            )
            metrics.update(representation_metrics)

        if args.stage in {"planner", "joint"}:
            assert planner is not None
            map_obs, map_targets = sample_map_batch(
                selected, streams.map_states, device
            )
            features = planner_features(map_obs, args.input_mode, representation)
            count = (
                args.planner_depth
                if args.planner_type.startswith("feedforward")
                else choose_iterations(
                    iterations,
                    args.iteration_schedule,
                    step,
                    args.steps,
                    streams.iteration_schedule,
                )
            )
            outputs = planner(
                features,
                iterations=count,
                deep_supervision_every=args.deep_supervision_every,
            )
            planning_total, planning_metrics = planner_loss(
                outputs,
                map_targets,
                plan_weights,
                args.distance_scale,
                iteration_budgeted=args.planner_type == "iterative",
            )
            if (
                args.stage == "planner"
                and representation is not None
                and args.lambda_planner_map > 0.0
            ):
                decoded = representation.map_decoder(features)
                map_losses = map_decoder_loss(
                    decoded,
                    map_obs,
                    map_targets["valid_action_mask"],
                )
                planner_map_total = (
                    rep_weights.wall * map_losses["wall"]
                    + rep_weights.agent * map_losses["agent"]
                    + rep_weights.goal * map_losses["goal"]
                    + rep_weights.valid * map_losses["valid"]
                )
                planning_total = (
                    planning_total + args.lambda_planner_map * planner_map_total
                )
                metrics["planner_map"] = planner_map_total
            metrics.update(
                {f"planner_{name}": value for name, value in planning_metrics.items()}
            )
            metrics["iterations"] = planning_total.new_tensor(float(count))

        if args.stage == "representation":
            assert representation_total is not None
            total = representation_total
        elif args.stage == "planner":
            assert planning_total is not None
            total = planning_total
        else:
            assert representation_total is not None and planning_total is not None
            total = (
                planning_total + args.lambda_joint_representation * representation_total
            )
            if step % args.gradient_audit_every == 0 or step == 1:
                assert representation is not None
                shared = [
                    parameter
                    for parameter in representation.encoder.parameters()
                    if parameter.requires_grad
                ]
                gradient_row = gradient_cosine(
                    representation_total, planning_total, shared
                )
                gradient_row["step"] = float(step)
                gradient_history.append(gradient_row)
                print(
                    "gradient_audit "
                    f"step={step} cosine={gradient_row['cosine']:.4f} "
                    f"rep_norm={gradient_row['first_norm']:.4f} "
                    f"plan_norm={gradient_row['second_norm']:.4f}"
                )

        metrics["total"] = total
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(
                f"non-finite training loss at step={step}: "
                f"{float(total.detach().cpu())}"
            )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                optimizer_parameters, args.grad_clip
            )
        else:
            squared = total.new_tensor(0.0)
            for parameter in optimizer_parameters:
                if parameter.grad is not None:
                    squared = squared + parameter.grad.detach().square().sum()
            grad_norm = squared.sqrt()
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(
                f"non-finite gradient norm at step={step}: "
                f"{float(grad_norm.detach().cpu())}"
            )
        metrics["grad_norm"] = grad_norm
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if (
            representation is not None
            and target is not None
            and args.stage in {"representation", "joint"}
        ):
            update_ema_target(representation, target, args.ema_momentum)

        history.append(
            {name: float(value.detach().cpu()) for name, value in metrics.items()}
        )
        if step % args.log_every == 0 or step == args.steps:
            averages = mean_window(history, min(args.log_every, len(history)))
            elapsed = max(time.time() - started, 1e-6)
            fields = " ".join(
                f"{name}={value:.4f}"
                for name, value in averages.items()
                if name
                in {
                    "total",
                    "prediction",
                    "sigreg",
                    "map_valid",
                    "planner_action",
                    "planner_bellman",
                    "planner_gap",
                    "planner_map",
                    "planner_value",
                    "iterations",
                }
            )
            print(
                f"step={step:>6d}/{args.steps} {fields} "
                f"steps/s={args.log_every / elapsed:.2f}"
            )
            started = time.time()

    planner_inference_macs: dict[str, dict[str, int]] | None = None
    representation_inference_macs: dict[str, int] | None = None
    representation_planning_parameter_count = 0
    if planner is not None:
        mac_iterations = (
            (args.planner_depth,)
            if args.planner_type.startswith("feedforward")
            else tuple(sorted({*iterations, 256}))
        )
        planner_inference_macs = {
            str(size): {
                str(count): estimate_planner_conv_macs(
                    planner,
                    input_channels=int(planner.config.input_channels),
                    maze_size=size,
                    iterations=count,
                    device=device,
                )
                for count in mac_iterations
            }
            for size in (21, 25)
        }
        if representation is None:
            representation_inference_macs = {str(size): 0 for size in (21, 25)}
        else:
            representation_inference_macs = {
                str(size): estimate_representation_planning_conv_macs(
                    representation,
                    maze_size=size,
                    device=device,
                )
                for size in (21, 25)
            }
            representation_planning_parameter_count = sum(
                parameter.numel()
                for module in (
                    representation.encoder,
                    representation.planning_projector,
                )
                for parameter in module.parameters()
            )

    payload: dict[str, Any] = {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": FORMAT_VERSION,
        "stage": args.stage,
        "variant_name": args.variant_name,
        "input_mode": args.input_mode,
        "training_args": vars(args),
        "experiment_spec_sha256": args.experiment_spec_sha256,
        "analysis_spec_sha256": args.analysis_spec_sha256,
        "protocol": protocol_metadata(
            train_manifest=args.train_manifest,
            eval_manifest=args.eval_manifest,
            development_manifest=args.development_manifest,
            seed=args.seed,
            max_steps=args.max_steps,
        ),
        "holdout": {
            "train_vs_confirmatory": overlap,
            "train_vs_development": development_overlap,
            "development_vs_confirmatory": eval_overlap,
        },
        "training_summary": mean_window(history, min(args.log_every, len(history))),
        "training_accounting": {
            "elapsed_seconds": float(time.time() - run_started),
            "optimizer_steps": int(args.steps),
            "planner_map_examples": int(args.steps * args.map_batch_size)
            if args.stage in {"planner", "joint"}
            else 0,
            "representation_trajectories": int(
                args.steps * args.map_batch_size * args.trajectories_per_map
            )
            if args.stage in {"representation", "joint"}
            else 0,
            "sequence_length": int(args.seq_len)
            if args.stage in {"representation", "joint"}
            else 0,
        },
        "gradient_history": gradient_history,
        "source_representation_ckpt": args.representation_ckpt,
        "source_representation_sha256": sha256_file(args.representation_ckpt)
        if args.representation_ckpt
        else None,
    }
    if representation is not None:
        payload["representation_config"] = representation.config.to_dict()
        payload["representation_state_dict"] = representation.state_dict()
    if target is not None:
        payload["target_state_dict"] = target.state_dict()
    if planner is not None:
        payload["planner_config"] = planner.config.to_dict()
        payload["planner_state_dict"] = planner.state_dict()
        payload["planner_parameter_count"] = parameter_count(planner)
        payload["representation_planning_parameter_count"] = (
            representation_planning_parameter_count
        )
        payload["total_inference_parameter_count"] = (
            parameter_count(planner) + representation_planning_parameter_count
        )
        payload["planner_inference_conv_macs"] = planner_inference_macs
        payload["representation_inference_conv_macs"] = representation_inference_macs
    save_checkpoint(args.output, payload)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
