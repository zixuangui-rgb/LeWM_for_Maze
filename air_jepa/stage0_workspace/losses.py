"""Locked AIR0 action, future, and distributional cost objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from air_jepa.stage0_workspace.models import AIRWorkspaceOutput
from air_jepa.stage0_workspace.schemas import MethodLossSpec


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"masked mean shape mismatch: {values.shape} vs {mask.shape}")
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def tie_aware_action_loss(
    energy: torch.Tensor,
    optimal_action_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy on -energy with uniform mass over all BFS-optimal actions."""

    if energy.ndim != 2 or energy.shape[1] != 4:
        raise ValueError("energy must have shape [B,4]")
    if optimal_action_mask.shape != energy.shape:
        raise ValueError("optimal action mask must match energy")
    target_mass = optimal_action_mask.sum(dim=1)
    if not bool((target_mass > 0).all()):
        raise ValueError("every sample needs at least one optimal action")
    targets = optimal_action_mask.to(energy.dtype) / target_mass[:, None]
    return -(targets * F.log_softmax(-energy, dim=1)).sum(dim=1).mean()


def distributional_cost_loss(
    logits: torch.Tensor,
    candidate_distances: torch.Tensor,
    *,
    max_distance: int,
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[:2] != candidate_distances.shape:
        raise ValueError("cost logits must be [B,4,bins] for [B,4] targets")
    if logits.shape[-1] != max_distance + 1:
        raise ValueError("cost bin count does not match max_distance")
    targets = candidate_distances.clamp(min=0, max=max_distance).long()
    raw = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    return raw / math.log(float(max_distance + 1))


@dataclass(frozen=True)
class FutureLoss:
    total: torch.Tensor
    normalized_field: torch.Tensor
    normalized_delta: torch.Tensor
    raw_field_mse: torch.Tensor
    raw_delta_mse: torch.Tensor
    copy_delta_normalized: torch.Tensor


def _normalized_channel_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("future tensors must have matching [B,4,C,H,W] shapes")
    batch, actions, channels, height, width = target.shape
    if valid_mask.shape != (batch, height, width):
        raise ValueError("future valid mask must be [B,H,W]")
    expanded_mask = valid_mask[:, None, None].expand(
        batch, actions, channels, height, width
    )
    weights = expanded_mask.to(target.dtype)
    count = weights.sum(dim=(0, 1, 3, 4)).clamp_min(1.0)
    mean = (target * weights).sum(dim=(0, 1, 3, 4)) / count
    centered = target - mean[None, None, :, None, None]
    variance = (centered.square() * weights).sum(dim=(0, 1, 3, 4)) / count
    variance = variance.detach().clamp_min(float(epsilon))
    squared = (prediction - target).square()
    normalized = squared / variance[None, None, :, None, None]
    return _masked_mean(normalized, expanded_mask), _masked_mean(squared, expanded_mask)


def future_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    epsilon: float,
) -> FutureLoss:
    if source.ndim != 4 or target.shape[0] != source.shape[0]:
        raise ValueError("source latent must be [B,C,H,W]")
    if target.shape[2:] != source.shape[1:]:
        raise ValueError("source and successor latent dimensions differ")
    normalized_field, raw_field = _normalized_channel_error(
        prediction,
        target,
        valid_mask=valid_mask,
        epsilon=epsilon,
    )
    source_expanded = source[:, None].expand_as(target)
    predicted_delta = prediction - source_expanded
    target_delta = target - source_expanded
    normalized_delta, raw_delta = _normalized_channel_error(
        predicted_delta,
        target_delta,
        valid_mask=valid_mask,
        epsilon=epsilon,
    )
    copy_delta, _ = _normalized_channel_error(
        torch.zeros_like(target_delta),
        target_delta,
        valid_mask=valid_mask,
        epsilon=epsilon,
    )
    return FutureLoss(
        total=0.5 * (normalized_field + normalized_delta),
        normalized_field=normalized_field,
        normalized_delta=normalized_delta,
        raw_field_mse=raw_field,
        raw_delta_mse=raw_delta,
        copy_delta_normalized=copy_delta,
    )


def deep_supervision_weights(
    outputs: list[AIRWorkspaceOutput],
) -> torch.Tensor:
    if not outputs:
        raise ValueError("AIR loss requires at least one output")
    iterations = torch.as_tensor(
        [output.iterations for output in outputs],
        dtype=outputs[0].energy.dtype,
        device=outputs[0].energy.device,
    )
    if not bool((iterations[1:] > iterations[:-1]).all()):
        raise ValueError("AIR outputs must have strictly increasing iteration counts")
    return iterations / iterations.sum()


@dataclass(frozen=True)
class AIRLossResult:
    total: torch.Tensor
    action: torch.Tensor
    future: torch.Tensor
    cost: torch.Tensor
    future_metrics: FutureLoss
    readout_weights: torch.Tensor


def air_loss(
    outputs: list[AIRWorkspaceOutput],
    *,
    successor_latent: torch.Tensor,
    source_latent: torch.Tensor,
    candidate_distances: torch.Tensor,
    optimal_action_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    weights: MethodLossSpec,
    max_distance: int,
    target_variance_epsilon: float,
) -> AIRLossResult:
    readout_weights = deep_supervision_weights(outputs)
    action_terms = torch.stack(
        [
            tie_aware_action_loss(output.energy, optimal_action_mask)
            for output in outputs
        ]
    )
    cost_terms = torch.stack(
        [
            distributional_cost_loss(
                output.cost_logits,
                candidate_distances,
                max_distance=max_distance,
            )
            for output in outputs
        ]
    )
    action = (readout_weights * action_terms).sum()
    cost = (readout_weights * cost_terms).sum()
    final_future = outputs[-1].predicted_future
    if final_future is None:
        raise ValueError("final AIR output must retain predicted future fields")
    future = future_prediction_loss(
        final_future,
        successor_latent,
        source_latent,
        valid_mask=valid_mask,
        epsilon=target_variance_epsilon,
    )
    total = (
        weights.action * action + weights.future * future.total + weights.cost * cost
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("AIR objective became non-finite")
    return AIRLossResult(
        total=total,
        action=action,
        future=future.total,
        cost=cost,
        future_metrics=future,
        readout_weights=readout_weights,
    )


def local_ranking_metrics(
    energy: torch.Tensor,
    optimal_action_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if energy.shape != optimal_action_mask.shape or energy.ndim != 2:
        raise ValueError("local metric tensors must have matching [B,4] shape")
    optimal = optimal_action_mask.bool()
    selected = energy.argmin(dim=1)
    correct = optimal.gather(1, selected[:, None]).squeeze(1)
    positive = torch.finfo(energy.dtype).max
    best_optimal = energy.masked_fill(~optimal, positive).min(dim=1).values
    best_bad = energy.masked_fill(optimal, positive).min(dim=1).values
    has_bad = (~optimal).any(dim=1)
    margin = _masked_mean(best_bad - best_optimal, has_bad)
    return {
        "local_top1": correct.float().mean(),
        "local_margin": margin,
    }


__all__ = [
    "AIRLossResult",
    "FutureLoss",
    "air_loss",
    "deep_supervision_weights",
    "distributional_cost_loss",
    "future_prediction_loss",
    "local_ranking_metrics",
    "tie_aware_action_loss",
]
