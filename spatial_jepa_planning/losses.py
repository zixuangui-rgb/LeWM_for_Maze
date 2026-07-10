"""Losses and local decision metrics for spatial maze planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from spatial_jepa_planning.models import PlannerOutput, neighbor_stack


@dataclass(frozen=True)
class PlannerLossWeights:
    value: float = 1.0
    action: float = 1.0
    valid: float = 0.25
    bellman: float = 0.5
    gap: float = 0.5
    convergence: float = 0.0
    gap_margin: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannerLossWeights:
        names = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in names if key in data})


@dataclass(frozen=True)
class RepresentationLossWeights:
    prediction: float = 1.0
    sigreg: float = 0.09
    variance: float = 0.0
    covariance: float = 0.0
    wall: float = 0.5
    agent: float = 0.25
    goal: float = 0.25
    valid: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepresentationLossWeights:
        names = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in names if key in data})


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"masked_mean shape mismatch: {values.shape} vs {mask.shape}")
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def spatial_soft_target_cross_entropy(
    logits: torch.Tensor,
    optimal_action_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != optimal_action_mask.shape or logits.ndim != 4:
        raise ValueError("policy logits and optimal mask must have shape [B,4,H,W]")
    target_mass = optimal_action_mask.sum(dim=1)
    valid_states = target_mass > 0
    targets = optimal_action_mask / target_mass.unsqueeze(1).clamp_min(1.0)
    per_cell = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    return masked_mean(per_cell, valid_states)


def policy_gap_loss(
    logits: torch.Tensor,
    optimal_action_mask: torch.Tensor,
    valid_action_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    optimal = optimal_action_mask.bool()
    suboptimal = valid_action_mask.bool() & ~optimal
    has_pair = optimal.any(dim=1) & suboptimal.any(dim=1)
    negative = torch.finfo(logits.dtype).min
    best_optimal = logits.masked_fill(~optimal, negative).max(dim=1).values
    best_bad = logits.masked_fill(~suboptimal, negative).max(dim=1).values
    per_cell = F.relu(float(margin) - best_optimal + best_bad)
    return masked_mean(per_cell, has_pair)


def bellman_consistency_loss(
    values: torch.Tensor,
    valid_action_mask: torch.Tensor,
    free_mask: torch.Tensor,
    goal_mask: torch.Tensor,
    distance_scale: float,
) -> torch.Tensor:
    if distance_scale <= 0:
        raise ValueError("distance_scale must be positive")
    large = float(values.shape[-2] * values.shape[-1] * 4 + 1)
    neighbors = neighbor_stack(values, large)
    next_values = neighbors.masked_fill(~valid_action_mask.bool(), large)
    backup = 1.0 + next_values.min(dim=1).values.detach()
    target = torch.where(goal_mask.bool(), torch.zeros_like(backup), backup)
    active = free_mask.bool() & (valid_action_mask.sum(dim=1) > 0)
    error = F.smooth_l1_loss(
        values,
        target,
        reduction="none",
    )
    return masked_mean(error, active)


def planner_loss(
    outputs: list[PlannerOutput],
    targets: dict[str, torch.Tensor],
    weights: PlannerLossWeights,
    distance_scale: float,
    iteration_budgeted: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not outputs:
        raise ValueError("planner must return at least one output")
    per_output: list[torch.Tensor] = []
    totals: dict[str, torch.Tensor] = {}
    distance = targets["distance"]
    for output in outputs:
        free = targets["free_mask"].bool()
        optimal_action_mask = targets["optimal_action_mask"]
        bellman_valid_action_mask = targets["valid_action_mask"].bool()
        if iteration_budgeted:
            within_budget = distance <= float(output.iterations)
            free = free & within_budget
            optimal_action_mask = optimal_action_mask * within_budget.unsqueeze(1)
            known_neighbor = neighbor_stack(
                (distance <= float(output.iterations - 1)).float(),
                0.0,
            ).bool()
            bellman_valid_action_mask = bellman_valid_action_mask & known_neighbor
        value_error = (
            F.smooth_l1_loss(
                output.value / distance_scale,
                distance / distance_scale,
                reduction="none",
            )
            * distance_scale
        )
        value = masked_mean(value_error, free)
        action = spatial_soft_target_cross_entropy(
            output.policy_logits,
            optimal_action_mask,
        )
        valid = F.binary_cross_entropy_with_logits(
            output.valid_logits,
            targets["valid_action_mask"],
        )
        bellman = bellman_consistency_loss(
            output.value,
            bellman_valid_action_mask,
            free,
            targets["goal_mask"],
            distance_scale,
        )
        gap = policy_gap_loss(
            output.policy_logits,
            optimal_action_mask,
            targets["valid_action_mask"],
            weights.gap_margin,
        )
        current = (
            weights.value * value
            + weights.action * action
            + weights.valid * valid
            + weights.bellman * bellman
            + weights.gap * gap
        )
        per_output.append(current)
        for name, component in {
            "value": value,
            "action": action,
            "valid": valid,
            "bellman": bellman,
            "gap": gap,
        }.items():
            totals[name] = totals.get(name, component.new_tensor(0.0)) + component

    loss = torch.stack(per_output).mean()
    convergence = loss.new_tensor(0.0)
    if len(outputs) > 1:
        convergence = F.mse_loss(
            F.log_softmax(outputs[-1].policy_logits, dim=1),
            F.log_softmax(outputs[-2].policy_logits.detach(), dim=1),
        )
        loss = loss + weights.convergence * convergence
    count = float(len(outputs))
    metrics = {name: value / count for name, value in totals.items()}
    metrics["convergence"] = convergence
    metrics["total"] = loss
    return loss, metrics


def variance_floor_loss(latent: torch.Tensor, minimum_std: float = 1.0) -> torch.Tensor:
    if latent.ndim != 4:
        raise ValueError("variance loss expects [N,C,H,W]")
    samples = latent.permute(0, 2, 3, 1).reshape(-1, latent.shape[1])
    if samples.shape[0] < 2:
        return latent.new_tensor(0.0)
    std = torch.sqrt(samples.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(float(minimum_std) - std).mean()


def covariance_loss(latent: torch.Tensor, max_samples: int = 4096) -> torch.Tensor:
    if latent.ndim != 4:
        raise ValueError("covariance loss expects [N,C,H,W]")
    samples = latent.permute(0, 2, 3, 1).reshape(-1, latent.shape[1])
    if samples.shape[0] > max_samples:
        indices = torch.linspace(
            0,
            samples.shape[0] - 1,
            max_samples,
            device=samples.device,
        ).long()
        samples = samples[indices]
    if samples.shape[0] < 2:
        return latent.new_tensor(0.0)
    centered = samples - samples.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(samples.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return off_diagonal.square().sum() / float(latent.shape[1])


def map_decoder_loss(
    decoded: dict[str, torch.Tensor],
    observations: torch.Tensor,
    valid_action_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if observations.ndim != 4:
        raise ValueError("map decoder targets must be [B,H,W,C]")
    wall = observations[..., 1]
    agent_target = observations[..., 2].flatten(1).argmax(dim=1)
    goal_target = observations[..., 3].flatten(1).argmax(dim=1)
    return {
        "wall": F.binary_cross_entropy_with_logits(decoded["wall_logits"], wall),
        "agent": F.cross_entropy(decoded["agent_logits"].flatten(1), agent_target),
        "goal": F.cross_entropy(decoded["goal_logits"].flatten(1), goal_target),
        "valid": F.binary_cross_entropy_with_logits(
            decoded["valid_logits"], valid_action_mask
        ),
    }


def local_policy_metrics(
    output: PlannerOutput,
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    valid = targets["valid_action_mask"].bool()
    optimal = targets["optimal_action_mask"].bool()
    negative = torch.finfo(output.policy_logits.dtype).min
    masked_logits = output.policy_logits.masked_fill(~valid, negative)
    prediction = masked_logits.argmax(dim=1)
    chosen_optimal = optimal.gather(1, prediction.unsqueeze(1)).squeeze(1)
    active = optimal.any(dim=1)
    top1 = masked_mean(chosen_optimal.float(), active)

    suboptimal = valid & ~optimal
    paired = active & suboptimal.any(dim=1)
    best_optimal = (
        output.policy_logits.masked_fill(~optimal, negative).max(dim=1).values
    )
    best_bad = output.policy_logits.masked_fill(~suboptimal, negative).max(dim=1).values
    margin = masked_mean(best_optimal - best_bad, paired)
    return {"local_top1": top1, "local_margin": margin}


__all__ = [
    "PlannerLossWeights",
    "RepresentationLossWeights",
    "covariance_loss",
    "local_policy_metrics",
    "map_decoder_loss",
    "planner_loss",
    "variance_floor_loss",
]
