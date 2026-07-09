"""Enhanced CEM planners with action validity masking and diversity bonus.

Extends hdwm.planning with:
  1. cem_plan_masked:     CEM with action validity masking
  2. cem_plan_diverse:    CEM with entropy diversity bonus
  3. cem_plan_enhanced:   CEM with both masking + diversity

All functions share the same signature as cem_plan() for drop-in replacement.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from hdwm.planning import _latent_rollout_cost


def cem_plan_masked(
    model,
    context_emb: torch.Tensor,
    context_act: torch.Tensor,
    goal_emb: torch.Tensor,
    horizon: int,
    history_size: int = 3,
    num_candidates: int = 256,
    num_elites: int = 32,
    cem_iters: int = 5,
    momentum: float = 0.1,
    num_actions: int = 5,
    device: torch.device | None = None,
    seed: int | None = None,
    score_fn: Callable | None = None,
    validity_head=None,
    validity_threshold: float = 0.5,
    min_valid_prob: float = 0.01,
) -> tuple[np.ndarray, float, list[float]]:
    """CEM planner with action validity masking.

    At each CEM iteration, predicts which actions are valid from the current latent
    and adjusts the categorical distribution to suppress invalid actions.

    Args:
        validity_head: ValidityHead module with predict_proba(z) -> [1, 5].
        validity_threshold: probability below which actions are suppressed.
        min_valid_prob: minimum probability for suppressed actions (keeps exploration).
        (All other args same as cem_plan)
    """
    if device is None:
        device = context_emb.device

    rng = np.random.default_rng(seed)
    probs = np.full((horizon, num_actions), 1.0 / num_actions, dtype=np.float64)
    best_seq: np.ndarray | None = None
    best_cost: float = float("inf")
    cost_history: list[float] = []

    # Get validity from current context (last embedding in context)
    current_z = context_emb[:, -1]  # [1, D]

    for iter_idx in range(cem_iters):
        # Apply validity masking to the distribution
        if validity_head is not None:
            with torch.no_grad():
                valid_probs = validity_head.predict_proba(current_z).cpu().numpy()  # [1, 5]
            # Blend validity into the distribution
            masked = probs.copy()
            for t in range(horizon):
                # Suppress invalid actions
                adjusted = masked[t] * valid_probs[0]
                adjusted = np.clip(adjusted, min_valid_prob / num_actions, None)
                adjusted = adjusted / adjusted.sum()
                masked[t] = adjusted
            sampling_probs = masked
        else:
            sampling_probs = probs

        # Sample candidates
        candidates = np.stack([
            rng.choice(num_actions, size=num_candidates, p=sampling_probs[t])
            for t in range(horizon)
        ], axis=1).astype(np.int64)

        costs = _latent_rollout_cost(
            model, context_emb, context_act, goal_emb, candidates,
            history_size, device, score_fn=score_fn,
        )

        # Select elites
        elite_indices = np.argsort(costs)[:num_elites]
        elites = candidates[elite_indices]

        # Track best
        if costs[elite_indices[0]] < best_cost:
            best_cost = float(costs[elite_indices[0]])
            best_seq = elites[0].copy()
        cost_history.append(best_cost)

        # Update distribution from elite frequencies
        new_probs = np.zeros_like(probs)
        for t in range(horizon):
            new_probs[t] = np.bincount(elites[:, t], minlength=num_actions) / num_elites
        probs = momentum * probs + (1.0 - momentum) * new_probs

    if best_seq is None:
        raise RuntimeError("CEM planning produced no valid sequence")
    return best_seq, best_cost, cost_history


def cem_plan_diverse(
    model,
    context_emb: torch.Tensor,
    context_act: torch.Tensor,
    goal_emb: torch.Tensor,
    horizon: int,
    history_size: int = 3,
    num_candidates: int = 256,
    num_elites: int = 32,
    cem_iters: int = 5,
    momentum: float = 0.1,
    num_actions: int = 5,
    device: torch.device | None = None,
    seed: int | None = None,
    score_fn: Callable | None = None,
    diversity_weight: float = 0.05,
    temperature: float = 1.5,
) -> tuple[np.ndarray, float, list[float]]:
    """CEM planner with diversity bonus via entropy regularization.

    Adds an entropy bonus to the CEM objective to prevent premature distribution collapse.
    Higher temperature = more exploration, diversity_weight controls the bonus strength.

    Args:
        diversity_weight: weight of the entropy bonus in elite selection.
        temperature: softens the distribution before sampling.
        (All other args same as cem_plan)
    """
    if device is None:
        device = context_emb.device

    rng = np.random.default_rng(seed)
    probs = np.full((horizon, num_actions), 1.0 / num_actions, dtype=np.float64)
    best_seq: np.ndarray | None = None
    best_cost: float = float("inf")
    cost_history: list[float] = []

    for _ in range(cem_iters):
        # Sample with temperature
        tempered = probs ** (1.0 / temperature)
        tempered = tempered / tempered.sum(axis=-1, keepdims=True)

        candidates = np.stack([
            rng.choice(num_actions, size=num_candidates, p=tempered[t])
            for t in range(horizon)
        ], axis=1).astype(np.int64)

        costs = _latent_rollout_cost(
            model, context_emb, context_act, goal_emb, candidates,
            history_size, device, score_fn=score_fn,
        )

        # Diversity bonus: compute per-candidate action diversity
        # Penalize candidates that use the same action too often
        action_counts = np.zeros((num_candidates, num_actions))
        for a in range(num_actions):
            action_counts[:, a] = (candidates == a).sum(axis=1)
        # Entropy of action usage per candidate
        action_freq = action_counts / horizon
        entropy = -np.sum(action_freq * np.log(action_freq + 1e-8), axis=-1)
        # Elite selection: low cost + high entropy = better
        combined_score = costs - diversity_weight * entropy * np.abs(costs.mean())

        elite_indices = np.argsort(combined_score)[:num_elites]
        elites = candidates[elite_indices]

        if costs[elite_indices[0]] < best_cost:
            best_cost = float(costs[elite_indices[0]])
            best_seq = elites[0].copy()
        cost_history.append(best_cost)

        # Update distribution
        new_probs = np.zeros_like(probs)
        for t in range(horizon):
            new_probs[t] = np.bincount(elites[:, t], minlength=num_actions) / num_elites
        probs = momentum * probs + (1.0 - momentum) * new_probs

    if best_seq is None:
        raise RuntimeError("CEM planning produced no valid sequence")
    return best_seq, best_cost, cost_history


def cem_plan_enhanced(
    model,
    context_emb: torch.Tensor,
    context_act: torch.Tensor,
    goal_emb: torch.Tensor,
    horizon: int,
    history_size: int = 3,
    num_candidates: int = 256,
    num_elites: int = 32,
    cem_iters: int = 5,
    momentum: float = 0.1,
    num_actions: int = 5,
    device: torch.device | None = None,
    seed: int | None = None,
    score_fn: Callable | None = None,
    validity_head=None,
    diversity_weight: float = 0.05,
    temperature: float = 1.5,
) -> tuple[np.ndarray, float, list[float]]:
    """CEM with both validity masking AND diversity bonus.

    Combines the two enhancements for maximum effect.
    """
    if device is None:
        device = context_emb.device

    rng = np.random.default_rng(seed)
    probs = np.full((horizon, num_actions), 1.0 / num_actions, dtype=np.float64)
    best_seq: np.ndarray | None = None
    best_cost: float = float("inf")
    cost_history: list[float] = []

    current_z = context_emb[:, -1]

    for _ in range(cem_iters):
        # Apply validity masking
        if validity_head is not None:
            with torch.no_grad():
                valid_probs = validity_head.predict_proba(current_z).cpu().numpy()
            masked = probs.copy()
            for t in range(horizon):
                adjusted = masked[t] * valid_probs[0]
                adjusted = np.clip(adjusted, 0.002, None)
                adjusted = adjusted / adjusted.sum()
                masked[t] = adjusted
            sampling_probs = masked
        else:
            sampling_probs = probs

        # Apply temperature
        tempered = sampling_probs ** (1.0 / temperature)
        tempered = tempered / tempered.sum(axis=-1, keepdims=True)

        candidates = np.stack([
            rng.choice(num_actions, size=num_candidates, p=tempered[t])
            for t in range(horizon)
        ], axis=1).astype(np.int64)

        costs = _latent_rollout_cost(
            model, context_emb, context_act, goal_emb, candidates,
            history_size, device, score_fn=score_fn,
        )

        # Diversity bonus
        action_counts = np.zeros((num_candidates, num_actions))
        for a in range(num_actions):
            action_counts[:, a] = (candidates == a).sum(axis=1)
        action_freq = action_counts / horizon
        entropy = -np.sum(action_freq * np.log(action_freq + 1e-8), axis=-1)
        combined_score = costs - diversity_weight * entropy * np.abs(costs.mean())

        elite_indices = np.argsort(combined_score)[:num_elites]
        elites = candidates[elite_indices]

        if costs[elite_indices[0]] < best_cost:
            best_cost = float(costs[elite_indices[0]])
            best_seq = elites[0].copy()
        cost_history.append(best_cost)

        new_probs = np.zeros_like(probs)
        for t in range(horizon):
            new_probs[t] = np.bincount(elites[:, t], minlength=num_actions) / num_elites
        probs = momentum * probs + (1.0 - momentum) * new_probs

    if best_seq is None:
        raise RuntimeError("CEM planning produced no valid sequence")
    return best_seq, best_cost, cost_history
