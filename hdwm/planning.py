"""CEM MPC planning and evaluation for HDWM world models.

Reference implementations:
- stable-worldmodel solver/categorical_cem.py (discrete CEM with Gumbel-max)
- procgen_maze evaluate_planning_v5.py (categorical distribution update CEM)
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def _bfs_shortest_path(
    grid: np.ndarray, start: int, goal: int, width: int
) -> int | None:
    """BFS shortest path length on a 2D grid mask (True = wall/blocked).

    Returns the number of steps, or None if unreachable.
    """
    height = grid.size // width
    walkable = ~grid.reshape(-1)
    sy, sx = divmod(start, width)
    gy, gx = divmod(goal, width)
    if not walkable[start] or not walkable[goal]:
        return None

    visited = np.zeros(height * width, dtype=bool)
    q = deque()
    q.append((start, 0))
    visited[start] = True
    while q:
        state, dist = q.popleft()
        if state == goal:
            return dist
        y, x = divmod(state, width)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                nstate = ny * width + nx
                if not visited[nstate] and walkable[nstate]:
                    visited[nstate] = True
                    q.append((nstate, dist + 1))
    return None


def _sample_start_goal(
    rng: np.random.Generator,
    empty_mask: np.ndarray,
    min_path_length: int = 3,
    max_attempts: int = 500,
) -> tuple[int, int, int]:
    """Sample a reachable (start, goal) pair with at least min_path_length steps.

    Returns (start_state, goal_state, bfs_distance).
    """
    empty_positions = np.flatnonzero(empty_mask.reshape(-1))
    if empty_positions.size < 2:
        raise ValueError("environment must contain at least two empty cells")

    grid_width = empty_mask.shape[1] if empty_mask.ndim == 2 else empty_mask.shape[0]
    # Use a boolean obstacle mask for BFS.
    obstacle_mask = ~empty_mask

    for _ in range(max_attempts):
        start = int(rng.choice(empty_positions))
        goal = int(rng.choice(empty_positions))
        if start == goal:
            continue
        dist = _bfs_shortest_path(obstacle_mask, start, goal, grid_width)
        if dist is not None and dist >= min_path_length:
            return start, goal, dist
    raise RuntimeError(
        f"could not find reachable start-goal pair after {max_attempts} attempts"
    )


def _encode_observation(
    model, env, observation: np.ndarray, device: torch.device, maze_size: int | None = None
) -> torch.Tensor:
    """Encode a single observation frame into the latent embedding space.

    Returns a tensor of shape [1, 1, D].
    """
    # observation shape: [H, W, C] -> add batch and time dims -> [1, 1, H, W, C]
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
    obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, C]
    with torch.no_grad():
        # Some encoders (e.g., SizeConditionedEncoder) require a size argument
        import inspect
        sig = inspect.signature(model.encoder.forward)
        if len(sig.parameters) > 1 and 'size' in sig.parameters:
            sz = maze_size if maze_size is not None else env.config.width
            encoded = model.encoder(obs_tensor, sz)
        else:
            encoded = model.encoder(obs_tensor)
        embedding, _ = model.embedding_projector(encoded)  # [1, 1, D]
    return embedding


def _latent_rollout_cost(
    model,
    context_emb: torch.Tensor,
    context_act: torch.Tensor,
    goal_emb: torch.Tensor,
    candidate_actions: np.ndarray,
    history_size: int,
    device: torch.device,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> np.ndarray:
    """Compute cost for each candidate action sequence by rolling out the predictor.

    Args:
        context_emb: [1, history_size, D]  current context embeddings
        context_act: [1, history_size]      current context action indices
        goal_emb:    [1, 1, D]              goal embedding
        candidate_actions: [num_candidates, horizon]  action indices (0-4)
        history_size: number of past frames used by the predictor
        device: torch device

    Returns:
        costs: [num_candidates]  MSE cost per candidate
    """
    num_candidates, horizon = candidate_actions.shape

    # Expand context across the candidate dimension.
    # emb: [S, H, D], act: [S, H-1]  (predictor expects actions of shape [B, T-1])
    emb = context_emb.expand(num_candidates, -1, -1).contiguous()  # [S, H, D]
    act = (
        context_act[:, : history_size - 1].expand(num_candidates, -1).contiguous()
    )  # [S, H-1]
    cand_tensor = torch.as_tensor(candidate_actions, dtype=torch.long, device=device)

    with torch.no_grad():
        # Rollout: first iteration warmup (predicts from real context),
        # then horizon-1 forward steps consume candidate actions.
        # Evaluates cost after horizon-1 steps to reduce predictor error accumulation.
        for t in range(horizon):
            cur_action = cand_tensor[:, t:t + 1]  # [S, 1]
            prediction = model.predictor(emb, act)  # [S, H-1, D]
            next_emb = prediction[:, -1:]  # [S, 1, D]
            emb = torch.cat([emb[:, 1:], next_emb], dim=1)  # [S, H, D]
            act = torch.cat([act[:, 1:], cur_action], dim=1)  # [S, H-1]

    terminal = emb[:, -1]  # [S, D] — latent after horizon-1 forward steps
    goal = goal_emb.squeeze(1).expand_as(terminal)  # [S, D]
    if score_fn is not None:
        return score_fn(terminal, goal).detach().cpu().numpy()
    return F.mse_loss(terminal, goal, reduction="none").sum(dim=-1).detach().cpu().numpy()


def cem_plan(
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
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    allowed_actions: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Categorical CEM planner for discrete action spaces.

    Fits a per-timestep categorical distribution over actions by iteratively
    sampling candidates, rolling out the world model predictor, and refitting
    from elite (lowest-cost) trajectories.

    Args:
        model: HDWM LEWM/LEWMCNN model with encoder, embedding_projector, predictor
        context_emb:  [1, history_size, D]  past observation embeddings
        context_act:  [1, history_size]      past action indices
        goal_emb:     [1, 1, D]              goal observation embedding
        horizon:      planning horizon (number of actions)
        history_size: predictor context window
        num_candidates: samples per CEM iteration
        num_elites:     elites selected per iteration
        cem_iters:      number of CEM iterations
        momentum:       EMA coefficient for distribution update
        num_actions:    size of discrete action space (default 5)
        device:         torch device
        seed:           random seed for reproducibility

    Returns:
        best_sequence: [horizon] best action sequence found
        best_cost:     scalar best cost achieved
        cost_history:  [cem_iters] best cost per iteration
    """
    if device is None:
        device = context_emb.device

    rng = np.random.default_rng(seed)
    action_values = (
        np.arange(num_actions, dtype=np.int64)
        if allowed_actions is None
        else np.asarray(allowed_actions, dtype=np.int64)
    )
    if action_values.ndim != 1 or action_values.size == 0:
        raise ValueError("allowed_actions must contain at least one action")
    if ((action_values < 0) | (action_values >= num_actions)).any():
        raise ValueError("allowed_actions contains an action outside num_actions")

    # Uniform initial distribution over allowed action ids.
    probs = np.full(
        (horizon, action_values.size),
        1.0 / action_values.size,
        dtype=np.float64,
    )
    best_seq: np.ndarray | None = None
    best_cost: float = float("inf")
    cost_history: list[float] = []

    for _ in range(cem_iters):
        # Sample candidates from the categorical distribution.
        candidates = np.stack(
            [
                rng.choice(action_values, size=num_candidates, p=probs[t])
                for t in range(horizon)
            ],
            axis=1,
        ).astype(np.int64)

        costs = _latent_rollout_cost(
            model,
            context_emb,
            context_act,
            goal_emb,
            candidates,
            history_size,
            device,
            score_fn=score_fn,
        )

        # Select elites.
        elite_indices = np.argsort(costs)[:num_elites]
        elites = candidates[elite_indices]

        # Track best.
        if costs[elite_indices[0]] < best_cost:
            best_cost = float(costs[elite_indices[0]])
            best_seq = elites[0].copy()
        cost_history.append(best_cost)

        # Update distribution from elite frequencies with momentum.
        new_probs = np.zeros_like(probs)
        for t in range(horizon):
            for action_idx, action in enumerate(action_values):
                new_probs[t, action_idx] = np.mean(elites[:, t] == action)
        probs = momentum * probs + (1.0 - momentum) * new_probs

    if best_seq is None:
        raise RuntimeError("CEM planning produced no valid sequence")
    return best_seq, best_cost, cost_history


def _run_mpc_episode(
    model,
    env,
    start_state,
    goal_state,
    goal_emb,
    device,
    history_size,
    horizon,
    max_steps,
    receding_horizon,
    cem_candidates,
    cem_elites,
    cem_iters,
    cem_momentum,
    num_actions,
    ep_seed,
    method="cem",
):
    """Run one MPC episode with CEM or random actions. Returns trajectory data."""
    width = env.config.width  # noqa: F841

    env.reset(seed=ep_seed, options={"start_state": start_state})
    start_obs = env._last_observation
    start_emb = _encode_observation(model, env, start_obs, device)

    context_emb = start_emb.repeat(1, history_size, 1)
    context_act = torch.full(
        (1, history_size), 0, dtype=torch.long, device=device  # STAY padding
    )

    env.reset(seed=ep_seed, options={"start_state": start_state})
    current_state = start_state
    path = [current_state]
    # Track latent distance to goal at each step (potential function).
    potential_distances: list[float] = []
    success = False
    rng = np.random.default_rng(ep_seed)

    # Initial potential
    with torch.no_grad():
        initial_dist = float(
            torch.nn.functional.mse_loss(
                start_emb.squeeze(1), goal_emb.squeeze(1), reduction="none"
            )
            .sum(dim=-1)
            .item()
        )
    potential_distances.append(initial_dist)

    for step in range(max_steps):
        if current_state == goal_state:
            success = True
            break

        if method == "cem":
            plan_seed = ep_seed * 10000 + step
            best_seq, _, _ = cem_plan(
                model,
                context_emb,
                context_act,
                goal_emb,
                horizon=horizon,
                history_size=history_size,
                num_candidates=cem_candidates,
                num_elites=cem_elites,
                cem_iters=cem_iters,
                momentum=cem_momentum,
                num_actions=num_actions,
                device=device,
                seed=plan_seed,
            )
            actions_to_exec = [
                int(best_seq[k]) for k in range(min(receding_horizon, horizon))
            ]
        else:
            actions_to_exec = [
                int(rng.integers(0, num_actions)) for _ in range(receding_horizon)
            ]

        for action in actions_to_exec:
            obs, _, _, _, info = env.step(action)
            current_state = int(info["state"])
            path.append(current_state)

            new_emb = _encode_observation(model, env, obs, device)
            context_emb = torch.cat([context_emb[:, 1:], new_emb], dim=1)
            context_act = torch.cat(
                [
                    context_act[:, 1:],
                    torch.tensor([[action]], dtype=torch.long, device=device),
                ],
                dim=1,
            )

            # Compute potential: latent distance to goal
            with torch.no_grad():
                dist = float(
                    torch.nn.functional.mse_loss(
                        new_emb.squeeze(1), goal_emb.squeeze(1), reduction="none"
                    )
                    .sum(dim=-1)
                    .item()
                )
            potential_distances.append(dist)

            if current_state == goal_state:
                success = True
                break
        if success:
            break

    path_length = len(path) - 1
    return {
        "success": success,
        "path_length": path_length,
        "path": path,
        "potential_distances": potential_distances,
    }


def evaluate_mpc(
    model,
    env,
    num_episodes: int = 50,
    max_steps: int | None = None,
    horizon: int | None = None,
    history_size: int = 3,
    receding_horizon: int = 2,
    cem_candidates: int = 256,
    cem_elites: int = 32,
    cem_iters: int = 5,
    cem_momentum: float = 0.1,
    min_path_length: int = 3,
    device: torch.device | None = None,
    seed: int = 0,
    verbose: bool = False,
    random_baseline: bool = True,
) -> dict:
    """Evaluate a world model with CEM MPC and optional Random baseline.

    Returns CEM metrics, Random baseline metrics, and per-episode potential
    function data (latent distance to goal over MPC steps).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    rng = np.random.Generator(np.random.PCG64(seed))
    num_actions = env.config.action_vocab_size

    cem_results: list[dict] = []
    rnd_results: list[dict] = []

    for episode in range(num_episodes):
        ep_seed = int(rng.integers(0, 2**31))

        if hasattr(env, "_maze_mask"):
            obstacle_mask = env._maze_mask
        elif hasattr(env, "_obstacle_mask"):
            obstacle_mask = env._obstacle_mask
        else:
            obstacle_mask = np.zeros((env.config.height, env.config.width), dtype=bool)

        empty_mask = ~obstacle_mask
        if hasattr(env, "_goal_position"):
            flat_empty = empty_mask.reshape(-1)
            flat_empty[env._goal_position] = False

        try:
            start_state, goal_state, optimal_dist = _sample_start_goal(
                rng, empty_mask, min_path_length=min_path_length
            )
        except RuntimeError:
            continue

        if horizon is None:
            horizon = min(optimal_dist * 3, 32)
        if max_steps is None:
            max_steps = min(optimal_dist * 5, 128)

        # Encode goal once.
        env.reset(seed=ep_seed, options={"start_state": goal_state})
        goal_obs = env._last_observation
        goal_emb = _encode_observation(model, env, goal_obs, device)

        # CEM run.
        cem_ep = _run_mpc_episode(
            model,
            env,
            start_state,
            goal_state,
            goal_emb,
            device,
            history_size,
            horizon,
            max_steps,
            receding_horizon,
            cem_candidates,
            cem_elites,
            cem_iters,
            cem_momentum,
            num_actions,
            ep_seed,
            method="cem",
        )
        cem_ep["episode"] = episode
        cem_ep["optimal_length"] = optimal_dist
        cem_ep["path_ratio"] = (
            cem_ep["path_length"] / optimal_dist if optimal_dist > 0 else float("inf")
        )
        cem_ep["start_state"] = start_state
        cem_ep["goal_state"] = goal_state
        cem_results.append(cem_ep)

        # Random baseline run.
        if random_baseline:
            rnd_ep = _run_mpc_episode(
                model,
                env,
                start_state,
                goal_state,
                goal_emb,
                device,
                history_size,
                horizon,
                max_steps,
                receding_horizon,
                cem_candidates,
                cem_elites,
                cem_iters,
                cem_momentum,
                num_actions,
                ep_seed + 1,
                method="random",
            )
            rnd_ep["episode"] = episode
            rnd_ep["optimal_length"] = optimal_dist
            rnd_ep["path_ratio"] = (
                rnd_ep["path_length"] / optimal_dist
                if optimal_dist > 0
                else float("inf")
            )
            rnd_ep["start_state"] = start_state
            rnd_ep["goal_state"] = goal_state
            rnd_results.append(rnd_ep)

        if verbose and episode % max(1, num_episodes // 5) == 0:
            cem_status = "✓" if cem_ep["success"] else "✗"
            if random_baseline:
                rnd_status = "✓" if rnd_ep["success"] else "✗"
                msg = (
                    f"  Ep {episode}: CEM={cem_status} "
                    f"({cem_ep['path_length']} steps)  "
                    f"RND={rnd_status} ({rnd_ep['path_length']} steps)"
                )
            else:
                msg = f"  Ep {episode}: CEM={cem_status}"
            print(msg)

    def _agg(eps):
        n = len(eps)
        if n == 0:
            return {
                "success_rate": 0.0,
                "avg_path_ratio": 0.0,
                "avg_path_length": 0.0,
                "avg_optimal_length": 0.0,
                "num_episodes": 0,
                "per_episode": [],
            }
        return {
            "success_rate": sum(e["success"] for e in eps) / n,
            "avg_path_length": float(np.mean([e["path_length"] for e in eps])),
            "avg_optimal_length": float(np.mean([e["optimal_length"] for e in eps])),
            "avg_path_ratio": float(np.mean([e["path_ratio"] for e in eps])),
            "num_episodes": n,
            "per_episode": eps,
        }

    result = {"cem": _agg(cem_results)}
    if random_baseline:
        result["random"] = _agg(rnd_results)

    # Compat: flat keys for backward compatibility
    result["success_rate"] = result["cem"]["success_rate"]
    result["avg_path_length"] = result["cem"]["avg_path_length"]
    result["avg_optimal_length"] = result["cem"]["avg_optimal_length"]
    result["avg_path_ratio"] = result["cem"]["avg_path_ratio"]
    result["num_episodes"] = result["cem"]["num_episodes"]
    return result
