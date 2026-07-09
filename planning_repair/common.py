#!/usr/bin/env python3
"""Shared helpers for the planning-repair experiment suite."""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.common import (  # noqa: E402
    ACTION_IDS,
    ACTION_TO_SLOT,
    SLOT_TO_ACTION,
    bfs_distances_from,
    create_env,
    next_state,
    observe_state,
    read_jsonl,
    set_agent_state,
    size_bucket,
    verify_holdout,
    write_json,
)
from hdwm.config import LEWMCNNConfig, ProcgenMazeConfig  # noqa: E402
from scripts.train.train_dim256 import Unisize256  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_default_model_config(
    *,
    latent_dim: int = 256,
    cnn_channels: tuple[int, ...] = (64, 128, 256),
    predictor_heads: int = 16,
) -> LEWMCNNConfig:
    base_env = ProcgenMazeConfig(
        height=25,
        width=25,
        observation_channels=5,
        p_noise=0.0,
        p_noop=0.0,
        p_action_turn=0.0,
        p_action_stay=0.0,
        resample_maze_per_sequence=False,
    )
    return LEWMCNNConfig(
        env_config=base_env,
        latent_dim=latent_dim,
        cnn_channels=cnn_channels,
        latent_batch_norm=True,
        embedding_stage="post_bn",
        sigreg_stage="post_bn",
        predictor_heads=predictor_heads,
    )


def build_or_load_model(
    model_ckpt: str | Path | None,
    device: torch.device,
    *,
    latent_dim: int = 256,
    max_size: int = 31,
) -> tuple[Unisize256, Any, dict[str, Any]]:
    """Build a Unisize256 backbone and optionally initialize from checkpoint."""

    metadata: dict[str, Any] = {}
    if model_ckpt:
        ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
        cfg = ckpt["model_config"]
        model = Unisize256(cfg, max_size=max_size).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        metadata["init_model_ckpt"] = str(model_ckpt)
        return model, cfg, metadata

    cfg = make_default_model_config(latent_dim=latent_dim)
    model = Unisize256(cfg, max_size=max_size).to(device)
    metadata["init_model_ckpt"] = None
    return model, cfg, metadata


def load_backbone_from_repair_ckpt(
    checkpoint: str | Path,
    device: torch.device,
    *,
    max_size: int = 31,
) -> tuple[Unisize256, dict[str, Any]]:
    """Load the backbone portion of a planning-repair checkpoint."""

    data = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Unisize256(data["model_config"], max_size=max_size).to(device)
    model.load_state_dict(data["model_state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, data


def load_manifest_pair(
    train_manifest: str | Path,
    eval_manifest: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    train_entries = read_jsonl(train_manifest)
    eval_entries = read_jsonl(eval_manifest)
    overlap = verify_holdout(train_entries, eval_entries)
    return train_entries, eval_entries, overlap


def grouped_limit(
    entries: list[dict[str, Any]],
    *,
    max_per_size: int = 0,
    limit: int = 0,
) -> list[dict[str, Any]]:
    if max_per_size > 0:
        counts: dict[int, int] = defaultdict(int)
        filtered: list[dict[str, Any]] = []
        for entry in entries:
            size = int(entry["maze_size"])
            if counts[size] >= max_per_size:
                continue
            counts[size] += 1
            filtered.append(entry)
        entries = filtered
    if limit > 0:
        entries = entries[:limit]
    return entries


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("expected at least one integer")
    return items


def json_dump(path: str | Path, data: Any) -> None:
    write_json(path, data)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def compute_maze_supervision(
    *,
    states: torch.Tensor,
    env: Any,
    size: int,
    device: torch.device,
    budgets: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Compute planning-relevant labels for a sampled sequence batch.

    Labels are derived from the oracle maze topology but are used only as
    auxiliary supervision or diagnostics. The backbone input is still the normal
    pixel/one-hot observation.
    """

    if states.ndim != 2:
        raise ValueError(f"expected states [B,T], got {tuple(states.shape)}")
    states_np = states.detach().cpu().numpy().astype(np.int64)
    batch_size, seq_len = states_np.shape
    flat_states = states_np.reshape(-1)
    goal = int(env._goal_position)
    width = int(env.config.width)
    goal_dists = bfs_distances_from(env._maze_mask, goal, width)
    reachable = goal_dists[goal_dists >= 0]
    max_dist = float(reachable.max()) if reachable.size else 1.0
    max_dist = max(max_dist, 1.0)

    agent_xy = np.zeros((len(flat_states), 2), dtype=np.float32)
    goal_xy = np.zeros((len(flat_states), 2), dtype=np.float32)
    valid_action = np.zeros((len(flat_states), len(ACTION_IDS)), dtype=np.float32)
    optimal_action_mask = np.zeros_like(valid_action)
    optimal_action = np.full((len(flat_states),), -100, dtype=np.int64)
    bfs_distance = np.zeros((len(flat_states),), dtype=np.float32)
    bfs_distance_norm = np.zeros((len(flat_states),), dtype=np.float32)
    reachability = np.zeros((len(flat_states), len(budgets)), dtype=np.float32)

    goal_x = float(goal % size) / max(size - 1, 1)
    goal_y = float(goal // size) / max(size - 1, 1)

    for i, state in enumerate(flat_states.tolist()):
        state = int(state)
        agent_xy[i, 0] = float(state % size) / max(size - 1, 1)
        agent_xy[i, 1] = float(state // size) / max(size - 1, 1)
        goal_xy[i, 0] = goal_x
        goal_xy[i, 1] = goal_y

        cur_dist = int(goal_dists[state])
        bfs_distance[i] = float(cur_dist)
        bfs_distance_norm[i] = float(cur_dist) / max_dist if cur_dist >= 0 else 1.0
        for budget_idx, budget in enumerate(budgets):
            reachability[i, budget_idx] = float(cur_dist >= 0 and cur_dist <= budget)

        candidate_dists: list[float] = []
        for action in ACTION_IDS:
            slot = ACTION_TO_SLOT[int(action)]
            nxt = next_state(env, state, int(action))
            valid_action[i, slot] = float(nxt != state)
            if nxt == state:
                candidate_dists.append(float("inf"))
            else:
                dist = int(goal_dists[nxt])
                candidate_dists.append(float(dist) if dist >= 0 else float("inf"))

        if state == goal or cur_dist <= 0:
            continue
        best = min(candidate_dists)
        if not np.isfinite(best):
            continue
        for slot, dist in enumerate(candidate_dists):
            if np.isclose(dist, best):
                optimal_action_mask[i, slot] = 1.0
        optimal_slots = np.flatnonzero(optimal_action_mask[i] > 0.0)
        if len(optimal_slots):
            optimal_action[i] = int(optimal_slots[0])

    def shaped(array: np.ndarray, *tail: int) -> torch.Tensor:
        return torch.as_tensor(
            array.reshape(batch_size, seq_len, *tail),
            dtype=torch.float32,
            device=device,
        )

    return {
        "agent_xy": shaped(agent_xy, 2),
        "goal_xy": shaped(goal_xy, 2),
        "valid_action": shaped(valid_action, len(ACTION_IDS)),
        "optimal_action_mask": shaped(optimal_action_mask, len(ACTION_IDS)),
        "optimal_action": torch.as_tensor(
            optimal_action.reshape(batch_size, seq_len),
            dtype=torch.long,
            device=device,
        ),
        "bfs_distance": torch.as_tensor(
            bfs_distance.reshape(batch_size, seq_len),
            dtype=torch.float32,
            device=device,
        ),
        "bfs_distance_norm": torch.as_tensor(
            bfs_distance_norm.reshape(batch_size, seq_len),
            dtype=torch.float32,
            device=device,
        ),
        "reachability": shaped(reachability, len(budgets)),
    }


def valid_moving_actions(
    env: Any,
    state: int,
    previous_state: int | None = None,
) -> list[int]:
    actions: list[int] = []
    non_backtracking: list[int] = []
    for action in ACTION_IDS:
        nxt = next_state(env, state, int(action))
        if nxt != state:
            actions.append(int(action))
            if previous_state is None or nxt != previous_state:
                non_backtracking.append(int(action))
    return non_backtracking or actions


def summarize_navigation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "sr": 0.0, "spl": 0.0}
    successes = [row for row in rows if row["success"]]
    failures = [row for row in rows if not row["success"]]
    total_steps = max(sum(int(row["path_length"]) for row in rows), 1)
    return {
        "n": len(rows),
        "sr": float(len(successes) / len(rows)),
        "spl": float(np.mean([row["spl"] for row in rows])),
        "avg_path_success": float(np.mean([row["path_length"] for row in successes]))
        if successes
        else 0.0,
        "avg_final_bfs": float(
            np.mean([row["final_bfs_distance"] for row in failures])
        )
        if failures
        else 0.0,
        "stuck_rate": float(sum(row["stuck_steps"] for row in rows) / total_steps),
        "invalid_rate": float(sum(row["invalid_actions"] for row in rows) / total_steps),
        "num_success": len(successes),
        "num_failure": len(failures),
    }


def encode_single_observation(
    model: Unisize256,
    obs: np.ndarray,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encoder(obs_t, size)
        embedding, _ = model.embedding_projector(encoded)
    return embedding


__all__ = [
    "ACTION_IDS",
    "ACTION_TO_SLOT",
    "SLOT_TO_ACTION",
    "build_or_load_model",
    "compute_maze_supervision",
    "create_env",
    "encode_single_observation",
    "grouped_limit",
    "json_dump",
    "load_backbone_from_repair_ckpt",
    "load_manifest_pair",
    "observe_state",
    "parse_int_list",
    "read_json",
    "set_agent_state",
    "set_seed",
    "size_bucket",
    "summarize_navigation",
    "valid_moving_actions",
]

