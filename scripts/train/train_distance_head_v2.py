#!/usr/bin/env python3
"""Train DistanceHead-v2 for Set B with ranking and scale-aware targets.

The baseline DistanceHead learns pairwise BFS regression. This v2 trainer keeps
the frozen LeWM backbone but adds losses that directly target navigation:

- size-balanced maze sampling
- normalized/log distance regression
- local valid-action ranking loss
- optional triangle inequality regularization

The resulting checkpoint is still a standard DistanceHead, so existing greedy
evaluators can use it as a score function: lower score means closer to goal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.metric_heads.distance_head import DistanceHead
from hdwm.planning import _bfs_shortest_path
from scripts.train.train_dim256 import Unisize256


HISTORY_SIZE = 3


@dataclass
class TrainConfig:
    model_ckpt: str = "checkpoints/unisize_dim256.pt"
    train_manifest: str = "data/splits/unisize_train_manifest.jsonl"
    val_manifest: str = "data/splits/unisize_eval_manifest.jsonl"
    output: str = "checkpoints/metric_heads/distance_head_v2_setb.pt"
    latent_dim: int = 256
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    input_mode: str = "concat"
    target_mode: str = "log_norm"
    steps: int = 30000
    batch_size: int = 512
    pairs_per_maze: int = 64
    local_batch_size: int = 256
    action_batch_size: int = 256
    predictor_action_batch_size: int = 0
    triangle_batch_size: int = 128
    lr: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-5
    regression_weight: float = 1.0
    ranking_weight: float = 1.0
    action_ce_weight: float = 0.0
    predictor_action_ce_weight: float = 0.0
    triangle_weight: float = 0.05
    ranking_margin: float = 0.02
    eval_every: int = 1000
    eval_batches: int = 8
    eval_action_batches: int = 8
    eval_predictor_action_batches: int = 0
    unfreeze_backbone: bool = False
    unfreeze_scope: str = "none"
    mazes_per_step: int = 4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def create_env(entry: dict[str, Any]) -> ProcgenMazeEnv:
    size = int(entry["maze_size"])
    return ProcgenMazeEnv(
        ProcgenMazeConfig(
            height=size,
            width=size,
            observation_channels=5,
            p_noise=0.0,
            p_noop=0.0,
            p_action_turn=0.0,
            p_action_stay=0.0,
            resample_maze_per_sequence=False,
            topology_seed=int(entry["topology_seed"]),
        ),
        seed=int(entry.get("env_seed", 42)),
    )


def observe_state(env: ProcgenMazeEnv, state: int) -> np.ndarray:
    obs, _ = env._observe_with_noise(np.array([state]))
    return obs[0]


def encode_batch(
    model: Unisize256,
    observations: list[np.ndarray],
    maze_size: int,
    device: torch.device,
    grad: bool = False,
) -> torch.Tensor:
    obs = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device)
    obs = obs.unsqueeze(1)
    if grad:
        encoded = model.encoder(obs, maze_size)
        embedding, _ = model.embedding_projector(encoded)
        return embedding.squeeze(1)
    with torch.no_grad():
        encoded = model.encoder(obs, maze_size)
        embedding, _ = model.embedding_projector(encoded)
    return embedding.squeeze(1).detach().cpu()


def load_backbone(path: str, device: torch.device, freeze: bool = True) -> Unisize256:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Unisize256(ckpt["model_config"], max_size=31).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model


def all_pairs_bfs(mask: np.ndarray, cells: list[int], width: int) -> np.ndarray:
    n = len(cells)
    bfs = np.full((n, n), -1, dtype=np.int16)
    height = mask.shape[0]
    cell_to_idx = {cell: idx for idx, cell in enumerate(cells)}
    for src_idx, src in enumerate(cells):
        queue: deque[tuple[int, int]] = deque([(src, 0)])
        seen = {src}
        while queue:
            state, dist = queue.popleft()
            dst_idx = cell_to_idx.get(state)
            if dst_idx is not None:
                bfs[src_idx, dst_idx] = dist
            row, col = divmod(state, width)
            for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = row + drow
                nc = col + dcol
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                next_state = nr * width + nc
                if next_state in seen or mask[nr, nc]:
                    continue
                seen.add(next_state)
                queue.append((next_state, dist + 1))
    return bfs


class SetBMazeCache:
    """Lazy CPU cache for per-maze observations/latents and BFS distances.

    In frozen-backbone mode the cache stores precomputed latents on CPU.
    In unfrozen-backbone mode it stores raw observations and encodes them
    on-the-fly so gradients can reach the encoder and projector.
    """

    def __init__(
        self,
        model: Unisize256,
        entries: list[dict[str, Any]],
        device: torch.device,
        store_observations: bool = False,
    ) -> None:
        self.model = model
        self.entries = entries
        self.device = device
        self.store_observations = store_observations
        self.by_size: dict[int, list[int]] = defaultdict(list)
        for idx, entry in enumerate(entries):
            self.by_size[int(entry["maze_size"])].append(idx)
        self.sizes = sorted(self.by_size)
        self._cache: dict[int, dict[str, Any]] = {}
        self._build_count = 0

    def get(self, idx: int) -> dict[str, Any]:
        if idx not in self._cache:
            self._cache[idx] = self._build(idx)
        return self._cache[idx]

    def sample_maze_idx(self, rng: np.random.Generator) -> int:
        size = int(rng.choice(self.sizes))
        return int(rng.choice(self.by_size[size]))

    def _build(self, idx: int) -> dict[str, Any]:
        t0 = time.time()
        entry = self.entries[idx]
        size = int(entry["maze_size"])
        log_this = self._build_count < 20
        if log_this:
            print(
                f"  [cache] building maze idx={idx} size={size} topo={entry['topology_seed']} "
                f"store_obs={self.store_observations}",
                flush=True,
            )
        env = create_env(entry)
        cells = np.flatnonzero((~env._maze_mask).reshape(-1)).astype(np.int64).tolist()
        observations = [observe_state(env, cell) for cell in cells]
        if self.store_observations:
            latents = None
        else:
            latents = encode_batch(self.model, observations, size, self.device)
            observations = None
        bfs = all_pairs_bfs(env._maze_mask, cells, env.config.width)
        cell_to_idx = {cell: i for i, cell in enumerate(cells)}
        next_indices = np.full((len(cells), env.config.action_vocab_size), -1, dtype=np.int32)
        for i, cell in enumerate(cells):
            for action in range(env.config.action_vocab_size):
                next_cell = int(env._next_state(cell, env._decode_action(action)))
                next_indices[i, action] = cell_to_idx.get(next_cell, -1)
        max_dist = max(float(np.max(bfs)), 1.0)
        elapsed = time.time() - t0
        self._build_count += 1
        if log_this or elapsed > 5.0:
            print(
                f"  [cache] built maze idx={idx} size={size} cells={len(cells)} "
                f"elapsed={elapsed:.1f}s cache_size={len(self._cache) + 1}",
                flush=True,
            )
        return {
            "entry": entry,
            "latents": latents,
            "observations": observations,
            "cells": cells,
            "bfs": bfs,
            "next_indices": next_indices,
            "max_dist": max_dist,
        }

    def encode_maze(self, idx: int, model: Unisize256, grad: bool = False) -> torch.Tensor:
        """Return latents for *idx*, encoding observations on demand if needed."""
        maze = self.get(idx)
        if maze["latents"] is not None:
            return maze["latents"].to(self.device)
        observations = maze["observations"]
        if observations is None:
            raise RuntimeError("maze has neither latents nor observations cached")
        return encode_batch(model, observations, int(maze["entry"]["maze_size"]), self.device, grad=grad)


class StepMazeCache:
    """Small per-step differentiable cache for unfrozen-backbone training."""

    def __init__(
        self,
        dataset: SetBMazeCache,
        model: Unisize256,
        rng: np.random.Generator,
        n_mazes: int,
    ) -> None:
        if n_mazes <= 0:
            raise ValueError("n_mazes must be positive")
        self._indices = [dataset.sample_maze_idx(rng) for _ in range(n_mazes)]
        self._mazes = [dataset.get(idx) for idx in self._indices]
        self._latents = [dataset.encode_maze(idx, model, grad=True) for idx in self._indices]
        self.device = dataset.device

    def sample_maze_idx(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(self._indices)))

    def get(self, idx: int) -> dict[str, Any]:
        maze = dict(self._mazes[idx])
        maze["latents"] = self._latents[idx]
        return maze


def transform_distance(dist: torch.Tensor, max_dist: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return dist
    if mode == "norm":
        return dist / max_dist.clamp_min(1.0)
    if mode == "log":
        return torch.log1p(dist)
    if mode == "log_norm":
        return torch.log1p(dist) / torch.log1p(max_dist.clamp_min(1.0))
    raise ValueError(f"unknown target_mode: {mode}")


def sample_regression_batch(
    dataset: SetBMazeCache,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    model: Unisize256 | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z1_list: list[torch.Tensor] = []
    z2_list: list[torch.Tensor] = []
    dist_list: list[float] = []
    max_list: list[float] = []
    while len(z1_list) < cfg.batch_size:
        maze_idx = dataset.sample_maze_idx(rng)
        maze = dataset.get(maze_idx)
        latents = dataset.encode_maze(maze_idx, model, grad=True) if model is not None else maze["latents"]
        bfs = maze["bfs"]
        n = latents.shape[0]
        for _ in range(cfg.pairs_per_maze):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            dist = float(bfs[i, j])
            if dist < 0:
                continue
            z1_list.append(latents[i])
            z2_list.append(latents[j])
            dist_list.append(dist)
            max_list.append(float(maze["max_dist"]))
            if len(z1_list) >= cfg.batch_size:
                break
    z1 = torch.stack(z1_list).to(device)
    z2 = torch.stack(z2_list).to(device)
    dist = torch.tensor(dist_list, dtype=torch.float32, device=device)
    max_dist = torch.tensor(max_list, dtype=torch.float32, device=device)
    target = transform_distance(dist, max_dist, cfg.target_mode)
    return z1, z2, target


def sample_local_ranking_batch(
    dataset: SetBMazeCache,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    model: Unisize256 | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    good_list: list[torch.Tensor] = []
    bad_list: list[torch.Tensor] = []
    goal_list: list[torch.Tensor] = []
    attempts = 0
    while len(good_list) < cfg.local_batch_size and attempts < cfg.local_batch_size * 64:
        attempts += 1
        maze_idx = dataset.sample_maze_idx(rng)
        maze = dataset.get(maze_idx)
        latents = dataset.encode_maze(maze_idx, model, grad=True) if model is not None else maze["latents"]
        bfs = maze["bfs"]
        next_indices = maze["next_indices"]
        n = latents.shape[0]
        state = int(rng.integers(0, n))
        goal = int(rng.integers(0, n))
        cur_dist = int(bfs[state, goal])
        if state == goal or cur_dist <= 0:
            continue
        candidates = [idx for idx in next_indices[state, 1:].tolist() if idx >= 0 and idx != state]
        if len(candidates) < 2:
            continue
        improving = [idx for idx in candidates if 0 <= bfs[idx, goal] < cur_dist]
        worse = [idx for idx in candidates if bfs[idx, goal] >= cur_dist]
        if not improving or not worse:
            continue
        good_idx = int(rng.choice(improving))
        bad_idx = int(rng.choice(worse))
        good_list.append(latents[good_idx])
        bad_list.append(latents[bad_idx])
        goal_list.append(latents[goal])
    if not good_list:
        raise RuntimeError("failed to sample local ranking batch")
    return (
        torch.stack(good_list).to(device),
        torch.stack(bad_list).to(device),
        torch.stack(goal_list).to(device),
    )


def valid_action_candidates(maze: dict[str, Any], state: int) -> list[int]:
    """Return next-state indices for moving actions from *state*."""
    next_indices = maze["next_indices"]
    candidates: list[int] = []
    for action in range(1, next_indices.shape[1]):
        next_idx = int(next_indices[state, action])
        if next_idx >= 0 and next_idx != state:
            candidates.append(next_idx)
    return candidates


def valid_action_transitions(maze: dict[str, Any], state: int) -> list[tuple[int, int]]:
    """Return (action, next_state_idx) for moving actions from *state*."""
    next_indices = maze["next_indices"]
    candidates: list[tuple[int, int]] = []
    for action in range(1, next_indices.shape[1]):
        next_idx = int(next_indices[state, action])
        if next_idx >= 0 and next_idx != state:
            candidates.append((action, next_idx))
    return candidates


def sample_action_order_batch(
    dataset: SetBMazeCache,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    model: Unisize256 | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample listwise local action-order examples.

    Returns:
      next_latents: [B, 4, D]
      goal_latents: [B, D]
      mask: [B, 4], true for valid candidate slots
      target_probs: [B, 4], uniform over all true-BFS-optimal actions
    """
    next_rows: list[torch.Tensor] = []
    goal_rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    target_probs: list[torch.Tensor] = []
    max_actions = 4
    attempts = 0
    max_attempts = cfg.action_batch_size * 128

    while len(target_probs) < cfg.action_batch_size and attempts < max_attempts:
        attempts += 1
        maze_idx = dataset.sample_maze_idx(rng)
        maze = dataset.get(maze_idx)
        latents = dataset.encode_maze(maze_idx, model, grad=True) if model is not None else maze["latents"]
        bfs = maze["bfs"]
        n = latents.shape[0]
        state = int(rng.integers(0, n))
        goal = int(rng.integers(0, n))
        if state == goal or bfs[state, goal] <= 0:
            continue
        candidates = valid_action_candidates(maze, state)
        if len(candidates) < 2:
            continue
        distances = np.asarray([bfs[next_idx, goal] for next_idx in candidates], dtype=np.float32)
        if (distances < 0).any():
            continue
        best_distance = float(np.min(distances))
        optimal = np.isclose(distances, best_distance)

        row = torch.zeros((max_actions, latents.shape[-1]), dtype=latents.dtype, device=latents.device)
        mask = torch.zeros(max_actions, dtype=torch.bool, device=latents.device)
        probs = torch.zeros(max_actions, dtype=torch.float32, device=latents.device)
        for slot, next_idx in enumerate(candidates[:max_actions]):
            row[slot] = latents[next_idx]
            mask[slot] = True
            if bool(optimal[slot]):
                probs[slot] = 1.0
        if probs.sum() <= 0:
            continue
        probs = probs / probs.sum()
        next_rows.append(row)
        goal_rows.append(latents[goal])
        masks.append(mask)
        target_probs.append(probs)

    if not target_probs:
        raise RuntimeError("failed to sample action order batch")
    return (
        torch.stack(next_rows).to(device),
        torch.stack(goal_rows).to(device),
        torch.stack(masks).to(device),
        torch.stack(target_probs).to(device),
    )


def action_order_loss(
    head: DistanceHead,
    next_latents: torch.Tensor,
    goal_latents: torch.Tensor,
    mask: torch.Tensor,
    target_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, max_actions, dim = next_latents.shape
    goal_rep = goal_latents[:, None, :].expand(batch_size, max_actions, dim)
    scores = head(next_latents.reshape(batch_size * max_actions, dim), goal_rep.reshape(batch_size * max_actions, dim))
    scores = scores.reshape(batch_size, max_actions)
    logits = (-scores).masked_fill(~mask, -1e9)
    loss = -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    pred = logits.argmax(dim=1)
    top1 = target_probs.gather(1, pred[:, None]).gt(0).float().mean()
    return loss, top1


def sample_predictor_action_order_batch(
    dataset: SetBMazeCache,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    model: Unisize256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample local action-order examples on predictor-generated next latents.

    This directly matches predictor_greedy: candidate actions are converted into
    predicted next embeddings by model.predictor, while labels still come from
    true BFS distances in the real maze.
    """
    next_rows: list[torch.Tensor] = []
    goal_rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    target_probs: list[torch.Tensor] = []
    max_actions = 4
    attempts = 0
    max_attempts = max(cfg.predictor_action_batch_size, 1) * 128

    while len(target_probs) < cfg.predictor_action_batch_size and attempts < max_attempts:
        attempts += 1
        maze_idx = dataset.sample_maze_idx(rng)
        maze = dataset.get(maze_idx)
        latents = maze["latents"]
        if latents is None:
            raise ValueError("predictor action training currently requires frozen cached latents")
        latents = latents.to(device)
        bfs = maze["bfs"]
        n = latents.shape[0]
        state = int(rng.integers(0, n))
        goal = int(rng.integers(0, n))
        if state == goal or bfs[state, goal] <= 0:
            continue
        candidates = valid_action_transitions(maze, state)
        if len(candidates) < 2:
            continue
        action_ids = [action for action, _ in candidates]
        next_ids = [next_idx for _, next_idx in candidates]
        distances = np.asarray([bfs[next_idx, goal] for next_idx in next_ids], dtype=np.float32)
        if (distances < 0).any():
            continue
        best_distance = float(np.min(distances))
        optimal = np.isclose(distances, best_distance)

        num_actions = maze["next_indices"].shape[1]
        ctx_emb = latents[state].view(1, 1, -1).repeat(1, HISTORY_SIZE, 1)
        ctx_act = torch.full((1, HISTORY_SIZE), num_actions - 1, dtype=torch.long, device=device)
        ctx_emb_rep = ctx_emb.expand(num_actions, -1, -1)
        ctx_act_rep = ctx_act[:, :-1].repeat(num_actions, 1)
        ctx_act_rep[:, -1] = torch.arange(num_actions, device=device)
        with torch.no_grad():
            pred_all = model.predictor(ctx_emb_rep, ctx_act_rep)[:, -1, :]

        row = torch.zeros((max_actions, latents.shape[-1]), dtype=latents.dtype, device=device)
        mask = torch.zeros(max_actions, dtype=torch.bool, device=device)
        probs = torch.zeros(max_actions, dtype=torch.float32, device=device)
        for slot, action in enumerate(action_ids[:max_actions]):
            row[slot] = pred_all[action]
            mask[slot] = True
            if bool(optimal[slot]):
                probs[slot] = 1.0
        if probs.sum() <= 0:
            continue
        probs = probs / probs.sum()
        next_rows.append(row)
        goal_rows.append(latents[goal])
        masks.append(mask)
        target_probs.append(probs)

    if not target_probs:
        raise RuntimeError("failed to sample predictor action order batch")
    return (
        torch.stack(next_rows).to(device),
        torch.stack(goal_rows).to(device),
        torch.stack(masks).to(device),
        torch.stack(target_probs).to(device),
    )


def sample_triangle_batch(
    dataset: SetBMazeCache,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    model: Unisize256 | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_list: list[torch.Tensor] = []
    b_list: list[torch.Tensor] = []
    c_list: list[torch.Tensor] = []
    while len(a_list) < cfg.triangle_batch_size:
        maze_idx = dataset.sample_maze_idx(rng)
        maze = dataset.get(maze_idx)
        latents = dataset.encode_maze(maze_idx, model, grad=True) if model is not None else maze["latents"]
        n = latents.shape[0]
        idx = rng.integers(0, n, size=3)
        a_list.append(latents[int(idx[0])])
        b_list.append(latents[int(idx[1])])
        c_list.append(latents[int(idx[2])])
    return torch.stack(a_list).to(device), torch.stack(b_list).to(device), torch.stack(c_list).to(device)


def evaluate_head(
    head: DistanceHead,
    dataset: SetBMazeCache,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    model: Unisize256 | None = None,
    predictor_model: Unisize256 | None = None,
) -> dict[str, float]:
    head.eval()
    losses = []
    rank_accs = []
    action_top1s = []
    action_losses = []
    predictor_action_top1s = []
    predictor_action_losses = []
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            z1, z2, target = sample_regression_batch(dataset, cfg, rng, device, model)
            pred = head(z1, z2)
            losses.append(float(F.smooth_l1_loss(pred, target).item()))

            good, bad, goal = sample_local_ranking_batch(dataset, cfg, rng, device, model)
            good_score = head(good, goal)
            bad_score = head(bad, goal)
            rank_accs.append(float((good_score < bad_score).float().mean().item()))
        for _ in range(cfg.eval_action_batches):
            next_latents, goal_latents, mask, target_probs = sample_action_order_batch(
                dataset, cfg, rng, device, model
            )
            action_loss, action_top1 = action_order_loss(head, next_latents, goal_latents, mask, target_probs)
            action_losses.append(float(action_loss.item()))
            action_top1s.append(float(action_top1.item()))
        if cfg.eval_predictor_action_batches > 0:
            if predictor_model is None:
                raise ValueError("predictor_model is required for predictor action evaluation")
            for _ in range(cfg.eval_predictor_action_batches):
                next_latents, goal_latents, mask, target_probs = sample_predictor_action_order_batch(
                    dataset, cfg, rng, device, predictor_model
                )
                pred_action_loss, pred_action_top1 = action_order_loss(
                    head, next_latents, goal_latents, mask, target_probs
                )
                predictor_action_losses.append(float(pred_action_loss.item()))
                predictor_action_top1s.append(float(pred_action_top1.item()))
    return {
        "val_reg_loss": float(np.mean(losses)),
        "val_rank_acc": float(np.mean(rank_accs)),
        "val_action_ce": float(np.mean(action_losses)) if action_losses else 0.0,
        "val_action_top1": float(np.mean(action_top1s)) if action_top1s else 0.0,
        "val_predictor_action_ce": float(np.mean(predictor_action_losses)) if predictor_action_losses else 0.0,
        "val_predictor_action_top1": float(np.mean(predictor_action_top1s)) if predictor_action_top1s else 0.0,
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", default="checkpoints/unisize_dim256.pt")
    parser.add_argument("--train-manifest", default="data/splits/unisize_train_manifest.jsonl")
    parser.add_argument("--val-manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--local-batch-size", type=int, default=256)
    parser.add_argument("--action-batch-size", type=int, default=256)
    parser.add_argument("--predictor-action-batch-size", type=int, default=0)
    parser.add_argument("--triangle-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--target-mode", choices=["raw", "norm", "log", "log_norm"], default="log_norm")
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--action-ce-weight", type=float, default=0.0)
    parser.add_argument("--predictor-action-ce-weight", type=float, default=0.0)
    parser.add_argument("--triangle-weight", type=float, default=0.05)
    parser.add_argument("--ranking-margin", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-action-batches", type=int, default=8)
    parser.add_argument("--eval-predictor-action-batches", type=int, default=0)
    parser.add_argument("--output", default="checkpoints/metric_heads/distance_head_v2_setb.pt")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument(
        "--unfreeze-scope",
        choices=["none", "projector", "encoder_projector"],
        default="none",
        help="Backbone portion to train. --unfreeze-backbone maps to encoder_projector for compatibility.",
    )
    parser.add_argument("--mazes-per-step", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = TrainConfig(
        model_ckpt=args.model_ckpt,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        steps=args.steps,
        batch_size=args.batch_size,
        local_batch_size=args.local_batch_size,
        action_batch_size=args.action_batch_size,
        predictor_action_batch_size=args.predictor_action_batch_size,
        triangle_batch_size=args.triangle_batch_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        target_mode=args.target_mode,
        regression_weight=args.regression_weight,
        ranking_weight=args.ranking_weight,
        action_ce_weight=args.action_ce_weight,
        predictor_action_ce_weight=args.predictor_action_ce_weight,
        triangle_weight=args.triangle_weight,
        ranking_margin=args.ranking_margin,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        eval_action_batches=args.eval_action_batches,
        eval_predictor_action_batches=args.eval_predictor_action_batches,
        output=args.output,
        unfreeze_backbone=args.unfreeze_backbone,
        unfreeze_scope=("encoder_projector" if args.unfreeze_backbone and args.unfreeze_scope == "none" else args.unfreeze_scope),
        mazes_per_step=args.mazes_per_step,
        device=args.device,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    print("=" * 70)
    print("DISTANCE HEAD V2 TRAINING")
    print("=" * 70)
    print(json.dumps(asdict(cfg), indent=2))

    with open(cfg.train_manifest) as f:
        train_entries = [json.loads(line) for line in f if line.strip()]
    with open(cfg.val_manifest) as f:
        val_entries_all = [json.loads(line) for line in f if line.strip()]
    val_entries = [entry for entry in val_entries_all if int(entry["maze_size"]) <= 21]

    train_topo = {(entry["maze_size"], entry["topology_seed"]) for entry in train_entries}
    val_topo = {(entry["maze_size"], entry["topology_seed"]) for entry in val_entries}
    train_layout = {entry.get("layout_hash") for entry in train_entries if entry.get("layout_hash")}
    val_layout = {entry.get("layout_hash") for entry in val_entries if entry.get("layout_hash")}
    train_task = {entry.get("task_hash") for entry in train_entries if entry.get("task_hash")}
    val_task = {entry.get("task_hash") for entry in val_entries if entry.get("task_hash")}
    if train_topo & val_topo or train_layout & val_layout or train_task & val_task:
        raise ValueError("train/val leakage detected")
    print(f"Train entries: {len(train_entries)}, val entries: {len(val_entries)}, overlap=0")

    train_backbone = cfg.unfreeze_scope != "none"
    if cfg.predictor_action_ce_weight > 0:
        if train_backbone:
            raise ValueError("predictor-aligned DH currently requires --unfreeze-scope none")
        if cfg.predictor_action_batch_size <= 0:
            raise ValueError("--predictor-action-batch-size must be positive when predictor loss is enabled")
        if cfg.eval_predictor_action_batches <= 0:
            raise ValueError("--eval-predictor-action-batches must be positive when predictor loss is enabled")
    model = load_backbone(cfg.model_ckpt, device, freeze=not train_backbone)
    if train_backbone:
        model.train()
        for param in model.parameters():
            param.requires_grad = False
        if cfg.unfreeze_scope == "encoder_projector":
            for param in model.encoder.parameters():
                param.requires_grad = True
        for param in model.embedding_projector.parameters():
            param.requires_grad = True
        n_backbone = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"Unfreeze scope: {cfg.unfreeze_scope}, trainable backbone params: {n_backbone:,}")
    train_ds = SetBMazeCache(model, train_entries, device, store_observations=train_backbone)
    val_ds = SetBMazeCache(model, val_entries, device, store_observations=train_backbone)
    print("Caches are lazy; first training/eval batches will build maze caches on demand.", flush=True)

    head = DistanceHead(
        latent_dim=cfg.latent_dim,
        hidden_dims=list(cfg.hidden_dims),
        input_mode=cfg.input_mode,
    ).to(device)
    if train_backbone:
        backbone_params = [param for param in model.parameters() if param.requires_grad]
        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.backbone_lr},
                {"params": head.parameters(), "lr": cfg.lr},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    best_val_action_top1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    logs: list[dict[str, float]] = []
    train_losses: list[float] = []
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        head.train()
        if train_backbone:
            model.train()
            ds_for_training: SetBMazeCache | StepMazeCache = StepMazeCache(
                train_ds, model, rng, cfg.mazes_per_step
            )
        else:
            ds_for_training = train_ds

        z1, z2, target = sample_regression_batch(ds_for_training, cfg, rng, device)
        pred = head(z1, z2)
        reg_loss = F.smooth_l1_loss(pred, target)

        good, bad, goal = sample_local_ranking_batch(ds_for_training, cfg, rng, device)
        good_score = head(good, goal)
        bad_score = head(bad, goal)
        ranking_loss = F.relu(good_score - bad_score + cfg.ranking_margin).mean()

        tri_loss = torch.tensor(0.0, device=device)
        if cfg.triangle_weight > 0 and cfg.triangle_batch_size > 0:
            za, zb, zc = sample_triangle_batch(ds_for_training, cfg, rng, device)
            tri_loss = F.relu(head(za, zc) - head(za, zb) - head(zb, zc)).mean()

        action_loss = torch.tensor(0.0, device=device)
        action_top1 = torch.tensor(0.0, device=device)
        if cfg.action_ce_weight > 0 and cfg.action_batch_size > 0:
            next_latents, goal_latents, mask, target_probs = sample_action_order_batch(
                ds_for_training, cfg, rng, device
            )
            action_loss, action_top1 = action_order_loss(head, next_latents, goal_latents, mask, target_probs)

        predictor_action_loss = torch.tensor(0.0, device=device)
        predictor_action_top1 = torch.tensor(0.0, device=device)
        if cfg.predictor_action_ce_weight > 0 and cfg.predictor_action_batch_size > 0:
            next_latents, goal_latents, mask, target_probs = sample_predictor_action_order_batch(
                train_ds, cfg, rng, device, model
            )
            predictor_action_loss, predictor_action_top1 = action_order_loss(
                head, next_latents, goal_latents, mask, target_probs
            )

        loss = (
            cfg.regression_weight * reg_loss
            + cfg.ranking_weight * ranking_loss
            + cfg.action_ce_weight * action_loss
            + cfg.predictor_action_ce_weight * predictor_action_loss
            + cfg.triangle_weight * tri_loss
        )
        opt.zero_grad()
        loss.backward()
        if train_backbone:
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        else:
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        scheduler.step()
        train_losses.append(float(loss.item()))

        if step % cfg.eval_every == 0:
            if train_backbone:
                model.eval()
            metrics = evaluate_head(
                head,
                val_ds,
                cfg,
                rng,
                device,
                model if train_backbone else None,
                model if cfg.eval_predictor_action_batches > 0 else None,
            )
            if train_backbone:
                model.train()
            elapsed = time.time() - t0
            log_row = {
                "step": float(step),
                "train_loss": float(np.mean(train_losses[-cfg.eval_every :])),
                "reg_loss": float(reg_loss.item()),
                "ranking_loss": float(ranking_loss.item()),
                "action_loss": float(action_loss.item()),
                "action_top1": float(action_top1.item()),
                "predictor_action_loss": float(predictor_action_loss.item()),
                "predictor_action_top1": float(predictor_action_top1.item()),
                "triangle_loss": float(tri_loss.item()),
                "elapsed": float(elapsed),
                **metrics,
            }
            logs.append(log_row)
            print(
                f"Step {step:>6d}/{cfg.steps}: train={log_row['train_loss']:.4f} "
                f"reg={log_row['reg_loss']:.4f} rank={log_row['ranking_loss']:.4f} "
                f"act={log_row['action_loss']:.4f} pred_act={log_row['predictor_action_loss']:.4f} "
                f"tri={log_row['triangle_loss']:.4f} "
                f"val_reg={metrics['val_reg_loss']:.4f} val_rank_acc={metrics['val_rank_acc']:.4f} "
                f"val_action_top1={metrics['val_action_top1']:.4f} "
                f"val_pred_action_top1={metrics['val_predictor_action_top1']:.4f} ({elapsed:.0f}s)",
                flush=True,
            )
            t0 = time.time()
            selection_metric = (
                metrics["val_predictor_action_top1"]
                if cfg.predictor_action_ce_weight > 0
                else metrics["val_action_top1"]
            )
            if selection_metric > best_val_action_top1:
                best_val_action_top1 = selection_metric
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                if train_backbone:
                    best_state.update(
                        {
                            f"backbone.{key}": value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                    )

    if best_state is not None:
        if train_backbone:
            head.load_state_dict(
                {
                    key: value.to(device)
                    for key, value in best_state.items()
                    if not key.startswith("backbone.")
                }
            )
            model.load_state_dict(
                {
                    key.split("backbone.", 1)[1]: value.to(device)
                    for key, value in best_state.items()
                    if key.startswith("backbone.")
                }
            )
        else:
            head.load_state_dict({key: value.to(device) for key, value in best_state.items()})

    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "head_state_dict": head.state_dict(),
        "config": {
            **asdict(cfg),
            "hidden_dims": list(cfg.hidden_dims),
            "best_selection_metric": best_val_action_top1,
            "selection_metric_name": (
                "val_predictor_action_top1"
                if cfg.predictor_action_ce_weight > 0
                else "val_action_top1"
            ),
        },
        "logs": logs,
        "final_train_loss": float(np.mean(train_losses[-min(1000, len(train_losses)) :])),
    }
    if train_backbone:
        ckpt["model_state_dict"] = model.state_dict()
        ckpt["model_config"] = model.config
    torch.save(ckpt, output)
    with open(output.with_suffix(".json"), "w") as f:
        json.dump(ckpt["config"], f, indent=2)
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
