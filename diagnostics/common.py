#!/usr/bin/env python3
"""Shared utilities for Maze-JEPA diagnostics.

The diagnostics are intentionally written as plain scripts with small helper
functions. They avoid adding a new framework on top of the existing research
code, while still enforcing a fixed protocol for scientific comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.metric_heads.distance_head import DistanceHead
from hdwm.metric_heads.qrl_head import QRLHead
from scripts.train.train_dim256 import Unisize256


ACTION_IDS = (1, 2, 3, 4)
ACTION_TO_SLOT = {action: idx for idx, action in enumerate(ACTION_IDS)}
SLOT_TO_ACTION = {idx: action for action, idx in ACTION_TO_SLOT.items()}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-ckpt", required=True, help="LeWM checkpoint path.")
    parser.add_argument("--train-manifest", default="data/splits/unisize_train_manifest.jsonl")
    parser.add_argument("--eval-manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--run-id", required=True, help="Diagnostics run id.")
    parser.add_argument("--out-dir", default="diagnostics_runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seen-max-size", type=int, default=21)


def run_dir(args: argparse.Namespace) -> Path:
    return Path(args.out_dir) / args.run_id


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def select_entries(
    entries: list[dict[str, Any]],
    max_per_size: int,
    seed: int,
    sizes: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Select a deterministic, size-balanced subset of manifest entries."""

    rng = np.random.default_rng(seed)
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        size = int(entry["maze_size"])
        if sizes is not None and size not in sizes:
            continue
        by_size[size].append(entry)

    selected: list[dict[str, Any]] = []
    for size in sorted(by_size):
        group = list(by_size[size])
        rng.shuffle(group)
        selected.extend(group[:max_per_size] if max_per_size > 0 else group)
    return selected


def verify_holdout(train_entries: list[dict[str, Any]], eval_entries: list[dict[str, Any]]) -> dict[str, int]:
    """Verify topology/layout/task holdout and return overlap counts."""

    train_topo = {(entry["maze_size"], entry["topology_seed"]) for entry in train_entries}
    eval_topo = {(entry["maze_size"], entry["topology_seed"]) for entry in eval_entries}
    train_layout = {entry.get("layout_hash") for entry in train_entries if entry.get("layout_hash")}
    eval_layout = {entry.get("layout_hash") for entry in eval_entries if entry.get("layout_hash")}
    train_task = {entry.get("task_hash") for entry in train_entries if entry.get("task_hash")}
    eval_task = {entry.get("task_hash") for entry in eval_entries if entry.get("task_hash")}
    counts = {
        "topology_overlap": len(train_topo & eval_topo),
        "layout_overlap": len(train_layout & eval_layout),
        "task_overlap": len(train_task & eval_task),
    }
    if any(counts.values()):
        raise ValueError(f"train/eval leakage detected: {counts}")
    return counts


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


def set_agent_state(env: ProcgenMazeEnv, state: int) -> np.ndarray:
    if env._maze_mask.reshape(-1)[state]:
        raise ValueError("state must be an empty cell")
    env._state = int(state)
    env._elapsed_steps = 0
    obs = observe_state(env, int(state))
    env._last_observation = obs
    env._last_noise_mask = np.zeros_like(env._maze_mask, dtype=bool)
    return obs


def free_cells(env: ProcgenMazeEnv) -> np.ndarray:
    return np.flatnonzero((~env._maze_mask).reshape(-1)).astype(np.int64)


def bfs_distances_from(mask: np.ndarray, source: int, width: int) -> np.ndarray:
    """Return BFS distance from source to every cell, -1 for walls/unreachable."""

    height = mask.shape[0]
    out = np.full(height * width, -1, dtype=np.int32)
    if mask.reshape(-1)[source]:
        return out
    queue: deque[int] = deque([int(source)])
    out[source] = 0
    while queue:
        state = queue.popleft()
        row, col = divmod(state, width)
        for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = row + drow
            nc = col + dcol
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            nxt = nr * width + nc
            if mask[nr, nc] or out[nxt] >= 0:
                continue
            out[nxt] = out[state] + 1
            queue.append(nxt)
    return out


def all_pairs_bfs(mask: np.ndarray, cells: np.ndarray, width: int) -> np.ndarray:
    dists = np.full((len(cells), len(cells)), -1, dtype=np.int32)
    cell_to_idx = {int(cell): i for i, cell in enumerate(cells.tolist())}
    for i, cell in enumerate(cells.tolist()):
        full = bfs_distances_from(mask, int(cell), width)
        for dst_cell, j in cell_to_idx.items():
            dists[i, j] = full[dst_cell]
    return dists


def next_state(env: ProcgenMazeEnv, state: int, action: int) -> int:
    return int(env._next_state(int(state), env._decode_action(int(action))))


def valid_action_mask(env: ProcgenMazeEnv, state: int) -> np.ndarray:
    mask = np.zeros(4, dtype=np.float32)
    for action in ACTION_IDS:
        mask[ACTION_TO_SLOT[action]] = 1.0 if next_state(env, state, action) != state else 0.0
    return mask


def optimal_action_mask(env: ProcgenMazeEnv, state: int, goal: int) -> tuple[np.ndarray, int, float]:
    """Return optimal moving action mask, selected class, and current BFS distance.

    The selected class is the first optimal slot and is used for cross entropy.
    The full mask is preserved so top-1 metrics can credit tied shortest-path
    actions.
    """

    goal_dists = bfs_distances_from(env._maze_mask, int(goal), env.config.width)
    cur_dist = float(goal_dists[int(state)])
    mask = np.zeros(4, dtype=np.float32)
    if state == goal or cur_dist <= 0:
        return mask, -1, cur_dist
    candidate_dists: list[float] = []
    for action in ACTION_IDS:
        nxt = next_state(env, state, action)
        if nxt == state:
            candidate_dists.append(math.inf)
        else:
            dist = float(goal_dists[nxt])
            candidate_dists.append(dist if dist >= 0 else math.inf)
    best = min(candidate_dists)
    if not math.isfinite(best):
        return mask, -1, cur_dist
    for slot, dist in enumerate(candidate_dists):
        if math.isclose(dist, best):
            mask[slot] = 1.0
    selected = int(np.flatnonzero(mask)[0])
    return mask, selected, cur_dist


def load_lewm(path: str | Path, device: torch.device) -> Unisize256:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Unisize256(ckpt["model_config"], max_size=31).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_distance_head(path: str | Path, device: torch.device) -> DistanceHead:
    data = torch.load(path, map_location=device, weights_only=False)
    cfg = data.get("config", {})
    head = DistanceHead(
        latent_dim=int(cfg.get("latent_dim", 256)),
        hidden_dims=list(cfg.get("hidden_dims", [512, 256, 128])),
        input_mode=str(cfg.get("input_mode", "concat")),
    ).to(device)
    head.load_state_dict(data["head_state_dict"])
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    return head


def load_qrl_head(path: str | Path, device: torch.device) -> QRLHead:
    data = torch.load(path, map_location=device, weights_only=False)
    cfg = data.get("config", {})
    head = QRLHead(
        latent_dim=int(cfg.get("latent_dim", 256)),
        hidden_dims=list(cfg.get("hidden_dims", [512, 256, 128])),
    ).to(device)
    head.load_state_dict(data["head_state_dict"])
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    return head


def extract_layers_batch(
    model: Unisize256,
    observations: list[np.ndarray],
    maze_size: int,
    device: torch.device,
    layers: list[str],
) -> dict[str, torch.Tensor]:
    """Extract diagnostic layers for a batch of same-size observations."""

    obs = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device).unsqueeze(1)
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        if "spatial_flat" in layers or "spatial_pool" in layers:
            cnn = model.encoder.cnn
            x = obs.permute(0, 1, 4, 2, 3).reshape(obs.shape[0], obs.shape[4], obs.shape[2], obs.shape[3])
            spatial = cnn.conv(x)
            if "spatial_flat" in layers:
                out["spatial_flat"] = spatial.flatten(1).detach().cpu()
            if "spatial_pool" in layers:
                out["spatial_pool"] = F.adaptive_avg_pool2d(spatial, 1).flatten(1).detach().cpu()
        if "encoded" in layers or "embedding" in layers:
            encoded = model.encoder(obs, maze_size).squeeze(1)
            if "encoded" in layers:
                out["encoded"] = encoded.detach().cpu()
            if "embedding" in layers:
                emb, _ = model.embedding_projector(encoded.unsqueeze(1))
                out["embedding"] = emb.squeeze(1).detach().cpu()
    return out


def encode_observations(
    model: Unisize256,
    observations: list[np.ndarray],
    maze_size: int,
    device: torch.device,
) -> torch.Tensor:
    return extract_layers_batch(model, observations, maze_size, device, ["embedding"])["embedding"].to(device)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation sufficient for Spearman diagnostics."""

    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson_corr(rankdata(np.asarray(x)), rankdata(np.asarray(y)))


def summarize_numeric(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": float(arr.size)}


def size_bucket(size: int, seen_max_size: int) -> str:
    return "seen" if int(size) <= int(seen_max_size) else "ood"
