#!/usr/bin/env python3
"""Simple Set B DistanceHead training with only BFS distance supervision.

This script intentionally avoids the ranking/action/predictor objectives used
by later experiments. It is meant to answer a narrow debugging question:

Does a frozen LeWM embedding plus a plain DistanceHead learn to regress BFS
distance, and do train/held-out eval losses decrease normally?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
from scripts.train.train_dim256 import Unisize256
from scripts.train.train_distance_head_v2 import all_pairs_bfs


@dataclass
class TrainConfig:
    model_ckpt: str = "checkpoints/backbones/unisize_dim256_clean_20260702.pt"
    train_manifest: str = "data/splits/unisize_train_manifest.jsonl"
    eval_manifest: str = "data/splits/unisize_eval_manifest.jsonl"
    output: str = "checkpoints/metric_heads/distance_head_simple_setb.pt"
    latent_dim: int = 256
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    input_mode: str = "concat"
    target_mode: str = "raw"
    loss: str = "mse"
    steps: int = 30000
    batch_size: int = 512
    pairs_per_maze: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    eval_every: int = 1000
    eval_batches: int = 16
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


def encode_observations(
    model: Unisize256,
    observations: list[np.ndarray],
    maze_size: int,
    device: torch.device,
) -> torch.Tensor:
    obs = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        encoded = model.encoder(obs, maze_size)
        embedding, _ = model.embedding_projector(encoded)
    return embedding.squeeze(1).detach().cpu()


def load_entries(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_backbone(path: str, device: torch.device) -> Unisize256:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Unisize256(ckpt["model_config"], max_size=31).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


class MazeLatentCache:
    """Lazy frozen latent/BFS cache for a manifest split."""

    def __init__(
        self,
        model: Unisize256,
        entries: list[dict[str, Any]],
        device: torch.device,
    ) -> None:
        self.model = model
        self.entries = entries
        self.device = device
        self.by_size: dict[int, list[int]] = defaultdict(list)
        for idx, entry in enumerate(entries):
            self.by_size[int(entry["maze_size"])].append(idx)
        self.sizes = sorted(self.by_size)
        self._cache: dict[int, dict[str, Any]] = {}
        self._build_count = 0

    def sample_idx(self, rng: np.random.Generator) -> int:
        size = int(rng.choice(self.sizes))
        return int(rng.choice(self.by_size[size]))

    def get(self, idx: int) -> dict[str, Any]:
        if idx not in self._cache:
            self._cache[idx] = self._build(idx)
        return self._cache[idx]

    def _build(self, idx: int) -> dict[str, Any]:
        t0 = time.time()
        entry = self.entries[idx]
        size = int(entry["maze_size"])
        env = create_env(entry)
        cells = np.flatnonzero((~env._maze_mask).reshape(-1)).astype(np.int64).tolist()
        observations = [observe_state(env, cell) for cell in cells]
        latents = encode_observations(self.model, observations, size, self.device)
        bfs = all_pairs_bfs(env._maze_mask, cells, env.config.width).astype(np.float32)
        max_dist = max(float(np.max(bfs)), 1.0)
        self._build_count += 1
        if self._build_count <= 20:
            print(
                f"  [cache] split_size={len(self.entries)} idx={idx} size={size} "
                f"cells={len(cells)} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        return {"latents": latents, "bfs": bfs, "max_dist": max_dist, "entry": entry}


def transform_target(dist: torch.Tensor, max_dist: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return dist
    if mode == "norm":
        return dist / max_dist.clamp_min(1.0)
    if mode == "log":
        return torch.log1p(dist)
    if mode == "log_norm":
        return torch.log1p(dist) / torch.log1p(max_dist.clamp_min(1.0))
    raise ValueError(f"unknown target_mode: {mode}")


def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target)
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    if loss_name == "mae":
        return F.l1_loss(pred, target)
    raise ValueError(f"unknown loss: {loss_name}")


def sample_pair_batch(
    dataset: MazeLatentCache,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z1_list: list[torch.Tensor] = []
    z2_list: list[torch.Tensor] = []
    dist_list: list[float] = []
    max_list: list[float] = []
    while len(z1_list) < cfg.batch_size:
        maze = dataset.get(dataset.sample_idx(rng))
        latents = maze["latents"]
        bfs = maze["bfs"]
        n = int(latents.shape[0])
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
    dist_t = torch.tensor(dist_list, dtype=torch.float32, device=device)
    max_t = torch.tensor(max_list, dtype=torch.float32, device=device)
    return z1, z2, transform_target(dist_t, max_t, cfg.target_mode)


def evaluate_loss(
    head: DistanceHead,
    dataset: MazeLatentCache,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    losses: list[float] = []
    head.eval()
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            z1, z2, target = sample_pair_batch(dataset, cfg, rng, device)
            pred = head(z1, z2)
            losses.append(float(compute_loss(pred, target, cfg.loss).item()))
    return float(np.mean(losses))


def check_holdout(train_entries: list[dict[str, Any]], eval_entries: list[dict[str, Any]]) -> None:
    train_topo = {(entry["maze_size"], entry["topology_seed"]) for entry in train_entries}
    eval_topo = {(entry["maze_size"], entry["topology_seed"]) for entry in eval_entries}
    train_layout = {entry.get("layout_hash") for entry in train_entries if entry.get("layout_hash")}
    eval_layout = {entry.get("layout_hash") for entry in eval_entries if entry.get("layout_hash")}
    train_task = {entry.get("task_hash") for entry in train_entries if entry.get("task_hash")}
    eval_task = {entry.get("task_hash") for entry in eval_entries if entry.get("task_hash")}
    if train_topo & eval_topo or train_layout & eval_layout or train_task & eval_task:
        raise ValueError("train/eval leakage detected")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", default="checkpoints/backbones/unisize_dim256_clean_20260702.pt")
    parser.add_argument("--train-manifest", default="data/splits/unisize_train_manifest.jsonl")
    parser.add_argument("--eval-manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--output", default="checkpoints/metric_heads/distance_head_simple_setb.pt")
    parser.add_argument("--target-mode", choices=["raw", "norm", "log", "log_norm"], default="raw")
    parser.add_argument("--loss", choices=["mse", "smooth_l1", "mae"], default="mse")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pairs-per-maze", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return TrainConfig(
        model_ckpt=args.model_ckpt,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        output=args.output,
        target_mode=args.target_mode,
        loss=args.loss,
        steps=args.steps,
        batch_size=args.batch_size,
        pairs_per_maze=args.pairs_per_maze,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        device=args.device,
    )


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_entries = load_entries(cfg.train_manifest)
    eval_entries_all = load_entries(cfg.eval_manifest)
    eval_seen_entries = [entry for entry in eval_entries_all if int(entry["maze_size"]) <= 21]
    eval_ood_entries = [entry for entry in eval_entries_all if int(entry["maze_size"]) > 21]
    check_holdout(train_entries, eval_entries_all)

    print("=" * 70)
    print("SIMPLE DISTANCE HEAD TRAINING")
    print("=" * 70)
    print(json.dumps(asdict(cfg), indent=2))
    print(
        f"Train entries={len(train_entries)} eval_seen={len(eval_seen_entries)} "
        f"eval_ood={len(eval_ood_entries)} overlap=0"
    )

    model = load_backbone(cfg.model_ckpt, device)
    train_ds = MazeLatentCache(model, train_entries, device)
    seen_ds = MazeLatentCache(model, eval_seen_entries, device)
    ood_ds = MazeLatentCache(model, eval_ood_entries, device)

    head = DistanceHead(
        latent_dim=cfg.latent_dim,
        hidden_dims=list(cfg.hidden_dims),
        input_mode=cfg.input_mode,
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    best_eval_seen = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    logs: list[dict[str, float]] = []
    recent_losses: list[float] = []
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        head.train()
        z1, z2, target = sample_pair_batch(train_ds, cfg, rng, device)
        pred = head(z1, z2)
        loss = compute_loss(pred, target, cfg.loss)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        scheduler.step()
        recent_losses.append(float(loss.item()))

        if step % cfg.eval_every == 0:
            train_eval = evaluate_loss(head, train_ds, cfg, rng, device)
            seen_eval = evaluate_loss(head, seen_ds, cfg, rng, device)
            ood_eval = evaluate_loss(head, ood_ds, cfg, rng, device) if eval_ood_entries else 0.0
            elapsed = time.time() - t0
            row = {
                "step": float(step),
                "train_batch_loss": float(np.mean(recent_losses[-cfg.eval_every :])),
                "train_eval_loss": train_eval,
                "eval_seen_loss": seen_eval,
                "eval_ood_loss": ood_eval,
                "elapsed": float(elapsed),
            }
            logs.append(row)
            print(
                f"Step {step:>6d}/{cfg.steps}: "
                f"train_batch={row['train_batch_loss']:.4f} "
                f"train_eval={train_eval:.4f} eval_seen={seen_eval:.4f} "
                f"eval_ood={ood_eval:.4f} ({elapsed:.0f}s)",
                flush=True,
            )
            t0 = time.time()
            if seen_eval < best_eval_seen:
                best_eval_seen = seen_eval
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict({key: value.to(device) for key, value in best_state.items()})

    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "head_state_dict": head.state_dict(),
        "config": {
            **asdict(cfg),
            "hidden_dims": list(cfg.hidden_dims),
            "best_eval_seen_loss": best_eval_seen,
        },
        "logs": logs,
        "final_train_loss": float(np.mean(recent_losses[-min(1000, len(recent_losses)) :])),
    }
    torch.save(ckpt, output)
    with open(output.with_suffix(".json"), "w") as f:
        json.dump(ckpt["config"], f, indent=2)
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
