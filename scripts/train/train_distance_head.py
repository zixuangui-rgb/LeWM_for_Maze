#!/usr/bin/env python3
"""Train a Distance Head on frozen LeWM latents to predict BFS shortest-path distance.

The distance head takes (z_current, z_goal) and predicts the BFS distance between
the two states. LeWM weights are frozen — only the distance head is trained.

Data pipeline:
    1. Pre-extract latents from all train maze walkable cells (one-time cost)
    2. During training: sample pairs, compute BFS labels, train with MSE loss

Usage:
    # Quick sanity: overfit on 1 maze
    python scripts/train/train_distance_head.py --overfit

    # Full training
    python scripts/train/train_distance_head.py --steps 50000

Output:
    checkpoints/metric_heads/distance_head.pt
    checkpoints/metric_heads/distance_head_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

_HDWM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDWM_ROOT))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.metric_heads.distance_head import DistanceHead
from hdwm.planning import _bfs_shortest_path
from scripts.train.train_ablation_models import OriginalLeWM


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "latent_source": "embedding",      # "encoded" or "embedding"
    "latent_dim": 256,
    "hidden_dims": [256, 128],
    "input_mode": "concat",            # "concat", "diff", "concat_diff"
    "dropout": 0.0,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "pairs_per_maze": 64,              # sample this many pairs per maze per epoch
    "max_distance": 121,               # cap BFS distance (max for 11x11 grid)
    "lewm_checkpoint": "checkpoints/ablation/original_lewm.pt",
    "train_manifest": "data/splits/fixed11_train_manifest.jsonl",
    "val_manifest": "data/splits/fixed11_val_manifest.jsonl",
}


# ---------------------------------------------------------------------------
# Latent extraction
# ---------------------------------------------------------------------------

def encode_obs(model, obs: np.ndarray, maze_size: int, device: torch.device) -> torch.Tensor:
    """Encode one observation frame → latent [D]."""
    t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encoder(t, maze_size)           # [1,1,M]
        embedding, _ = model.embedding_projector(encoded)  # [1,1,D]
    return embedding.squeeze(0).squeeze(0)  # [D]


def encode_obs_encoded(model, obs: np.ndarray, maze_size: int, device: torch.device) -> torch.Tensor:
    """Encode one observation frame → encoded latent [M] (pre-projector)."""
    t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encoder(t, maze_size)  # [1,1,M]
    return encoded.squeeze(0).squeeze(0)  # [M]


def pre_extract_maze_latents(
    model: OriginalLeWM,
    entry: dict,
    device: torch.device,
    latent_source: str = "embedding",
) -> tuple[torch.Tensor, list[int], np.ndarray]:
    """Extract latents for all walkable cells in a maze.

    Returns:
        latents: [num_walkable, D] tensor on device
        cell_indices: list of flat cell indices
        bfs_cache: [num_walkable, num_walkable] BFS distance matrix (numpy)
    """
    sz = entry["maze_size"]
    env_cfg = ProcgenMazeConfig(
        height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry["topology_seed"],
    )
    env = ProcgenMazeEnv(env_cfg, seed=entry.get("level_seed", 42))
    obstacle = env._maze_mask
    empty_mask = ~obstacle
    walkable = np.flatnonzero(empty_mask.reshape(-1)).tolist()

    goal_pos = env._goal_position
    latents_list = []
    for cell in walkable:
        # Directly render observation for this cell (bypass reset's goal restriction)
        env._state = cell
        obs, _ = env._observe_with_noise(np.array([cell]))
        obs = obs[0]  # [H,W,C]
        env._last_observation = obs
        if latent_source == "encoded":
            z = encode_obs_encoded(model, obs, sz, device)
        else:
            z = encode_obs(model, obs, sz, device)
        latents_list.append(z)

    latents = torch.stack(latents_list, dim=0)  # [N, D]

    # Pre-compute BFS distances
    width = env.config.width
    n = len(walkable)
    bfs_cache = np.full((n, n), -1, dtype=np.int32)
    for i in range(n):
        for j in range(i, n):
            d = _bfs_shortest_path(obstacle, walkable[i], walkable[j], width)
            if d is not None:
                bfs_cache[i, j] = d
                bfs_cache[j, i] = d

    return latents, walkable, bfs_cache


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LatentPairDataset:
    """Pre-extracted latent pairs for distance head training.

    Lazily loads mazes from manifest and caches their latents.
    """

    def __init__(
        self,
        model: OriginalLeWM,
        entries: list[dict],
        device: torch.device,
        config: dict,
        is_train: bool = True,
    ):
        self.model = model
        self.entries = entries
        self.device = device
        self.cfg = config
        self.is_train = is_train
        self._cache: dict[int, tuple[torch.Tensor, list[int], np.ndarray]] = {}

    def __len__(self):
        return len(self.entries)

    def get_maze(self, idx: int) -> tuple[torch.Tensor, list[int], np.ndarray]:
        """Get or compute cached (latents, cells, bfs_cache) for a maze."""
        if idx not in self._cache:
            self._cache[idx] = pre_extract_maze_latents(
                self.model, self.entries[idx], self.device, self.cfg["latent_source"],
            )
        return self._cache[idx]

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of (z1, z2, bfs_distance).

        Strategy: pick one maze, sample multiple pairs within that maze.
        This is efficient because most latents are already in GPU memory.
        """
        z1_list, z2_list, labels_list = [], [], []
        pairs_per = max(1, batch_size // 4)  # 4 mazes → batch

        for _ in range(4):
            maze_idx = int(rng.integers(0, len(self.entries)))
            latents, cells, bfs = self.get_maze(maze_idx)
            n = len(cells)

            for _ in range(pairs_per):
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n))
                if i == j:
                    continue
                d = bfs[i, j]
                if d < 0:  # unreachable
                    d = self.cfg["max_distance"]

                z1_list.append(latents[i])
                z2_list.append(latents[j])
                labels_list.append(float(d))

                if len(z1_list) >= batch_size:
                    break
            if len(z1_list) >= batch_size:
                break

        z1 = torch.stack(z1_list[:batch_size], dim=0)
        z2 = torch.stack(z2_list[:batch_size], dim=0)
        labels = torch.tensor(labels_list[:batch_size], dtype=torch.float32, device=self.device)
        return z1, z2, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def overfit_test(head, dataset, device, max_steps=2000, target_loss=0.01):
    """Overfit on a single maze (first entry). Verify model can memorize BFS distances."""
    latents, cells, bfs = dataset.get_maze(0)
    n = len(cells)

    opt = optim.Adam(head.parameters(), lr=1e-3)
    rng = np.random.default_rng(42)

    print(f"  Overfit test: 1 maze, {n} walkable cells, {n*(n-1)} pairs")
    t0 = time.time()
    for step in range(1, max_steps + 1):
        # Sample pairs from this single maze
        idx_i = rng.integers(0, n, size=256)
        idx_j = rng.integers(0, n, size=256)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        z1 = latents[idx_i]
        z2 = latents[idx_j]
        labels = torch.tensor(
            [bfs[i, j] if bfs[i, j] >= 0 else 121 for i, j in zip(idx_i, idx_j)],
            dtype=torch.float32, device=device,
        )

        pred = head(z1, z2)
        loss = F.mse_loss(pred, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 400 == 0:
            with torch.no_grad():
                # Full evaluation on all pairs
                all_z1 = latents.unsqueeze(1).expand(-1, n, -1).reshape(-1, latents.shape[-1])
                all_z2 = latents.unsqueeze(0).expand(n, -1, -1).reshape(-1, latents.shape[-1])
                all_pred = head(all_z1, all_z2).reshape(n, n)
                # Pearson correlation
                all_labels = torch.tensor(bfs, dtype=torch.float32, device=device)
                valid_mask = all_labels >= 0
                if valid_mask.sum() > 1:
                    pred_v = all_pred[valid_mask]
                    label_v = all_labels[valid_mask]
                    # Pearson
                    p_mean, l_mean = pred_v.mean(), label_v.mean()
                    p_centered, l_centered = pred_v - p_mean, label_v - l_mean
                    pearson = (p_centered * l_centered).sum() / (
                        torch.sqrt((p_centered**2).sum()) * torch.sqrt((l_centered**2).sum()) + 1e-8
                    )
                else:
                    pearson = torch.tensor(0.0)
            print(f"    Step {step:>5d}: loss={loss.item():.4f}  pearson={pearson.item():.4f}")
            if loss.item() < target_loss:
                print(f"    Reached target loss at step {step}")
                break

    elapsed = time.time() - t0
    print(f"  Overfit done in {elapsed:.0f}s ({step} steps)")
    return loss.item()


def train(config: dict | None = None):
    if config is None:
        config = DEFAULT_CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config
    print("=" * 70)
    print("DISTANCE HEAD TRAINING")
    print("=" * 70)
    print(f"  Latent source: {cfg['latent_source']}")
    print(f"  Input mode: {cfg['input_mode']}")
    print(f"  Hidden dims: {cfg['hidden_dims']}")
    print(f"  LR: {cfg['learning_rate']}")
    print(f"  Device: {device}")
    print()

    # 1. Load frozen LeWM
    print("[1] Loading frozen LeWM...")
    ckpt = torch.load(cfg["lewm_checkpoint"], map_location=device, weights_only=False)
    model = OriginalLeWM(ckpt["model_config"], max_size=31).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  OriginalLeWM loaded, latent_dim={cfg['latent_dim']}")
    print()

    # 2. Load manifests
    print("[2] Loading manifests...")
    with open(cfg["train_manifest"]) as f:
        train_entries = [json.loads(l) for l in f if l.strip()]
    with open(cfg["val_manifest"]) as f:
        val_entries = [json.loads(l) for l in f if l.strip()]
    print(f"  Train: {len(train_entries)} mazes")
    print(f"  Val:   {len(val_entries)} mazes")
    # Verify no topology overlap
    train_topo = set(e["topology_seed"] for e in train_entries)
    val_topo = set(e["topology_seed"] for e in val_entries)
    assert len(train_topo & val_topo) == 0, "Topology leakage detected!"
    print(f"  Topology overlap: 0 ✓")
    print()

    # 3. Create datasets
    print("[3] Creating datasets (pre-extracting latents)...")
    t0 = time.time()
    train_dataset = LatentPairDataset(model, train_entries, device, cfg, is_train=True)
    val_dataset = LatentPairDataset(model, val_entries, device, cfg, is_train=False)
    # Pre-load a few mazes to verify
    train_dataset.get_maze(0)
    val_dataset.get_maze(0)
    print(f"  Latent extraction working ({time.time() - t0:.0f}s for first maze)")
    print()

    # 4. Create distance head
    print("[4] Creating DistanceHead...")
    head = DistanceHead(
        latent_dim=cfg["latent_dim"],
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        input_mode=cfg["input_mode"],
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  Parameters: {n_params:,}")
    print()

    # 5. Overfit test
    print("[5] Overfit test on single maze...")
    ov_loss = overfit_test(head, train_dataset, device, max_steps=2000)
    print()

    # 6. Re-init and train properly
    print("[6] Full training...")
    head = DistanceHead(
        latent_dim=cfg["latent_dim"],
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        input_mode=cfg["input_mode"],
    ).to(device)

    opt = optim.AdamW(head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.get("steps", 20000))
    rng = np.random.default_rng(42)

    train_losses = []
    val_metrics_log = []
    steps = cfg.get("steps", 20000)
    eval_every = cfg.get("eval_every", 1000)
    batch_size = cfg["batch_size"]

    t0 = time.time()
    for step in range(1, steps + 1):
        head.train()
        z1, z2, labels = train_dataset.sample_batch(batch_size, rng)
        pred = head(z1, z2)
        loss = F.mse_loss(pred, labels)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        scheduler.step()

        train_losses.append(loss.item())

        if step % eval_every == 0:
            head.eval()
            # Validation metrics
            all_preds, all_labels = [], []
            with torch.no_grad():
                for _ in range(20):  # 20 batches for validation
                    z1_v, z2_v, labels_v = val_dataset.sample_batch(batch_size, rng)
                    all_preds.append(head(z1_v, z2_v))
                    all_labels.append(labels_v)

            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)

            val_mse = F.mse_loss(preds, labels).item()
            val_mae = F.l1_loss(preds, labels).item()

            # Pearson
            p_mean, l_mean = preds.mean(), labels.mean()
            p_c, l_c = preds - p_mean, labels - l_mean
            pearson = (p_c * l_c).sum() / (torch.sqrt((p_c**2).sum()) * torch.sqrt((l_c**2).sum()) + 1e-8)

            # Spearman (using simplified ranking)
            from scipy.stats import spearmanr
            sp, _ = spearmanr(preds.cpu().numpy(), labels.cpu().numpy())

            avg_train = float(np.mean(train_losses[-eval_every:]))
            elapsed = time.time() - t0
            print(f"  Step {step:>6d}/{steps}: train_loss={avg_train:.4f}  "
                  f"val_mse={val_mse:.4f}  val_mae={val_mae:.2f}  "
                  f"pearson={pearson.item():.4f}  spearman={sp:.4f}  ({elapsed:.0f}s)")
            t0 = time.time()

            val_metrics_log.append({
                "step": step,
                "train_loss": avg_train,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "pearson": pearson.item() if hasattr(pearson, 'item') else float(pearson),
                "spearman": float(sp),
            })

    # 7. Save checkpoint
    print("\n[7] Saving checkpoint...")
    output_dir = Path("checkpoints/metric_heads")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_data = {
        "head_state_dict": head.state_dict(),
        "config": cfg,
        "val_metrics": val_metrics_log,
        "final_train_loss": float(np.mean(train_losses[-1000:])),
    }
    torch.save(ckpt_data, output_dir / "distance_head.pt")
    with open(output_dir / "distance_head_config.json", "w") as f:
        json.dump({**cfg, "n_params": n_params}, f, indent=2)

    print(f"  Saved: {output_dir / 'distance_head.pt'}")
    print(f"  Saved: {output_dir / 'distance_head_config.json'}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train Distance Head on frozen LeWM latents")
    p.add_argument("--overfit", action="store_true", help="Only run overfit test")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-source", choices=["encoded", "embedding"], default="embedding")
    p.add_argument("--input-mode", choices=["concat", "diff", "concat_diff"], default="concat")
    p.add_argument("--hidden-dims", type=str, default="256,128")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",")]

    config = {**DEFAULT_CONFIG}
    config.update({
        "latent_source": args.latent_source,
        "input_mode": args.input_mode,
        "hidden_dims": hidden_dims,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "steps": args.steps,
    })

    if args.overfit:
        device = torch.device(args.device)
        ckpt = torch.load(config["lewm_checkpoint"], map_location=device, weights_only=False)
        model = OriginalLeWM(ckpt["model_config"], max_size=31).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        with open(config["train_manifest"]) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        dataset = LatentPairDataset(model, entries[:1], device, config, is_train=True)
        head = DistanceHead(
            latent_dim=config["latent_dim"],
            hidden_dims=config["hidden_dims"],
            input_mode=config["input_mode"],
        ).to(device)
        overfit_test(head, dataset, device, max_steps=2000)
    else:
        train(config)


if __name__ == "__main__":
    main()
