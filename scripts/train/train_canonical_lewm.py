#!/usr/bin/env python3
"""Canonical unisize LeWM training with position auxiliary losses.

Trains a size-conditioned LeWM backbone with:
  - LeWM prediction loss (MSE between predicted and target embeddings)
  - SIGReg regularization
  - Agent absolute position auxiliary loss (x, y)
  - Relative-to-goal position auxiliary loss (dx, dy)
  - Goal position auxiliary loss (goal_x, goal_y)

Best config (dx/dy acc > 98% on unseen topology):
  --lambda-rel 1.0 (key breakthrough)
  --lambda-abs 0.1 --lambda-goal 0.5 --lambda-sigreg 0.09
  --latent-dim 128 --steps 30000

All losses are logged separately. Final checkpoint includes full model state
for subsequent probe training.

Usage:
    python scripts/train_canonical_lewm.py \
        --train-manifest data/splits/unisize_train_manifest.jsonl \
        --eval-manifest data/splits/unisize_eval_manifest.jsonl \
        --steps 30000 --latent-dim 128 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_HDWM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HDWM_ROOT))

from hdwm.config import LEWMCNNConfig, ProcgenMazeConfig, SequenceDataConfig
from hdwm.data import ManifestSequenceDataset, sequence_batch_collate
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.losses import SIGReg
from hdwm.models.lewm import CNNEncoder, NextEmbeddingPredictor
from hdwm.models.shared import LatentEmbeddingProjector
from torch.utils.data import DataLoader


class SizeConditionedEncoder(nn.Module):
    """CNN encoder with learnable size embedding concatenated before projection."""

    def __init__(self, config, max_size: int = 31):
        super().__init__()
        self.cnn = CNNEncoder(config)
        cnn_dim = config.effective_model_dim
        self.size_embed = nn.Embedding(max_size + 1, 16)
        self.fuse = nn.Sequential(
            nn.Linear(cnn_dim + 16, cnn_dim),
            nn.LayerNorm(cnn_dim),
            nn.ReLU(),
            nn.Linear(cnn_dim, cnn_dim),
        )

    def forward(self, observations: torch.Tensor, size: int) -> torch.Tensor:
        cnn_out = self.cnn(observations)  # [B, T, M]
        B, T, M = cnn_out.shape
        size_tensor = torch.full((B, T), size, device=cnn_out.device, dtype=torch.long)
        size_emb = self.size_embed(size_tensor)  # [B, T, 16]
        fused = self.fuse(torch.cat([cnn_out, size_emb], dim=-1))  # [B, T, M]
        return fused


class UnisizeLEWMAux(nn.Module):
    """Size-conditioned LeWM with position auxiliary losses.

    Auxiliary heads operate on the *encoder output* (before projector),
    matching the canonical probe setting.
    """

    def __init__(self, config, max_size: int = 31):
        super().__init__()
        self.config = config
        self.encoder = SizeConditionedEncoder(config, max_size)
        self.embedding_projector = LatentEmbeddingProjector(config)
        self.predictor = NextEmbeddingPredictor(config)

        # Auxiliary heads on encoder output (for probing) - MLP heads for better expressiveness
        cnn_dim = config.effective_model_dim
        self.abs_pos_head = nn.Sequential(
            nn.Linear(cnn_dim, 256), nn.ReLU(), nn.Linear(256, 2)
        )   # (x, y)
        self.rel_pos_head = nn.Sequential(
            nn.Linear(cnn_dim, 256), nn.ReLU(), nn.Linear(256, 2)
        )   # (dx, dy)
        self.goal_pos_head = nn.Sequential(
            nn.Linear(cnn_dim, 256), nn.ReLU(), nn.Linear(256, 2)
        )  # (goal_x, goal_y)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, size: int):
        encoded = self.encoder(observations, size)  # [B, T, M]
        embedding, sigreg_embedding = self.embedding_projector(encoded)
        prediction = self.predictor(embedding, actions)
        target = embedding[:, 1:]

        # Auxiliary predictions from encoded (before projector)
        abs_pos_pred = self.abs_pos_head(encoded)  # [B, T, 2]
        rel_pos_pred = self.rel_pos_head(encoded)  # [B, T, 2]
        goal_pos_pred = self.goal_pos_head(encoded)  # [B, T, 2]

        # Get predictor hidden states for probing
        # predictor hidden: [B, T-1, D] - transformer output before output_projection
        action_condition = self.predictor.action_condition(actions)
        inputs = self.predictor.input_projection(embedding[:, :-1])
        from hdwm.models.lewm import add_temporal_position_embedding
        inputs = add_temporal_position_embedding(inputs, self.predictor.temporal_position_embedding)
        predictor_hidden = self.predictor.transformer(inputs, action_condition)

        return {
            "encoded": encoded,
            "embedding": embedding,
            "sigreg_embedding": sigreg_embedding,
            "prediction": prediction,
            "target": target,
            "abs_pos_pred": abs_pos_pred,
            "rel_pos_pred": rel_pos_pred,
            "goal_pos_pred": goal_pos_pred,
            "predictor_hidden": predictor_hidden,
        }


def compute_position_labels(
    states: torch.Tensor,  # [B, T]
    goal_map: torch.Tensor,  # [B, T, H, W]
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute normalized position labels.

    Returns:
        x, y, dx, dy  each [B, T] in [0, 1] or [-1, 1]

    Coordinate convention:
        - state = y * size + x
        - x = state % size  (column, left-to-right)
        - y = state // size (row, top-to-bottom)
        - dx = (goal_x - agent_x) / (size - 1)
        - dy = (goal_y - agent_y) / (size - 1)
    """
    B, T = states.shape
    # Agent position
    x_raw = states % size
    y_raw = states // size
    x = x_raw.float() / max(size - 1, 1)
    y = y_raw.float() / max(size - 1, 1)

    # Goal position from goal_map (channel 3 of observation)
    # goal_map: [B, T, H, W] -> argmax over flattened spatial dims
    goal_flat = goal_map.reshape(B, T, -1)  # [B, T, H*W]
    goal_state = goal_flat.argmax(dim=-1)  # [B, T]
    goal_x = goal_state % size
    goal_y = goal_state // size

    # Ensure all tensors on same device
    device = goal_map.device
    x_raw = x_raw.to(device)
    y_raw = y_raw.to(device)

    # Relative position
    dx = (goal_x - x_raw).float() / max(size - 1, 1)
    dy = (goal_y - y_raw).float() / max(size - 1, 1)

    return x, y, dx, dy


def eval_on_manifest(
    model: nn.Module,
    manifest_path: str,
    data_config: SequenceDataConfig,
    device: torch.device,
    num_batches: int = 20,
) -> dict:
    """Evaluate on eval manifest."""
    dataset = ManifestSequenceDataset(manifest_path, data_config, seed=42, max_batches=num_batches)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=sequence_batch_collate, num_workers=0)

    losses = {"pred": [], "abs": [], "rel": [], "goal": [], "sigreg": []}
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            obs = batch.observations.to(device)
            actions = batch.actions.to(device)
            size = obs.shape[2]
            batch_size = obs.shape[0]
            seq_len = obs.shape[1]

            output = model(obs, actions, size)
            pred_loss = F.mse_loss(output["prediction"], output["target"]).item()

            # Compute position labels
            x, y, dx, dy = compute_position_labels(batch.states, obs[:, :, :, :, 3], size)
            abs_target = torch.stack([x, y], dim=-1).to(device)  # [B, T, 2]
            rel_target = torch.stack([dx, dy], dim=-1).to(device)  # [B, T, 2]
            abs_loss = F.mse_loss(output["abs_pos_pred"], abs_target).item()
            rel_loss = F.mse_loss(output["rel_pos_pred"], rel_target).item()

            # Goal position loss
            goal_map = obs[:, :, :, :, 3]
            goal_flat = goal_map.reshape(batch_size, seq_len, -1)
            goal_state = goal_flat.argmax(dim=-1)
            goal_x = (goal_state % size).float() / max(size - 1, 1)
            goal_y = (goal_state // size).float() / max(size - 1, 1)
            goal_target = torch.stack([goal_x, goal_y], dim=-1).to(device)
            goal_loss = F.mse_loss(output["goal_pos_pred"], goal_target).item()

            losses["pred"].append(pred_loss)
            losses["abs"].append(abs_loss)
            losses["rel"].append(rel_loss)
            losses["goal"].append(goal_loss)

    model.train()
    return {k: float(np.mean(v)) for k, v in losses.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", default="data/splits/unisize_train_manifest.jsonl")
    p.add_argument("--eval-manifest", default="data/splits/unisize_eval_manifest.jsonl")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--cnn-channels", type=str, default="32,64,128")
    p.add_argument("--lambda-abs", type=float, default=0.1, help="Weight for absolute position aux loss")
    p.add_argument("--lambda-rel", type=float, default=1.0, help="Weight for relative position aux loss")
    p.add_argument("--lambda-goal", type=float, default=0.1, help="Weight for goal position aux loss")
    p.add_argument("--lambda-sigreg", type=float, default=0.09, help="Weight for SIGReg loss")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default="checkpoints/canonical_lewm.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 70)
    print("CANONICAL LEWM TRAINING")
    print("=" * 70)
    print(f"  latent_dim: {args.latent_dim}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  steps: {args.steps}")
    print(f"  lr: {args.lr}")
    print(f"  lambda_abs: {args.lambda_abs}")
    print(f"  lambda_rel: {args.lambda_rel}")
    print(f"  lambda_sigreg: {args.lambda_sigreg}")
    print(f"  train_manifest: {args.train_manifest}")
    print(f"  eval_manifest: {args.eval_manifest}")
    print(f"  device: {device}")
    print()

    # Load manifests
    with open(args.train_manifest) as f:
        train_entries = [json.loads(line) for line in f if line.strip()]
    with open(args.eval_manifest) as f:
        eval_entries = [json.loads(line) for line in f if line.strip()]

    train_sizes = sorted(set(e["maze_size"] for e in train_entries))
    print(f"  Train sizes: {train_sizes}")
    print(f"  Train entries: {len(train_entries)}")
    print(f"  Eval entries: {len(eval_entries)}")
    print()

    # Model config
    base_cfg = ProcgenMazeConfig(
        height=25, width=25, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False)
    cnn_ch = tuple(int(x.strip()) for x in args.cnn_channels.split(","))
    model_cfg = LEWMCNNConfig(
        env_config=base_cfg, latent_dim=args.latent_dim,
        cnn_channels=cnn_ch, latent_batch_norm=True,
        embedding_stage="post_bn", sigreg_stage="post_bn",
        predictor_heads=max(4, args.latent_dim // 16))
    model = UnisizeLEWMAux(model_cfg, max_size=31).to(device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    model.train()

    data_config = SequenceDataConfig(
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        batch_sample_strategy="same_within_batch",
    )

    losses_log = {"total": [], "pred": [], "abs": [], "rel": [], "goal": [], "sigreg": []}
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # Sample from train manifest
        entry = np.random.choice(train_entries)
        sz = entry["maze_size"]
        env_config = ProcgenMazeConfig(
            height=sz, width=sz, observation_channels=5,
            p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
            resample_maze_per_sequence=False, topology_seed=entry["topology_seed"])
        env = ProcgenMazeEnv(env_config, seed=int(np.random.randint(0, 2**31)))
        batch = env.sample_sequence(batch_size=args.batch_size, sequence_length=args.seq_len)
        obs = batch.observations.to(device)
        actions = batch.actions.to(device)
        batch_size = obs.shape[0]
        seq_len = obs.shape[1]

        output = model(obs, actions, sz)

        # LeWM prediction loss
        pred_loss = F.mse_loss(output["prediction"], output["target"])

        # SIGReg loss
        sigreg_loss = sigreg(output["sigreg_embedding"].transpose(0, 1))

        # Position auxiliary losses
        x, y, dx, dy = compute_position_labels(batch.states, obs[:, :, :, :, 3], sz)
        abs_target = torch.stack([x, y], dim=-1).to(device)  # [B, T, 2]
        rel_target = torch.stack([dx, dy], dim=-1).to(device)  # [B, T, 2]
        abs_loss = F.mse_loss(output["abs_pos_pred"], abs_target)
        rel_loss = F.mse_loss(output["rel_pos_pred"], rel_target)

        # Goal position auxiliary loss
        goal_map = obs[:, :, :, :, 3]
        goal_flat = goal_map.reshape(batch_size, seq_len, -1)
        goal_state = goal_flat.argmax(dim=-1)
        goal_x = (goal_state % sz).float() / max(sz - 1, 1)
        goal_y = (goal_state // sz).float() / max(sz - 1, 1)
        goal_target = torch.stack([goal_x, goal_y], dim=-1).to(device)  # [B, T, 2]
        goal_loss = F.mse_loss(output["goal_pos_pred"], goal_target)

        # Total loss
        loss = pred_loss + args.lambda_sigreg * sigreg_loss + args.lambda_abs * abs_loss + args.lambda_rel * rel_loss + args.lambda_goal * goal_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses_log["total"].append(loss.item())
        losses_log["pred"].append(pred_loss.item())
        losses_log["abs"].append(abs_loss.item())
        losses_log["rel"].append(rel_loss.item())
        losses_log["goal"].append(goal_loss.item())
        losses_log["sigreg"].append(sigreg_loss.item())

        if step % args.log_every == 0:
            elapsed = max(time.time() - t0, 0.01)
            avg_total = np.mean(losses_log["total"][-args.log_every:])
            avg_pred = np.mean(losses_log["pred"][-args.log_every:])
            avg_abs = np.mean(losses_log["abs"][-args.log_every:])
            avg_rel = np.mean(losses_log["rel"][-args.log_every:])
            avg_goal = np.mean(losses_log["goal"][-args.log_every:])
            avg_sigreg = np.mean(losses_log["sigreg"][-args.log_every:])
            print(f"  Step {step:>6d}/{args.steps} | "
                  f"total={avg_total:.4f} pred={avg_pred:.4f} abs={avg_abs:.4f} rel={avg_rel:.4f} goal={avg_goal:.4f} sigreg={avg_sigreg:.4f} | "
                  f"{args.log_every/elapsed:.1f} it/s")
            t0 = time.time()

        if step % args.eval_every == 0:
            eval_losses = eval_on_manifest(model, args.eval_manifest, data_config, device, num_batches=20)
            print(f"  >>> EVAL step {step} | pred={eval_losses['pred']:.6f} abs={eval_losses['abs']:.6f} rel={eval_losses['rel']:.6f} goal={eval_losses['goal']:.6f}")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model_cfg,
        "train_sizes": train_sizes,
        "eval_sizes": sorted(set(e["maze_size"] for e in eval_entries)),
        "latent_dim": args.latent_dim,
        "steps": args.steps,
        "lambda_abs": args.lambda_abs,
        "lambda_rel": args.lambda_rel,
        "lambda_goal": args.lambda_goal,
        "lambda_sigreg": args.lambda_sigreg,
        "final_loss": np.mean(losses_log["total"][-500:]),
        "final_pred_loss": np.mean(losses_log["pred"][-500:]),
        "final_abs_loss": np.mean(losses_log["abs"][-500:]),
        "final_rel_loss": np.mean(losses_log["rel"][-500:]),
        "final_goal_loss": np.mean(losses_log["goal"][-500:]),
    }, args.output)
    print(f"\n  Saved: {args.output}")
    print(f"  Final losses: total={np.mean(losses_log['total'][-500:]):.4f} "
          f"pred={np.mean(losses_log['pred'][-500:]):.4f} "
          f"abs={np.mean(losses_log['abs'][-500:]):.4f} "
          f"rel={np.mean(losses_log['rel'][-500:]):.4f} "
          f"goal={np.mean(losses_log['goal'][-500:]):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
