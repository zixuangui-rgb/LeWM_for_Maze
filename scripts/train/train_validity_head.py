#!/usr/bin/env python3
"""Train Action Validity Head on frozen LeWM latents.

Labels: for each walkable cell, check each of 5 actions — valid if next_state != current.
Training: multi-label BCE loss on predicted validity logits.

Usage:
    python scripts/train/train_validity_head.py --steps 10000
"""

import argparse, json, sys, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from torch import nn, optim
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.metric_heads.validity_head import ValidityHead
from scripts.train.train_ablation_models import OriginalLeWM
from scripts.train.train_distance_head import pre_extract_maze_latents, LatentPairDataset


def generate_validity_labels(env, cells, num_actions=5):
    """For each cell, compute validity of each action. Returns [N, 5] bool array."""
    labels = np.zeros((len(cells), num_actions), dtype=np.float32)
    for i, cell in enumerate(cells):
        for a in range(num_actions):
            a_enum = env._decode_action(a)
            ns = env._next_state(cell, a_enum)
            labels[i, a] = 1.0 if ns != cell else 0.0  # valid = moves agent
    return labels


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen LeWM
    ckpt = torch.load(config["lewm_ckpt"], map_location=device, weights_only=False)
    model = OriginalLeWM(ckpt["model_config"], max_size=31).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    # Load manifests
    with open(config["train_manifest"]) as f:
        train_entries = [json.loads(l) for l in f if l.strip()]
    with open(config["val_manifest"]) as f:
        val_entries = [json.loads(l) for l in f if l.strip()]
    # Verify topology hold-out
    train_topo = set(e["topology_seed"] for e in train_entries)
    val_topo = set(e["topology_seed"] for e in val_entries)
    assert len(train_topo & val_topo) == 0, f"Topology leakage! Overlap: {len(train_topo & val_topo)}"
    print(f"Train mazes: {len(train_entries)}, Val mazes: {len(val_entries)}, Topology overlap: 0")

    # Create validity dataset (latents + action validity labels)
    # Pre-extract from a subset for efficiency
    print("Pre-extracting latents and validity labels...")
    n_pre = min(500, len(train_entries))  # 500 mazes ~ 24K states
    all_z = []
    all_labels = []
    t0 = time.time()
    for idx in range(n_pre):
        entry = train_entries[idx]
        sz = entry["maze_size"]
        env_cfg = ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
            p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
            resample_maze_per_sequence=False, topology_seed=entry["topology_seed"])
        env = ProcgenMazeEnv(env_cfg, seed=entry.get("level_seed", 42))
        obstacle = env._maze_mask
        walkable = np.flatnonzero(~obstacle.reshape(-1)).tolist()
        labels = generate_validity_labels(env, walkable)

        latents_list = []
        for cell in walkable:
            env._state = cell
            obs, _ = env._observe_with_noise(np.array([cell]))
            t = torch.as_tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                e = model.encoder(t, sz)
                emb, _ = model.embedding_projector(e)
            latents_list.append(emb.squeeze(0).squeeze(0))

        all_z.append(torch.stack(latents_list))
        all_labels.append(torch.tensor(labels, dtype=torch.float32))

        if (idx+1) % 100 == 0:
            print(f"  {idx+1}/{n_pre} ({time.time()-t0:.0f}s)")

    z_data = torch.cat(all_z, dim=0)  # [total_states, D]
    y_data = torch.cat(all_labels, dim=0)  # [total_states, 5]
    print(f"Dataset: {z_data.shape[0]} states, {y_data.sum().item():.0f} valid actions")

    # Also pre-extract val data
    val_z, val_y = [], []
    for idx in range(min(100, len(val_entries))):
        entry = val_entries[idx]; sz = entry["maze_size"]
        env_cfg = ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
            p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
            resample_maze_per_sequence=False, topology_seed=entry["topology_seed"])
        env = ProcgenMazeEnv(env_cfg, seed=entry.get("level_seed",42))
        walkable = np.flatnonzero(~env._maze_mask.reshape(-1)).tolist()
        labels = generate_validity_labels(env, walkable)
        latents_list = []
        for cell in walkable:
            env._state = cell
            obs, _ = env._observe_with_noise(np.array([cell]))
            t = torch.as_tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                e = model.encoder(t, sz); emb, _ = model.embedding_projector(e)
            latents_list.append(emb.squeeze(0).squeeze(0))
        val_z.append(torch.stack(latents_list)); val_y.append(torch.tensor(labels, dtype=torch.float32))
    val_z_t = torch.cat(val_z, dim=0).to(device); val_y_t = torch.cat(val_y, dim=0).to(device)

    # Create head
    head = ValidityHead(latent_dim=128, hidden_dims=[128], num_actions=5).to(device)
    print(f"ValidityHead params: {sum(p.numel() for p in head.parameters()):,}")
    opt = optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=1e-5)
    rng = np.random.default_rng(42)

    # Train
    batch_size = config["batch_size"]
    steps = config["steps"]
    z_data_dev = z_data.to(device); y_data_dev = y_data.to(device)
    n_train = z_data_dev.shape[0]
    losses = []

    print(f"Training {steps} steps...")
    t0 = time.time()
    for step in range(1, steps+1):
        idx = torch.randint(0, n_train, (batch_size,))
        z_b = z_data_dev[idx]; y_b = y_data_dev[idx]
        logits = head(z_b)
        loss = F.binary_cross_entropy_with_logits(logits, y_b)

        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

        if step % 2000 == 0:
            head.eval()
            with torch.no_grad():
                v_logits = head(val_z_t)
                v_loss = F.binary_cross_entropy_with_logits(v_logits, val_y_t).item()
                v_pred = (torch.sigmoid(v_logits) > 0.5)
                v_acc = (v_pred == (val_y_t > 0.5)).float().mean().item()
                # Per-action accuracy
                per_action = []
                for a in range(5):
                    acc_a = (v_pred[:, a] == (val_y_t[:, a] > 0.5)).float().mean().item()
                    per_action.append(acc_a)
            head.train()
            print(f"  Step {step:>5d}: loss={np.mean(losses[-2000:]):.4f}  "
                  f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}  "
                  f"per_action={[f'{x:.3f}' for x in per_action]}  ({time.time()-t0:.0f}s)")
            t0 = time.time()

    # Save
    out = Path("checkpoints/metric_heads/validity_head.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state_dict": head.state_dict(), "config": config,
                "final_loss": float(np.mean(losses[-1000:]))}, out)
    print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    config = {
        "lewm_ckpt": "checkpoints/ablation/original_lewm.pt",
        "train_manifest": "data/splits/fixed11_train_manifest.jsonl",
        "val_manifest": "data/splits/fixed11_val_manifest.jsonl",
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
    }
    train(config)


if __name__ == "__main__":
    main()
