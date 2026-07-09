#!/usr/bin/env python3
"""Probe using SPATIAL CNN features (pre-pooling) for richer representation.

Instead of 128-dim pooled latent, we extract the CNN feature map before
AdaptiveAvgPool2d, which preserves spatial grid (2×2 to 6×6 × 128 channels
depending on input size). This gives 512-4608 dim features, dramatically
richer than the 128-dim bottleneck.

Usage:
  python scripts/probe/probe_spatial.py --ckpt checkpoints/canonical_lewm_rel1.0.pt \
    --sizes 13,15,17,19,21,23,25 --device cuda
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_HDWM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDWM_ROOT))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from scripts.eval.eval_latent_l2_cem import load_model_from_ckpt, create_env_from_entry
from scripts.train.train_canonical_lewm import SizeConditionedEncoder


def extract_spatial_features(model, obs_frame, sz, device):
    """Extract CNN pre-pooling features [128, H', W'] for a single frame [H, W, C]."""
    obs_t = torch.as_tensor(obs_frame, dtype=torch.float32, device=device)  # [H,W,C]
    x = obs_t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W,C]
    if isinstance(model.encoder, SizeConditionedEncoder):
        cnn = model.encoder.cnn  # CNNEncoder
        x = x.permute(0, 1, 4, 2, 3).reshape(1, x.shape[4], x.shape[2], x.shape[3])
        x = cnn.conv(x)  # [1, 128, H', W']
        return x.squeeze(0)  # [128, H', W']
    else:
        raise RuntimeError("only unisize encoder supported")


def generate_spatial_data(model, sz, n_topos, traj_per_topo, seq_len, device):
    """Generate spatial features + position labels."""
    feat_list, labels = [], {"ax": [], "ay": [], "gx": [], "gy": []}
    for ts in range(n_topos):
        cfg = ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
                                p_noise=0, p_noop=0, p_action_turn=0, p_action_stay=0,
                                topology_seed=ts, resample_maze_per_sequence=False)
        env = ProcgenMazeEnv(cfg, seed=ts * 1000)
        goal_pos = env._goal_position
        for ep in range(traj_per_topo):
            env2 = ProcgenMazeEnv(cfg, seed=ts * 10000 + ep)
            batch = env2.sample_sequence(batch_size=1, sequence_length=seq_len)
            obs = batch.observations; states = batch.states[0].numpy()
            for t in range(seq_len):
                # obs: [1, T, H, W, C] → obs[0, t]: [H, W, C]
                feat = extract_spatial_features(model, obs[0, t].cpu().numpy(), sz, device)
                feat_list.append(feat.cpu())  # [128, H', W']
                s = int(states[t])
                labels["ax"].append(float(s % sz)); labels["ay"].append(float(s // sz))
                labels["gx"].append(float(goal_pos % sz)); labels["gy"].append(float(goal_pos // sz))
    return (torch.stack(feat_list),
            torch.tensor(labels["ax"], dtype=torch.long),
            torch.tensor(labels["ay"], dtype=torch.long),
            torch.tensor(labels["gx"], dtype=torch.long),
            torch.tensor(labels["gy"], dtype=torch.long))


class SpatialMLP(nn.Module):
    """MLP on flattened spatial features (128×H'×W' + size) → n_cls."""
    def __init__(self, in_dim, n_cls, hidden=512, n_layers=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2)]
        layers.append(nn.Linear(hidden, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_head(Xtr, ytr, Xval, yval, n_cls, device, epochs=50, lr=1e-3, bs=512):
    m = SpatialMLP(Xtr.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc = 0; best_sd = None
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], bs):
            idx = perm[i:i+bs]
            loss = F.cross_entropy(m(Xtr[idx].to(device)), ytr[idx].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        scheduler.step()
        m.eval()
        with torch.no_grad():
            pred = m(Xval.to(device)).argmax(-1).cpu()
            acc = float((pred == yval).float().mean())
        if acc > best_acc: best_acc = acc; best_sd = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"    ep {ep+1}/{epochs} acc={acc:.4f} best={best_acc:.4f}", flush=True)
    if best_sd: m.load_state_dict(best_sd)
    return m, best_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sizes", default="13,15,17,19,21,23,25")
    p.add_argument("--n-topos", type=int, default=60)
    p.add_argument("--n-topos-val", type=int, default=15)
    p.add_argument("--traj-per-topo", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="results/phase2_probe/spatial_probe.csv")
    args = p.parse_args()
    device = torch.device(args.device)
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    model, ckpt, mtype = load_model_from_ckpt(args.ckpt, device)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(args.output, "w")
    csv_f.write("size,val_agent_x,val_agent_y,val_goal_x,val_goal_y,spatial_dim,tag\n")

    with open("data/splits/unisize_eval_manifest.jsonl") as f:
        eval_entries = [json.loads(l) for l in f if l.strip()]

    for sz in sizes:
        print(f"\n{'='*60}\nsize={sz}")
        t0 = time.time()
        n_topo = args.n_topos + args.n_topos_val
        X, ax, ay, gx, gy = generate_spatial_data(model, sz, n_topo, args.traj_per_topo, args.seq_len, device)
        n = X.shape[0]; n_val_frames = args.n_topos_val * args.traj_per_topo * args.seq_len
        # Flatten spatial
        X_flat = X.reshape(n, -1)
        Xtr, Xval = X_flat[:-n_val_frames], X_flat[-n_val_frames:]
        print(f"  {n} frames, spatial={tuple(X.shape[1:])} → flattened={X_flat.shape[1]} ({time.time()-t0:.0f}s)")

        heads = {}
        for tgt, y in [("agent_x", ax), ("agent_y", ay), ("goal_x", gx), ("goal_y", gy)]:
            ytr, yval = y[:-n_val_frames], y[-n_val_frames:]
            h, acc = train_head(Xtr, ytr, Xval, yval, sz, device, args.epochs)
            heads[tgt] = h
            print(f"  {tgt}: val_acc={acc:.4f}")

        # Eval on manifest
        correct = {t: 0 for t in heads}; total = 0
        for entry in eval_entries:
            if entry["maze_size"] != sz: continue
            env = create_env_from_entry(entry, device)
            eg = env._goal_position
            empty = np.flatnonzero((~env._maze_mask).reshape(-1)); safe = empty[empty != eg]
            if safe.size < 2: continue
            s = int(np.random.default_rng(42+total).choice(safe))
            env.reset(seed=0, options={"start_state": s})
            feat = extract_spatial_features(model, env._last_observation.copy(), sz, device).cpu()
            z = feat.reshape(1, -1)
            with torch.no_grad():
                for tgt, head in heads.items():
                    pred = int(head(z.to(device)).argmax(-1).item())
                    true_val = (s % sz) if tgt == "agent_x" else (s // sz) if tgt == "agent_y" else (eg % sz) if tgt == "goal_x" else (eg // sz)
                    if pred == true_val: correct[tgt] += 1
            total += 1
        tag = "OOD" if sz > 21 else "seen"
        acc_str = " ".join([f"{t}={correct[t]/max(total,1):.3f}" for t in heads])
        print(f"  Held-out [{tag}]: {acc_str} (n={total})")
        csv_f.write(f"{sz},{correct['agent_x']/max(total,1):.4f},{correct['agent_y']/max(total,1):.4f},{correct['goal_x']/max(total,1):.4f},{correct['goal_y']/max(total,1):.4f},{X_flat.shape[1]},{tag}\n")
        csv_f.flush()
        # Save per-size spatial heads
        ckpt_name = Path(args.ckpt).stem
        torch.save({tgt: h.state_dict() for tgt, h in heads.items()},
                   f"checkpoints/heads/{ckpt_name}_spatial_sz{sz}.pt")

    csv_f.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
