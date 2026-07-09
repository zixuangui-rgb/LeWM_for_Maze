#!/usr/bin/env python3
"""Reproduce optimal per-size probing results (target: acc>0.9, r>0.9).

Uses per-size MLP classifiers on ENCODED layer (pre-projector, where aux
losses operate). Trains on many hold-out topologies, validates on held-out
split of the same topology range.

This replicates the earlier probe_holdout success, then extends to eval-manifest
topologies for true hold-out generalization test.
"""

from __future__ import annotations

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

MAX_SZ = 25


class PerSizeMLP(nn.Module):
    """Per-size MLP: in_dim → 256 → 256 → 256 → n_cls (CE loss)."""
    def __init__(self, in_dim, n_cls, hidden=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1)]
        layers.append(nn.Linear(hidden, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def generate_per_size_data(model, sz, n_topos, traj_per_topo, seq_len, device, layer="encoded"):
    """Generate encoded features + labels from hold-out topology seeds."""
    enc_list, ax_list, ay_list, gx_list, gy_list = [], [], [], [], []
    for ts in range(n_topos):
        cfg = ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
                                p_noise=0, p_noop=0, p_action_turn=0, p_action_stay=0,
                                topology_seed=ts, resample_maze_per_sequence=False)
        env = ProcgenMazeEnv(cfg, seed=ts * 1000)
        goal_pos = env._goal_position
        for ep in range(traj_per_topo):
            env2 = ProcgenMazeEnv(cfg, seed=ts * 10000 + ep)
            batch = env2.sample_sequence(batch_size=1, sequence_length=seq_len)
            obs = batch.observations.to(device); states = batch.states[0].numpy()
            with torch.no_grad():
                try:
                    enc = model.encoder(obs, sz)
                except TypeError:
                    enc = model.encoder(obs)
            enc_f = enc.reshape(-1, enc.shape[-1]).cpu()
            enc_list.append(enc_f)
            for t in range(seq_len):
                s = int(states[t])
                ax_list.append(float(s % sz)); ay_list.append(float(s // sz))
                gx_list.append(float(goal_pos % sz)); gy_list.append(float(goal_pos // sz))
    return (torch.cat(enc_list),
            torch.tensor(ax_list, dtype=torch.long),
            torch.tensor(ay_list, dtype=torch.long),
            torch.tensor(gx_list, dtype=torch.long),
            torch.tensor(gy_list, dtype=torch.long))


def train_one_head(Xtr, ytr, Xval, yval, n_cls, device, epochs=50, lr=1e-3, bs=512):
    m = PerSizeMLP(Xtr.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc = 0; best_sd = None
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(Xtr.shape[0])
        total_loss = 0.0
        for i in range(0, Xtr.shape[0], bs):
            idx = perm[i:i+bs]
            logits = m(Xtr[idx].to(device))
            loss = F.cross_entropy(logits, ytr[idx].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()
        m.eval()
        with torch.no_grad():
            pred = m(Xval.to(device)).argmax(-1).cpu()
            acc = float((pred == yval).float().mean())
        if acc > best_acc: best_acc = acc; best_sd = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"    ep {ep+1}/{epochs} loss={total_loss/Xtr.shape[0]:.4f} acc={acc:.4f} best={best_acc:.4f}", flush=True)
    if best_sd: m.load_state_dict(best_sd)
    return m, best_acc


def eval_on_manifest(model, heads, sz, eval_entries, device):
    """Evaluate trained heads on eval-manifest topologies (true hold-out)."""
    correct = {t: 0 for t in heads}; total = 0
    for entry in eval_entries:
        if entry["maze_size"] != sz: continue
        env = create_env_from_entry(entry, device)
        eg = env._goal_position
        empty = np.flatnonzero((~env._maze_mask).reshape(-1))
        safe = empty[empty != eg]
        if safe.size < 2: continue
        rng = np.random.default_rng(int(42 + total))
        s = int(rng.choice(safe))
        env.reset(seed=0, options={"start_state": s})
        obs_t = torch.as_tensor(env._last_observation, dtype=torch.float32,
                                device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            try:
                enc = model.encoder(obs_t, sz)
            except TypeError:
                enc = model.encoder(obs_t)
        z = enc.reshape(1, -1)
        with torch.no_grad():
            for tgt, head in heads.items():
                pred = int(head(z.to(device)).argmax(-1).item())
                true = (s % sz) if tgt in ("agent_x",) else \
                       (s // sz) if tgt == "agent_y" else \
                       (eg % sz) if tgt == "goal_x" else \
                       (eg // sz)
                if pred == true: correct[tgt] += 1
        total += 1
    return {t: correct[t] / max(total, 1) for t in heads}, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval-manifest", default="data/splits/unisize_eval_manifest.jsonl")
    p.add_argument("--sizes", default="9,11,13,15,17,19,21,23,25")
    p.add_argument("--n-topos", type=int, default=80, help="topologies for training")
    p.add_argument("--n-topos-val", type=int, default=20, help="topologies for validation")
    p.add_argument("--traj-per-topo", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="results/phase2_probe/optimal_probe.csv")
    args = p.parse_args()
    device = torch.device(args.device)
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    model, ckpt, mtype = load_model_from_ckpt(args.ckpt, device)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    with open(args.eval_manifest) as f:
        eval_entries_all = [json.loads(l) for l in f if l.strip()]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(args.output, "w")
    csv_f.write("size,val_agent_x,val_agent_y,val_goal_x,val_goal_y,heldout_agent_x,heldout_agent_y,heldout_goal_x,heldout_goal_y,tag\n")

    for sz in sizes:
        print(f"\n{'='*60}\nsize={sz}")
        # ── Generate data from hold-out seeds ──
        t0 = time.time()
        Xall, ax, ay, gx, gy = generate_per_size_data(
            model, sz, args.n_topos + args.n_topos_val,
            args.traj_per_topo, args.seq_len, device, "encoded")
        n = Xall.shape[0]; n_val = args.n_topos_val * args.traj_per_topo * args.seq_len
        print(f"  {n} frames ({time.time()-t0:.0f}s), val={n_val}")

        # Split: last n_val frames = val (from separate seeds)
        Xtr, Xval = Xall[:-n_val], Xall[-n_val:]
        ax_tr, ax_val = ax[:-n_val], ax[-n_val:]
        ay_tr, ay_val = ay[:-n_val], ay[-n_val:]
        gx_tr, gx_val = gx[:-n_val], gx[-n_val:]
        gy_tr, gy_val = gy[:-n_val], gy[-n_val:]

        # ── Train 4 heads ──
        heads = {}
        for tgt, ytr, yval in [("agent_x", ax_tr, ax_val), ("agent_y", ay_tr, ay_val),
                                ("goal_x", gx_tr, gx_val), ("goal_y", gy_tr, gy_val)]:
            print(f"  {tgt}:", flush=True)
            h, acc = train_one_head(Xtr, ytr, Xval, yval, sz, device, args.epochs)
            heads[tgt] = h
            print(f"    val_acc={acc:.4f}")

        # ── True hold-out eval (on eval-manifest topologies) ──
        heldout_acc, n_ho = eval_on_manifest(model, heads, sz, eval_entries_all, device)
        tag = "OOD" if sz > 21 else "seen"
        print(f"  Held-out manifest: agent_x={heldout_acc['agent_x']:.4f} agent_y={heldout_acc['agent_y']:.4f} goal_x={heldout_acc['goal_x']:.4f} goal_y={heldout_acc['goal_y']:.4f} (n={n_ho}) [{tag}]")

        # Get val accuracies
        va = {}
        for tgt, yval in [("agent_x", ax_val), ("agent_y", ay_val),
                          ("goal_x", gx_val), ("goal_y", gy_val)]:
            with torch.no_grad():
                pred = heads[tgt](Xval.to(device)).argmax(-1).cpu()
                va[tgt] = float((pred == yval).float().mean())

        csv_f.write(f"{sz},{va['agent_x']:.4f},{va['agent_y']:.4f},{va['goal_x']:.4f},{va['goal_y']:.4f},"
                    f"{heldout_acc['agent_x']:.4f},{heldout_acc['agent_y']:.4f},{heldout_acc['goal_x']:.4f},{heldout_acc['goal_y']:.4f},{tag}\n")
        csv_f.flush()
        # Save per-size heads
        ckpt_name = Path(args.ckpt).stem
        torch.save({tgt: h.state_dict() for tgt, h in heads.items()},
                   f"checkpoints/heads/{ckpt_name}_persize_sz{sz}.pt")

    csv_f.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
