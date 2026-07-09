#!/usr/bin/env python3
"""Hold-out probing: unisize LEWM+aux trained on 13-21, evaluated on 9-25.

For each eval maze size, generates hold-out topology data (seeds beyond training
range), extracts latent embeddings, and fits a ridge regression probe for:
  agent_x, agent_y, goal_x, goal_y, dx (=goal_x-agent_x), dy (=goal_y-agent_y)

Reports acc (cell-exact classification), r (Pearson), spearman_r per size per target.
"""

from __future__ import annotations

import argparse, json, sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_HDWM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDWM_ROOT))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from scripts.eval.eval_latent_l2_cem import load_model_from_ckpt
from scripts.train.train_canonical_lewm import SizeConditionedEncoder


def ridge_probe_classify(X_tr, y_tr, X_te, y_te, n_cls):
    """Ridge regression → classification into n_cls discrete labels. Returns acc,r,spearman_r,mse."""
    lam = 0.001
    I = torch.eye(X_tr.shape[-1]); I[-1, -1] = 0
    Y = torch.nn.functional.one_hot(y_tr.long(), n_cls).float()
    W = torch.linalg.solve(X_tr.T @ X_tr + lam * I, X_tr.T @ Y)
    pred_cls = (X_te @ W).argmax(-1)
    acc = float((pred_cls == y_te.long()).float().mean())
    r = _pearson(pred_cls.float().numpy(), y_te.float().numpy())
    s = _spearman(pred_cls.float().numpy(), y_te.float().numpy())
    mse = float((pred_cls.float() - y_te.float()).pow(2).mean())
    return acc, r, s, mse


def ridge_probe_regress(X_tr, y_tr, X_te, y_te):
    """Ridge regression for continuous targets. Returns r,spearman_r,mse,mae."""
    lam = 0.001
    I = torch.eye(X_tr.shape[-1]); I[-1, -1] = 0
    W = torch.linalg.solve(X_tr.T @ X_tr + lam * I, X_tr.T @ y_tr.float().unsqueeze(1))
    pred = (X_te @ W).squeeze(-1)
    r = _pearson(pred.float().numpy(), y_te.float().numpy())
    s = _spearman(pred.float().numpy(), y_te.float().numpy())
    mse = float((pred - y_te.float()).pow(2).mean())
    mae = float((pred - y_te.float()).abs().mean())
    return r, s, mse, mae


def _pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 2 or x.std() == 0 or y.std() == 0: return 0.0
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float((xm * ym).sum() / denom) if denom > 0 else 0.0


def _spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 2 or x.std() == 0 or y.std() == 0: return 0.0
    return _pearson(x.argsort().argsort().astype(float), y.argsort().argsort().astype(float))


class MLPClassifier(torch.nn.Module):
    """2-layer MLP → n_cls logits (CE loss, AdamW)."""
    def __init__(self, in_dim, n_cls, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_cls))
    def forward(self, x):
        return self.net(x)


class MLPRegressor(torch.nn.Module):
    """2-layer MLP → 1 scalar (MSE loss, AdamW)."""
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(model, X_tr, y_tr, X_val, y_val, lr=1e-3, epochs=30, batch=256):
    device = X_tr.device
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_metric = -1e9; best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(X_tr.shape[0], device=device)
        for i in range(0, X_tr.shape[0], batch):
            idx = perm[i:i+batch]
            logits = model(X_tr[idx])
            loss = torch.nn.functional.cross_entropy(logits, y_tr[idx].long()) if isinstance(model, MLPClassifier) else torch.nn.functional.mse_loss(logits, y_tr[idx].float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_val)
            if isinstance(model, MLPClassifier):
                acc = float((pred.argmax(-1) == y_val.long()).float().mean())
                metric = acc
            else:
                metric = _spearman(pred.cpu().float().numpy(), y_val.cpu().float().numpy())
        if metric > best_metric:
            best_metric = metric; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return best_metric


def generate_holdout_data(model, size, topo_start, n_topos, traj_per_topo, seq_len, device, layer="embedding"):
    """Generate embeddings + labels for hold-out topologies."""
    all_emb, labels = [], {"agent_x": [], "agent_y": [], "goal_x": [], "goal_y": [],
                            "dx": [], "dy": []}
    for ts in range(topo_start, topo_start + n_topos):
        cfg = ProcgenMazeConfig(height=size, width=size, observation_channels=5,
                                p_noise=0, p_noop=0, p_action_turn=0, p_action_stay=0,
                                topology_seed=ts, resample_maze_per_sequence=False)
        env = ProcgenMazeEnv(cfg, seed=ts * 1000)
        goal_pos = env._goal_position
        for ep in range(traj_per_topo):
            env2 = ProcgenMazeEnv(cfg, seed=ts * 10000 + ep)
            batch = env2.sample_sequence(batch_size=1, sequence_length=seq_len)
            obs = batch.observations.to(device)
            states = batch.states[0].numpy()
            with torch.no_grad():
                if isinstance(model.encoder, SizeConditionedEncoder):
                    enc = model.encoder(obs, size)
                else:
                    enc = model.encoder(obs)
                if layer == "encoded":
                    feat = enc.reshape(-1, enc.shape[-1])
                else:
                    emb, _ = model.embedding_projector(enc)
                    feat = emb.reshape(-1, emb.shape[-1])
            all_emb.append(feat.cpu())
            for t in range(seq_len):
                s = int(states[t])
                ax, ay = s % size, s // size
                gx, gy = goal_pos % size, goal_pos // size
                labels["agent_x"].append(float(ax))
                labels["agent_y"].append(float(ay))
                labels["goal_x"].append(float(gx))
                labels["goal_y"].append(float(gy))
                labels["dx"].append(float(gx - ax))
                labels["dy"].append(float(gy - ay))
    emb_all = torch.cat(all_emb, dim=0)
    labs = {k: torch.tensor(v, dtype=torch.float32) for k, v in labels.items()}
    return emb_all, labs


def probe_size(model, size, topo_start, n_topos, traj_per_topo, seq_len, device, probe_type="ridge", layer="embedding"):
    """Run probing for a single maze size. probe_type: ridge | mlp, layer: encoded | embedding"""
    emb, labs = generate_holdout_data(model, size, topo_start, n_topos, traj_per_topo, seq_len, device, layer)
    X_raw = emb.float()
    X_ridge = torch.cat([X_raw, torch.ones(X_raw.shape[0], 1)], -1)
    n = X_raw.shape[0]
    perm = torch.randperm(n)
    split = int(n * 0.8)
    if probe_type == "ridge":
        X_tr, X_te = X_ridge[perm[:split]], X_ridge[perm[split:]]
    else:
        X_tr, X_te = X_raw[perm[:split]], X_raw[perm[split:]]

    results = {}
    for key in ["agent_x", "agent_y", "goal_x", "goal_y"]:
        y = labs[key]
        y_tr, y_te = y[perm[:split]], y[perm[split:]]
        if probe_type == "mlp":
            mlp = MLPClassifier(X_raw.shape[-1], size).to(device)
            train_mlp(mlp, X_tr.to(device), y_tr.to(device), X_te.to(device), y_te.to(device))
            with torch.no_grad():
                pred = mlp(X_te.to(device)).argmax(-1).cpu()
            acc = float((pred == y_te.long()).float().mean())
            r = _pearson(pred.float().numpy(), y_te.float().numpy())
            sp = _spearman(pred.float().numpy(), y_te.float().numpy())
            mse = float((pred.float() - y_te.float()).pow(2).mean())
            results[key] = {"acc": acc, "r": r, "spearman_r": sp, "mse": mse}
        else:
            acc, r, sp, mse = ridge_probe_classify(
                X_tr, y_tr.long(), X_te, y_te.long(), size)
            results[key] = {"acc": acc, "r": r, "spearman_r": sp, "mse": mse}
    for key in ["dx", "dy"]:
        y = labs[key]
        y_tr, y_te = y[perm[:split]], y[perm[split:]]
        if probe_type == "mlp":
            mlp = MLPRegressor(X_raw.shape[-1]).to(device)
            train_mlp(mlp, X_tr.to(device), y_tr.to(device), X_te.to(device), y_te.to(device))
            with torch.no_grad():
                pred = mlp(X_te.to(device)).cpu()
            r = _pearson(pred.float().numpy(), y_te.float().numpy())
            sp = _spearman(pred.float().numpy(), y_te.float().numpy())
            mse = float((pred - y_te.float()).pow(2).mean())
            mae = float((pred - y_te.float()).abs().mean())
            results[key] = {"r": r, "spearman_r": sp, "mse": mse, "mae": mae}
        else:
            r, sp, mse, mae = ridge_probe_regress(X_tr, y_tr, X_te, y_te)
            results[key] = {"r": r, "spearman_r": sp, "mse": mse, "mae": mae}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sizes", default="9,11,13,15,17,19,21,23,25")
    p.add_argument("--topo-start", type=int, default=1000, help="Hold-out seed offset")
    p.add_argument("--n-topos", type=int, default=60)
    p.add_argument("--traj-per-topo", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--probe-type", default="ridge", choices=["ridge", "mlp"])
    p.add_argument("--layer", default="embedding", choices=["encoded", "embedding"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    model, ckpt, mtype = load_model_from_ckpt(args.ckpt, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    output = args.output or f"results/holdout_{args.probe_type}_{args.layer}_{Path(args.ckpt).stem}.csv"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(output, "w")
    csv_file.write("size,target,acc,r,spearman_r,mse,mae\n")

    print(f"Hold-out probe: {Path(args.ckpt).stem} | sizes={sizes} | "
          f"topo_start={args.topo_start} n_topos={args.n_topos}")
    for sz in sizes:
        res = probe_size(model, sz, args.topo_start, args.n_topos,
                         args.traj_per_topo, args.seq_len, device, args.probe_type, args.layer)
        for tgt in ["agent_x", "agent_y", "goal_x", "goal_y", "dx", "dy"]:
            r = res[tgt]
            acc_v = f"{r['acc']:.4f}" if 'acc' in r else ''
            mae_v = f"{r['mae']:.4f}" if 'mae' in r else ''
            line = f"{sz},{tgt},{acc_v},{r['r']:.4f},{r['spearman_r']:.4f},{r['mse']:.4f},{mae_v}"
            csv_file.write(line + "\n")
            acc_s = f"acc={r['acc']:.4f}" if 'acc' in r else ""
            print(f"  size={sz:2d} {tgt:8s} {acc_s} r={r['r']:.4f} sp_r={r['spearman_r']:.4f}")
        csv_file.flush()
    csv_file.close()
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
