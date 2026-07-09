#!/usr/bin/env python3
"""Systematic comparison: Encoder latent vs Predictor output latent.

Four experiments:
  E1: Encoder latent + vanilla MLP probe
  E2: Predictor output latent + vanilla MLP probe
  E3: Encoder latent + spatial features (improved probing)
  E4: Predictor output latent + improved probing (transformer hidden states)

All experiments:
  - Freeze backbone weights
  - Identical probe architectures and training settings (within each pair)
  - Strict time-step alignment
  - Train on unisize_train_manifest, eval on unisize_eval_manifest (topology hold-out)
  - Report agent_x, agent_y, goal_x, goal_y separately
  - Include symbolic BFS evaluation

Usage:
  python scripts/eval/experiment_encoder_vs_predictor.py \
    --ckpt checkpoints/unisize_dim256.pt --device cuda
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from collections import deque
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_HDWM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDWM_ROOT))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from scripts.train.train_dim256 import Unisize256, SizeCondEnc
from scripts.train.train_dim256_pred_aux import Unisize256PredAux
from scripts.eval.eval_latent_l2_cem import create_env_from_entry

# ──────────────────────────────────────────────────────────────────────────────
# Probe model definitions
# ──────────────────────────────────────────────────────────────────────────────

class VanillaMLP(nn.Module):
    """Standard 4-layer MLP probe: in_dim → 256 → 256 → 256 → n_cls."""
    def __init__(self, in_dim, n_cls, hidden=256, n_layers=4, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpatialMLP(nn.Module):
    """MLP on flattened spatial features: in_dim → 512 → 512 → n_cls."""
    def __init__(self, in_dim, n_cls, hidden=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Latent extraction functions
# ──────────────────────────────────────────────────────────────────────────────

def extract_encoder_latent(model, obs_tensor, sz, device):
    """Extract ENCODED latent (CNN output after pooling): [1, 256]."""
    obs = torch.as_tensor(obs_tensor, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encoder(obs, sz)  # [1, 1, 256]
    return encoded.squeeze(0).squeeze(0)  # [256]


def extract_predictor_latent(model, obs_seq, actions_seq, sz, device):
    """Extract PREDICTOR OUTPUT latent for the LAST step.

    Input:
      obs_seq: [T, H, W, C] numpy array
      actions_seq: [T-1] numpy array of actions taken between frames
      sz: maze size

    Returns:
      pred_out: [256] — predictor output for step T-1 (predicting latent at step T-1+1 = T)
    """
    obs = torch.as_tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, H, W, C]
    actions = torch.as_tensor(actions_seq, dtype=torch.long, device=device).unsqueeze(0)  # [1, T-1]
    with torch.no_grad():
        encoded = model.encoder(obs, sz)  # [1, T, 256]
        embedding, sigreg = model.embedding_projector(encoded)  # [1, T, 256]
        prediction = model.predictor(embedding, actions)  # [1, T-1, 256]
    return prediction.squeeze(0)[-1]  # [256] — last prediction step


def extract_predictor_hidden(model, obs_seq, actions_seq, sz, device):
    """Extract PREDICTOR TRANSFORMER HIDDEN for improved probing.

    Accesses the transformer's output BEFORE the final output_projection.
    """
    obs = torch.as_tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0)
    actions = torch.as_tensor(actions_seq, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encoder(obs, sz)
        embedding, sigreg = model.embedding_projector(encoded)
        # Manually run predictor internals to get hidden state
        action_condition = model.predictor.action_condition(actions)
        inputs = model.predictor.input_projection(embedding[:, :-1])
        from hdwm.models.lewm import add_temporal_position_embedding
        inputs = add_temporal_position_embedding(inputs, model.predictor.temporal_position_embedding)
        hidden = model.predictor.transformer(inputs, action_condition)
    return hidden.squeeze(0)[-1]  # [256]


def extract_spatial_features(model, obs_frame, sz, device):
    """Extract CNN pre-pooling spatial features: [C, H', W']."""
    obs_t = torch.as_tensor(obs_frame, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    cnn = model.encoder.cnn
    x = obs_t.permute(0, 1, 4, 2, 3).reshape(1, obs_t.shape[4], obs_t.shape[2], obs_t.shape[3])
    with torch.no_grad():
        x = cnn.conv(x)
    return x.squeeze(0)  # [C, H', W']


# ──────────────────────────────────────────────────────────────────────────────
# Data generation with strict time-step alignment
# ──────────────────────────────────────────────────────────────────────────────

def generate_data_encoder(model, entries, device, max_frames_per_entry=4):
    """Generate encoder-latent data. Latent at time t → label at time t."""
    latents, labels_ax, labels_ay, labels_gx, labels_gy = [], [], [], [], []
    rng = np.random.default_rng(42)

    for entry in entries:
        sz = entry['maze_size']
        cfg = ProcgenMazeConfig(
            height=sz, width=sz, observation_channels=5,
            p_noise=0, p_noop=0, p_action_turn=0, p_action_stay=0,
            topology_seed=entry['topology_seed'], resample_maze_per_sequence=False)
        env = ProcgenMazeEnv(cfg, seed=int(rng.integers(2**31)))
        gp = env._goal_position
        batch = env.sample_sequence(batch_size=1, sequence_length=8)
        obs = batch.observations  # [1, 8, H, W, C]
        states = batch.states[0].numpy()  # [8]

        # Use frames 0 to T-1 (encoder at time t → label at time t)
        for t in range(8):
            s = int(states[t])
            lat = extract_encoder_latent(model, obs[0, t].cpu().numpy(), sz, device)
            latents.append(lat.cpu())
            labels_ax.append(float(s % sz))
            labels_ay.append(float(s // sz))
            labels_gx.append(float(gp % sz))
            labels_gy.append(float(gp // sz))

    return (torch.stack(latents),
            torch.tensor(labels_ax, dtype=torch.long),
            torch.tensor(labels_ay, dtype=torch.long),
            torch.tensor(labels_gx, dtype=torch.long),
            torch.tensor(labels_gy, dtype=torch.long))


def generate_data_predictor(model, entries, device):
    """Generate predictor-output-latent data with correct time alignment.

    predictor[t] predicts z[t+1], so predictor[t] → label[t+1].
    We pair each predictor output with the NEXT frame's state and goal.
    """
    latents, labels_ax, labels_ay, labels_gx, labels_gy = [], [], [], [], []
    rng = np.random.default_rng(42)

    for entry in entries:
        sz = entry['maze_size']
        cfg = ProcgenMazeConfig(
            height=sz, width=sz, observation_channels=5,
            p_noise=0, p_noop=0, p_action_turn=0, p_action_stay=0,
            topology_seed=entry['topology_seed'], resample_maze_per_sequence=False)
        env = ProcgenMazeEnv(cfg, seed=int(rng.integers(2**31)))
        gp = env._goal_position
        batch = env.sample_sequence(batch_size=1, sequence_length=8)
        obs = batch.observations  # [1, 8, H, W, C]
        actions = batch.actions  # [1, 7]
        states = batch.states[0].numpy()  # [8]

        # Run predictor on full sequence
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
        with torch.no_grad():
            encoded = model.encoder(obs_t, sz)
            embedding, sigreg = model.embedding_projector(encoded)
            prediction = model.predictor(embedding, actions_t)  # [1, 7, 256]

        # prediction[t=0] predicts latent at step 1 → label at step 1
        for t in range(7):
            s = int(states[t + 1])  # next frame's state
            latents.append(prediction[0, t].cpu())
            labels_ax.append(float(s % sz))
            labels_ay.append(float(s // sz))
            labels_gx.append(float(gp % sz))
            labels_gy.append(float(gp // sz))

    return (torch.stack(latents),
            torch.tensor(labels_ax, dtype=torch.long),
            torch.tensor(labels_ay, dtype=torch.long),
            torch.tensor(labels_gx, dtype=torch.long),
            torch.tensor(labels_gy, dtype=torch.long))


def generate_data_predictor_hidden(model, entries, device):
    """Generate predictor hidden-state data (before output_projection)."""
    latents, labels_ax, labels_ay, labels_gx, labels_gy = [], [], [], [], []
    rng = np.random.default_rng(42)

    for entry in entries:
        sz = entry['maze_size']
        cfg = ProcgenMazeConfig(
            height=sz, width=sz, observation_channels=5,
            p_noise=0, p_noop=0, p_action_turn=0, p_action_stay=0,
            topology_seed=entry['topology_seed'], resample_maze_per_sequence=False)
        env = ProcgenMazeEnv(cfg, seed=int(rng.integers(2**31)))
        gp = env._goal_position
        batch = env.sample_sequence(batch_size=1, sequence_length=8)
        obs = batch.observations
        actions = batch.actions
        states = batch.states[0].numpy()

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
        with torch.no_grad():
            encoded = model.encoder(obs_t, sz)
            embedding, sigreg = model.embedding_projector(encoded)
            action_condition = model.predictor.action_condition(actions_t)
            inputs = model.predictor.input_projection(embedding[:, :-1])
            from hdwm.models.lewm import add_temporal_position_embedding
            inputs = add_temporal_position_embedding(inputs, model.predictor.temporal_position_embedding)
            hidden = model.predictor.transformer(inputs, action_condition)  # [1, 7, 256]

        for t in range(7):
            s = int(states[t + 1])
            latents.append(hidden[0, t].cpu())
            labels_ax.append(float(s % sz))
            labels_ay.append(float(s // sz))
            labels_gx.append(float(gp % sz))
            labels_gy.append(float(gp // sz))

    return (torch.stack(latents),
            torch.tensor(labels_ax, dtype=torch.long),
            torch.tensor(labels_ay, dtype=torch.long),
            torch.tensor(labels_gx, dtype=torch.long),
            torch.tensor(labels_gy, dtype=torch.long))


def generate_data_spatial(model, entries, device):
    """Generate spatial features (CNN conv output, pre-pooling)."""
    feat_list, labels_ax, labels_ay, labels_gx, labels_gy = [], [], [], [], []
    rng = np.random.default_rng(42)

    for entry in entries:
        sz = entry['maze_size']
        cfg = ProcgenMazeConfig(
            height=sz, width=sz, observation_channels=5,
            p_noise=0, p_noop=0, p_action_turn=0, p_action_stay=0,
            topology_seed=entry['topology_seed'], resample_maze_per_sequence=False)
        env = ProcgenMazeEnv(cfg, seed=int(rng.integers(2**31)))
        gp = env._goal_position
        batch = env.sample_sequence(batch_size=1, sequence_length=8)
        obs = batch.observations
        states = batch.states[0].numpy()

        for t in range(8):
            s = int(states[t])
            feat = extract_spatial_features(model, obs[0, t].cpu().numpy(), sz, device)
            feat_list.append(feat.cpu())
            labels_ax.append(float(s % sz))
            labels_ay.append(float(s // sz))
            labels_gx.append(float(gp % sz))
            labels_gy.append(float(gp // sz))

    return (torch.stack(feat_list),
            torch.tensor(labels_ax, dtype=torch.long),
            torch.tensor(labels_ay, dtype=torch.long),
            torch.tensor(labels_gx, dtype=torch.long),
            torch.tensor(labels_gy, dtype=torch.long))


# ──────────────────────────────────────────────────────────────────────────────
# Probe training
# ──────────────────────────────────────────────────────────────────────────────

def train_probe(Xtr, ytr, Xval, yval, n_cls, device, epochs=50, lr=1e-3, bs=512,
                weight_decay=1e-4, probe_type='vanilla', hidden=256):
    """Train a probe head and return (model, best_val_acc)."""
    if probe_type == 'spatial':
        m = SpatialMLP(Xtr.shape[1], n_cls).to(device)
    else:
        m = VanillaMLP(Xtr.shape[1], n_cls, hidden=hidden).to(device)

    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc = 0
    best_sd = None

    for ep in range(epochs):
        m.train()
        perm = torch.randperm(Xtr.shape[0])
        total_loss = 0.0
        for i in range(0, Xtr.shape[0], bs):
            idx = perm[i:i+bs]
            logits = m(Xtr[idx].to(device))
            loss = F.cross_entropy(logits, ytr[idx].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()

        m.eval()
        with torch.no_grad():
            pred = m(Xval.to(device)).argmax(-1).cpu()
            acc = float((pred == yval).float().mean())
        if acc > best_acc:
            best_acc = acc
            best_sd = {k: v.cpu().clone() for k, v in m.state_dict().items()}

    if best_sd:
        m.load_state_dict(best_sd)
    return m, best_acc


# ──────────────────────────────────────────────────────────────────────────────
# Eval on held-out manifest
# ──────────────────────────────────────────────────────────────────────────────

def eval_encoder_on_manifest(model, heads, sz, eval_entries, device):
    """Evaluate encoder probes on eval manifest."""
    correct = {t: 0 for t in heads}
    total = 0
    for entry in eval_entries:
        if entry['maze_size'] != sz:
            continue
        env = create_env_from_entry(entry, device)
        eg = env._goal_position
        empty = np.flatnonzero((~env._maze_mask).reshape(-1))
        safe = empty[empty != eg]
        if safe.size < 2:
            continue
        rng = np.random.default_rng(int(42 + total))
        s = int(rng.choice(safe))
        env.reset(seed=0, options={'start_state': s})
        lat = extract_encoder_latent(model, env._last_observation.copy(), sz, device)
        z = lat.unsqueeze(0)  # [1, 256]
        with torch.no_grad():
            for tgt, head in heads.items():
                pred = int(head(z.to(device)).argmax(-1).item())
                true_val = (s % sz) if tgt == 'agent_x' else \
                           (s // sz) if tgt == 'agent_y' else \
                           (eg % sz) if tgt == 'goal_x' else (eg // sz)
                if pred == true_val:
                    correct[tgt] += 1
        total += 1
    return {t: correct[t] / max(total, 1) for t in heads}, total


def eval_predictor_on_manifest(model, heads, sz, eval_entries, device):
    """Evaluate predictor probes on eval manifest.

    Predictor needs 2 frames: observe frame 0, then take action, get prediction for frame 1.
    """
    correct = {t: 0 for t in heads}
    total = 0
    rng = np.random.default_rng(42)

    for entry in eval_entries:
        if entry['maze_size'] != sz:
            continue
        env = create_env_from_entry(entry, device)
        eg = env._goal_position
        empty = np.flatnonzero((~env._maze_mask).reshape(-1))
        safe = empty[empty != eg]
        if safe.size < 2:
            continue
        s = int(rng.choice(safe))
        env.reset(seed=0, options={'start_state': s})
        obs0 = env._last_observation.copy()

        # Take one step to get action and next state
        action = int(rng.integers(1, 5))
        _, _, _, _, info = env.step(action)
        s1 = int(info['state'])

        # Get predictor output
        obs_seq = np.stack([obs0, env._last_observation.copy()], axis=0)  # [2, H, W, C]
        actions_seq = np.array([action])  # [1]
        lat = extract_predictor_latent(model, obs_seq, actions_seq, sz, device)
        z = lat.unsqueeze(0)  # [1, 256]

        with torch.no_grad():
            for tgt, head in heads.items():
                pred = int(head(z.to(device)).argmax(-1).item())
                true_val = (s1 % sz) if tgt == 'agent_x' else \
                           (s1 // sz) if tgt == 'agent_y' else \
                           (eg % sz) if tgt == 'goal_x' else (eg // sz)
                if pred == true_val:
                    correct[tgt] += 1
        total += 1
    return {t: correct[t] / max(total, 1) for t in heads}, total


def eval_spatial_on_manifest(model, heads, sz, eval_entries, device):
    """Evaluate spatial probes on eval manifest."""
    correct = {t: 0 for t in heads}
    total = 0
    for entry in eval_entries:
        if entry['maze_size'] != sz:
            continue
        env = create_env_from_entry(entry, device)
        eg = env._goal_position
        empty = np.flatnonzero((~env._maze_mask).reshape(-1))
        safe = empty[empty != eg]
        if safe.size < 2:
            continue
        rng = np.random.default_rng(int(42 + total))
        s = int(rng.choice(safe))
        env.reset(seed=0, options={'start_state': s})
        feat = extract_spatial_features(model, env._last_observation.copy(), sz, device)
        z = feat.reshape(1, -1)
        with torch.no_grad():
            for tgt, head in heads.items():
                pred = int(head(z.to(device)).argmax(-1).item())
                true_val = (s % sz) if tgt == 'agent_x' else \
                           (s // sz) if tgt == 'agent_y' else \
                           (eg % sz) if tgt == 'goal_x' else (eg // sz)
                if pred == true_val:
                    correct[tgt] += 1
        total += 1
    return {t: correct[t] / max(total, 1) for t in heads}, total


# ──────────────────────────────────────────────────────────────────────────────
# BFS planning
# ──────────────────────────────────────────────────────────────────────────────

def bfs_full_path(occ, sy, sx, gy, gx, sz):
    """BFS on occupancy grid from (sy,sx) to (gy,gx)."""
    H, W = sz, sz
    grid = occ[:H, :W].astype(np.float32)
    if grid[sy, sx] >= 0.5 or grid[gy, gx] >= 0.5:
        return []
    parent = np.full(H * W, -1, np.int32)
    fa = np.full(H * W, -1, np.int32)
    q = deque()
    si = sy * W + sx
    gi = gy * W + gx
    q.append(si)
    parent[si] = si
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
    acts = [1, 2, 3, 4]
    while q:
        cur = q.popleft()
        if cur == gi:
            break
        y, x = divmod(cur, W)
        for d, (dy, dx) in enumerate(dirs):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                ns = ny * W + nx
                if grid[ny, nx] < 0.5 and parent[ns] == -1:
                    parent[ns] = cur
                    q.append(ns)
                    fa[ns] = acts[d]
    if parent[gi] == -1:
        return []
    path = []
    c = gi
    while c != si and parent[c] != -1 and parent[c] != c:
        path.append(int(fa[c]))
        c = parent[c]
    path.reverse()
    return path


def get_latent_for_bfs(model, heads, sz, env, device, latent_type):
    """Get prediction from the appropriate latent type."""
    obs = env._last_observation.copy()
    if latent_type == 'spatial':
        feat = extract_spatial_features(model, obs, sz, device)
        z = feat.reshape(1, -1)
    elif latent_type == 'encoder':
        lat = extract_encoder_latent(model, obs, sz, device)
        z = lat.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported latent_type for BFS: {latent_type}")
    return z


def run_bfs_eval(model, heads, sz, eval_entries, device, latent_type, n_ep=30):
    """BFS evaluation using probes + oracle occupancy.

    Uses oracle occupancy grid but probes to predict agent/goal position.
    """
    rng = np.random.default_rng(42)
    sz_ev = [e for e in eval_entries if e['maze_size'] == sz]
    n_ep = min(n_ep, len(sz_ev))
    sampled = rng.choice(sz_ev, size=n_ep, replace=False)
    succ, total, pos_ok = 0, 0, 0

    for entry in sampled:
        env = create_env_from_entry(entry, device)
        om = env._maze_mask
        eg = env._goal_position
        occ = om.astype(np.float32)
        empty = np.flatnonzero((~om).reshape(-1))
        safe = empty[empty != eg]
        if safe.size < 2:
            continue
        s = int(rng.choice(safe))
        env.reset(seed=0, options={'start_state': s})
        z = get_latent_for_bfs(model, heads, sz, env, device, latent_type)

        with torch.no_grad():
            pax = int(heads['agent_x'](z.to(device)).argmax(-1).item())
            pay = int(heads['agent_y'](z.to(device)).argmax(-1).item())
            pgx = int(heads['goal_x'](z.to(device)).argmax(-1).item())
            pgy = int(heads['goal_y'](z.to(device)).argmax(-1).item())

        true_ax, true_ay = s % sz, s // sz
        true_gx, true_gy = eg % sz, eg // sz
        if pax == true_ax and pay == true_ay and pgx == true_gx and pgy == true_gy:
            pos_ok += 1

        pred_path = bfs_full_path(occ, pay, pax, pgy, pgx, sz)
        env.reset(seed=0, options={'start_state': s})
        cur = s
        ok = False
        for act in (pred_path if pred_path else [int(rng.integers(1, 5))]):
            if cur == eg:
                ok = True
                break
            _, _, _, _, info = env.step(act)
            cur = int(info['state'])
            if cur == eg:
                ok = True
                break
        total += 1
        if ok:
            succ += 1

    return succ / total if total > 0 else 0, pos_ok / total if total > 0 else 0


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(name, model, tr_entries, ev_entries, sizes, device,
                   latent_type, probe_type, epochs=50):
    """Run one complete experiment across all sizes.

    Args:
        latent_type: 'encoder' | 'predictor' | 'predictor_hidden' | 'spatial'
        probe_type: 'vanilla' | 'spatial'
    """
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Latent: {latent_type} | Probe: {probe_type}")
    print(f"{'='*70}")

    results = []

    for sz in sizes:
        print(f"\n--- size={sz} ---")
        t0 = time.time()

        # Filter train entries by size
        sz_tr = [e for e in tr_entries if e['maze_size'] == sz]

        # Generate data
        if latent_type == 'encoder':
            Xall, ax, ay, gx, gy = generate_data_encoder(model, sz_tr, device)
        elif latent_type == 'predictor':
            Xall, ax, ay, gx, gy = generate_data_predictor(model, sz_tr, device)
        elif latent_type == 'predictor_hidden':
            Xall, ax, ay, gx, gy = generate_data_predictor_hidden(model, sz_tr, device)
        elif latent_type == 'spatial':
            Xall, ax, ay, gx, gy = generate_data_spatial(model, sz_tr, device)
        else:
            raise ValueError(f"Unknown latent_type: {latent_type}")

        n = Xall.shape[0]
        # Flatten spatial features
        if latent_type == 'spatial':
            spatial_shape = tuple(Xall.shape[1:])
            Xall = Xall.reshape(n, -1)
        else:
            spatial_shape = Xall.shape[1:]

        # Split: 80% train, 20% val
        n_val = max(1, int(n * 0.2))
        perm = torch.randperm(n)
        Xtr, Xval = Xall[perm[n_val:]], Xall[perm[:n_val]]
        ax_tr, ax_val = ax[perm[n_val:]], ax[perm[:n_val]]
        ay_tr, ay_val = ay[perm[n_val:]], ay[perm[:n_val]]
        gx_tr, gx_val = gx[perm[n_val:]], gx[perm[:n_val]]
        gy_tr, gy_val = gy[perm[n_val:]], gy[perm[:n_val]]

        print(f"  {n} frames (val={n_val}), latent_dim={Xall.shape[1]} ({time.time()-t0:.0f}s)")

        # Train 4 heads
        heads = {}
        val_accs = {}
        for tgt, ytr, yval in [('agent_x', ax_tr, ax_val), ('agent_y', ay_tr, ay_val),
                                ('goal_x', gx_tr, gx_val), ('goal_y', gy_tr, gy_val)]:
            h, acc = train_probe(Xtr, ytr, Xval, yval, sz, device, epochs=epochs,
                                 probe_type=probe_type)
            heads[tgt] = h
            val_accs[tgt] = acc
            print(f"  {tgt}: val_acc={acc:.4f}")

        # Evaluate on held-out manifold
        if latent_type == 'encoder':
            heldout_acc, n_ho = eval_encoder_on_manifest(model, heads, sz, ev_entries, device)
        elif latent_type == 'predictor':
            heldout_acc, n_ho = eval_predictor_on_manifest(model, heads, sz, ev_entries, device)
        elif latent_type == 'predictor_hidden':
            heldout_acc, n_ho = eval_predictor_on_manifest(model, heads, sz, ev_entries, device)
        elif latent_type == 'spatial':
            heldout_acc, n_ho = eval_spatial_on_manifest(model, heads, sz, ev_entries, device)
        else:
            heldout_acc = {}
            n_ho = 0

        tag = "OOD" if sz > 21 else "seen"
        print(f"  Held-out [{tag}]: ax={heldout_acc.get('agent_x',0):.4f} "
              f"ay={heldout_acc.get('agent_y',0):.4f} "
              f"gx={heldout_acc.get('goal_x',0):.4f} "
              f"gy={heldout_acc.get('goal_y',0):.4f} (n={n_ho})")

        # BFS evaluation
        if latent_type in ('encoder', 'spatial'):
            sr, pok = run_bfs_eval(model, heads, sz, ev_entries, device, latent_type)
        else:
            sr, pok = 0, 0  # Predictor BFS needs sequential rollout - harder to eval fairly

        results.append({
            'size': sz,
            'val_ax': val_accs.get('agent_x', 0),
            'val_ay': val_accs.get('agent_y', 0),
            'val_gx': val_accs.get('goal_x', 0),
            'val_gy': val_accs.get('goal_y', 0),
            'ho_ax': heldout_acc.get('agent_x', 0),
            'ho_ay': heldout_acc.get('agent_y', 0),
            'ho_gx': heldout_acc.get('goal_x', 0),
            'ho_gy': heldout_acc.get('goal_y', 0),
            'bfs_sr': sr,
            'bfs_posOK': pok,
            'n_frames': n,
            'latent_dim': Xall.shape[1],
            'tag': tag,
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--train-manifest', default='data/splits/unisize_train_manifest.jsonl')
    p.add_argument('--eval-manifest', default='data/splits/unisize_eval_manifest.jsonl')
    p.add_argument('--sizes', default='9,11,13,15,17,19,21')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--device', default='cuda')
    p.add_argument('--experiments', default='all', help='E1,E2,E3,E4 or all')
    p.add_argument('--output-dir', default='results/encoder_vs_predictor')
    args = p.parse_args()

    device = torch.device(args.device)
    sizes = [int(s.strip()) for s in args.sizes.split(',')]

    # Load data
    with open(args.train_manifest) as f:
        tr_entries = [json.loads(l) for l in f if l.strip()]
    with open(args.eval_manifest) as f:
        ev_entries = [json.loads(l) for l in f if l.strip()]

    # Verify topology hold-out
    train_seeds = set(e['topology_seed'] for e in tr_entries)
    eval_seeds = set(e['topology_seed'] for e in ev_entries)
    overlap = train_seeds & eval_seeds
    print(f"Topology hold-out check: train={len(train_seeds)} eval={len(eval_seeds)} overlap={len(overlap)}")
    if overlap:
        print(f"  WARNING: {len(overlap)} overlapping topology seeds found!")
    else:
        print(f"  ✓ Strict hold-out confirmed.")

    # Load model (auto-detect architecture)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    aux_type = ckpt.get('aux_type', '')
    max_size = ckpt.get('max_size', 31)
    if aux_type == 'predictor_only':
        model = Unisize256PredAux(ckpt['model_config'], max_size=max_size).to(device)
        print(f"Model loaded: Unisize256PredAux (predictor-only aux)")
    else:
        model = Unisize256(ckpt['model_config'], max_size=max_size).to(device)
        print(f"Model loaded: Unisize256 (encoder aux)")
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Model loaded: {ckpt.get('latent_dim')}-dim, {ckpt.get('cnn_channels')} channels")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    exps = set(args.experiments.split(','))
    all_results = {}

    # ──── Experiment 1: Encoder + Vanilla MLP ────
    if 'all' in exps or 'E1' in exps:
        res = run_experiment(
            'E1: Encoder latent + Vanilla MLP', model,
            tr_entries, ev_entries, sizes, device,
            latent_type='encoder', probe_type='vanilla', epochs=args.epochs)
        all_results['E1_encoder_vanilla'] = res

    # ──── Experiment 2: Predictor + Vanilla MLP ────
    if 'all' in exps or 'E2' in exps:
        res = run_experiment(
            'E2: Predictor output + Vanilla MLP', model,
            tr_entries, ev_entries, sizes, device,
            latent_type='predictor', probe_type='vanilla', epochs=args.epochs)
        all_results['E2_predictor_vanilla'] = res

    # ──── Experiment 3: Encoder + Spatial features (improved) ────
    if 'all' in exps or 'E3' in exps:
        res = run_experiment(
            'E3: Encoder spatial features + MLP (improved)', model,
            tr_entries, ev_entries, sizes, device,
            latent_type='spatial', probe_type='spatial', epochs=args.epochs)
        all_results['E3_encoder_spatial'] = res

    # ──── Experiment 4: Predictor hidden + Vanilla MLP (improved) ────
    if 'all' in exps or 'E4' in exps:
        res = run_experiment(
            'E4: Predictor hidden states + MLP (improved)', model,
            tr_entries, ev_entries, sizes, device,
            latent_type='predictor_hidden', probe_type='vanilla', epochs=args.epochs)
        all_results['E4_predictor_hidden'] = res

    # ──── Print summary ────
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")

    for exp_name, res_list in all_results.items():
        print(f"\n── {exp_name} ──")
        print(f"  {'Size':>4s}  {'AgentX':>8s}  {'AgentY':>8s}  {'GoalX':>8s}  {'GoalY':>8s}  {'BFS_SR':>8s}  {'Tag':>6s}")
        print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")
        for r in res_list:
            print(f"  {r['size']:4d}  {r['ho_ax']:8.4f}  {r['ho_ay']:8.4f}  {r['ho_gx']:8.4f}  {r['ho_gy']:8.4f}  {r['bfs_sr']:8.3f}  {r['tag']:>6s}")

    # Save JSON
    result_path = Path(args.output_dir) / 'experiment_results.json'
    with open(result_path, 'w') as f:
        json.dump({k: [{kk: float(vv) if isinstance(vv, (np.floating, float)) else int(vv) if isinstance(vv, (np.integer, int)) else vv for kk, vv in r.items()} for r in v] for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved: {result_path}")


if __name__ == '__main__':
    main()
