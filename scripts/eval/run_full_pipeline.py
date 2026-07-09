#!/usr/bin/env python3
"""Comprehensive pipeline: train metric heads + evaluate all methods on both splits.

Set A (size11): fixed11_train (320) / fixed11_val (80) / fixed11_test (100)
Set B (multisize): unisize_train (2800, sz 9-21) / unisize_eval (900, sz 9-25)

Methods: BFS, L2 CEM, DistanceHead CEM, GCRL CEM, QRL CEM, BC, RL(sparse/dense)
"""

import argparse, json, sys, time, os, csv, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict, deque
from torch import nn, optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdwm.config import ProcgenMazeConfig, LEWMCNNConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.planning import _bfs_shortest_path, cem_plan
from hdwm.metric_heads.distance_head import DistanceHead
from hdwm.metric_heads.gcrl_head import GCRLHead
from hdwm.metric_heads.qrl_head import QRLHead
from scripts.train.train_dim256 import Unisize256, SizeCondEnc

# ══ Inline utilities (avoid importing eval_full_bfs_correct.py which has module-level side effects) ══

def extract_spatial(model, obs, sz, dev):
    """Extract CNN spatial features (pre-pooling output)."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    cnn = model.encoder.cnn
    x = obs_t.permute(0, 1, 4, 2, 3).reshape(1, obs_t.shape[4], obs_t.shape[2], obs_t.shape[3])
    with torch.no_grad():
        x = cnn.conv(x)
    return x.squeeze(0)

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

CEM_CFG = dict(horizon=12, num_candidates=64, cem_iters=1, receding_horizon=1, history_size=3)
DEVICE = None  # set in main

# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_backbone():
    global DEVICE
    ckpt = torch.load('checkpoints/unisize_dim256.pt', map_location=DEVICE, weights_only=False)
    model = Unisize256(ckpt['model_config'], max_size=31).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model

def encode_obs(model, obs, sz):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encoder(obs_t, sz)
        embedding, _ = model.embedding_projector(encoded)
    return embedding

def create_env(entry):
    sz = entry['maze_size']
    return ProcgenMazeEnv(ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry['topology_seed']),
        seed=entry.get('env_seed', 42))

# ═══════════════════════════════════════════════════════════════════════════════
# Latent extraction & metric head training
# ═══════════════════════════════════════════════════════════════════════════════

def pre_extract_maze_latents(model, entry):
    """Extract latents for all walkable cells in a maze."""
    sz = entry['maze_size']
    env = create_env(entry)
    obstacle = env._maze_mask
    empty_mask = ~obstacle
    walkable = np.flatnonzero(empty_mask.reshape(-1)).tolist()
    goal_pos = env._goal_position
    latents_list = []
    for cell in walkable:
        env._state = cell
        obs, _ = env._observe_with_noise(np.array([cell]))
        obs = obs[0]
        env._last_observation = obs
        z = encode_obs(model, obs, sz).squeeze(0).squeeze(0)
        latents_list.append(z)
    latents = torch.stack(latents_list, dim=0)
    width = env.config.width
    n = len(walkable)
    bfs_cache = np.full((n, n), -1, dtype=np.int32)
    for i in range(n):
        for j in range(i, n):
            d = _bfs_shortest_path(obstacle, walkable[i], walkable[j], width)
            if d is not None:
                bfs_cache[i, j] = d; bfs_cache[j, i] = d
    return latents, walkable, bfs_cache

class LatentPairDataset:
    def __init__(self, model, entries):
        self.model = model; self.entries = entries; self._cache = {}
    def get_maze(self, idx):
        if idx not in self._cache:
            self._cache[idx] = pre_extract_maze_latents(self.model, self.entries[idx])
        return self._cache[idx]
    def sample_batch(self, batch_size, rng, max_dist=121):
        z1_l, z2_l, labels_l = [], [], []
        for _ in range(4):
            mi = int(rng.integers(0, len(self.entries)))
            latents, cells, bfs = self.get_maze(mi)
            n = len(cells)
            for _ in range(max(1, batch_size // 4)):
                i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
                if i == j: continue
                d = bfs[i, j]; d = d if d >= 0 else max_dist
                z1_l.append(latents[i]); z2_l.append(latents[j]); labels_l.append(float(d))
                if len(z1_l) >= batch_size: break
            if len(z1_l) >= batch_size: break
        return (torch.stack(z1_l[:batch_size]), torch.stack(z2_l[:batch_size]),
                torch.tensor(labels_l[:batch_size], dtype=torch.float32, device=DEVICE))

def train_distance_head(model, train_entries, val_entries, save_path, steps=10000):
    print('  Training DistanceHead...')
    train_ds = LatentPairDataset(model, train_entries)
    val_ds = LatentPairDataset(model, val_entries)
    head = DistanceHead(latent_dim=256, hidden_dims=[256,128], input_mode='concat').to(DEVICE)
    opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(42)
    for step in range(1, steps + 1):
        head.train()
        z1, z2, labels = train_ds.sample_batch(256, rng)
        loss = F.mse_loss(head(z1, z2), labels)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        scheduler.step()
        if step % 2000 == 0:
            head.eval()
            with torch.no_grad():
                z1v, z2v, lv = val_ds.sample_batch(512, rng)
                pred_v = head(z1v, z2v)
                val_mse = F.mse_loss(pred_v, lv).item()
                from scipy.stats import spearmanr
                sp, _ = spearmanr(pred_v.cpu().numpy(), lv.cpu().numpy())
            print(f'    Step {step:>5d}: loss={loss.item():.4f} val_mse={val_mse:.4f} spearman={sp:.4f}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'head_state_dict': head.state_dict(), 'config': {'latent_dim': 256, 'hidden_dims': [256,128], 'input_mode': 'concat'}}, save_path)
    print(f'    Saved: {save_path}')
    return head

def train_gcrl_head(model, train_entries, val_entries, save_path, steps=10000):
    print('  Training GCRL head...')
    train_ds = LatentPairDataset(model, train_entries)
    val_ds = LatentPairDataset(model, val_entries)
    horizons = [1, 2, 4, 8, 16, 32, 64]
    head = GCRLHead(latent_dim=256, hidden_dims=[256,128], horizons=horizons).to(DEVICE)
    opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.default_rng(42)
    for step in range(1, steps + 1):
        head.train()
        z1, z2, bfs_labels = train_ds.sample_batch(256, rng)
        h_indices = torch.randint(0, len(horizons), (z1.shape[0],), device=DEVICE)
        h_vals = torch.tensor([horizons[i] for i in h_indices.tolist()], dtype=torch.float32, device=DEVICE)
        targets = (bfs_labels <= h_vals).float()
        logits = head(z1, z2, h_indices)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        if step % 2000 == 0:
            head.eval()
            with torch.no_grad():
                z1v, z2v, bfs_v = val_ds.sample_batch(512, rng)
                hi = torch.randint(0, len(horizons), (z1v.shape[0],), device=DEVICE)
                hv = torch.tensor([horizons[i] for i in hi.tolist()], device=DEVICE)
                targets_v = (bfs_v <= hv).float()
                logits_v = head(z1v, z2v, hi)
                val_acc = ((torch.sigmoid(logits_v) > 0.5) == (targets_v > 0.5)).float().mean().item()
            print(f'    Step {step:>5d}: loss={loss.item():.4f} val_acc={val_acc:.4f}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'head_state_dict': head.state_dict(), 'head_type': 'gcrl', 'config': {'latent_dim': 256}}, save_path)
    print(f'    Saved: {save_path}')
    return head

def train_qrl_head(model, train_entries, val_entries, save_path, steps=10000):
    print('  Training QRL head...')
    train_ds = LatentPairDataset(model, train_entries)
    val_ds = LatentPairDataset(model, val_entries)
    head = QRLHead(latent_dim=256, hidden_dims=[256,128]).to(DEVICE)
    opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.default_rng(42)
    for step in range(1, steps + 1):
        head.train()
        z1, z2, bfs_labels = train_ds.sample_batch(256, rng)
        pred = head(z1, z2)
        mse_loss = F.mse_loss(pred, bfs_labels)
        bsz = min(64, 256 // 4)
        za, _, _ = train_ds.sample_batch(bsz, rng)
        zb, _, _ = train_ds.sample_batch(bsz, rng)
        zc, _, _ = train_ds.sample_batch(bsz, rng)
        min_sz = min(za.shape[0], zb.shape[0], zc.shape[0])
        tri_loss = head.triangle_loss(za[:min_sz], zb[:min_sz], zc[:min_sz])
        loss = mse_loss + 0.1 * tri_loss
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        if step % 2000 == 0:
            head.eval()
            with torch.no_grad():
                z1v, z2v, bfs_v = val_ds.sample_batch(512, rng)
                pred_v = head(z1v, z2v)
                val_mse = F.mse_loss(pred_v, bfs_v).item()
            print(f'    Step {step:>5d}: loss={loss.item():.4f} val_mse={val_mse:.4f}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'head_state_dict': head.state_dict(), 'head_type': 'qrl', 'config': {'latent_dim': 256}}, save_path)
    print(f'    Saved: {save_path}')
    return head

# ═══════════════════════════════════════════════════════════════════════════════
# Sampling helpers
# ═══════════════════════════════════════════════════════════════════════════════

def sample_start_goal(env, rng, min_path=3):
    obs_mask = env._maze_mask; empty = ~obs_mask
    flat = empty.reshape(-1).copy()
    if hasattr(env, '_goal_position'): flat[env._goal_position] = False
    pos = np.flatnonzero(flat); w = env.config.width
    for _ in range(500):
        s = int(rng.choice(pos)); g = int(rng.choice(pos))
        if s == g: continue
        d = _bfs_shortest_path(obs_mask, s, g, w)
        if d is not None and d >= min_path: return s, g, d
    raise RuntimeError('no start-goal pair found')

# ═══════════════════════════════════════════════════════════════════════════════
# CEM evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_cem_episode(model, env, start, goal, sz, score_fn, seed):
    na = env.config.action_vocab_size
    elites = max(CEM_CFG['num_candidates'] // 8, 8)
    env.reset(seed=seed, options={'start_state': start})
    start_emb = encode_obs(model, env._last_observation, sz)
    ctx_emb = start_emb.repeat(1, CEM_CFG['history_size'], 1)
    ctx_act = torch.full((1, CEM_CFG['history_size']), 0, dtype=torch.long, device=DEVICE)  # STAY padding
    env.reset(seed=seed, options={'start_state': goal})
    goal_emb = encode_obs(model, env._last_observation, sz)
    env.reset(seed=seed, options={'start_state': start})
    cur = start; succ = False; inv = 0; stuck = 0; last = cur; path_len = 0
    for step in range(128):
        if cur == goal: succ = True; break
        best_seq, _, _ = cem_plan(model, ctx_emb, ctx_act, goal_emb,
            horizon=CEM_CFG['horizon'], history_size=CEM_CFG['history_size'],
            num_candidates=CEM_CFG['num_candidates'], num_elites=elites,
            cem_iters=CEM_CFG['cem_iters'], momentum=0.1, num_actions=na,
            device=DEVICE, seed=seed * 10000 + step, score_fn=score_fn)
        a = int(best_seq[0]); prev = cur
        obs, _, _, _, info = env.step(a); cur = int(info['state'])
        path_len += 1
        if cur == prev and a != 0: inv += 1
        if cur == last: stuck += 1
        last = cur
        new_emb = encode_obs(model, obs, sz)
        ctx_emb = torch.cat([ctx_emb[:, 1:], new_emb], dim=1)
        ctx_act = torch.cat([ctx_act[:, 1:], torch.tensor([[a]], dtype=torch.long, device=DEVICE)], dim=1)
        if cur == goal: succ = True; break
    fbfs = None
    if not succ:
        d = _bfs_shortest_path(env._maze_mask, cur, goal, env.config.width)
        fbfs = d if d is not None else -1
    return dict(success=succ, path_length=path_len, invalid_actions=inv, stuck_steps=stuck, final_bfs_distance=fbfs)

# ═══════════════════════════════════════════════════════════════════════════════
# BFS evaluation (same as eval_full_bfs_correct)
# ═══════════════════════════════════════════════════════════════════════════════

def bfs_full_path(occ, sy, sx, gy, gx, sz):
    H, W = sz, sz; grid = occ[:H, :W].astype(np.float32)
    if grid[sy, sx] >= 0.5 or grid[gy, gx] >= 0.5: return []
    parent = np.full(H * W, -1, np.int32); fa = np.full(H * W, -1, np.int32)
    q = deque(); si = sy * W + sx; gi = gy * W + gx; q.append(si); parent[si] = si
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]; acts = [1, 2, 3, 4]
    while q:
        cur = q.popleft()
        if cur == gi: break
        y, x = divmod(cur, W)
        for d, (dy, dx) in enumerate(dirs):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                ns = ny * W + nx
                if grid[ny, nx] < 0.5 and parent[ns] == -1: parent[ns] = cur; q.append(ns); fa[ns] = acts[d]
    if parent[gi] == -1: return []
    path = []; c = gi
    while c != si and parent[c] != -1 and parent[c] != c: path.append(int(fa[c])); c = parent[c]
    path.reverse(); return path

class SpatialMLP(nn.Module):
    def __init__(self, in_dim, n_cls, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_cls))
    def forward(self, x): return self.net(x)

def evaluate_bfs(model, train_entries, eval_entries, output_dir):
    """Symbolic BFS planner with oracle occupancy."""
    print('\n' + '=' * 60)
    print('SYMBOLIC BFS PLANNER')
    print('=' * 60)
    rng = np.random.default_rng(42)
    sizes = sorted(set(e['maze_size'] for e in eval_entries))
    all_results = {}
    for sz in sizes:
        sz_tr = [e for e in train_entries if e['maze_size'] == sz]
        if not sz_tr: continue
        # Train probes on train entries
        feat_list, labs = [], {'ax': [], 'ay': [], 'gx': [], 'gy': []}
        for entry in sz_tr:
            env = create_env(entry); gp = env._goal_position
            for _ in range(2):
                batch = env.sample_sequence(batch_size=1, sequence_length=8)
                obs = batch.observations; st = batch.states[0].numpy()
                for t in range(8):
                    feat = extract_spatial(model, obs[0, t].cpu().numpy(), sz, DEVICE)
                    feat_list.append(feat.cpu())
                    s = int(st[t])
                    labs['ax'].append(float(s % sz)); labs['ay'].append(float(s // sz))
                    labs['gx'].append(float(gp % sz)); labs['gy'].append(float(gp // sz))
        X = torch.stack(feat_list); n = X.shape[0]; Xf = X.reshape(n, -1)
        nv = max(1, int(len(sz_tr) * 0.2 * 2 * 8))
        perm = torch.randperm(n); Xt, Xv = Xf[perm[nv:]], Xf[perm[:nv]]
        print(f'  sz={sz}: {n} frames from {len(sz_tr)} entries, val={nv}')
        heads = {}
        for tgt, lab in [('agent_x', 'ax'), ('agent_y', 'ay'), ('goal_x', 'gx'), ('goal_y', 'gy')]:
            yt = torch.tensor(labs[lab], dtype=torch.long)
            m = SpatialMLP(Xt.shape[1], sz).to(DEVICE)
            opt_t = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            best_acc = 0; best_sd = None
            for ep in range(20):
                m.train(); perm2 = torch.randperm(Xt.shape[0])
                for i in range(0, Xt.shape[0], 256):
                    idx = perm2[i:i+256]
                    loss = F.cross_entropy(m(Xt[idx].to(DEVICE)), yt[perm[nv:]][idx].to(DEVICE))
                    opt_t.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt_t.step()
                m.eval()
                with torch.no_grad():
                    acc = float((m(Xv.to(DEVICE)).argmax(-1).cpu() == yt[perm[:nv]]).float().mean())
                if acc > best_acc: best_acc = acc; best_sd = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            if best_sd: m.load_state_dict(best_sd)
            heads[tgt] = m
        # Eval
        sz_ev = [e for e in eval_entries if e['maze_size'] == sz]
        n_ep = min(30, len(sz_ev)); sampled = rng.choice(sz_ev, size=n_ep, replace=False)
        succ, total, pos_ok = 0, 0, 0
        for entry in sampled:
            env = create_env(entry); om = env._maze_mask; eg = env._goal_position
            occ = om.astype(np.float32); empty = np.flatnonzero((~om).reshape(-1))
            safe = empty[empty != eg]
            if safe.size < 2: continue
            s = int(rng.choice(safe))
            env.reset(seed=0, options={'start_state': s})
            feat = extract_spatial(model, env._last_observation.copy(), sz, DEVICE)
            z = feat.reshape(1, -1)
            with torch.no_grad():
                pax = int(heads['agent_x'](z.to(DEVICE)).argmax(-1).item())
                pay = int(heads['agent_y'](z.to(DEVICE)).argmax(-1).item())
                pgx = int(heads['goal_x'](z.to(DEVICE)).argmax(-1).item())
                pgy = int(heads['goal_y'](z.to(DEVICE)).argmax(-1).item())
            true_ax, true_ay = s % sz, s // sz
            true_gx, true_gy = eg % sz, eg // sz
            if pax == true_ax and pay == true_ay and pgx == true_gx and pgy == true_gy: pos_ok += 1
            pred_path = bfs_full_path(occ, pay, pax, pgy, pgx, sz)
            env.reset(seed=0, options={'start_state': s}); cur = s; ok = False
            for act in (pred_path if pred_path else [1]):
                if cur == eg: ok = True; break
                _, _, _, _, info = env.step(act); cur = int(info['state'])
                if cur == eg: ok = True; break
            total += 1
            if ok: succ += 1
        sr = succ / max(total, 1)
        all_results[f'sz{sz}'] = {'sr': float(sr), 'posOK': float(pos_ok / max(total, 1)), 'n': total}
        print(f'  sz={sz}: SR={sr:.4f} posOK={pos_ok/max(total,1):.4f} ({succ}/{total})')
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/bfs_results.json', 'w') as f: json.dump(all_results, f, indent=2)
    return all_results

# ═══════════════════════════════════════════════════════════════════════════════
# CEM Planning evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_cem_variants(model, eval_entries, heads, output_dir, num_ep=100):
    """Evaluate L2 CEM + all metric head variants."""
    print('\n' + '=' * 60)
    print('CEM PLANNING EVALUATION')
    print('=' * 60)
    rng = np.random.default_rng(42)

    # Pre-sample episodes (same for all variants)
    eps = []
    for i in range(num_ep):
        for attempt in range(100):
            entry = rng.choice(eval_entries)  # RANDOM sample across ALL sizes
            env = create_env(entry)
            try:
                s, g, opt = sample_start_goal(env, rng)
                eps.append(dict(entry=entry, start=s, goal=g, opt=opt, sz=entry['maze_size']))
                break
            except RuntimeError: continue
    print(f'  Pre-sampled {len(eps)} episodes')

    # Define score fns
    def make_gcrl_score(head):
        def fn(terminal, goal):
            h_idx = head.get_horizon_idx(CEM_CFG['horizon'])
            logits = head(terminal, goal, h_idx)
            return -logits  # lower logit = less reachable = higher cost
        return fn

    def make_qrl_score(head):
        def fn(terminal, goal): return head(terminal, goal)
        return fn

    def make_dh_score(head):
        def fn(terminal, goal): return head(terminal, goal)
        return fn

    def l2_score(terminal, goal):
        return F.mse_loss(terminal, goal, reduction='none').sum(dim=-1)

    def make_l2_dh_score(head):
        def fn(terminal, goal):
            l2 = F.mse_loss(terminal, goal, reduction='none').sum(dim=-1)
            return 0.5 * l2 + 0.5 * head(terminal, goal)
        return fn

    variants = [('L2', l2_score)]
    if heads.get('distance'):
        variants += [('DistanceHead', make_dh_score(heads['distance'])),
                     ('L2+DistanceHead', make_l2_dh_score(heads['distance']))]
    if heads.get('gcrl'):
        variants.append(('GCRL', make_gcrl_score(heads['gcrl'])))
    if heads.get('qrl'):
        variants.append(('QRL', make_qrl_score(heads['qrl'])))

    all_res = {}
    for vname, score_fn in variants:
        print(f'\n  [{vname}]')
        ep_res = []; t0 = time.time()
        for i, ed in enumerate(eps):
            env = create_env(ed['entry']); seed = 42 * 10000 + i
            r = run_cem_episode(model, env, ed['start'], ed['goal'], ed['sz'], score_fn, seed)
            r['op_len'] = ed['opt']
            r['spl'] = ed['opt'] / max(r['path_length'], ed['opt']) if r['success'] else 0.0
            ep_res.append(r)
            if (i + 1) % 25 == 0:
                srs = [e['success'] for e in ep_res]
                print(f'    Ep {i+1:>3d}: SR={np.mean(srs):.4f}')
        succ = sum(1 for e in ep_res if e['success'])
        sr = succ / len(ep_res)
        spl = np.mean([e['spl'] for e in ep_res])
        fails = [e for e in ep_res if not e['success']]
        a_bfs = np.mean([e['final_bfs_distance'] for e in fails if e['final_bfs_distance'] is not None]) if fails else 0
        all_steps = max(sum(e['path_length'] for e in ep_res), 1)
        stuck_r = sum(e['stuck_steps'] for e in ep_res) / all_steps
        inv_r = sum(e['invalid_actions'] for e in ep_res) / all_steps
        all_res[vname] = dict(sr=float(sr), spl=float(spl),
            avg_path_success=float(np.mean([e['path_length'] for e in ep_res if e['success']]) if succ > 0 else 0),
            avg_final_bfs=float(a_bfs), stuck_rate=float(stuck_r), invalid_rate=float(inv_r),
            num_success=int(succ), num_failure=len(ep_res) - int(succ), time=float(time.time() - t0))
        print(f'    SR={sr:.4f} SPL={spl:.4f} stuck={stuck_r:.4f} inv={inv_r:.4f} ({time.time()-t0:.0f}s)')

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/cem_results.json', 'w') as f: json.dump(all_res, f, indent=2)

    # Print table
    print(f"\n{'Method':<20s} {'SR':>7s} {'SPL':>7s} {'Stuck':>7s} {'Invalid':>8s} {'S/F':>8s}")
    print('-' * 60)
    for vn, r in all_res.items():
        print(f"{vn:<20s} {r['sr']:>7.4f} {r['spl']:>7.4f} {r['stuck_rate']:>7.4f} {r['invalid_rate']:>8.4f} {r['num_success']:>3d}/{r['num_failure']:>3d}")
    return all_res

# ═══════════════════════════════════════════════════════════════════════════════
# BC baseline
# ═══════════════════════════════════════════════════════════════════════════════

def bfs_optimal_action(occ, sy, sx, gy, gx, sz):
    H, W = sz, sz; grid = occ[:H, :W].astype(np.float32)
    if grid[sy, sx] >= 0.5 or grid[gy, gx] >= 0.5: return 0
    if sy == gy and sx == gx: return 0
    parent = np.full(H * W, -1, np.int32); first_act = np.full(H * W, -1, np.int32)
    si, gi = sy * W + sx, gy * W + gx; q = deque(); q.append(si); parent[si] = si
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]; acts = [1, 2, 3, 4]
    while q:
        cur = q.popleft()
        if cur == gi: break
        y, x = divmod(cur, W)
        for d, (dy, dx) in enumerate(dirs):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                ns = ny * W + nx
                if grid[ny, nx] < 0.5 and parent[ns] == -1: parent[ns] = cur; q.append(ns)
                first_act[ns] = acts[d] if cur == si else first_act[cur]
    if parent[gi] == -1: return 0
    return int(first_act[gi]) if first_act[gi] >= 0 else 0

def evaluate_bc(model, train_entries, eval_entries, output_dir):
    """Behaviour Cloning from BFS expert."""
    print('\n' + '=' * 60)
    print('BEHAVIOUR CLONING BASELINE')
    print('=' * 60)
    rng = np.random.default_rng(42)

    # Generate training data
    print('  Generating BC training data from BFS expert...')
    t0 = time.time()
    all_feats, all_actions = [], []
    for entry in train_entries:
        sz = entry['maze_size']
        env = create_env(entry)
        occ = env._maze_mask; goal = env._goal_position
        gy, gx = divmod(goal, sz)
        empty = np.flatnonzero((~occ).reshape(-1)); safe = empty[empty != goal]
        if len(safe) < 2: continue
        states = rng.choice(safe, size=min(30, len(safe)), replace=False)
        for s in states:
            sy, sx = divmod(int(s), sz)
            act = bfs_optimal_action(occ, sy, sx, gy, gx, sz)
            env._state = int(s)
            obs_np, _ = env._observe_with_noise(np.array([int(s)]))
            feat = extract_spatial(model, obs_np[0], sz, DEVICE)
            # Normalize spatial dims via adaptive pooling for stack compatibility
            feat = F.adaptive_avg_pool2d(feat.unsqueeze(0), (2, 2)).squeeze(0)
            all_feats.append(feat.cpu()); all_actions.append(act)

    Xf = torch.stack(all_feats).reshape(len(all_feats), -1)
    y = torch.tensor(all_actions, dtype=torch.long)
    n = Xf.shape[0]; nv = max(1, int(n * 0.2))
    perm = torch.randperm(n); Xt, Xv = Xf[perm[nv:]], Xf[perm[:nv]]
    yt, yv = y[perm[nv:]], y[perm[:nv]]
    print(f'  {n} training samples, val={nv}, time={time.time()-t0:.0f}s')
    print(f'  Action dist: STAY={sum(all_actions).__class__}')  # placeholder

    # Train policy head
    policy_head = nn.Sequential(
        nn.Linear(Xf.shape[1], 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 5)
    ).to(DEVICE)
    opt = optim.AdamW(policy_head.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, 30)
    best_acc = 0; best_sd = None
    for ep in range(30):
        policy_head.train(); perm2 = torch.randperm(Xt.shape[0])
        for i in range(0, Xt.shape[0], 256):
            idx = perm2[i:i+256]
            loss = F.cross_entropy(policy_head(Xt[idx].to(DEVICE)), yt[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy_head.parameters(), 1.0); opt.step()
        sch.step()
        policy_head.eval()
        with torch.no_grad():
            acc = float((policy_head(Xv.to(DEVICE)).argmax(-1).cpu() == yv).float().mean())
        if acc > best_acc: best_acc = acc; best_sd = {k: v.cpu().clone() for k, v in policy_head.state_dict().items()}
        if (ep + 1) % 10 == 0: print(f'  BC ep {ep+1}/30: val_acc={acc:.4f} best={best_acc:.4f}')
    if best_sd: policy_head.load_state_dict(best_sd)
    print(f'  Best val accuracy: {best_acc:.4f}')

    # Save policy
    os.makedirs(os.path.dirname(f'{output_dir}/../checkpoints'), exist_ok=True)
    torch.save({'head_state_dict': policy_head.state_dict(), 'val_acc': best_acc}, f'{output_dir}/bc_policy.pt')

    # Evaluate
    print('  Evaluating BC policy...')
    per_size = defaultdict(lambda: {'succ': 0, 'total': 0, 'path_lens': []})
    for entry in eval_entries:
        sz = entry['maze_size']
        if per_size[sz]['total'] >= 30: continue
        env = create_env(entry)
        occ = env._maze_mask; goal = env._goal_position
        empty = np.flatnonzero((~occ).reshape(-1)); safe = empty[empty != goal]
        if len(safe) < 2: continue
        s = int(rng.choice(safe))
        env.reset(seed=0, options={'start_state': s}); cur = s; succ = False; plen = 0
        for step in range(128):
            if cur == goal: succ = True; break
            obs = env._last_observation.copy()
            feat = extract_spatial(model, obs, sz, DEVICE); z = feat.reshape(1, -1)
            with torch.no_grad():
                act = int(policy_head(z.to(DEVICE)).argmax(-1).item())
            _, _, _, _, info = env.step(act); cur = int(info['state']); plen += 1
        per_size[sz]['total'] += 1
        if succ: per_size[sz]['succ'] += 1; per_size[sz]['path_lens'].append(plen)

    results = {}
    for sz in sorted(per_size.keys()):
        d = per_size[sz]
        results[f'sz{sz}'] = {'sr': float(d['succ'] / max(d['total'], 1)),
                              'avg_path': float(np.mean(d['path_lens'])) if d['path_lens'] else 0.0, 'n': d['total']}
    total_s = sum(d['succ'] for d in per_size.values())
    total_e = sum(d['total'] for d in per_size.values())
    results['overall'] = {'sr': float(total_s / max(total_e, 1)), 'total_ep': total_e}

    with open(f'{output_dir}/bc_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"  BC Overall SR: {results['overall']['sr']:.4f} ({total_s}/{total_e})")
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# RL baseline (PPO)
# ═══════════════════════════════════════════════════════════════════════════════

class RLPolicy(nn.Module):
    """CNN spatial features → actor (action logits) + critic (value)."""
    def __init__(self, backbone, spatial_dim, n_actions=5, hidden=512):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters(): p.requires_grad = False
        shared = [nn.Linear(spatial_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2)]
        self.shared = nn.Sequential(*shared)
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def get_spatial(self, obs, sz):
        if isinstance(obs, np.ndarray): obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.unsqueeze(0).unsqueeze(0).to(next(self.backbone.parameters()).device)
        cnn = self.backbone.encoder.cnn
        x = obs.permute(0, 1, 4, 2, 3).reshape(-1, obs.shape[4], obs.shape[2], obs.shape[3])
        with torch.no_grad(): x = cnn.conv(x)
        return x.reshape(x.shape[0], -1)  # flatten spatial dims

    def forward(self, obs, sz):
        feat = self.get_spatial(obs, sz)
        h = self.shared(feat)
        return self.actor(h), self.critic(h)

def compute_bfs_reward(cur_state, goal, env, reward_type='sparse'):
    """Compute reward.
    sparse: +1 if goal, 0 otherwise
    dense: BFS distance change based reward
    """
    if cur_state == goal:
        return 10.0 if reward_type == 'dense' else 1.0
    if reward_type == 'sparse':
        return 0.0
    else:  # dense
        # We need BFS distance - use oracle occupancy
        w = env.config.width
        d = _bfs_shortest_path(env._maze_mask, cur_state, goal, w)
        return 0.1 / max(d, 1) if d is not None else -0.1

def evaluate_rl(model, train_entries, eval_entries, output_dir, reward_type='sparse'):
    """Train RL policy with PPO, evaluate on eval entries."""
    print(f'\n{"="*60}')
    print(f'RL BASELINE ({reward_type.upper()} REWARD)')
    print(f'{"="*60}')
    rng = np.random.default_rng(42)

    # Determine spatial feature dim
    test_entry = train_entries[0]; sz = test_entry['maze_size']
    env = create_env(test_entry)
    env.reset(seed=0, options={'start_state': int(np.flatnonzero((~env._maze_mask).reshape(-1))[0])})
    feat = extract_spatial(model, env._last_observation, sz, DEVICE)
    spatial_dim = feat.reshape(1, -1).shape[1]

    policy = RLPolicy(model, spatial_dim, n_actions=5).to(DEVICE)
    # Only train head (backbone frozen)
    opt = optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-5)

    print(f'  Spatial dim: {spatial_dim}, trainable params: {sum(p.numel() for p in policy.parameters() if p.requires_grad)}')
    print(f'  Training PPO for 3000 steps...')

    # Simple PPO loop
    gamma = 0.99; eps_clip = 0.2; K_epochs = 4
    log_data = {'loss': [], 'reward': []}
    for step in range(3000):
        entry = rng.choice(train_entries)
        sz = entry['maze_size']
        env = create_env(entry)
        occ = env._maze_mask; goal = env._goal_position
        empty = np.flatnonzero((~occ).reshape(-1)); safe = empty[empty != goal]
        if len(safe) < 2: continue
        s = int(rng.choice(safe))
        env.reset(seed=int(rng.integers(2**31)), options={'start_state': s})

        # Collect trajectory
        obs_list, act_list, rew_list, logp_list, val_list = [], [], [], [], []
        cur = s; done = False
        for t in range(32):
            if done: break
            obs = env._last_observation.copy()
            logits, value = policy(obs, sz)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            _, _, term, trunc, info = env.step(int(action.item()))
            cur = int(info['state']); done = term or trunc
            r = 1.0 if (reward_type == 'sparse' and cur == goal) else \
                (10.0 if (reward_type == 'dense' and cur == goal) else 0.0)

            obs_list.append(obs); act_list.append(action); rew_list.append(r)
            logp_list.append(log_prob); val_list.append(value)

        if len(obs_list) < 2: continue

        # Compute returns and advantages
        returns, advantages = [], []
        R = 0
        for r in reversed(rew_list):
            R = r + gamma * R; returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        values = torch.cat(val_list)
        advantages = (returns - values.detach()).unsqueeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(K_epochs):
            batch_logits, batch_values = [], []
            for obs_i in obs_list:
                lo, va = policy(obs_i, sz)
                batch_logits.append(lo); batch_values.append(va)
            batch_logits = torch.cat([l for l in batch_logits]).reshape(len(obs_list), -1)
            batch_values = torch.cat([v for v in batch_values]).squeeze(-1)

            new_probs = torch.softmax(batch_logits, dim=-1)
            new_dist = torch.distributions.Categorical(new_probs)
            new_logp = new_dist.log_prob(torch.cat(act_list))
            old_logp = torch.cat(logp_list)

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * advantages.squeeze(-1)
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages.squeeze(-1)
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(batch_values, returns)
            loss = policy_loss + 0.5 * value_loss

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
            opt.step()

        log_data['loss'].append(loss.item()); log_data['reward'].append(sum(rew_list))
        if (step + 1) % 500 == 0:
            avg_l = np.mean(log_data['loss'][-500:]); avg_r = np.mean(log_data['reward'][-500:])
            print(f'    Step {step+1}: avg_loss={avg_l:.4f} avg_reward={avg_r:.4f}')

    # Save policy
    os.makedirs(os.path.dirname(f'{output_dir}/../checkpoints'), exist_ok=True)
    torch.save({'policy_state_dict': {k: v.cpu() for k, v in policy.state_dict().items() if 'backbone' not in k},
                'reward_type': reward_type}, f'{output_dir}/rl_{reward_type}_policy.pt')

    # Evaluate
    print(f'  Evaluating RL policy...')
    per_size = defaultdict(lambda: {'succ': 0, 'total': 0, 'path_lens': []})
    for entry in eval_entries:
        sz = entry['maze_size']
        if per_size[sz]['total'] >= 30: continue
        env = create_env(entry)
        occ = env._maze_mask; goal = env._goal_position
        empty = np.flatnonzero((~occ).reshape(-1)); safe = empty[empty != goal]
        if len(safe) < 2: continue
        s = int(rng.choice(safe))
        env.reset(seed=0, options={'start_state': s}); cur = s; succ = False; plen = 0
        for step in range(128):
            if cur == goal: succ = True; break
            obs = env._last_observation.copy()
            with torch.no_grad():
                logits, _ = policy(obs, sz)
                act = int(logits.argmax(-1).item())
            _, _, _, _, info = env.step(act); cur = int(info['state']); plen += 1
        per_size[sz]['total'] += 1
        if succ: per_size[sz]['succ'] += 1; per_size[sz]['path_lens'].append(plen)

    results = {}
    for sz in sorted(per_size.keys()):
        d = per_size[sz]
        results[f'sz{sz}'] = {'sr': float(d['succ'] / max(d['total'], 1)),
                              'avg_path': float(np.mean(d['path_lens'])) if d['path_lens'] else 0.0, 'n': d['total']}
    total_s = sum(d['succ'] for d in per_size.values())
    total_e = sum(d['total'] for d in per_size.values())
    results['overall'] = {'sr': float(total_s / max(total_e, 1)), 'total_ep': total_e}

    with open(f'{output_dir}/rl_{reward_type}_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"  RL ({reward_type}) Overall SR: {results['overall']['sr']:.4f} ({total_s}/{total_e})")
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_set(name, train_manifest, eval_manifest, output_dir, train_heads=True):
    """Run all evaluations for one experiment set."""
    global DEVICE
    DEVICE = torch.device('cuda')
    print(f'\n{"#"*70}')
    print(f'# EXPERIMENT SET: {name}')
    print(f'# Train: {train_manifest}')
    print(f'# Eval:  {eval_manifest}')
    print(f'# Output: {output_dir}')
    print(f'{"#"*70}')

    os.makedirs(output_dir, exist_ok=True)

    # Load backbone
    model = load_backbone()
    print(f'Backbone loaded: 256-dim, GPU={next(model.parameters()).device}')

    # Load data
    with open(train_manifest) as f: train_entries = [json.loads(l) for l in f if l.strip()]
    with open(eval_manifest) as f: eval_entries = [json.loads(l) for l in f if l.strip()]
    print(f'Train: {len(train_entries)} entries, sizes={sorted(set(e["maze_size"] for e in train_entries))}')
    print(f'Eval:  {len(eval_entries)} entries, sizes={sorted(set(e["maze_size"] for e in eval_entries))}')

    # Verify split
    train_topo = set((e['maze_size'], e['topology_seed']) for e in train_entries)
    eval_topo = set((e['maze_size'], e['topology_seed']) for e in eval_entries)
    overlap = train_topo & eval_topo
    if overlap:
        print(f'⚠ WARNING: Same-size topology overlap: {len(overlap)}')
    else:
        print(f'✓ Data split: strict hold-out (0 overlap)')

    # ── Method 1: BFS ──
    bf = evaluate_bfs(model, train_entries, eval_entries, output_dir)

    # ── Train/load metric heads ──
    heads = {}
    head_dir = 'checkpoints/metric_heads'
    os.makedirs(head_dir, exist_ok=True)

    if train_heads:
        # For metric head training, use a validation split from train entries
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(train_entries))
        n_val = max(1, len(train_entries) // 5)
        tr_idx = perm[n_val:]; val_idx = perm[:n_val]
        tr_entries = [train_entries[i] for i in tr_idx]
        val_entries = [train_entries[i] for i in val_idx]
        print(f'\nMetric head training: {len(tr_entries)} train / {len(val_entries)} val')

        suffix = name.lower().replace(' ', '_')
        heads['distance'] = train_distance_head(model, tr_entries, val_entries,
                                                 f'{head_dir}/distance_head_{suffix}.pt', steps=10000)
        heads['gcrl'] = train_gcrl_head(model, tr_entries, val_entries,
                                         f'{head_dir}/gcrl_head_{suffix}.pt', steps=10000)
        heads['qrl'] = train_qrl_head(model, tr_entries, val_entries,
                                       f'{head_dir}/qrl_head_{suffix}.pt', steps=10000)
    else:
        # Load existing
        print('\nLoading pre-trained metric heads...')
        for hname in ['distance', 'gcrl', 'qrl']:
            hp = f'{head_dir}/{hname}_head.pt'
            if os.path.exists(hp):
                c = torch.load(hp, map_location=DEVICE, weights_only=False)
                if hname == 'distance':
                    head = DistanceHead(latent_dim=c['config']['latent_dim'],
                                        hidden_dims=c['config']['hidden_dims'],
                                        input_mode=c['config']['input_mode']).to(DEVICE)
                elif hname == 'gcrl':
                    head = GCRLHead(latent_dim=c['config']['latent_dim'],
                                    hidden_dims=[256,128]).to(DEVICE)
                else:
                    head = QRLHead(latent_dim=c['config']['latent_dim'],
                                   hidden_dims=[256,128]).to(DEVICE)
                head.load_state_dict(c['head_state_dict'])
                head.eval()
                heads[hname] = head
                print(f'  Loaded {hname} from {hp}')

    # ── Method 2-6: CEM variants ──
    cem = evaluate_cem_variants(model, eval_entries, heads, output_dir, num_ep=100)

    # ── Method 7: BC ──
    bc = evaluate_bc(model, train_entries, eval_entries, output_dir)

    # ── Method 8-9: RL ──
    rl_s = evaluate_rl(model, train_entries, eval_entries, output_dir, 'sparse')
    rl_d = evaluate_rl(model, train_entries, eval_entries, output_dir, 'dense')

    # ── Summary ──
    print(f'\n{"="*70}')
    print(f'SUMMARY: {name}')
    print(f'{"="*70}')
    # Compute overall BFS SR across all sizes
    bf_overall_sr = 0.0
    bf_total = 0
    for k, v in bf.items():
        if k.startswith('sz'):
            bf_overall_sr += v['sr'] * v['n']
            bf_total += v['n']
    if bf_total > 0:
        bf_overall_sr /= bf_total
    all_methods = {'BFS': {'sr': bf_overall_sr},
                   **{k: v for k, v in cem.items()},
                   'BC': bc['overall'],
                   'RL-sparse': rl_s['overall'],
                   'RL-dense': rl_d['overall']}
    for mn, mr in all_methods.items():
        sr = mr.get('sr', 0) if isinstance(mr, dict) else 0
        print(f'  {mn:<20s} SR={sr:.4f}')

    with open(f'{output_dir}/summary.json', 'w') as f: json.dump({
        'set': name, 'train_manifest': train_manifest, 'eval_manifest': eval_manifest,
        'methods': {k: {'sr': v.get('sr', 0)} if isinstance(v, dict) else {'sr': 0} for k, v in all_methods.items()}
    }, f, indent=2)
    print(f'\nSaved summary to {output_dir}/summary.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--set', choices=['size11', 'multisize', 'all'], default='all')
    p.add_argument('--skip-heads', action='store_true', help='Skip metric head training')
    args = p.parse_args()

    # Clean old results
    for d in ['results/set_a_size11', 'results/set_b_multisize']:
        os.makedirs(d, exist_ok=True)

    if args.set in ('size11', 'all'):
        run_set('Set A: Size 11',
                'data/splits/fixed11_train_manifest.jsonl',
                'data/splits/fixed11_test_manifest.jsonl',
                'results/set_a_size11',
                train_heads=not args.skip_heads)

    if args.set in ('multisize', 'all'):
        run_set('Set B: Multi-size',
                'data/splits/unisize_train_manifest.jsonl',
                'data/splits/unisize_eval_manifest.jsonl',
                'results/set_b_multisize',
                train_heads=not args.skip_heads)


if __name__ == '__main__':
    main()
