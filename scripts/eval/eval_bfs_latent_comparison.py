#!/usr/bin/env python3
"""Compare 3 latent sources for Symbolic BFS planning:
1. spatial  — CNN pre-pooling output [256, H', W']
2. encoded  — SizeCondEnc output [256] (pre-projector)
3. embedding— LatentEmbeddingProjector output [256] (post-projector, what CEM uses)

Each source → per-size SpatialMLP probes → BFS on oracle occupancy → SR.
"""
import sys, json, time, os, numpy as np, torch, torch.nn.functional as F
from collections import deque
os.chdir('/data2/songxinshuai/yanyh/WORLDMODEL'); sys.path.insert(0, '.')
from torch import nn, optim
from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from scripts.train.train_dim256 import Unisize256

BASE = '/data2/songxinshuai/yanyh/WORLDMODEL'; DEVICE = torch.device('cuda')

# ── Latent extraction ───────────────────────────────────────────────────────

def extract_spatial(model, obs, sz, dev):
    """CNN pre-pooling output: [256, H', W']."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    cnn = model.encoder.cnn
    x = obs_t.permute(0, 1, 4, 2, 3).reshape(1, obs_t.shape[4], obs_t.shape[2], obs_t.shape[3])
    with torch.no_grad(): x = cnn.conv(x)
    return x.squeeze(0)  # [256, H', W']

def extract_encoded(model, obs, sz, dev):
    """SizeCondEnc output: [256] (pre-projector, post size-fusion)."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    with torch.no_grad(): encoded = model.encoder(obs_t, sz)
    return encoded.squeeze(0).squeeze(0)  # [256]

def extract_embedding(model, obs, sz, dev):
    """LatentEmbeddingProjector output: [256] (post-projector, what CEM uses)."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encoder(obs_t, sz)
        embedding, _ = model.embedding_projector(encoded)
    return embedding.squeeze(0).squeeze(0)  # [256]

EXTRACTORS = {
    'spatial':   extract_spatial,
    'encoded':   extract_encoded,
    'embedding': extract_embedding,
}

# ── BFS ──────────────────────────────────────────────────────────────────────

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

# ── Probe training ───────────────────────────────────────────────────────────

class SpatialMLP(nn.Module):
    def __init__(self, in_dim, n_cls, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_cls))
    def forward(self, x): return self.net(x)

def create_env(entry):
    sz = entry['maze_size']
    return ProcgenMazeEnv(ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry['topology_seed']),
        seed=entry.get('env_seed', 42))

def train_probes(model, train_entries, sz, extract_fn, device, epochs=30):
    """Train 4 per-size probes (agent_x, agent_y, goal_x, goal_y) from given latent source."""
    feat_list = []; labs = {'ax': [], 'ay': [], 'gx': [], 'gy': []}
    sz_tr = [e for e in train_entries if e['maze_size'] == sz]
    rng = np.random.default_rng(42)

    for entry in sz_tr:
        env = create_env(entry); gp = env._goal_position
        for _ in range(2):
            batch = env.sample_sequence(batch_size=1, sequence_length=8)
            obs = batch.observations; st = batch.states[0].numpy()
            for t in range(8):
                feat = extract_fn(model, obs[0, t].cpu().numpy(), sz, device)
                # Normalize spatial features for consistent dimensionality
                if feat.ndim == 3:
                    feat = F.adaptive_avg_pool2d(feat.unsqueeze(0), (2, 2)).squeeze(0)
                feat_list.append(feat.cpu().reshape(-1))
                s = int(st[t])
                labs['ax'].append(float(s % sz)); labs['ay'].append(float(s // sz))
                labs['gx'].append(float(gp % sz)); labs['gy'].append(float(gp // sz))

    X = torch.stack(feat_list); n = X.shape[0]; Xf = X.reshape(n, -1)
    nv = max(1, int(len(sz_tr) * 0.2 * 2 * 8))
    perm = torch.randperm(n); Xt, Xv = Xf[perm[nv:]], Xf[perm[:nv]]
    print(f'    {n} frames ({Xf.shape[1]} dim), val={nv}', flush=True)

    heads = {}
    for tgt, lab in [('agent_x', 'ax'), ('agent_y', 'ay'), ('goal_x', 'gx'), ('goal_y', 'gy')]:
        yt = torch.tensor(labs[lab], dtype=torch.long)
        m = SpatialMLP(Xt.shape[1], sz).to(device)
        opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best_acc = 0; best_sd = None
        for ep in range(epochs):
            m.train(); perm2 = torch.randperm(Xt.shape[0])
            for i in range(0, Xt.shape[0], 256):
                idx = perm2[i:i+256]
                loss = F.cross_entropy(m(Xt[idx].to(device)), yt[perm[nv:]][idx].to(device))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            sch.step()
            m.eval()
            with torch.no_grad():
                acc = float((m(Xv.to(device)).argmax(-1).cpu() == yt[perm[:nv]]).float().mean())
            if acc > best_acc: best_acc = acc; best_sd = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        if best_sd: m.load_state_dict(best_sd)
        heads[tgt] = m
    return heads

def evaluate_bfs(model, heads, eval_entries, sz, extract_fn, device, max_ep=30):
    """BFS evaluation using trained probes."""
    sz_ev = [e for e in eval_entries if e['maze_size'] == sz]
    n_ep = min(max_ep, len(sz_ev))
    rng = np.random.default_rng(42)
    sampled = rng.choice(sz_ev, size=n_ep, replace=False)
    succ, total, pos_ok = 0, 0, 0

    for entry in sampled:
        env = create_env(entry); om = env._maze_mask; eg = env._goal_position
        occ = om.astype(np.float32); empty = np.flatnonzero((~om).reshape(-1))
        safe = empty[empty != eg]
        if safe.size < 2: continue
        s = int(rng.choice(safe))
        env.reset(seed=0, options={'start_state': s})
        feat = extract_fn(model, env._last_observation.copy(), sz, device)
        if feat.ndim == 3:
            feat = F.adaptive_avg_pool2d(feat.unsqueeze(0), (2, 2)).squeeze(0)
        z = feat.reshape(1, -1)
        with torch.no_grad():
            pax = int(heads['agent_x'](z.to(device)).argmax(-1).item())
            pay = int(heads['agent_y'](z.to(device)).argmax(-1).item())
            pgx = int(heads['goal_x'](z.to(device)).argmax(-1).item())
            pgy = int(heads['goal_y'](z.to(device)).argmax(-1).item())
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
    pok = pos_ok / max(total, 1)
    return {'sr': float(sr), 'posOK': float(pok), 'n': total}

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = DEVICE
    ckpt = torch.load(f'{BASE}/checkpoints/unisize_dim256.pt', map_location=device, weights_only=False)
    model = Unisize256(ckpt['model_config'], max_size=31).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    for p in model.parameters(): p.requires_grad = False

    for split_name, train_m, eval_m, eval_sizes in [
        ('Set A: Size 11', f'{BASE}/data/splits/fixed11_train_manifest.jsonl', f'{BASE}/data/splits/fixed11_test_manifest.jsonl', [11]),
        ('Set B: Multi-size', f'{BASE}/data/splits/unisize_train_manifest.jsonl', f'{BASE}/data/splits/unisize_eval_manifest.jsonl', [9, 11, 13, 15, 17, 19, 21]),
    ]:
        with open(train_m) as f: train_entries = [json.loads(l) for l in f if l.strip()]
        with open(eval_m) as f: eval_entries = [json.loads(l) for l in f if l.strip()]

        print(f'\n{"="*70}')
        print(f'{split_name}  |  Train: {len(train_entries)}  Eval: {len(eval_entries)}')
        print(f'{"="*70}')

        all_results = {}
        for src_name, extract_fn in EXTRACTORS.items():
            print(f'\n  ── Latent: {src_name} ──')
            t0 = time.time()
            src_results = {}
            for sz in eval_sizes:
                print(f'    sz={sz}:', end=' ', flush=True)
                heads = train_probes(model, train_entries, sz, extract_fn, device, epochs=30)
                res = evaluate_bfs(model, heads, eval_entries, sz, extract_fn, device)
                src_results[f'sz{sz}'] = res
                print(f'SR={res["sr"]:.4f} posOK={res["posOK"]:.4f} ({res["n"]} ep)', flush=True)
            elapsed = time.time() - t0
            # Aggregate
            total_s = sum(r['sr'] * r['n'] for r in src_results.values())
            total_n = sum(r['n'] for r in src_results.values())
            src_results['overall'] = {'sr': float(total_s / max(total_n, 1)), 'total_ep': total_n}
            all_results[src_name] = src_results
            print(f'    Overall SR={src_results["overall"]["sr"]:.4f}  ({elapsed:.0f}s)')

        # Print comparison table
        print(f'\n  {"="*60}')
        print(f'  COMPARISON: {split_name}')
        print(f'  {"Latent":<15s} {"SR":>8s} {"posOK":>8s} {"vs Spatial":>10s}')
        print(f'  {"-"*45}')
        base_sr = all_results['spatial']['overall']['sr']
        for src_name in ['spatial', 'encoded', 'embedding']:
            sr = all_results[src_name]['overall']['sr']
            delta = f'+{sr-base_sr:.1%}' if sr >= base_sr else f'{sr-base_sr:.1%}'
            print(f'  {src_name:<15s} {sr:>8.4f} {"—":>8s} {delta:>10s}')

        # Per-size detail
        if len(eval_sizes) > 1:
            print(f'\n  {"Size":<8s}', end='')
            for src in ['spatial', 'encoded', 'embedding']: print(f'{src:>12s}', end='')
            print()
            for sz in eval_sizes:
                print(f'  {sz:<8d}', end='')
                for src in ['spatial', 'encoded', 'embedding']:
                    print(f'{all_results[src][f"sz{sz}"]["sr"]:>12.4f}', end='')
                print()

        # Save
        out_dir = f'{BASE}/results/latent_comparison'
        os.makedirs(out_dir, exist_ok=True)
        fname = f'{out_dir}/{split_name.lower().replace(": ","_").replace(" ","_")}.json'
        with open(fname, 'w') as f: json.dump(all_results, f, indent=2)
        print(f'  Saved: {fname}')

if __name__ == '__main__':
    main()
