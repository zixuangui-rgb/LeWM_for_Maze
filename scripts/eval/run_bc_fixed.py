#!/usr/bin/env python3
"""Fixed BC baseline: Conv2d head on spatial features + all walkable cells as training data."""
import sys, json, os, time, numpy as np, torch, torch.nn.functional as F
from collections import deque, Counter, defaultdict
os.chdir('/data2/songxinshuai/yanyh/WORLDMODEL')
sys.path.insert(0, '.')
from torch import nn, optim
from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.planning import _bfs_shortest_path

BASE = '/data2/songxinshuai/yanyh/WORLDMODEL'
DEVICE = torch.device('cuda')

def extract_spatial_2d(model, obs, sz, dev):
    """Extract CNN spatial features as [C, H, W]."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    cnn = model.encoder.cnn
    x = obs_t.permute(0, 1, 4, 2, 3).reshape(1, obs_t.shape[4], obs_t.shape[2], obs_t.shape[3])
    with torch.no_grad(): x = cnn.conv(x)
    return x.squeeze(0)  # [256, H', W']

def bfs_optimal_action(grid, sy, sx, gy, gx, sz):
    H, W = sz, sz
    if grid[sy, sx] or grid[gy, gx]: return 0
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
                if not grid[ny, nx] and parent[ns] == -1:
                    parent[ns] = cur; q.append(ns)
                    first_act[ns] = acts[d] if cur == si else first_act[cur]
    if parent[gi] == -1: return 0
    return int(first_act[gi]) if first_act[gi] >= 0 else 0

class ConvBCPolicy(nn.Module):
    """Conv2d head on spatial features: adaptive pool → Conv → MLP → 5 actions."""
    def __init__(self, in_channels=256, n_actions=5, hidden=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 64, 1), nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64 * 2 * 2, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x):
        # x: [B, 256, H, W] — H,W can vary
        x = self.pool(x)  # → [B, 256, 2, 2]
        h = self.conv_head(x)  # → [B, 64, 2, 2]
        h = h.reshape(h.shape[0], -1)  # → [B, 256]
        return self.mlp(h)

def create_env(entry):
    sz = entry['maze_size']
    return ProcgenMazeEnv(ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry['topology_seed']),
        seed=entry.get('env_seed', 42))

def run_bc(model, train_entries, eval_entries, output_dir):
    print(f'\n{"="*50}\nBC (Conv2d head, all cells)\n{"="*50}')
    rng = np.random.default_rng(42)
    t0 = time.time()

    # Generate training data: ALL walkable cells from all train mazes
    all_feats_2d, all_actions = [], []
    for entry in train_entries:
        sz = entry['maze_size']; env = create_env(entry)
        grid = env._maze_mask; goal = env._goal_position
        gy, gx = divmod(goal, sz)
        empty = np.flatnonzero((~grid).reshape(-1)); safe = empty[empty != goal]
        if len(safe) < 2: continue
        for s in safe:
            sy, sx = divmod(int(s), sz)
            act = bfs_optimal_action(grid, sy, sx, gy, gx, sz)
            env._state = int(s)
            obs_np, _ = env._observe_with_noise(np.array([int(s)]))
            feat_2d = extract_spatial_2d(model, obs_np[0], sz, DEVICE)  # [256, H, W]
            # Normalize to [256, 2, 2] via adaptive pooling for stack compatibility
            feat_2d = F.adaptive_avg_pool2d(feat_2d.unsqueeze(0), (2, 2)).squeeze(0)
            all_feats_2d.append(feat_2d.cpu()); all_actions.append(act)

    # Stack: list of [256, H, W] → [N, 256, H, W]
    X = torch.stack(all_feats_2d)
    y = torch.tensor(all_actions, dtype=torch.long)
    n = X.shape[0]; nv = max(1, n // 5)
    perm = torch.randperm(n); Xt, Xv = X[perm[nv:]], X[perm[:nv]]
    yt, yv = y[perm[nv:]], y[perm[:nv]]
    print(f'  {n} samples ({time.time()-t0:.0f}s), train={Xt.shape[0]}, val={Xv.shape[0]}')
    act_dist = Counter(all_actions)
    act_names = ['STAY','UP','DOWN','LEFT','RIGHT']
    print(f'  Action dist: ' + ', '.join(f'{act_names[a]}={act_dist.get(a,0)}' for a in range(5)))

    # Train Conv2d policy
    policy = ConvBCPolicy(in_channels=256).to(DEVICE)
    print(f'  Params: {sum(p.numel() for p in policy.parameters())}')
    opt = optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, 50)
    best_acc, best_sd = 0, None
    bs = 128

    for ep in range(50):
        policy.train(); perm2 = torch.randperm(Xt.shape[0])
        total_loss = 0.0
        for i in range(0, Xt.shape[0], bs):
            idx = perm2[i:i+bs]
            logits = policy(Xt[idx].to(DEVICE))
            loss = F.cross_entropy(logits, yt[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
            total_loss += loss.item() * len(idx)
        sch.step()
        policy.eval()
        with torch.no_grad():
            pred = policy(Xv.to(DEVICE)).argmax(-1).cpu()
            acc = float((pred == yv).float().mean())
        if acc > best_acc: best_acc = acc; best_sd = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        if (ep + 1) % 10 == 0:
            per_cls = {}
            for a in range(5):
                mask = yv == a
                if mask.sum() > 0: per_cls[act_names[a]] = f'{float((pred[mask]==a).float().mean()):.3f}'
            print(f'  ep {ep+1}/50: acc={acc:.4f} best={best_acc:.4f}  ' + ' | '.join(f'{k}={v}' for k,v in per_cls.items()))

    if best_sd: policy.load_state_dict(best_sd)
    print(f'  Best val_acc: {best_acc:.4f}')

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save({'policy_state_dict': policy.state_dict(), 'val_acc': best_acc, 'n_samples': n},
               f'{output_dir}/bc_policy.pt')

    # Evaluate
    print('  Evaluating BC policy on eval split...')
    per_size = defaultdict(lambda: {'succ': 0, 'total': 0, 'path_lens': []})
    for entry in eval_entries:
        sz = entry['maze_size']
        if per_size[sz]['total'] >= 30: continue
        env = create_env(entry); grid = env._maze_mask; goal = env._goal_position
        empty = np.flatnonzero((~grid).reshape(-1)); safe = empty[empty != goal]
        if len(safe) < 2: continue
        s = int(rng.choice(safe)); env.reset(seed=0, options={'start_state': s}); cur = s; succ = False; plen = 0
        for _ in range(128):
            if cur == goal: succ = True; break
            obs = env._last_observation.copy()
            feat_2d = extract_spatial_2d(model, obs, sz, DEVICE).unsqueeze(0)  # [1,256,H,W]
            with torch.no_grad(): act = int(policy(feat_2d.to(DEVICE)).argmax(-1).item())
            _, _, _, _, info = env.step(act); cur = int(info['state']); plen += 1
        per_size[sz]['total'] += 1
        if succ: per_size[sz]['succ'] += 1; per_size[sz]['path_lens'].append(plen)
    results = {}
    for sz_k in sorted(per_size.keys()):
        d = per_size[sz_k]; results[f'sz{sz_k}'] = {'sr': float(d['succ']/max(d['total'],1)),
            'avg_path': float(np.mean(d['path_lens'])) if d['path_lens'] else 0.0, 'n': d['total']}
    total_s = sum(d['succ'] for d in per_size.values()); total_e = sum(d['total'] for d in per_size.values())
    results['overall'] = {'sr': float(total_s/max(total_e,1)), 'total_ep': total_e}
    with open(f'{output_dir}/bc_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"  BC Overall SR: {results['overall']['sr']:.4f} ({total_s}/{total_e})")
    return results

def main():
    from scripts.train.train_dim256 import Unisize256
    ckpt = torch.load(f'{BASE}/checkpoints/unisize_dim256.pt', map_location=DEVICE, weights_only=False)
    model = Unisize256(ckpt['model_config'], max_size=31).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    for p in model.parameters(): p.requires_grad = False
    for split_name, train_m, eval_m, out_d in [
        ('Set A: Size 11', f'{BASE}/data/splits/fixed11_train_manifest.jsonl', f'{BASE}/data/splits/fixed11_test_manifest.jsonl', f'{BASE}/results/set_a_size11'),
        ('Set B: Multi-size', f'{BASE}/data/splits/unisize_train_manifest.jsonl', f'{BASE}/data/splits/unisize_eval_manifest.jsonl', f'{BASE}/results/set_b_multisize'),
    ]:
        with open(train_m) as f: tr = [json.loads(l) for l in f if l.strip()]
        with open(eval_m) as f: ev = [json.loads(l) for l in f if l.strip()]
        print(f'\n{"#"*50}\n{split_name}\n  Train: {len(tr)}, Eval: {len(ev)}\n{"#"*50}')
        run_bc(model, tr, ev, out_d)

if __name__ == '__main__':
    main()
