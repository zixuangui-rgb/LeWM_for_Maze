#!/usr/bin/env python3
"""BC with deep CNN on full observation — aiming for 90%+ val_acc."""
import sys, json, os, time, numpy as np, torch, torch.nn.functional as F
from collections import deque, Counter
os.chdir('/data2/songxinshuai/yanyh/WORLDMODEL'); sys.path.insert(0, '.')
from torch import nn, optim
from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv

BASE = '/data2/songxinshuai/yanyh/WORLDMODEL'; DEVICE = torch.device('cuda')

def bfs_optimal_action(grid, sy, sx, gy, gx, sz):
    H, W = sz, sz
    if grid[sy, sx] or grid[gy, gx]: return 0
    if sy == gy and sx == gx: return 0
    parent = np.full(H * W, -1, np.int32)
    first_act = np.full(H * W, -1, np.int32)
    si, gi = sy * W + sx, gy * W + gx
    q = deque(); q.append(si); parent[si] = si
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    acts = [1, 2, 3, 4]
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


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class DeepCNNPolicy(nn.Module):
    def __init__(self, in_ch=5, n_actions=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.res1(h)
        h = self.down(h)
        h = self.pool(h).squeeze(-1).squeeze(-1)
        return self.mlp(h)


def create_env(entry):
    sz = entry['maze_size']
    return ProcgenMazeEnv(ProcgenMazeConfig(
        height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry['topology_seed'],
    ), seed=entry.get('env_seed', 42))


def run_bc_cnn(train_entries, eval_entries, output_dir, tag):
    print(f'\n{"="*60}\nBC DeepCNN [{tag}]\n{"="*60}')
    rng = np.random.default_rng(42)

    # Generate data: full observation → BFS action
    t0 = time.time()
    obs_list, act_list = [], []
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
            obs_t = torch.as_tensor(obs_np[0], dtype=torch.float32).permute(2, 0, 1)
            obs_list.append(obs_t); act_list.append(act)

    # Pad observations to max size for stacking (multi-size data)
    max_sz = max(o.shape[1] for o in obs_list)
    obs_padded = []
    for o in obs_list:
        if o.shape[1] < max_sz:
            pad_h = max_sz - o.shape[1]; pad_w = max_sz - o.shape[2]
            o = F.pad(o, (0, pad_w, 0, pad_h), value=0.0)  # pad with 0 (empty channel=0)
        obs_padded.append(o)
    X = torch.stack(obs_padded); y = torch.tensor(act_list, dtype=torch.long)
    n = X.shape[0]; nv = max(1, n // 5)
    perm = torch.randperm(n); Xt, Xv = X[perm[nv:]], X[perm[:nv]]
    yt, yv = y[perm[nv:]], y[perm[:nv]]
    print(f'  {n} samples ({time.time()-t0:.0f}s), train={Xt.shape[0]}, val={Xv.shape[0]}')
    print(f'  Action dist: {dict(sorted(Counter(act_list).items()))}')

    # Train
    policy = DeepCNNPolicy().to(DEVICE)
    print(f'  Params: {sum(p.numel() for p in policy.parameters()):,}')
    opt = optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, 200)
    best_acc = 0; best_sd = None
    bs = 128

    for ep in range(200):
        policy.train(); perm2 = torch.randperm(Xt.shape[0])
        for i in range(0, Xt.shape[0], bs):
            idx = perm2[i:i + bs]
            loss = F.cross_entropy(policy(Xt[idx].to(DEVICE)), yt[idx].to(DEVICE))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        sch.step()
        policy.eval()
        with torch.no_grad():
            # Batch validation to avoid OOM
            preds = []
            for i in range(0, Xv.shape[0], 1024):
                preds.append(policy(Xv[i:i+1024].to(DEVICE)).argmax(-1).cpu())
            pred = torch.cat(preds)
            acc = float((pred == yv).float().mean())
        if acc > best_acc:
            best_acc = acc
            best_sd = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        if (ep + 1) % 40 == 0:
            print(f'  ep {ep+1}/200: acc={acc:.4f} best={best_acc:.4f}')

    if best_sd: policy.load_state_dict(best_sd)
    print(f'  Best val_acc: {best_acc:.4f}')

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save({'policy_state_dict': policy.state_dict(), 'val_acc': best_acc, 'n_samples': n},
               f'{output_dir}/bc_cnn_policy.pt')

    # Evaluate
    print('  Evaluating...')
    per_size = Counter()
    succ_per_size = Counter()
    for entry in eval_entries:
        sz = entry['maze_size']
        if per_size[sz] >= 30: continue
        env = create_env(entry); goal = env._goal_position
        grid = env._maze_mask
        empty = np.flatnonzero((~grid).reshape(-1)); safe = empty[empty != goal]
        if len(safe) < 2: continue
        s = int(rng.choice(safe))
        env.reset(seed=0, options={'start_state': s}); cur = s; ok = False; plen = 0
        for step in range(128):
            if cur == goal: ok = True; break
            obs = env._last_observation.copy()
            obs_t = torch.as_tensor(obs, dtype=torch.float32).permute(2, 0, 1)
            # Pad to max_sz for multi-size compatibility
            if obs_t.shape[1] < max_sz:
                pad_h = max_sz - obs_t.shape[1]; pad_w = max_sz - obs_t.shape[2]
                obs_t = F.pad(obs_t, (0, pad_w, 0, pad_h), value=0.0)
            obs_t = obs_t.unsqueeze(0).to(DEVICE)
            with torch.no_grad(): act = int(policy(obs_t).argmax(-1).item())
            _, _, _, _, info = env.step(act); cur = int(info['state']); plen += 1
        per_size[sz] += 1
        if ok: succ_per_size[sz] += 1

    results = {}
    for sz in sorted(per_size.keys()):
        results[f'sz{sz}'] = {'sr': float(succ_per_size[sz] / max(per_size[sz], 1)),
                               'n': per_size[sz]}
    total_s = sum(succ_per_size.values()); total_e = sum(per_size.values())
    results['overall'] = {'sr': float(total_s / max(total_e, 1)), 'total_ep': total_e}
    with open(f'{output_dir}/bc_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"  BC Overall SR: {results['overall']['sr']:.4f} ({total_s}/{total_e})")
    return results, best_acc


def main():
    for split_name, train_m, eval_m, out_d in [
        ('Set A: Size 11', f'{BASE}/data/splits/fixed11_train_manifest.jsonl',
         f'{BASE}/data/splits/fixed11_test_manifest.jsonl', f'{BASE}/results/set_a_size11'),
        ('Set B: Multi-size', f'{BASE}/data/splits/unisize_train_manifest.jsonl',
         f'{BASE}/data/splits/unisize_eval_manifest.jsonl', f'{BASE}/results/set_b_multisize'),
    ]:
        with open(train_m) as f: tr = [json.loads(l) for l in f if l.strip()]
        with open(eval_m) as f: ev = [json.loads(l) for l in f if l.strip()]
        print(f'\n{"#"*60}\n{split_name}\n  Train: {len(tr)}, Eval: {len(ev)}\n{"#"*60}')
        run_bc_cnn(tr, ev, out_d, split_name)


if __name__ == '__main__':
    main()
