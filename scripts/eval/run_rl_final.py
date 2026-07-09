#!/usr/bin/env python3
"""Final RL baseline: REINFORCE with proper dense reward (BFS distance-based)."""
import sys, json, os, numpy as np, torch, torch.nn.functional as F
from collections import deque, defaultdict
os.chdir('/data2/songxinshuai/yanyh/WORLDMODEL'); sys.path.insert(0, '.')
from torch import nn, optim
from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.planning import _bfs_shortest_path
from scripts.train.train_dim256 import Unisize256

BASE = '/data2/songxinshuai/yanyh/WORLDMODEL'; DEVICE = torch.device('cuda')

def extract_spatial(model, obs, sz, dev):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    cnn = model.encoder.cnn
    x = obs_t.permute(0,1,4,2,3).reshape(1,obs_t.shape[4],obs_t.shape[2],obs_t.shape[3])
    with torch.no_grad(): x = cnn.conv(x);
    return F.adaptive_avg_pool2d(x, (2,2)).reshape(1, -1)  # normalize spatial → [1, 1024]

class RLPolicy(nn.Module):
    def __init__(self, spatial_dim=1024, n_actions=5, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(spatial_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, n_actions); self.critic = nn.Linear(hidden, 1)
    def forward(self, feat):
        h = self.net(feat); return self.actor(h), self.critic(h)

def create_env(entry):
    sz = entry['maze_size']
    return ProcgenMazeEnv(ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry['topology_seed']),
        seed=entry.get('env_seed', 42))

def run_rl(model, train_entries, eval_entries, output_dir, reward_type='sparse', steps=5000):
    print(f'\n{"="*50}\nRL ({reward_type.upper()}) - {steps} steps\n{"="*50}')
    rng = np.random.default_rng(42)
    policy = RLPolicy().to(DEVICE)
    opt = optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f'  Params: {sum(p.numel() for p in policy.parameters())}')

    gamma = 0.99; log_every = 500; ep_rewards = []
    for step in range(1, steps + 1):
        entry = rng.choice(train_entries); sz_i = entry['maze_size']; env = create_env(entry)
        grid = env._maze_mask; goal = env._goal_position; w = env.config.width
        empty = np.flatnonzero((~grid).reshape(-1)); safe = empty[empty != goal]
        if len(safe) < 2: continue
        s = int(rng.choice(safe)); env.reset(seed=int(rng.integers(2**31)), options={'start_state': s})
        # Pre-compute initial BFS distance for dense reward
        prev_bfs = _bfs_shortest_path(grid, s, goal, w) if reward_type == 'dense' else None

        feats, actions, rewards, values = [], [], [], []
        cur = s; ep_r = 0
        for t in range(32):
            obs = env._last_observation.copy()
            feat = extract_spatial(model, obs, sz_i, DEVICE)
            logits, value = policy(feat)
            probs = torch.softmax(logits, dim=-1); dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            _, _, term, trunc, info = env.step(int(action.item()))
            cur = int(info['state']); done = term or trunc

            # Reward calculation
            if cur == goal:
                r = 10.0
            elif reward_type == 'sparse':
                r = 0.0
            else:  # dense: BFS distance change
                cur_bfs = _bfs_shortest_path(grid, cur, goal, w)
                if cur_bfs is not None and prev_bfs is not None:
                    if cur_bfs < prev_bfs: r = 1.0   # getting closer
                    elif cur_bfs > prev_bfs: r = -1.0  # getting farther
                    else: r = 0.0
                else:
                    r = 0.0
                prev_bfs = cur_bfs

            feats.append(feat); actions.append(action); rewards.append(r); values.append(value)
            ep_r += r
            if done: break
        if len(feats) < 2: continue
        ep_rewards.append(ep_r)

        # Compute returns
        returns = []; R = 0
        for r in reversed(rewards): R = r + gamma * R; returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        values_t = torch.cat(values).squeeze(-1)
        advantages = (returns_t - values_t.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient
        logits_all = torch.cat([policy(f)[0] for f in feats])
        action_t = torch.cat(actions)
        log_probs = torch.distributions.Categorical(torch.softmax(logits_all, dim=-1)).log_prob(action_t)
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values_t, returns_t)
        loss = policy_loss + 0.5 * value_loss
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()

        if step % log_every == 0:
            avg_r = np.mean(ep_rewards[-log_every:])
            print(f'    Step {step:>5d}: loss={loss.item():.4f} avg_reward={avg_r:.4f}')

    os.makedirs(output_dir, exist_ok=True)
    torch.save({'policy_state_dict': policy.state_dict(), 'reward_type': reward_type},
               f'{output_dir}/rl_{reward_type}_policy.pt')

    # Evaluate
    print(f'  Evaluating...')
    per_size = defaultdict(lambda: {'succ': 0, 'total': 0, 'path_lens': []})
    for entry in eval_entries:
        sz_i = entry['maze_size']
        if per_size[sz_i]['total'] >= 30: continue
        env = create_env(entry); grid = env._maze_mask; goal = env._goal_position
        empty = np.flatnonzero((~grid).reshape(-1)); safe = empty[empty != goal]
        if len(safe) < 2: continue
        s = int(rng.choice(safe)); env.reset(seed=0, options={'start_state': s}); cur = s; succ = False; plen = 0
        for _ in range(128):
            if cur == goal: succ = True; break
            obs = env._last_observation.copy()
            feat = extract_spatial(model, obs, sz_i, DEVICE)
            with torch.no_grad(): act = int(policy(feat)[0].argmax(-1).item())
            _, _, _, _, info = env.step(act); cur = int(info['state']); plen += 1
        per_size[sz_i]['total'] += 1
        if succ: per_size[sz_i]['succ'] += 1; per_size[sz_i]['path_lens'].append(plen)

    results = {}
    for sz_k in sorted(per_size.keys()):
        d = per_size[sz_k]; results[f'sz{sz_k}'] = {'sr': float(d['succ']/max(d['total'],1)),
            'avg_path': float(np.mean(d['path_lens'])) if d['path_lens'] else 0.0, 'n': d['total']}
    total_s = sum(d['succ'] for d in per_size.values()); total_e = sum(d['total'] for d in per_size.values())
    results['overall'] = {'sr': float(total_s/max(total_e,1)), 'total_ep': total_e}
    with open(f'{output_dir}/rl_{reward_type}_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"  RL ({reward_type}) Overall SR: {results['overall']['sr']:.4f} ({total_s}/{total_e})")
    return results

def main():
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
        for rt in ['sparse', 'dense']:
            run_rl(model, tr, ev, out_d, rt, steps=5000)

if __name__ == '__main__':
    main()
