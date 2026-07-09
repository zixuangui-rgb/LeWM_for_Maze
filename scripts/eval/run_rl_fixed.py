#!/usr/bin/env python3
"""Fixed RL: DeepCNN (same as BC) + PPO + proper training (unfrozen CNN, entropy, multi-trajectory).

CRITICAL FIXES vs run_rl_cnn.py:
1. CNN is NOT frozen — gradients flow through all layers
2. Entropy bonus to prevent policy collapse
3. Multi-trajectory collection before each PPO update
4. Same DeepCNN architecture as successful BC
5. Longer horizon (128 steps)
6. Cosine LR schedule
"""
import sys, json, os, time, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
os.chdir('/data2/songxinshuai/yanyh/WORLDMODEL'); sys.path.insert(0, '.')
from torch import nn, optim
from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.planning import _bfs_shortest_path

BASE = '/data2/songxinshuai/yanyh/WORLDMODEL'; DEVICE = torch.device('cuda')

# ══ Same DeepCNN as BC (NOT frozen!) ══════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
    def forward(self, x): return F.relu(self.net(x) + x)


class CNNRL(nn.Module):
    """Unfrozen CNN — ALL params receive gradients, BatchNorm for stability."""
    def __init__(self, in_ch=5, n_actions=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.actor = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                    nn.Linear(128, n_actions))
        self.critic = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                     nn.Linear(128, 1))

    def forward(self, x):
        h = self.conv(x).squeeze(-1).squeeze(-1)
        return self.actor(h), self.critic(h)


def obs_to_tensor(obs, max_sz=21):
    t = torch.as_tensor(obs, dtype=torch.float32).permute(2, 0, 1)
    if t.shape[1] < max_sz:
        pad_h = max_sz - t.shape[1]; pad_w = max_sz - t.shape[2]
        t = F.pad(t, (0, pad_w, 0, pad_h), value=0.0)
    return t.unsqueeze(0).to(DEVICE)


def create_env(entry):
    sz = entry['maze_size']
    return ProcgenMazeEnv(ProcgenMazeConfig(
        height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry['topology_seed'],
    ), seed=entry.get('env_seed', 42))

# ══ Training ══════════════════════════════════════════════════════════════════

def run_rl(train_entries, eval_entries, output_dir, steps=20000):
    print(f'\n{"="*50}\nRL DeepCNN+PPO (FIXED) {steps} steps\n{"="*50}')
    rng = np.random.default_rng(42)
    policy = CNNRL().to(DEVICE)
    print(f'  Params: {sum(p.numel() for p in policy.parameters()):,}')
    opt = optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    gamma = 0.99; eps_clip = 0.2; K_epochs = 4; entropy_coef = 0.01
    trajs_per_update = 2  # collect 2 trajectories before PPO update
    log_every = 500; ep_rewards = []

    for step in range(1, steps + 1):
        # Collect multiple trajectories
        all_feats, all_acts, all_rews, all_logps, all_vals = [], [], [], [], []
        all_dones = []

        for _ in range(trajs_per_update):
            entry = rng.choice(train_entries); sz_i = entry['maze_size']
            env = create_env(entry); grid = env._maze_mask
            goal = env._goal_position; w = env.config.width
            empty = np.flatnonzero((~grid).reshape(-1)); safe = empty[empty != goal]
            if len(safe) < 2: continue
            s = int(rng.choice(safe))
            env.reset(seed=int(rng.integers(2 ** 31)), options={'start_state': s})

            feats, acts, rews, logps, vals = [], [], [], [], []
            cur = s; ep_r = 0
            prev_bfs = _bfs_shortest_path(grid, s, goal, w)

            for t in range(64):  # reasonable horizon for sz=11
                obs_t = obs_to_tensor(env._last_observation.copy())
                logits, value = policy(obs_t)
                probs = torch.softmax(logits, dim=-1); dist = torch.distributions.Categorical(probs)
                action = dist.sample(); log_prob = dist.log_prob(action)

                _, _, term, trunc, info = env.step(int(action.item()))
                cur = int(info['state']); done = term or trunc

                # BFS distance-based dense reward
                cur_bfs = _bfs_shortest_path(grid, cur, goal, w)
                if cur == goal:
                    r = 10.0
                elif cur_bfs is not None and prev_bfs is not None:
                    if cur_bfs < prev_bfs: r = 1.0
                    elif cur_bfs > prev_bfs: r = -1.0
                    else: r = 0.0
                else:
                    r = 0.0
                prev_bfs = cur_bfs

                feats.append(obs_t); acts.append(action); rews.append(r)
                logps.append(log_prob); vals.append(value)
                ep_r += r
                if done: break

            if len(feats) < 2: continue
            ep_rewards.append(ep_r)
            all_feats.extend(feats); all_acts.extend(acts); all_rews.extend(rews)
            all_logps.extend(logps); all_vals.extend(vals); all_dones.append(len(feats))

        if len(all_feats) < 4: continue

        # Compute returns & advantages (per-trajectory, then concatenate)
        all_returns, all_advantages = [], []
        idx = 0
        for traj_len in all_dones:
            rews_traj = all_rews[idx:idx + traj_len]
            vals_traj = torch.cat(all_vals[idx:idx + traj_len]).squeeze(-1)
            returns = []; R = 0
            for r in reversed(rews_traj): R = r + gamma * R; returns.insert(0, R)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
            advantages = (returns_t - vals_traj.detach())
            all_returns.append(returns_t); all_advantages.append(advantages)
            idx += traj_len

        returns_t = torch.cat(all_returns)
        advantages = torch.cat(all_advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_logp = torch.cat(all_logps).detach()
        act_stack = torch.cat(all_acts)
        feat_stack = torch.cat(all_feats).detach()  # detach CNN features for PPO epoch reuse

        # PPO update with entropy bonus
        for _ in range(K_epochs):
            logits_all, values_all = policy(feat_stack)
            new_probs = torch.softmax(logits_all, dim=-1)
            new_dist = torch.distributions.Categorical(new_probs)
            new_logp = new_dist.log_prob(act_stack)
            entropy = new_dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_all.squeeze(-1), returns_t)
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()

        sch.step()
        if step % log_every == 0:
            avg_r = np.mean(ep_rewards[-log_every:])
            print(f'    Step {step:>6d}: loss={loss.item():.4f} avg_r={avg_r:.4f} entropy={entropy.item():.4f}')

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save({'policy_state_dict': policy.state_dict()}, f'{output_dir}/rl_fixed_policy.pt')

    # Evaluate
    print('  Evaluating...')
    per_size = defaultdict(lambda: {'succ': 0, 'total': 0})
    for entry in eval_entries:
        sz_i = entry['maze_size']
        if per_size[sz_i]['total'] >= 30: continue
        env = create_env(entry); goal = env._goal_position
        grid = env._maze_mask
        empty = np.flatnonzero((~grid).reshape(-1)); safe = empty[empty != goal]
        if len(safe) < 2: continue
        s = int(rng.choice(safe)); env.reset(seed=0, options={'start_state': s})
        cur = s; succ = False
        for _ in range(128):
            if cur == goal: succ = True; break
            obs_t = obs_to_tensor(env._last_observation.copy())
            with torch.no_grad():
                logits, _ = policy(obs_t); act = int(logits.argmax(-1).item())
            _, _, _, _, info = env.step(act); cur = int(info['state'])
        per_size[sz_i]['total'] += 1
        if succ: per_size[sz_i]['succ'] += 1

    results = {}
    for sz in sorted(per_size.keys()):
        d = per_size[sz]
        results[f'sz{sz}'] = {'sr': float(d['succ'] / max(d['total'], 1)), 'n': d['total']}
    total_s = sum(d['succ'] for d in per_size.values())
    total_e = sum(d['total'] for d in per_size.values())
    results['overall'] = {'sr': float(total_s / max(total_e, 1)), 'total_ep': total_e}
    with open(f'{output_dir}/rl_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"  RL Overall SR: {results['overall']['sr']:.4f} ({total_s}/{total_e})")
    return results


def main():
    for split_name, train_m, eval_m, out_d in [
        ('Set A: Size 11', f'{BASE}/data/splits/fixed11_train_manifest.jsonl',
         f'{BASE}/data/splits/fixed11_test_manifest.jsonl', f'{BASE}/results/set_a_size11'),
    ]:
        with open(train_m) as f: tr = [json.loads(l) for l in f if l.strip()]
        with open(eval_m) as f: ev = [json.loads(l) for l in f if l.strip()]
        print(f'\n{"#"*50}\n{split_name}\n  Train: {len(tr)}, Eval: {len(ev)}\n{"#"*50}')
        run_rl(tr, ev, out_d, steps=20000)


if __name__ == '__main__':
    main()
