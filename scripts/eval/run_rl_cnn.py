#!/usr/bin/env python3
"""RL with DeepCNN policy + PPO + BFS dense reward (same CNN as successful BC)."""
import sys, json, os, time, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
os.chdir('/data2/songxinshuai/yanyh/WORLDMODEL'); sys.path.insert(0, '.')
from torch import nn, optim
from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.planning import _bfs_shortest_path

BASE = '/data2/songxinshuai/yanyh/WORLDMODEL'; DEVICE = torch.device('cuda')
MAX_OBS_SZ = 21  # max maze size, for padding


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class CNNPolicy(nn.Module):
    """Lightweight CNN for fast RL training."""
    def __init__(self, in_ch=5, n_actions=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.actor = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def extract_features(self, x):
        with torch.no_grad():
            return self.conv(x).squeeze(-1).squeeze(-1).detach()

    def forward_actor(self, feat): return self.actor(feat)
    def forward_critic(self, feat): return self.critic(feat)


def create_env(entry):
    sz = entry['maze_size']
    return ProcgenMazeEnv(ProcgenMazeConfig(
        height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry['topology_seed'],
    ), seed=entry.get('env_seed', 42))


def obs_to_tensor(obs, max_sz=MAX_OBS_SZ):
    """Convert [H,W,5] numpy to padded [5,max_sz,max_sz] tensor."""
    t = torch.as_tensor(obs, dtype=torch.float32).permute(2, 0, 1)
    if t.shape[1] < max_sz:
        pad_h = max_sz - t.shape[1]; pad_w = max_sz - t.shape[2]
        t = F.pad(t, (0, pad_w, 0, pad_h), value=0.0)
    return t.unsqueeze(0).to(DEVICE)


def run_rl(train_entries, eval_entries, output_dir, reward_type='sparse', steps=20000):
    print(f'\n{"="*50}\nRL DeepCNN+PPO ({reward_type.upper()}) {steps} steps\n{"="*50}')
    rng = np.random.default_rng(42)

    policy = CNNPolicy().to(DEVICE)
    print(f'  Params: {sum(p.numel() for p in policy.parameters()):,}')
    opt = optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-5)

    gamma = 0.99; eps_clip = 0.2; K_epochs = 4
    log_every = 1000
    ep_rewards = []

    for step in range(1, steps + 1):
        entry = rng.choice(train_entries); sz_i = entry['maze_size']
        env = create_env(entry); grid = env._maze_mask; goal = env._goal_position; w = env.config.width
        empty = np.flatnonzero((~grid).reshape(-1)); safe = empty[empty != goal]
        if len(safe) < 2: continue
        s = int(rng.choice(safe))
        env.reset(seed=int(rng.integers(2 ** 31)), options={'start_state': s})

        # Collect trajectory — pre-extract CNN features once
        feat_list, act_list, rew_list, logp_list, val_list = [], [], [], [], []
        cur = s; ep_r = 0
        prev_bfs = _bfs_shortest_path(grid, s, goal, w) if reward_type == 'dense' else None

        for t in range(32):
            obs_t = obs_to_tensor(env._last_observation.copy(), MAX_OBS_SZ)
            feat = policy.extract_features(obs_t)  # [1, 128]
            logits = policy.forward_actor(feat)
            value = policy.forward_critic(feat)
            probs = torch.softmax(logits, dim=-1); dist = torch.distributions.Categorical(probs)
            action = dist.sample(); log_prob = dist.log_prob(action)

            _, _, term, trunc, info = env.step(int(action.item()))
            cur = int(info['state']); done = term or trunc

            if cur == goal:
                r = 10.0
            elif reward_type == 'sparse':
                r = 0.0
            else:  # dense: BFS distance change
                cur_bfs = _bfs_shortest_path(grid, cur, goal, w)
                if cur_bfs is not None and prev_bfs is not None:
                    if cur_bfs < prev_bfs: r = 1.0
                    elif cur_bfs > prev_bfs: r = -1.0
                    else: r = 0.0
                else:
                    r = 0.0
                prev_bfs = cur_bfs

            feat_list.append(feat); act_list.append(action); rew_list.append(r)
            logp_list.append(log_prob); val_list.append(value)
            ep_r += r
            if done: break

        if len(feat_list) < 2: continue
        ep_rewards.append(ep_r)

        # Compute returns & advantages
        returns = []; R = 0
        for r in reversed(rew_list): R = r + gamma * R; returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        values_t = torch.cat(val_list).squeeze(-1)
        advantages = (returns_t - values_t.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update (K epochs) — features are pre-extracted, only MLP heads run
        old_logp = torch.cat(logp_list).detach()
        feats_stack = torch.cat(feat_list).detach()  # [T, 128]
        act_stack = torch.cat(act_list)
        for _ in range(K_epochs):
            logits_all = policy.forward_actor(feats_stack)
            new_probs = torch.softmax(logits_all, dim=-1)
            new_dist = torch.distributions.Categorical(new_probs)
            new_logp = new_dist.log_prob(act_stack)

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = policy.forward_critic(feats_stack).squeeze(-1)
            value_loss = F.mse_loss(new_values, returns_t)
            loss = policy_loss + 0.5 * value_loss

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()

        if step % log_every == 0:
            avg_r = np.mean(ep_rewards[-log_every:])
            print(f'    Step {step:>6d}: loss={loss.item():.4f} avg_r={avg_r:.4f}')

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save({'policy_state_dict': policy.state_dict(), 'reward_type': reward_type},
               f'{output_dir}/rl_cnn_{reward_type}_policy.pt')

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
            obs_t = obs_to_tensor(env._last_observation.copy(), MAX_OBS_SZ)
            with torch.no_grad():
                feat = policy.extract_features(obs_t)
                logits = policy.forward_actor(feat)
                act = int(logits.argmax(-1).item())
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
    with open(f'{output_dir}/rl_{reward_type}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  RL ({reward_type}) Overall SR: {results['overall']['sr']:.4f} ({total_s}/{total_e})")
    return results


def main():
    for split_name, train_m, eval_m, out_d in [
        ('Set A: Size 11', f'{BASE}/data/splits/fixed11_train_manifest.jsonl',
         f'{BASE}/data/splits/fixed11_test_manifest.jsonl', f'{BASE}/results/set_a_size11'),
        ('Set B: Multi-size', f'{BASE}/data/splits/unisize_train_manifest.jsonl',
         f'{BASE}/data/splits/unisize_eval_manifest.jsonl', f'{BASE}/results/set_b_multisize'),
    ]:
        with open(train_m) as f: tr = [json.loads(l) for l in f if l.strip()]
        with open(eval_m) as f: ev = [json.loads(l) for l in f if l.strip()]
        print(f'\n{"#"*50}\n{split_name}\n  Train: {len(tr)}, Eval: {len(ev)}\n{"#"*50}')
        for rt in ['sparse', 'dense']:
            run_rl(tr, ev, out_d, rt, steps=20000)


if __name__ == '__main__':
    main()
