#!/usr/bin/env python3
"""Real GCRL+HER: Goal-Conditioned RL with Hindsight Experience Replay.

NOT the same as the old "GCRL Head" (now renamed ReachabilityHead).
This learns a goal-conditioned policy pi(a|s,g) using PPO + HER relabeling.

HER: failed trajectories get their goal replaced with the actually-reached state,
turning failures into successes for the hindsight goal.
"""
import sys, json, os, time, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict, deque
os.chdir('/data2/songxinshuai/yanyh/WORLDMODEL'); sys.path.insert(0, '.')
from torch import nn, optim
from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.planning import _bfs_shortest_path

BASE = '/data2/songxinshuai/yanyh/WORLDMODEL'; DEVICE = torch.device('cuda')

# ══ Goal-Conditioned Policy ═══════════════════════════════════════════════════

class GoalConditionedPolicy(nn.Module):
    """pi(a | s, g): CNN encodes [obs, goal_obs] separately → concat → actor/critic."""
    def __init__(self, in_ch=5, n_actions=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),  # 64 (obs) + 64 (goal)
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def encode(self, x):
        return self.encoder(x).squeeze(-1).squeeze(-1)  # [B, 64]

    def forward(self, obs, goal_obs):
        z_obs = self.encode(obs)       # [B, 64]
        z_goal = self.encode(goal_obs)  # [B, 64]
        z = torch.cat([z_obs, z_goal], dim=-1)  # [B, 128]
        return self.actor(z), self.critic(z)


def obs_to_tensor(obs, max_sz=21):
    t = torch.as_tensor(obs, dtype=torch.float32).permute(2, 0, 1)
    if t.shape[1] < max_sz:
        pad_h = max_sz - t.shape[1]; pad_w = max_sz - t.shape[2]
        t = F.pad(t, (0, pad_w, 0, pad_h), value=0.0)
    return t.unsqueeze(0).to(DEVICE)


def render_goal_obs(env, goal_state):
    """Render observation at goal position."""
    prev_state = env._state
    env._state = goal_state
    obs_np, _ = env._observe_with_noise(np.array([goal_state]))
    env._state = prev_state
    return obs_to_tensor(obs_np[0])


def create_env(entry):
    sz = entry['maze_size']
    return ProcgenMazeEnv(ProcgenMazeConfig(
        height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry['topology_seed'],
    ), seed=entry.get('env_seed', 42))

# ══ Training ══════════════════════════════════════════════════════════════════

def run_gcrl_her(train_entries, eval_entries, output_dir, steps=10000):
    print(f'\n{"="*50}\nGCRL+HER {steps} steps\n{"="*50}')
    rng = np.random.default_rng(42)
    policy = GoalConditionedPolicy().to(DEVICE)
    print(f'  Params: {sum(p.numel() for p in policy.parameters()):,}')
    opt = optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    gamma = 0.99; eps_clip = 0.2; K_epochs = 4; entropy_coef = 0.01
    her_ratio = 0.8  # fraction of transitions to HER-relabel
    replay_buffer = deque(maxlen=10000)
    log_every = 500; ep_rewards = []

    for step in range(1, steps + 1):
        # ── Collect episode ──
        entry = rng.choice(train_entries); sz_i = entry['maze_size']
        env = create_env(entry); grid = env._maze_mask
        w = env.config.width
        empty = np.flatnonzero((~grid).reshape(-1))

        # Sample start and goal
        safe_all = empty[empty != env._goal_position] if len(empty) > 1 else empty
        if len(safe_all) < 2: continue
        start_s = int(rng.choice(safe_all))
        goal_s = int(rng.choice(safe_all[safe_all != start_s] if len(safe_all) > 1 else safe_all))
        if goal_s == start_s: continue

        env.reset(seed=int(rng.integers(2 ** 31)), options={'start_state': start_s})
        goal_obs_t = render_goal_obs(env, goal_s)

        traj = []  # (obs_t, action, logp, value, reward, done, achieved_state)
        cur = start_s; total_r = 0

        for t in range(64):
            obs_t = obs_to_tensor(env._last_observation.copy())
            logits, value = policy(obs_t, goal_obs_t)
            probs = torch.softmax(logits, dim=-1); dist = torch.distributions.Categorical(probs)
            action = dist.sample(); log_prob = dist.log_prob(action)

            _, _, term, trunc, info = env.step(int(action.item()))
            cur = int(info['state']); done = term or trunc
            r = 10.0 if cur == goal_s else 0.0  # sparse reward
            total_r += r

            traj.append({
                'obs_t': obs_t, 'action': action, 'logp': log_prob, 'value': value,
                'reward': r, 'done': done, 'achieved': cur, 'goal_obs_t': goal_obs_t,
            })
            if done: break

        if len(traj) < 2: continue
        ep_rewards.append(total_r)
        achieved_final = traj[-1]['achieved']

        # ── HER relabeling ──
        # Store original transitions
        for t_data in traj:
            replay_buffer.append({
                'obs_t': t_data['obs_t'], 'action': t_data['action'],
                'logp': t_data['logp'], 'value': t_data['value'],
                'reward': t_data['reward'], 'done': t_data['done'],
                'goal_obs_t': t_data['goal_obs_t'],
            })

        # HER-relabeled: replace goal with achieved final state
        if rng.random() < her_ratio and achieved_final != goal_s:
            her_goal_obs = render_goal_obs(env, achieved_final)
            for t_data in traj:
                her_reward = 10.0 if t_data['achieved'] == achieved_final else 0.0
                replay_buffer.append({
                    'obs_t': t_data['obs_t'], 'action': t_data['action'],
                    'logp': t_data['logp'], 'value': t_data['value'],
                    'reward': her_reward, 'done': t_data['done'],
                    'goal_obs_t': her_goal_obs,
                })

        if len(replay_buffer) < 64: continue

        # ── PPO update from replay buffer ──
        batch_size = min(64, len(replay_buffer))
        indices = rng.choice(len(replay_buffer), size=batch_size, replace=False)
        batch = [replay_buffer[i] for i in indices]

        obs_stack = torch.cat([b['obs_t'] for b in batch])
        goal_stack = torch.cat([b['goal_obs_t'] for b in batch])
        act_stack = torch.tensor([b['action'].item() for b in batch], device=DEVICE)
        old_logp = torch.tensor([b['logp'].item() for b in batch], device=DEVICE)
        old_val = torch.tensor([b['value'].item() for b in batch], device=DEVICE)
        rew_stack = torch.tensor([b['reward'] for b in batch], dtype=torch.float32, device=DEVICE)

        # Compute returns and advantages (simplified: Monte Carlo per-episode)
        # For offline replay, use 1-step TD or MC
        returns = rew_stack  # simplified: for offline replay, use reward as return
        advantages = returns - old_val
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(K_epochs):
            logits_all, values_all = policy(obs_stack, goal_stack)
            new_probs = torch.softmax(logits_all, dim=-1)
            new_dist = torch.distributions.Categorical(new_probs)
            new_logp = new_dist.log_prob(act_stack)
            entropy = new_dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_all.squeeze(-1), returns)
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()

        sch.step()
        if step % log_every == 0:
            avg_r = np.mean(ep_rewards[-log_every:])
            print(f'    Step {step:>6d}: loss={loss.item():.4f} avg_r={avg_r:.4f} buf={len(replay_buffer)}')

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save({'policy_state_dict': policy.state_dict()}, f'{output_dir}/gcrl_her_policy.pt')

    # Evaluate
    print('  Evaluating...')
    per_size = defaultdict(lambda: {'succ': 0, 'total': 0})
    for entry in eval_entries:
        sz_i = entry['maze_size']
        if per_size[sz_i]['total'] >= 30: continue
        env = create_env(entry)
        grid = env._maze_mask; w = env.config.width
        empty = np.flatnonzero((~grid).reshape(-1))
        safe = empty[empty != env._goal_position] if len(empty) > 1 else empty
        if len(safe) < 2: continue
        s = int(rng.choice(safe)); goal_s = env._goal_position
        env.reset(seed=0, options={'start_state': s})
        goal_obs_t = render_goal_obs(env, goal_s)
        cur = s; succ = False
        for _ in range(128):
            if cur == goal_s: succ = True; break
            obs_t = obs_to_tensor(env._last_observation.copy())
            with torch.no_grad():
                logits, _ = policy(obs_t, goal_obs_t)
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
    with open(f'{output_dir}/gcrl_her_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"  GCRL+HER Overall SR: {results['overall']['sr']:.4f} ({total_s}/{total_e})")
    return results


def main():
    for split_name, train_m, eval_m, out_d in [
        ('Set A: Size 11', f'{BASE}/data/splits/fixed11_train_manifest.jsonl',
         f'{BASE}/data/splits/fixed11_test_manifest.jsonl', f'{BASE}/results/set_a_size11'),
    ]:
        with open(train_m) as f: tr = [json.loads(l) for l in f if l.strip()]
        with open(eval_m) as f: ev = [json.loads(l) for l in f if l.strip()]
        print(f'\n{"#"*50}\n{split_name}\n  Train: {len(tr)}, Eval: {len(ev)}\n{"#"*50}')
        run_gcrl_her(tr, ev, out_d, steps=10000)


if __name__ == '__main__':
    main()
