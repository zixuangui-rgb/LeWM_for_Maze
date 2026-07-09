#!/usr/bin/env python3
"""CEM L2 evaluation per maze size on Set B (multi-size split)."""
import sys, json, os, time, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
os.chdir('/data2/songxinshuai/yanyh/WORLDMODEL'); sys.path.insert(0, '.')
from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.planning import _bfs_shortest_path, cem_plan
from scripts.train.train_dim256 import Unisize256

BASE = '/data2/songxinshuai/yanyh/WORLDMODEL'; DEVICE = torch.device('cuda')
CEM_CFG = dict(horizon=12, num_candidates=64, cem_iters=1, receding_horizon=1, history_size=3)

def create_env(entry):
    sz = entry['maze_size']
    return ProcgenMazeEnv(ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry['topology_seed']),
        seed=entry.get('env_seed', 42))

def encode_obs(model, obs, sz):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encoder(obs_t, sz); embedding, _ = model.embedding_projector(encoded)
    return embedding

def l2_score(terminal, goal):
    return F.mse_loss(terminal, goal, reduction='none').sum(dim=-1)

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
    raise RuntimeError('no pair')

def run_cem_episode(model, env, start, goal, sz, seed):
    na = env.config.action_vocab_size; elites = max(64 // 8, 8)
    env.reset(seed=seed, options={'start_state': start})
    start_emb = encode_obs(model, env._last_observation, sz)
    ctx_emb = start_emb.repeat(1, CEM_CFG['history_size'], 1)
    ctx_act = torch.full((1, CEM_CFG['history_size']), 0, dtype=torch.long, device=DEVICE)  # STAY padding
    env.reset(seed=seed, options={'start_state': goal})
    goal_emb = encode_obs(model, env._last_observation, sz)
    env.reset(seed=seed, options={'start_state': start})
    cur = start; succ = False; plen = 0
    for step in range(128):
        if cur == goal: succ = True; break
        best_seq, _, _ = cem_plan(model, ctx_emb, ctx_act, goal_emb,
            horizon=CEM_CFG['horizon'], history_size=CEM_CFG['history_size'],
            num_candidates=64, num_elites=elites, cem_iters=1, momentum=0.1,
            num_actions=na, device=DEVICE, seed=seed * 10000 + step, score_fn=l2_score)
        a = int(best_seq[0])
        obs, _, _, _, info = env.step(a); cur = int(info['state']); plen += 1
        new_emb = encode_obs(model, obs, sz)
        ctx_emb = torch.cat([ctx_emb[:, 1:], new_emb], dim=1)
        ctx_act = torch.cat([ctx_act[:, 1:], torch.tensor([[a]], dtype=torch.long, device=DEVICE)], dim=1)
        if cur == goal: succ = True; break
    return dict(success=succ, path_length=plen)

def main():
    ckpt = torch.load(f'{BASE}/checkpoints/unisize_dim256.pt', map_location=DEVICE, weights_only=False)
    model = Unisize256(ckpt['model_config'], max_size=31).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    for p in model.parameters(): p.requires_grad = False

    with open(f'{BASE}/data/splits/unisize_eval_manifest.jsonl') as f:
        entries = [json.loads(l) for l in f if l.strip()]

    rng = np.random.default_rng(42)
    N_PER_SIZE = 20

    # Pre-sample episodes per size
    size_eps = defaultdict(list)
    for entry in entries:
        sz = entry['maze_size']
        if len(size_eps[sz]) >= N_PER_SIZE: continue
        env = create_env(entry)
        try:
            s, g, opt = sample_start_goal(env, rng)
            size_eps[sz].append(dict(entry=entry, start=s, goal=g, opt=opt, sz=sz))
        except RuntimeError: continue

    print(f'{"Size":<8s} {"SR":>8s} {"AvgPath":>10s} {"AvgOpt":>8s} {"Path/Opt":>10s} {"N":>5s}')
    print('-' * 55)

    all_per_size = {}
    for sz in sorted(size_eps.keys()):
        eps = size_eps[sz]
        succ, path_lens, opt_lens = 0, [], []
        t0 = time.time()
        for i, ed in enumerate(eps):
            env = create_env(ed['entry']); seed = 42 * 10000 + i
            r = run_cem_episode(model, env, ed['start'], ed['goal'], ed['sz'], seed)
            if r['success']: succ += 1; path_lens.append(r['path_length'])
            opt_lens.append(ed['opt'])
        sr = succ / len(eps)
        avg_path = np.mean(path_lens) if path_lens else 0
        avg_opt = np.mean(opt_lens)
        ratio = avg_path / avg_opt if avg_opt > 0 else 0
        tag = 'OOD' if sz > 21 else 'seen'
        print(f'{sz:<8d} {sr:>8.4f} {avg_path:>10.1f} {avg_opt:>8.1f} {ratio:>10.2f} {len(eps):>5d}  [{tag}]')
        all_per_size[str(sz)] = dict(sr=float(sr), avg_path=float(avg_path), avg_opt=float(avg_opt),
                                       path_opt_ratio=float(ratio), n=len(eps), tag=tag)

    total_s = sum(v['sr'] * v['n'] for v in all_per_size.values())
    total_n = sum(v['n'] for v in all_per_size.values())
    print(f'{"OVERALL":<8s} {total_s/total_n:>8.4f}')
    print(f'\n  ⏱  Total: {time.time()-t0:.0f}s')

    os.makedirs(f'{BASE}/results/cem_per_size', exist_ok=True)
    with open(f'{BASE}/results/cem_per_size/l2_cem_per_size.json', 'w') as f:
        json.dump(all_per_size, f, indent=2)

if __name__ == '__main__':
    main()
