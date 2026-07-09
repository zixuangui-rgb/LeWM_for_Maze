#!/usr/bin/env python3
"""Re-run Set B CEM with CORRECT random sampling across ALL sizes (fixes the sequential sampling bug)."""
import sys, json, os, time, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
os.chdir('/data2/songxinshuai/yanyh/WORLDMODEL'); sys.path.insert(0, '.')
from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.planning import _bfs_shortest_path, cem_plan
from hdwm.metric_heads.distance_head import DistanceHead
from hdwm.metric_heads.gcrl_head import GCRLHead
from hdwm.metric_heads.qrl_head import QRLHead
from scripts.train.train_dim256 import Unisize256

BASE = '/data2/songxinshuai/yanyh/WORLDMODEL'; DEV = torch.device('cuda')
PC = dict(horizon=12, num_candidates=64, cem_iters=1, receding_horizon=1, history_size=3)

def load_head(cls, path, **kw):
    c = torch.load(path, map_location=DEV, weights_only=False)
    h = cls(**kw).to(DEV); h.load_state_dict(c['head_state_dict']); h.eval()
    for p in h.parameters(): p.requires_grad = False
    return h

def create_env(entry):
    sz = entry['maze_size']
    return ProcgenMazeEnv(ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry['topology_seed']),
        seed=entry.get('env_seed', 42))

def enc_obs(model, obs, sz):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEV).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encoder(obs_t, sz); embedding, _ = model.embedding_projector(encoded)
    return embedding

def sample_sg(env, rng):
    obs = env._maze_mask; empty = ~obs
    flat = empty.reshape(-1).copy()
    if hasattr(env,'_goal_position'): flat[env._goal_position] = False
    pos = np.flatnonzero(flat); w = env.config.width
    for _ in range(500):
        s = int(rng.choice(pos)); g = int(rng.choice(pos))
        if s==g: continue
        d = _bfs_shortest_path(obs,s,g,w)
        if d is not None and d>=3: return s,g,d
    raise RuntimeError('no pair')

def run_ep(model, env, start, goal, sz, score_fn, seed):
    na = env.config.action_vocab_size; elites = max(64//8, 8)
    env.reset(seed=seed, options={'start_state': start})
    start_emb = enc_obs(model, env._last_observation, sz)
    ctx_emb = start_emb.repeat(1, PC['history_size'], 1)
    ctx_act = torch.full((1, PC['history_size']), 0, dtype=torch.long, device=DEV)
    env.reset(seed=seed, options={'start_state': goal})
    goal_emb = enc_obs(model, env._last_observation, sz)
    env.reset(seed=seed, options={'start_state': start})
    cur = start; succ = False; inv = 0; stuck = 0; last = cur; plen = 0
    for step in range(128):
        if cur == goal: succ = True; break
        best_seq, _, _ = cem_plan(model, ctx_emb, ctx_act, goal_emb,
            horizon=PC['horizon'], history_size=PC['history_size'],
            num_candidates=64, num_elites=elites, cem_iters=1, momentum=0.1,
            num_actions=na, device=DEV, seed=seed*10000+step, score_fn=score_fn)
        a = int(best_seq[0]); prev = cur
        obs, _, _, _, info = env.step(a); cur = int(info['state']); plen += 1
        if cur==prev and a!=0: inv += 1
        if cur==last: stuck += 1
        last = cur
        new_emb = enc_obs(model, obs, sz)
        ctx_emb = torch.cat([ctx_emb[:,1:], new_emb], dim=1)
        ctx_act = torch.cat([ctx_act[:,1:], torch.tensor([[a]], dtype=torch.long, device=DEV)], dim=1)
        if cur == goal: succ = True; break
    return dict(success=succ, path_length=plen, invalid_actions=inv, stuck_steps=stuck)

def main():
    ckpt = torch.load(f'{BASE}/checkpoints/unisize_dim256.pt', map_location=DEV, weights_only=False)
    model = Unisize256(ckpt['model_config'], max_size=31).to(DEV)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    for p in model.parameters(): p.requires_grad = False

    hd = f'{BASE}/checkpoints/metric_heads'
    dh = load_head(DistanceHead, f'{hd}/distance_head_set_b:_multi-size.pt', latent_dim=256, hidden_dims=[256,128], input_mode='concat')
    gcrl = load_head(GCRLHead, f'{hd}/gcrl_head_set_b:_multi-size.pt', latent_dim=256, hidden_dims=[256,128])
    qrl = load_head(QRLHead, f'{hd}/qrl_head_set_b:_multi-size.pt', latent_dim=256, hidden_dims=[256,128])

    with open(f'{BASE}/data/splits/unisize_eval_manifest.jsonl') as f:
        entries = [json.loads(l) for l in f if l.strip()]

    # Pre-sample 100 episodes: RANDOM across all sizes
    rng = np.random.default_rng(42)
    eps = []
    for i in range(100):
        while True:
            entry = rng.choice(entries)  # CORRECT: random sampling
            env = create_env(entry)
            try:
                s,g,opt = sample_sg(env, rng); eps.append(dict(entry=entry,start=s,goal=g,opt=opt,sz=entry['maze_size']))
                break
            except RuntimeError: continue

    sz_dist = defaultdict(int)
    for e in eps: sz_dist[e['sz']] += 1
    print(f'Episode distribution: {dict(sorted(sz_dist.items()))}')
    print()

    # Score functions
    def l2_score(terminal, goal):
        return F.mse_loss(terminal, goal, reduction='none').sum(dim=-1)
    def dh_score(terminal, goal): return dh(terminal, goal)
    def l2_dh_score(terminal, goal):
        return 0.5*F.mse_loss(terminal,goal,reduction='none').sum(dim=-1) + 0.5*dh(terminal,goal)
    def gcrl_score(terminal, goal):
        return -gcrl(terminal, goal, gcrl.get_horizon_idx(PC['horizon']))
    def qrl_score(terminal, goal): return qrl(terminal, goal)

    variants = [('L2', l2_score), ('DistanceHead', dh_score), ('L2+DistanceHead', l2_dh_score),
                ('GCRL', gcrl_score), ('QRL', qrl_score)]

    print(f"{'Method':<20s} {'SR':>7s} {'SPL':>7s} {'Stuck':>7s} {'Invalid':>8s} {'S/F':>8s}")
    print('-'*60)
    all_res = {}
    for vname, score_fn in variants:
        ep_res = []; t0 = time.time()
        for i, ed in enumerate(eps):
            env = create_env(ed['entry']); seed = 42*10000 + i
            r = run_ep(model, env, ed['start'], ed['goal'], ed['sz'], score_fn, seed)
            r['op_len'] = ed['opt']; r['spl'] = ed['opt']/max(r['path_length'],ed['opt']) if r['success'] else 0.0
            ep_res.append(r)
            if (i+1) % 25 == 0:
                print(f'  [{vname}] Ep {i+1:>3d}: SR={np.mean([e["success"] for e in ep_res]):.4f}')
        succ = sum(1 for e in ep_res if e['success']); sr = succ/len(ep_res)
        spl = np.mean([e['spl'] for e in ep_res])
        all_steps = max(sum(e['path_length'] for e in ep_res),1)
        stuck_r = sum(e['stuck_steps'] for e in ep_res)/all_steps
        inv_r = sum(e['invalid_actions'] for e in ep_res)/all_steps
        all_res[vname] = dict(sr=float(sr), spl=float(spl), stuck_rate=float(stuck_r), invalid_rate=float(inv_r),
                               num_success=int(succ), num_failure=100-int(succ), time=float(time.time()-t0))
        print(f'{vname:<20s} {sr:>7.4f} {spl:>7.4f} {stuck_r:>7.4f} {inv_r:>8.4f} {succ:>3d}/{100-succ:<3d}')

    os.makedirs(f'{BASE}/results/set_b_multisize', exist_ok=True)
    with open(f'{BASE}/results/set_b_multisize/cem_results.json', 'w') as f: json.dump(all_res, f, indent=2)
    print(f"\nSaved: results/set_b_multisize/cem_results.json")

if __name__ == '__main__':
    main()
