#!/usr/bin/env python3
"""Final comparison: L2, DistanceHead, GCRL, QRL, L2+DistanceHead on CEM planner."""
import argparse, json, sys, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.metric_heads.distance_head import DistanceHead
from hdwm.metric_heads.gcrl_head import GCRLHead
from hdwm.metric_heads.qrl_head import QRLHead
from hdwm.planning import _bfs_shortest_path, cem_plan
from scripts.train.train_ablation_models import OriginalLeWM

PC = dict(horizon=12, num_candidates=64, cem_iters=1, receding_horizon=1, history_size=3)

def load_lewm(ckpt, dev):
    c = torch.load(ckpt, map_location=dev, weights_only=False)
    m = OriginalLeWM(c["model_config"], max_size=31).to(dev)
    m.load_state_dict(c["model_state_dict"], strict=True); m.eval()
    for p in m.parameters(): p.requires_grad = False
    return m

def load_head(cls, ckpt, dev):
    c = torch.load(ckpt, map_location=dev, weights_only=False)
    if cls == DistanceHead:
        h = DistanceHead(latent_dim=c["config"]["latent_dim"], hidden_dims=c["config"]["hidden_dims"], input_mode=c["config"]["input_mode"]).to(dev)
    elif cls == GCRLHead:
        h = GCRLHead(latent_dim=c["config"]["latent_dim"], hidden_dims=[256,128]).to(dev)
    else:
        h = QRLHead(latent_dim=c["config"]["latent_dim"], hidden_dims=[256,128]).to(dev)
    h.load_state_dict(c["head_state_dict"]); h.eval()
    for p in h.parameters(): p.requires_grad = False
    return h

def enc_obs(model, obs, sz, dev):
    t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        e = model.encoder(t, sz); emb, _ = model.embedding_projector(e)
    return emb

def create_env(entry):
    sz = entry["maze_size"]
    return ProcgenMazeEnv(ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry["topology_seed"]),
        seed=entry.get("level_seed",42))

def sample_sg(env, rng):
    obs = env._maze_mask; empty = ~obs
    flat = empty.reshape(-1).copy()
    if hasattr(env,"_goal_position"): flat[env._goal_position] = False
    pos = np.flatnonzero(flat); w = env.config.width
    for _ in range(500):
        s = int(rng.choice(pos)); g = int(rng.choice(pos))
        if s==g: continue
        d = _bfs_shortest_path(obs,s,g,w)
        if d is not None and d>=3: return s,g,d
    raise RuntimeError("no pair")

def run_ep(model, env, start, goal, sz, dev, score_fn, seed):
    na = env.config.action_vocab_size; w = env.config.width; obs_mask = env._maze_mask
    elites = max(PC["num_candidates"]//8, 8)
    env.reset(seed=seed, options={"start_state": start})
    start_emb = enc_obs(model, env._last_observation, sz, dev)
    ctx_emb = start_emb.repeat(1, PC["history_size"], 1)
    ctx_act = torch.full((1, PC["history_size"]), na-1, dtype=torch.long, device=dev)
    env.reset(seed=seed, options={"start_state": goal})
    goal_emb = enc_obs(model, env._last_observation, sz, dev)
    env.reset(seed=seed, options={"start_state": start})
    cur = start; path = [cur]; succ = False; inv = 0; stuck = 0; last = cur
    for step in range(128):
        if cur == goal: succ = True; break
        best_seq, _, _ = cem_plan(model, ctx_emb, ctx_act, goal_emb,
            horizon=PC["horizon"], history_size=PC["history_size"],
            num_candidates=PC["num_candidates"], num_elites=elites, cem_iters=PC["cem_iters"],
            momentum=0.1, num_actions=na, device=dev, seed=seed*10000+step, score_fn=score_fn)
        a = int(best_seq[0]); prev = cur
        obs, _, _, _, info = env.step(a); cur = int(info["state"]); path.append(cur)
        if cur==prev and a!=0: inv += 1
        if cur==last: stuck += 1
        last = cur
        new_emb = enc_obs(model, obs, sz, dev)
        ctx_emb = torch.cat([ctx_emb[:,1:], new_emb], dim=1)
        ctx_act = torch.cat([ctx_act[:,1:], torch.tensor([[a]], dtype=torch.long, device=dev)], dim=1)
        if cur == goal: succ = True; break
    plen = len(path)-1
    fbfs = None
    if not succ:
        d = _bfs_shortest_path(obs_mask, cur, goal, w); fbfs = d if d is not None else -1
    return dict(success=succ, path_length=plen, invalid_actions=inv, stuck_steps=stuck, final_bfs_distance=fbfs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-episodes", type=int, default=100)
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path("results/phase4_metric_heads/final_comparison")
    out.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("FINAL COMPARISON: L2 vs DistanceHead vs GCRL vs QRL")
    print("="*60)

    model = load_lewm("checkpoints/ablation/original_lewm.pt", dev)
    dh = load_head(DistanceHead, "checkpoints/metric_heads/distance_head.pt", dev)
    gcrl = load_head(GCRLHead, "checkpoints/metric_heads/gcrl_head.pt", dev)
    qrl = load_head(QRLHead, "checkpoints/metric_heads/qrl_head.pt", dev)

    with open("data/splits/fixed11_test_manifest.jsonl") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    # Pre-sample episodes
    rng = np.random.default_rng(42)
    eps = []
    for i in range(args.num_episodes):
        while True:
            e = entries[i%len(entries)] if i<len(entries) else rng.choice(entries)
            env = create_env(e)
            try:
                s,g,opt = sample_sg(env, rng); eps.append(dict(entry=e,start=s,goal=g,opt=opt,sz=e["maze_size"]))
                break
            except: continue

    # Define scoring functions
    def gcrl_score(terminal, goal):
        # Use horizon=PC["horizon"]=12 -> find nearest bucket
        h_idx = gcrl.get_horizon_idx(PC["horizon"])
        logits = gcrl(terminal, goal, h_idx)
        return -logits  # lower logit = less reachable = higher cost

    def qrl_score(terminal, goal):
        return qrl(terminal, goal)

    def dh_score(terminal, goal):
        return dh(terminal, goal)

    def l2_dh_score(terminal, goal):
        l2 = F.mse_loss(terminal, goal, reduction="none").sum(dim=-1)
        return 0.5*l2 + 0.5*dh(terminal, goal)

    variants = [
        (None, "L2"),
        (dh_score, "DistanceHead"),
        (l2_dh_score, "L2+DistanceHead"),
        (gcrl_score, "GCRL"),
        (qrl_score, "QRL"),
    ]

    all_res = {}
    print(f"\nRunning {args.num_episodes} episodes × {len(variants)} variants...")
    for score_fn, vname in variants:
        print(f"\n  [{vname}]")
        ep_res = []; t0 = time.time()
        for i, ed in enumerate(eps):
            env = create_env(ed["entry"]); seed = 42*10000 + i
            r = run_ep(model, env, ed["start"], ed["goal"], ed["sz"], dev, score_fn, seed)
            r["op_len"] = ed["opt"]; r["spl"] = ed["opt"]/max(r["path_length"],ed["opt"]) if r["success"] else 0.0
            ep_res.append(r)
            if (i+1)%25==0:
                srs = [e["success"] for e in ep_res]; print(f"    Ep {i+1:>3d}: SR={np.mean(srs):.4f}")
        elapsed = time.time()-t0
        succ = sum(1 for e in ep_res if e["success"]); sr = succ/len(ep_res)
        spl = np.mean([e["spl"] for e in ep_res])
        fails = [e for e in ep_res if not e["success"]]
        a_bfs = np.mean([e["final_bfs_distance"] for e in fails if e["final_bfs_distance"] is not None]) if fails else 0
        all_steps = max(sum(e["path_length"] for e in ep_res),1)
        stuck_r = sum(e["stuck_steps"] for e in ep_res)/all_steps
        inv_r = sum(e["invalid_actions"] for e in ep_res)/all_steps
        all_res[vname] = dict(sr=float(sr), spl=float(spl), avg_path_success=float(np.mean([e["path_length"] for e in ep_res if e["success"]]) if succ>0 else 0),
                              avg_final_bfs=float(a_bfs), stuck_rate=float(stuck_r), invalid_rate=float(inv_r),
                              num_success=int(succ), num_failure=len(ep_res)-int(succ), time=float(elapsed))
        print(f"    SR={sr:.4f}  SPL={spl:.4f}  stuck={stuck_r:.4f}  ({elapsed:.0f}s)")

    print("\n"+"="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Method':<20s} {'SR':>7s} {'SPL':>7s} {'Stuck':>7s} {'Invalid':>8s} {'S/F':>8s}")
    print("-"*60)
    best = max(all_res.items(), key=lambda x: x[1]["sr"])
    for vn, r in all_res.items():
        mk = " ★" if vn==best[0] else ""
        print(f"{vn:<20s} {r['sr']:>7.4f} {r['spl']:>7.4f} {r['stuck_rate']:>7.4f} {r['invalid_rate']:>8.4f} {r['num_success']:>3d}/{r['num_failure']:>3d}{mk}")
    print("="*60)
    print(f"Best: {best[0]} (SR={best[1]['sr']:.4f})")

    with open(out/"final_comparison.json","w") as f: json.dump(all_res, f, indent=2)
    print(f"\nSaved: {out/'final_comparison.json'}")

if __name__ == "__main__":
    main()
