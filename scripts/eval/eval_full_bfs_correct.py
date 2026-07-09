#!/usr/bin/env python3
"""Full-path BFS with probes trained on TRAIN MANIFEST, evaluated on EVAL MANIFEST.

Protocol:
- Train probes on `unisize_train_manifest` data (same seed dist as backbone)
- Evaluate on `unisize_eval_manifest` (hold-out topologies)
- Use oracle occupancy for BFS (isolates position decoding quality)

This eliminates the seed-range overfitting issue.
"""

import argparse, json, sys, time, numpy as np, torch; sys.path.insert(0,'.')
from pathlib import Path
from torch import nn; import torch.nn.functional as F
from collections import deque
from scripts.train.train_dim256 import Unisize256
from hdwm.config import ProcgenMazeConfig; from hdwm.envs.procgen_maze import ProcgenMazeEnv
from scripts.eval.eval_latent_l2_cem import create_env_from_entry

class SpatialMLP(nn.Module):
    def __init__(self,in_dim,n_cls,hidden=512):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.Dropout(0.2),nn.Linear(hidden,hidden),nn.ReLU(),nn.Dropout(0.2),nn.Linear(hidden,n_cls))
    def forward(self,x): return self.net(x)

def extract_spatial(model,obs,sz,dev):
    obs_t=torch.as_tensor(obs,dtype=torch.float32,device=dev).unsqueeze(0).unsqueeze(0)
    cnn=model.encoder.cnn; x=obs_t.permute(0,1,4,2,3).reshape(1,obs_t.shape[4],obs_t.shape[2],obs_t.shape[3])
    with torch.no_grad(): x=cnn.conv(x); return x.squeeze(0)

def train_spatial(Xtr,ytr,Xv,yv,n_cls,dev,ep=30):
    m=SpatialMLP(Xtr.shape[1],n_cls).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep); best=0; best_sd=None
    for e in range(ep):
        m.train(); perm=torch.randperm(Xtr.shape[0])
        for i in range(0,Xtr.shape[0],256):
            idx=perm[i:i+256]; loss=F.cross_entropy(m(Xtr[idx].to(dev)),ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        sch.step(); m.eval()
        with torch.no_grad(): pred=m(Xv.to(dev)).argmax(-1).cpu(); acc=float((pred==yv).float().mean())
        if acc>best: best=acc; best_sd={k:v.cpu().clone() for k,v in m.state_dict().items()}
    if best_sd: m.load_state_dict(best_sd)
    return m,best

def bfs_full_path(occ,sy,sx,gy,gx,sz):
    H,W=sz,sz; grid=occ[:H,:W].astype(np.float32)
    if grid[sy,sx]>=0.5 or grid[gy,gx]>=0.5: return []
    parent=np.full(H*W,-1,np.int32); fa=np.full(H*W,-1,np.int32)
    q=deque(); si=sy*W+sx; gi=gy*W+gx; q.append(si); parent[si]=si
    dirs=[(-1,0),(1,0),(0,-1),(0,1)]; acts=[1,2,3,4]
    while q:
        cur=q.popleft()
        if cur==gi: break
        y,x=divmod(cur,W)
        for d,(dy,dx) in enumerate(dirs):
            ny,nx=y+dy,x+dx
            if 0<=ny<H and 0<=nx<W:
                ns=ny*W+nx
                if grid[ny,nx]<0.5 and parent[ns]==-1: parent[ns]=cur; q.append(ns); fa[ns]=acts[d]
    if parent[gi]==-1: return []
    path=[]; c=gi
    while c!=si and parent[c]!=-1 and parent[c]!=c: path.append(int(fa[c])); c=parent[c]
    path.reverse(); return path

def parse_sizes(text):
    return [int(item.strip()) for item in text.split(',') if item.strip()]

def verify_holdout(train_entries, eval_entries):
    train_topo = {(entry["maze_size"], entry["topology_seed"]) for entry in train_entries}
    eval_topo = {(entry["maze_size"], entry["topology_seed"]) for entry in eval_entries}
    train_layout = {entry.get("layout_hash") for entry in train_entries if entry.get("layout_hash")}
    eval_layout = {entry.get("layout_hash") for entry in eval_entries if entry.get("layout_hash")}
    train_task = {entry.get("task_hash") for entry in train_entries if entry.get("task_hash")}
    eval_task = {entry.get("task_hash") for entry in eval_entries if entry.get("task_hash")}
    if train_topo & eval_topo or train_layout & eval_layout or train_task & eval_task:
        raise ValueError("train/eval leakage detected")

parser = argparse.ArgumentParser()
parser.add_argument("--model-ckpt", default="checkpoints/unisize_dim256.pt")
parser.add_argument("--train-manifest", default="data/splits/unisize_train_manifest.jsonl")
parser.add_argument("--eval-manifest", default="data/splits/unisize_eval_manifest.jsonl")
parser.add_argument("--sizes", default="9,11,13,15,17,19,21")
parser.add_argument("--episodes-per-size", type=int, default=30)
parser.add_argument("--probe-trajectories-per-maze", type=int, default=2)
parser.add_argument("--probe-epochs", type=int, default=20)
parser.add_argument("--output", default="results/set_b_multisize/symbolic_bfs_probe_eval.json")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

device=torch.device(args.device)
ckpt=torch.load(args.model_ckpt,map_location=device,weights_only=False)
model=Unisize256(ckpt['model_config'],max_size=31).to(device)
model.load_state_dict(ckpt['model_state_dict']); model.eval()

with open(args.train_manifest) as f: tr_entries=[json.loads(l) for l in f if l.strip()]
with open(args.eval_manifest) as f: ev_entries=[json.loads(l) for l in f if l.strip()]
verify_holdout(tr_entries, ev_entries)

rng=np.random.default_rng(args.seed)
results = {
    "model_ckpt": args.model_ckpt,
    "train_manifest": args.train_manifest,
    "eval_manifest": args.eval_manifest,
    "sizes": parse_sizes(args.sizes),
    "episodes_per_size": args.episodes_per_size,
    "by_size": {},
}

for sz in parse_sizes(args.sizes):
    print(f'\n=== sz={sz} ===')
    # ── Generate probe training data from TRAIN MANIFEST ──
    sz_tr=[e for e in tr_entries if e['maze_size']==sz]
    feat_list=[]; labs={'ax':[],'ay':[],'gx':[],'gy':[]}
    for entry in sz_tr:
        cfg=ProcgenMazeConfig(height=sz,width=sz,observation_channels=5,p_noise=0,p_noop=0,p_action_turn=0,p_action_stay=0,topology_seed=entry['topology_seed'],resample_maze_per_sequence=False)
        env=ProcgenMazeEnv(cfg,seed=int(rng.integers(2**31))); gp=env._goal_position
        for _ in range(args.probe_trajectories_per_maze):
            batch=env.sample_sequence(batch_size=1,sequence_length=8); obs=batch.observations; st=batch.states[0].numpy()
            for t in range(8):
                feat=extract_spatial(model,obs[0,t].cpu().numpy(),sz,device); feat_list.append(feat.cpu())
                s=int(st[t]); labs['ax'].append(float(s%sz)); labs['ay'].append(float(s//sz))
                labs['gx'].append(float(gp%sz)); labs['gy'].append(float(gp//sz))
    X=torch.stack(feat_list); n=X.shape[0]; Xf=X.reshape(n,-1)
    n_entries=len(sz_tr); n_val_frames=int(n_entries*0.2*2*8)  # 20% entries for val
    nv=max(1,n_val_frames); perm=torch.randperm(n); Xt,Xv=Xf[perm[nv:]],Xf[perm[:nv]]
    print(f'  Train probe: {n} frames from {n_entries} entries, val={nv}',flush=True)

    # ── Train position heads ──
    heads={}
    for tgt,lab in [('agent_x','ax'),('agent_y','ay'),('goal_x','gx'),('goal_y','gy')]:
        yt=torch.tensor(labs[lab],dtype=torch.long)
        h,acc=train_spatial(Xt,yt[perm[nv:]],Xv,yt[perm[:nv]],sz,device,args.probe_epochs)
        heads[tgt]=h; print(f'  {tgt}: val_acc={acc:.4f}',flush=True)

    # ── BFS eval on EVAL MANIFEST ──
    sz_ev=[e for e in ev_entries if e['maze_size']==sz]
    n_ep=min(args.episodes_per_size,len(sz_ev)); sampled=rng.choice(sz_ev,size=n_ep,replace=False)
    succ,total,pos_ok=0,0,0
    for entry in sampled:
        env=create_env_from_entry(entry,device)
        om=env._maze_mask; eg=env._goal_position; occ=om.astype(np.float32)
        empty=np.flatnonzero((~om).reshape(-1)); safe=empty[empty!=eg]
        if safe.size<2: continue
        s=int(rng.choice(safe))
        env.reset(seed=0,options={'start_state':s})
        feat=extract_spatial(model,env._last_observation.copy(),sz,device); z=feat.reshape(1,-1)
        with torch.no_grad():
            pax=int(heads['agent_x'](z.to(device)).argmax(-1).item())
            pay=int(heads['agent_y'](z.to(device)).argmax(-1).item())
            pgx=int(heads['goal_x'](z.to(device)).argmax(-1).item())
            pgy=int(heads['goal_y'](z.to(device)).argmax(-1).item())
        true_ax,true_ay=s%sz,s//sz; true_gx,true_gy=eg%sz,eg//sz
        if pax==true_ax and pay==true_ay and pgx==true_gx and pgy==true_gy: pos_ok+=1
        pred_path=bfs_full_path(occ,pay,pax,pgy,pgx,sz)
        env.reset(seed=0,options={'start_state':s}); cur=s; ok=False
        for act in (pred_path if pred_path else [int(rng.integers(1,5))]):
            if cur==eg: ok=True; break
            _,_,_,_,info=env.step(act); cur=int(info['state'])
            if cur==eg: ok=True; break
        total+=1
        if ok: succ+=1
    sr = succ / total
    pos_ok_rate = pos_ok / total
    results["by_size"][str(sz)] = {
        "n": int(total),
        "sr": float(sr),
        "pos_ok": float(pos_ok_rate),
        "num_success": int(succ),
        "num_failure": int(total - succ),
    }
    print(f'  BFS SR={sr:.3f} posOK={pos_ok_rate:.3f} ({succ}/{total})',flush=True)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out}")
print('DONE')
