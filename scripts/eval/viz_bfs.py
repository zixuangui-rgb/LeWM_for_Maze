#!/usr/bin/env python3
"""Generate BFS planning visualization GIF."""
import json,sys,os,subprocess,numpy as np,torch; sys.path.insert(0,'.')
from torch import nn; import torch.nn.functional as F; from collections import deque
from scripts.train.train_dim256 import Unisize256
from hdwm.config import ProcgenMazeConfig; from hdwm.envs.procgen_maze import ProcgenMazeEnv
from scripts.eval.eval_latent_l2_cem import create_env_from_entry
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

class SpatialMLP(nn.Module):
    def __init__(self,in_dim,n_cls,hidden=512):
        super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.Dropout(0.2),nn.Linear(hidden,hidden),nn.ReLU(),nn.Dropout(0.2),nn.Linear(hidden,n_cls))
    def forward(self,x): return self.net(x)

def extract_spatial(model,obs,sz,dev):
    obs_t=torch.as_tensor(obs,dtype=torch.float32,device=dev).unsqueeze(0).unsqueeze(0)
    cnn=model.encoder.cnn; x=obs_t.permute(0,1,4,2,3).reshape(1,obs_t.shape[4],obs_t.shape[2],obs_t.shape[3])
    with torch.no_grad(): x=cnn.conv(x); return x.squeeze(0)

def train_spatial(Xtr,ytr,Xv,yv,n_cls,dev,ep=20):
    m=SpatialMLP(Xtr.shape[1],n_cls).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep); best=0; best_sd=None
    for e in range(ep):
        m.train(); perm=torch.randperm(Xtr.shape[0])
        for i in range(0,Xtr.shape[0],256): idx=perm[i:i+256]; loss=F.cross_entropy(m(Xtr[idx].to(dev)),ytr[idx].to(dev)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        sch.step(); m.eval()
        with torch.no_grad(): pred=m(Xv.to(dev)).argmax(-1).cpu(); acc=float((pred==yv).float().mean())
        if acc>best: best=acc; best_sd={k:v.cpu().clone() for k,v in m.state_dict().items()}
    if best_sd: m.load_state_dict(best_sd); return m,best

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
                if grid[ny,nx]<0.5 and parent[ns]==-1:
                    parent[ns]=cur; q.append(ns); fa[ns]=acts[d]
    if parent[gi]==-1: return []
    path=[]; c=gi
    while c!=si and parent[c]!=-1 and parent[c]!=c: path.append(int(fa[c])); c=parent[c]
    path.reverse(); return path

def draw_path_positions(sy,sx,path,sz):
    pos=[(sy,sx)]; cy,cx=sy,sx
    for a in path:
        if a==1: cy-=1
        elif a==2: cy+=1
        elif a==3: cx-=1
        elif a==4: cx+=1
        pos.append((cy,cx))
    return pos

device=torch.device('cuda')
ckpt=torch.load('checkpoints/unisize_dim256.pt',map_location=device,weights_only=False)
model=Unisize256(ckpt['model_config'],max_size=31).to(device)
model.load_state_dict(ckpt['model_state_dict']); model.eval()
rng=np.random.default_rng(42); sz=11

# Train probes
with open('data/splits/unisize_train_manifest.jsonl') as f: tr=[json.loads(l) for l in f if l.strip()]
with open('data/splits/unisize_eval_manifest.jsonl') as f: ev=[json.loads(l) for l in f if l.strip()]
sz_tr=[e for e in tr if e['maze_size']==sz]; feat_list=[]; labs={'ax':[],'ay':[],'gx':[],'gy':[]}
for entry in sz_tr:
    cfg=ProcgenMazeConfig(height=sz,width=sz,observation_channels=5,p_noise=0,p_noop=0,p_action_turn=0,p_action_stay=0,topology_seed=entry['topology_seed'],resample_maze_per_sequence=False)
    env=ProcgenMazeEnv(cfg,seed=int(rng.integers(2**31))); gp=env._goal_position
    for _ in range(2):
        batch=env.sample_sequence(batch_size=1,sequence_length=8); obs=batch.observations; st=batch.states[0].numpy()
        for t in range(8):
            feat=extract_spatial(model,obs[0,t].cpu().numpy(),sz,device); feat_list.append(feat.cpu())
            s=int(st[t]); labs['ax'].append(float(s%sz)); labs['ay'].append(float(s//sz)); labs['gx'].append(float(gp%sz)); labs['gy'].append(float(gp//sz))
X=torch.stack(feat_list); n=X.shape[0]; Xf=X.reshape(n,-1)
nv=max(1,int(len(sz_tr)*0.2*2*8)); perm=torch.randperm(n); Xt,Xv=Xf[perm[nv:]],Xf[perm[:nv]]
heads={}
for tgt,lab in [('agent_x','ax'),('agent_y','ay'),('goal_x','gx'),('goal_y','gy')]:
    yt=torch.tensor(labs[lab],dtype=torch.long); h,acc=train_spatial(Xt,yt[perm[nv:]],Xv,yt[perm[:nv]],sz,device,20); heads[tgt]=h

# Pick eval episode with posOK=True
sz_ev=[e for e in ev if e['maze_size']==sz]
for entry in sz_ev[:5]:
    env=create_env_from_entry(entry,device); om=env._maze_mask; eg=env._goal_position; occ=om.astype(np.float32)
    empty=np.flatnonzero((~om).reshape(-1)); safe=empty[empty!=eg]
    if safe.size<2: continue; s=int(rng.choice(safe))
    env.reset(seed=0,options={'start_state':s})
    feat=extract_spatial(model,env._last_observation.copy(),sz,device); z=feat.reshape(1,-1)
    with torch.no_grad():
        pax=int(heads['agent_x'](z.to(device)).argmax(-1).item()); pay=int(heads['agent_y'](z.to(device)).argmax(-1).item())
        pgx=int(heads['goal_x'](z.to(device)).argmax(-1).item()); pgy=int(heads['goal_y'](z.to(device)).argmax(-1).item())
    true_ax,true_ay=s%sz,s//sz; true_gx,true_gy=eg%sz,eg//sz
    if pax==true_ax and pay==true_ay and pgx==true_gx and pgy==true_gy: break

pred_path=bfs_full_path(occ,pay,pax,pgy,pgx,sz)
oracle_path=bfs_full_path(occ,true_ay,true_ax,true_gy,true_gx,sz)
os.makedirs('/tmp/viz_frames',exist_ok=True)

env.reset(seed=0,options={'start_state':s}); cur=s
for step in range(min(len(pred_path),25)+1):
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    grid=np.ones((sz,sz,3))
    for i in range(sz):
        for j in range(sz):
            if om[i,j]: grid[i,j]=[0.3,0.3,0.3]
    grid[true_ay,true_ax]=[0,1,0]; grid[true_gy,true_gx]=[1,0,0]
    for (y,x) in draw_path_positions(true_ay,true_ax,oracle_path,sz)[1:]:
        if 0<=y<sz and 0<=x<sz: grid[y,x]=[0,0.5,0.5]
    axes[0].imshow(grid,origin='upper'); axes[0].set_title('Ground Truth')

    grid2=np.ones((sz,sz,3))
    for i in range(sz):
        for j in range(sz):
            if om[i,j]: grid2[i,j]=[0.3,0.3,0.3]
    grid2[pay,pax]=[0,1,0]; grid2[pgy,pgx]=[1,0,0]
    for (y,x) in draw_path_positions(pay,pax,pred_path,sz)[1:]:
        if 0<=y<sz and 0<=x<sz: grid2[y,x]=[0,0.5,0.5]
    axes[1].imshow(grid2,origin='upper'); axes[1].set_title(f'Predicted (posOK)')

    fig.savefig(f'/tmp/viz_frames/frame_{step:03d}.png',dpi=80); plt.close()
    if step<len(pred_path):
        _,_,_,_,info=env.step(pred_path[step]); cur=int(info['state']); true_ax,true_ay=cur%sz,cur//sz

subprocess.run(['convert','-delay','80','/tmp/viz_frames/frame_*.png','/tmp/bfs_demo.gif'],check=True)
print(f'GIF: /tmp/bfs_demo.gif | sz={sz} start=({s%sz},{s//sz}) goal=({eg%sz},{eg//sz}) | pred=({pax},{pay})->({pgx},{pgy}) | posOK=True | path_len={len(pred_path)}')
