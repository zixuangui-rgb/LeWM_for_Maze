#!/usr/bin/env python3
"""Extreme OOD: sz=99 symbolic BFS planning."""
import sys,os,subprocess,numpy as np,torch,torch.nn.functional as F; sys.path.insert(0,'.')
from torch import nn; from collections import deque
from scripts.train.train_dim256 import SizeCondEnc, Unisize256
from hdwm.config import ProcgenMazeConfig; from hdwm.envs.procgen_maze import ProcgenMazeEnv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def extr(model,obs,sz,dev):
    obs_t=torch.as_tensor(obs,dtype=torch.float32,device=dev).unsqueeze(0).unsqueeze(0)
    cnn=model.encoder.cnn; x=obs_t.permute(0,1,4,2,3).reshape(1,5,sz,sz)
    with torch.no_grad(): x=cnn.conv(x); return x.squeeze(0)

class BigMLP(nn.Module):
    def __init__(self,in_dim,n_cls,hidden=1024):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.Dropout(0.3),nn.Linear(hidden,hidden),nn.ReLU(),nn.Dropout(0.3),nn.Linear(hidden,n_cls))
    def forward(self,x): return self.net(x)

def bfs_fp(occ,sy,sx,gy,gx,sz):
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

device=torch.device('cuda')
ckpt=torch.load('checkpoints/unisize_dim256.pt',map_location=device,weights_only=False)
model=Unisize256(ckpt['model_config'],max_size=99).to(device)
sd=ckpt['model_state_dict']; msd=model.state_dict()
for k,v in sd.items():
    if k in msd and msd[k].shape==v.shape: msd[k]=v
model.load_state_dict(msd); model.eval()
print('Model loaded (max_size=99)')

sz=99; rng=np.random.default_rng(42)
n_seeds,traj,seq=500,1,4
feat_list=[]; labs={'ax':[],'ay':[],'gx':[],'gy':[]}
for ts in range(n_seeds):
    cfg=ProcgenMazeConfig(height=sz,width=sz,observation_channels=5,p_noise=0,p_noop=0,p_action_turn=0,p_action_stay=0,topology_seed=ts,resample_maze_per_sequence=False)
    env=ProcgenMazeEnv(cfg,seed=ts*1000); gp=env._goal_position
    for ep in range(traj):
        e2=ProcgenMazeEnv(cfg,seed=ts*10000+ep)
        batch=e2.sample_sequence(1,seq); obs=batch.observations; st=batch.states[0].numpy()
        for t in range(seq):
            feat=extr(model,obs[0,t].cpu().numpy(),sz,device); feat_list.append(feat.cpu())
            s=int(st[t]); labs['ax'].append(float(s%sz)); labs['ay'].append(float(s//sz))
            labs['gx'].append(float(gp%sz)); labs['gy'].append(float(gp//sz))
X=torch.stack(feat_list); n=X.shape[0]; Xf=X.reshape(n,-1)
nv=n//5; perm=torch.randperm(n); Xt,Xv=Xf[perm[nv:]],Xf[perm[:nv]]
print(f'{Xf.shape[1]} dims, {n} frames',flush=True)

heads={}
for tgt,lab in [('agent_x','ax'),('agent_y','ay'),('goal_x','gx'),('goal_y','gy')]:
    yt=torch.tensor(labs[lab],dtype=torch.long)
    m=BigMLP(Xf.shape[1],sz).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30); best=0; best_sd=None
    for ep in range(30):
        m.train(); bp=torch.randperm(Xt.shape[0])
        for i in range(0,Xt.shape[0],128):
            bidx=bp[i:i+128]; loss=F.cross_entropy(m(Xt[bidx].to(device)),yt[perm[nv:]][bidx].to(device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        sch.step(); m.eval()
        with torch.no_grad(): pred=m(Xv.to(device)).argmax(-1).cpu(); acc=float((pred==yt[perm[:nv]]).float().mean())
        if acc>best: best=acc; best_sd={k:v.cpu().clone() for k,v in m.state_dict().items()}
    if best_sd: m.load_state_dict(best_sd)
    heads[tgt]=m; print(f'{tgt} val={best:.4f}',flush=True)

print('BFS eval...',flush=True)
succ_eps=[]; fail_eps=[]; total=0
for ts in range(5000,5030):
    cfg=ProcgenMazeConfig(height=sz,width=sz,observation_channels=5,p_noise=0,p_noop=0,p_action_turn=0,p_action_stay=0,topology_seed=ts,resample_maze_per_sequence=False)
    env=ProcgenMazeEnv(cfg,seed=0); om=env._maze_mask; eg=env._goal_position; occ=om.astype(np.float32)
    empty=np.flatnonzero((~om).reshape(-1)); safe=empty[empty!=eg]
    if safe.size<2: continue
    s=int(rng.choice(safe))
    env.reset(seed=0,options={'start_state':s})
    feat=extr(model,env._last_observation.copy(),sz,device); z=feat.reshape(1,-1)
    with torch.no_grad():
        pax=int(heads['agent_x'](z.to(device)).argmax(-1).item())
        pay=int(heads['agent_y'](z.to(device)).argmax(-1).item())
        pgx=int(heads['goal_x'](z.to(device)).argmax(-1).item())
        pgy=int(heads['goal_y'](z.to(device)).argmax(-1).item())
    posOK=(pax==s%sz and pay==s//sz and pgx==eg%sz and pgy==eg//sz)
    pred_path=bfs_fp(occ,pay,pax,pgy,pgx,sz)
    env.reset(seed=0,options={'start_state':s}); cur=s; ok=False; path_s=[s]
    for act in (pred_path if pred_path else [int(rng.integers(1,5))]):
        if cur==eg: ok=True; break
        _,_,_,_,info=env.step(act); cur=int(info['state']); path_s.append(cur)
        if cur==eg: ok=True; break
    total+=1
    ep_d={'seed':ts,'start':s,'goal':eg,'posOK':posOK,'ok':ok,'plen':len(path_s),'path':path_s,'pax':pax,'pay':pay,'pgx':pgx,'pgy':pgy,'occ':occ,'ppath':pred_path}
    if ok: succ_eps.append(ep_d)
    else: fail_eps.append(ep_d)

sr=len(succ_eps)/total; pok=sum(1 for e in succ_eps+fail_eps if e['posOK'])/total
print('SR=%.3f posOK=%.3f (%d/%d)'%(sr,pok,len(succ_eps),total))

# GIFs
for tag,eps in [('success',succ_eps[:1]),('fail',fail_eps[:1])]:
    if not eps: continue
    ep=eps[0]; occ_g=ep['occ']; pp=ep['ppath']
    fdir='/tmp/sz99_%s_frames'%tag; os.makedirs(fdir,exist_ok=True)
    env2=ProcgenMazeEnv(ProcgenMazeConfig(height=sz,width=sz,observation_channels=5,p_noise=0,p_noop=0,p_action_turn=0,p_action_stay=0,topology_seed=ep['seed'],resample_maze_per_sequence=False),seed=0)
    env2.reset(seed=0,options={'start_state':int(ep['start'])})
    max_fr=min(len(ep['path']),40)
    for fr in range(max_fr):
        cur=ep['path'][fr]; cy,cx=cur//sz,cur%sz; gy2,gx2=ep['goal']//sz,ep['goal']%sz
        r0,r1=max(0,cy-20),min(sz,cy+20); c0,c1=max(0,cx-20),min(sz,cx+20)
        grid=np.ones((r1-r0,c1-c0,3))
        for i in range(r0,r1):
            for j in range(c0,c1):
                if occ_g[i,j]>0.5: grid[i-r0,j-c0]=[0.3,0.3,0.3]
        grid[cy-r0,cx-c0]=[0,1,0]
        if r0<=gy2<r1 and c0<=gx2<c1: grid[gy2-r0,gx2-c0]=[1,0,0]
        if pp and fr<len(pp):
            py,px=cy,cx
            for a in pp[fr:fr+3]:
                if a==1: py-=1
                elif a==2: py+=1
                elif a==3: px-=1
                elif a==4: px+=1
                if r0<=py<r1 and c0<=px<c1: grid[py-r0,px-c0]=[0,0.5,0.5]
        fig,ax=plt.subplots(1,1,figsize=(8,8))
        ax.imshow(grid,origin='upper')
        ax.set_title('sz=99 %s step=%d/%d posOK=%s pred=(%d,%d)->(%d,%d)'%(tag,fr,max_fr,str(ep['posOK']),ep['pax'],ep['pay'],ep['pgx'],ep['pgy']))
        fig.savefig('%s/frame_%03d.png'%(fdir,fr),dpi=50); plt.close()
    subprocess.run(['convert','-delay','120','%s/frame_*.png'%fdir,'/tmp/sz99_%s.gif'%tag],check=True)
    print('GIF: /tmp/sz99_%s.gif (%d steps)'%(tag,max_fr))
print('DONE')
