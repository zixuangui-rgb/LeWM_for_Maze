#!/usr/bin/env python3
"""Compare encoder vs predictor latent probing quality."""
import json,sys,numpy as np,torch; sys.path.insert(0,'.')
from torch import nn; import torch.nn.functional as F
from scripts.train.train_dim256 import Unisize256
from hdwm.config import ProcgenMazeConfig; from hdwm.envs.procgen_maze import ProcgenMazeEnv

class MLPProbe(nn.Module):
    def __init__(self,in_dim,n_cls):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,n_cls))
    def forward(self,x): return self.net(x)

def train_probe(Xtr,ytr,Xv,yv,n_cls,dev,ep=30):
    m=MLPProbe(Xtr.shape[1],n_cls).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3); best=0; best_sd=None
    for e in range(ep):
        m.train(); perm=torch.randperm(Xtr.shape[0])
        for i in range(0,Xtr.shape[0],256):
            idx=perm[i:i+256]; loss=F.cross_entropy(m(Xtr[idx].to(dev)),ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): pred=m(Xv.to(dev)).argmax(-1).cpu(); acc=float((pred==yv).float().mean())
        if acc>best: best=acc; best_sd={k:v.cpu().clone() for k,v in m.state_dict().items()}
    if best_sd: m.load_state_dict(best_sd); return m,best

device=torch.device('cuda')
ckpt=torch.load('checkpoints/unisize_dim256.pt',map_location=device,weights_only=False)
model=Unisize256(ckpt['model_config'],max_size=31).to(device)
model.load_state_dict(ckpt['model_state_dict']); model.eval()

with open('data/splits/unisize_train_manifest.jsonl') as f: entries=[json.loads(l) for l in f if l.strip()]
rng=np.random.default_rng(42)
z_t_list,z_tp1_list,z_hat_list=[],[],[]
act_list,sz_list,pos_tp1_list=[],[],[]

for sz in [9,11,15,21]:
    sz_entries=[e for e in entries if e['maze_size']==sz][:100]
    for entry in sz_entries:
        cfg=ProcgenMazeConfig(height=sz,width=sz,observation_channels=5,p_noise=0,p_noop=0,p_action_turn=0,p_action_stay=0,topology_seed=entry['topology_seed'],resample_maze_per_sequence=False)
        env=ProcgenMazeEnv(cfg,seed=int(rng.integers(2**31)))
        batch=env.sample_sequence(batch_size=1,sequence_length=8)
        obs=batch.observations.to(device); actions=batch.actions.to(device); states=batch.states[0].numpy()
        with torch.no_grad():
            encoded=model.encoder(obs,sz)
            embedding,sigreg=model.embedding_projector(encoded)
            prediction=model.predictor(embedding,actions)
        for t in range(7):
            z_t_list.append(encoded[0,t].cpu())
            z_tp1_list.append(encoded[0,t+1].cpu())
            z_hat_list.append(prediction[0,t].cpu())
            act_list.append(int(actions[0,t].item()))
            sz_list.append(sz)
            pos_tp1_list.append(int(states[t+1]))

z_t=torch.stack(z_t_list); z_tp1=torch.stack(z_tp1_list); z_hat=torch.stack(z_hat_list)
acts=torch.tensor(act_list); sizes=torch.tensor(sz_list); positions=torch.tensor(pos_tp1_list)
n=z_t.shape[0]; print(f'{n} paired samples')

# Cosine similarity analysis
cos_pred_target=torch.nn.functional.cosine_similarity(z_hat,z_tp1,dim=-1)
cos_enc_target=torch.nn.functional.cosine_similarity(z_t,z_tp1,dim=-1)
print(f'cos(pred,target)={cos_pred_target.mean():.4f}+-{cos_pred_target.std():.4f}')
print(f'cos(enc,target) ={cos_enc_target.mean():.4f}+-{cos_enc_target.std():.4f}')
l2=((z_hat-z_tp1).pow(2).sum(-1)).sqrt()
print(f'L2(pred,target) ={l2.mean():.4f}+-{l2.std():.4f}')

# Per-size probing comparison
print('\n=== Probing accuracy: encoder(z_t+1) vs predictor(z_hat) ===')
print("%4s %10s %10s %8s" % ("sz","encoder","predictor","gap"))
for sz in sorted(set(sz_list)):
    m=(torch.tensor(sizes)==sz); sz_n=m.sum().item()
    if sz_n<50: continue
    idx=torch.randperm(sz_n); nv=max(1,sz_n//5); tr=idx[nv:]; vl=idx[:nv]
    Xtr_enc=z_tp1[m][tr]; Xv_enc=z_tp1[m][vl]
    Xtr_hat=z_hat[m][tr]; Xv_hat=z_hat[m][vl]
    yt=positions[m]; ytr=yt[tr]; yv=yt[vl]

    enc_accs,pred_accs=[],[]
    for tgt_fn,name in [(lambda p:p%sz,'col'),(lambda p:p//sz,'row')]:
        yt_tr=torch.tensor([tgt_fn(int(p)) for p in ytr],dtype=torch.long)
        yt_vl=torch.tensor([tgt_fn(int(p)) for p in yv],dtype=torch.long)
        for Xtr,Xv,label in [(Xtr_enc,Xv_enc,'enc'),(Xtr_hat,Xv_hat,'pred')]:
            _,acc=train_probe(Xtr,yt_tr,Xv,yt_vl,sz,device,20)
            if label=='enc': enc_accs.append(acc)
            else: pred_accs.append(acc)
    enc_avg=np.mean(enc_accs); pred_avg=np.mean(pred_accs)
    print(f'{sz:4d} {enc_avg:10.4f} {pred_avg:10.4f} {enc_avg-pred_avg:+8.4f}')
