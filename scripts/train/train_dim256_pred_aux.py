#!/usr/bin/env python3
"""Train 256-dim UnisizeLeWM with predictor-side auxiliary losses ONLY.

Key difference from train_dim256.py:
  - Position heads (abs, rel, goal) on PREDICTOR OUTPUT ONLY
  - NO encoder-side aux losses
  - Time alignment: predictor[t] → position[t+1] (next frame)

Comparison:
  - train_dim256.py:     aux losses on ENCODER only  → checkpoints/unisize_dim256.pt
  - train_dim256_pred_aux.py: aux losses on PREDICTOR only → checkpoints/unisize_dim256_pred_aux.pt
"""

import sys, json, time, numpy as np, torch, torch.nn.functional as F; sys.path.insert(0,'.')
from torch import nn
from hdwm.config import LEWMCNNConfig, ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.losses import SIGReg
from hdwm.models.lewm import CNNEncoder, NextEmbeddingPredictor
from hdwm.models.shared import LatentEmbeddingProjector
from scripts.train.train_canonical_lewm import compute_position_labels

class SizeCondEnc(nn.Module):
    def __init__(self,config,max_size=31):
        super().__init__(); self.cnn=CNNEncoder(config); cd=config.effective_model_dim
        self.size_embed=nn.Embedding(max_size+1,32)
        self.fuse=nn.Sequential(nn.Linear(cd+32,cd),nn.LayerNorm(cd),nn.ReLU(),nn.Linear(cd,cd))
    def forward(self,obs,size):
        cnn_out=self.cnn(obs); B,T,M=cnn_out.shape
        sz_t=torch.full((B,T),size,device=cnn_out.device,dtype=torch.long)
        return self.fuse(torch.cat([cnn_out,self.size_embed(sz_t)],dim=-1))

class Unisize256PredAux(nn.Module):
    """Unisize256 with position heads on PREDICTOR output only.

    Predictor heads: pred_abs, pred_rel, pred_goal on prediction[t] → position[t+1]
    No encoder-side aux heads — clean comparison with vanilla encoder-only aux model.
    """
    def __init__(self,config,max_size=31):
        super().__init__(); self.config=config; self.encoder=SizeCondEnc(config,max_size)
        self.embedding_projector=LatentEmbeddingProjector(config)
        self.predictor=NextEmbeddingPredictor(config)
        cd=config.effective_model_dim
        self.pred_abs_head=nn.Sequential(nn.Linear(cd,256),nn.ReLU(),nn.Linear(256,2))
        self.pred_rel_head=nn.Sequential(nn.Linear(cd,256),nn.ReLU(),nn.Linear(256,2))
        self.pred_goal_head=nn.Sequential(nn.Linear(cd,256),nn.ReLU(),nn.Linear(256,2))

    def forward(self,obs,actions,size):
        encoded=self.encoder(obs,size)
        embedding,sigreg=self.embedding_projector(encoded)
        prediction=self.predictor(embedding,actions)
        target=embedding[:,1:]
        return {
            'encoded':encoded,
            'embedding':embedding,
            'sigreg_embedding':sigreg,
            'prediction':prediction,
            'target':target,
            'pred_abs_pos_pred':self.pred_abs_head(prediction),
            'pred_rel_pos_pred':self.pred_rel_head(prediction),
            'pred_goal_pos_pred':self.pred_goal_head(prediction),
        }

if __name__ == '__main__':
    device=torch.device('cuda')
    with open('data/splits/unisize_train_manifest.jsonl') as f:
        entries=[json.loads(l) for l in f if l.strip()]
    print(f'Train entries: {len(entries)}')
    base=ProcgenMazeConfig(height=25,width=25,observation_channels=5,
        p_noise=0,p_noop=0,p_action_turn=0,p_action_stay=0,
        resample_maze_per_sequence=False)
    cfg=LEWMCNNConfig(env_config=base,latent_dim=256,cnn_channels=(64,128,256),
        latent_batch_norm=True,embedding_stage='post_bn',sigreg_stage='post_bn',
        predictor_heads=16)
    model=Unisize256PredAux(cfg,max_size=31).to(device)
    sigreg=SIGReg(knots=17,num_proj=1024).to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.0); model.train()
    rng=np.random.default_rng(42); t0=time.time(); steps=30000

    loss_log={'total':[],'pred':[],'pred_abs':[],'pred_rel':[],'pred_goal':[]}

    for step in range(1,steps+1):
        entry=entries[step%len(entries)]; sz=entry['maze_size']
        ec=ProcgenMazeConfig(height=sz,width=sz,observation_channels=5,
            p_noise=0,p_noop=0,p_action_turn=0,p_action_stay=0,
            resample_maze_per_sequence=False,topology_seed=entry['topology_seed'])
        env=ProcgenMazeEnv(ec,seed=int(rng.integers(2**31)))
        batch=env.sample_sequence(batch_size=256,sequence_length=8)
        obs=batch.observations.to(device); actions=batch.actions.to(device)
        output=model(obs,actions,sz)

        # LeWM prediction loss
        pred_loss=F.mse_loss(output['prediction'],output['target'])
        sigreg_loss=sigreg(output['sigreg_embedding'].transpose(0,1))

        # Position labels for predictor (time t+1: frames 1..T)
        x,y,dx,dy=compute_position_labels(batch.states,obs[:,:,:,:,3],sz)
        abs_target=torch.stack([x,y],dim=-1).to(device)      # [B,T,2]
        rel_target=torch.stack([dx,dy],dim=-1).to(device)     # [B,T,2]

        B,seq=obs.shape[:2]
        goal_flat=obs[:,:,:,:,3].reshape(B,seq,-1)
        goal_state=goal_flat.argmax(dim=-1)                   # [B,T]
        goal_target=torch.stack([
            (goal_state%sz).float()/max(sz-1,1),
            (goal_state//sz).float()/max(sz-1,1)],dim=-1).to(device)  # [B,T,2]

        # Predictor-side aux losses ONLY (time t+1: frames 1..T)
        pred_abs_loss=F.mse_loss(output['pred_abs_pos_pred'],abs_target[:,1:])
        pred_rel_loss=F.mse_loss(output['pred_rel_pos_pred'],rel_target[:,1:])
        pred_goal_loss=F.mse_loss(output['pred_goal_pos_pred'],goal_target[:,1:])

        loss=pred_loss+0.09*sigreg_loss+0.5*pred_abs_loss+1.0*pred_rel_loss+0.5*pred_goal_loss

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

        for k,v in [('total',loss),('pred',pred_loss),
                     ('pred_abs',pred_abs_loss),('pred_rel',pred_rel_loss),('pred_goal',pred_goal_loss)]:
            loss_log[k].append(v.item())

        if step%500==0:
            elapsed=max(time.time()-t0,0.01)
            avg={k:np.mean(loss_log[k][-500:]) for k in loss_log}
            print(f'Step {step:>6d}/{steps} | total={avg["total"]:.4f} '
                  f'pred_aux(ab={avg["pred_abs"]:.4f} rl={avg["pred_rel"]:.4f} gl={avg["pred_goal"]:.4f}) '
                  f'| {500/elapsed:.0f}it/s',flush=True)
            t0=time.time()

    torch.save({
        'model_state_dict':model.state_dict(),
        'model_config':cfg,
        'latent_dim':256,'cnn_channels':(64,128,256),
        'steps':steps,
        'final_loss':np.mean(loss_log['total'][-500:]),
        'final_pred_abs_loss':np.mean(loss_log['pred_abs'][-500:]),
        'final_pred_rel_loss':np.mean(loss_log['pred_rel'][-500:]),
        'final_pred_goal_loss':np.mean(loss_log['pred_goal'][-500:]),
        'aux_type':'predictor_only',
    },'checkpoints/unisize_dim256_pred_aux.pt')
    print(f'Saved: checkpoints/unisize_dim256_pred_aux.pt')
