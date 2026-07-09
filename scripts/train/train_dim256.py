#!/usr/bin/env python3
"""Train 256-dim UnisizeLeWM. Safe to import: classes defined at module level."""
import argparse, sys, json, time, numpy as np, torch, torch.nn.functional as F; sys.path.insert(0,'.')
from pathlib import Path
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

class Unisize256(nn.Module):
    def __init__(self,config,max_size=31):
        super().__init__(); self.config=config; self.encoder=SizeCondEnc(config,max_size)
        self.embedding_projector=LatentEmbeddingProjector(config); self.predictor=NextEmbeddingPredictor(config)
        cd=config.effective_model_dim
        self.abs_pos_head=nn.Sequential(nn.Linear(cd,256),nn.ReLU(),nn.Linear(256,2))
        self.rel_pos_head=nn.Sequential(nn.Linear(cd,256),nn.ReLU(),nn.Linear(256,2))
        self.goal_pos_head=nn.Sequential(nn.Linear(cd,256),nn.ReLU(),nn.Linear(256,2))
    def forward(self,obs,actions,size):
        encoded=self.encoder(obs,size); embedding,sigreg=self.embedding_projector(encoded)
        prediction=self.predictor(embedding,actions); target=embedding[:,1:]
        return {'encoded':encoded,'embedding':embedding,'sigreg_embedding':sigreg,'prediction':prediction,'target':target,'abs_pos_pred':self.abs_pos_head(encoded),'rel_pos_pred':self.rel_pos_head(encoded),'goal_pos_pred':self.goal_pos_head(encoded)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-manifest', default='data/splits/unisize_train_manifest.jsonl')
    parser.add_argument('--output', default='checkpoints/unisize_dim256.pt')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seq-len', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device=torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with open(args.train_manifest) as f: entries=[json.loads(l) for l in f if l.strip()]
    print(f'Train entries: {len(entries)}')
    base=ProcgenMazeConfig(height=25,width=25,observation_channels=5,p_noise=0,p_noop=0,p_action_turn=0,p_action_stay=0,resample_maze_per_sequence=False)
    cfg=LEWMCNNConfig(env_config=base,latent_dim=256,cnn_channels=(64,128,256),latent_batch_norm=True,embedding_stage='post_bn',sigreg_stage='post_bn',predictor_heads=16)
    model=Unisize256(cfg,max_size=31).to(device); sigreg=SIGReg(knots=17,num_proj=1024).to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0); model.train()
    rng=np.random.default_rng(args.seed); t0=time.time(); steps=args.steps
    loss_log={'total':[],'pred':[],'abs':[],'rel':[],'goal':[]}
    for step in range(1,steps+1):
        entry=entries[step%len(entries)]; sz=entry['maze_size']
        ec=ProcgenMazeConfig(height=sz,width=sz,observation_channels=5,p_noise=0,p_noop=0,p_action_turn=0,p_action_stay=0,resample_maze_per_sequence=False,topology_seed=entry['topology_seed'])
        env=ProcgenMazeEnv(ec,seed=int(rng.integers(2**31)))
        batch=env.sample_sequence(batch_size=args.batch_size,sequence_length=args.seq_len)
        obs=batch.observations.to(device); actions=batch.actions.to(device)
        output=model(obs,actions,sz)
        pred_loss=F.mse_loss(output['prediction'],output['target'])
        sigreg_loss=sigreg(output['sigreg_embedding'].transpose(0,1))
        x,y,dx,dy=compute_position_labels(batch.states,obs[:,:,:,:,3],sz)
        abs_target=torch.stack([x,y],dim=-1).to(device); rel_target=torch.stack([dx,dy],dim=-1).to(device)
        abs_loss=F.mse_loss(output['abs_pos_pred'],abs_target); rel_loss=F.mse_loss(output['rel_pos_pred'],rel_target)
        B,seq=obs.shape[:2]; goal_flat=obs[:,:,:,:,3].reshape(B,seq,-1); goal_state=goal_flat.argmax(dim=-1)
        goal_target=torch.stack([(goal_state%sz).float()/max(sz-1,1),(goal_state//sz).float()/max(sz-1,1)],dim=-1).to(device)
        goal_loss=F.mse_loss(output['goal_pos_pred'],goal_target)
        loss=pred_loss+0.09*sigreg_loss+0.1*abs_loss+1.0*rel_loss+0.5*goal_loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        for k,v in [('total',loss),('pred',pred_loss),('abs',abs_loss),('rel',rel_loss),('goal',goal_loss)]:
            loss_log[k].append(v.item())
        if step%500==0:
            elapsed=max(time.time()-t0,0.01)
            avg={k:np.mean(loss_log[k][-500:]) for k in loss_log}
            print(f'Step {step:>6d}/{steps} | total={avg["total"]:.4f} pred={avg["pred"]:.4f} abs={avg["abs"]:.4f} rel={avg["rel"]:.4f} goal={avg["goal"]:.4f} | {500/elapsed:.0f}it/s',flush=True)
            t0=time.time()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict':model.state_dict(),
        'model_config':cfg,
        'latent_dim':256,
        'cnn_channels':(64,128,256),
        'steps':steps,
        'batch_size':args.batch_size,
        'seq_len':args.seq_len,
        'lr':args.lr,
        'seed':args.seed,
        'train_manifest':args.train_manifest,
        'final_loss':np.mean(loss_log['total'][-500:]),
        'final_pred_loss':np.mean(loss_log['pred'][-500:]),
        'final_abs_loss':np.mean(loss_log['abs'][-500:]),
        'final_rel_loss':np.mean(loss_log['rel'][-500:]),
        'final_goal_loss':np.mean(loss_log['goal'][-500:]),
    }, output)
    print(f'Saved: {output}')
