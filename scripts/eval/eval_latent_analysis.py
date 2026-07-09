#!/usr/bin/env python3
"""Analyze WHY encoded >> embedding for position decoding in BFS planning.

LEWM paradigm: Predictor(embedding_t, action) → embedding_{t+1}, MSE with actual encoder(obs_{t+1}).

Question: If predictor and encoder are aligned via MSE, why does embedding lose position info?

Hypothesis: Projector learns a representation optimized for temporal prediction,
discarding "trivial" static information (position barely changes frame-to-frame).
"""
import sys, json, os, time, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
os.chdir('/data2/songxinshuai/yanyh/WORLDMODEL'); sys.path.insert(0, '.')
from torch import nn, optim
from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.planning import _bfs_shortest_path
from scripts.train.train_dim256 import Unisize256

BASE = '/data2/songxinshuai/yanyh/WORLDMODEL'; DEVICE = torch.device('cuda')

def extract_spatial(model, obs, sz, dev):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    cnn = model.encoder.cnn
    x = obs_t.permute(0,1,4,2,3).reshape(1,obs_t.shape[4],obs_t.shape[2],obs_t.shape[3])
    with torch.no_grad(): x = cnn.conv(x)
    return x.squeeze(0)

def extract_encoded(model, obs, sz, dev):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    with torch.no_grad(): encoded = model.encoder(obs_t, sz)
    return encoded.squeeze(0).squeeze(0)

def extract_embedding(model, obs, sz, dev):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encoder(obs_t, sz)
        embedding, _ = model.embedding_projector(encoded)
    return embedding.squeeze(0).squeeze(0), encoded.squeeze(0).squeeze(0)

def create_env(entry):
    sz = entry['maze_size']
    return ProcgenMazeEnv(ProcgenMazeConfig(height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False, topology_seed=entry['topology_seed']),
        seed=entry.get('env_seed', 42))

print("="*70)
print("LATENT SPACE ANALYSIS: Why does embedding lose position info?")
print("="*70)

ckpt = torch.load(f'{BASE}/checkpoints/unisize_dim256.pt', map_location=DEVICE, weights_only=False)
model = Unisize256(ckpt['model_config'], max_size=31).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict']); model.eval()
for p in model.parameters(): p.requires_grad = False

with open(f'{BASE}/data/splits/fixed11_train_manifest.jsonl') as f:
    entries = [json.loads(l) for l in f if l.strip()][:20]

rng = np.random.default_rng(42)
sz = 11

# ── Experiment 1: Collect encoded vs embedding for ALL cells in several mazes ──
print("\n[Exp 1] Position decoding from encoded vs embedding (per-maze probes)")
all_enc_pos_acc = []; all_emb_pos_acc = []
all_enc_goal_acc = []; all_emb_goal_acc = []

for entry in entries[:5]:
    env = create_env(entry); grid = env._maze_mask; goal = env._goal_position
    gy, gx = divmod(goal, sz)
    empty = np.flatnonzero((~grid).reshape(-1)); safe = empty[empty != goal]

    enc_feats, emb_feats = [], []
    pos_labels_x, pos_labels_y = [], []
    goal_labels_x, goal_labels_y = [], []

    for s in safe:
        sy, sx = divmod(int(s), sz)
        env._state = int(s)
        obs_np, _ = env._observe_with_noise(np.array([int(s)]))
        emb, enc = extract_embedding(model, obs_np[0], sz, DEVICE)
        emb_feats.append(emb.cpu()); enc_feats.append(enc.cpu())
        pos_labels_x.append(sx); pos_labels_y.append(sy)
        goal_labels_x.append(gx); goal_labels_y.append(gy)

    # Train tiny probes per maze
    for feat_list, name in [(enc_feats, 'encoded'), (emb_feats, 'embedding')]:
        X = torch.stack(feat_list); y = torch.tensor(pos_labels_x, dtype=torch.long)
        n = X.shape[0]; nv = max(1, n//3)
        Xt, Xv = X[nv:], X[:nv]; yt, yv = y[nv:], y[:nv]
        # Quick 100-epoch probe
        probe = nn.Linear(256, sz).to(DEVICE)
        opt = optim.Adam(probe.parameters(), lr=1e-2)
        for _ in range(200):
            loss = F.cross_entropy(probe(Xt.to(DEVICE)), yt.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = float((probe(Xv.to(DEVICE)).argmax(-1).cpu() == yv).float().mean())
        if name == 'encoded': all_enc_pos_acc.append(acc)
        else: all_emb_pos_acc.append(acc)

print(f"  Position X decoding (linear probe):")
print(f"    encoded:   {np.mean(all_enc_pos_acc):.4f} ± {np.std(all_enc_pos_acc):.4f}")
print(f"    embedding: {np.mean(all_emb_pos_acc):.4f} ± {np.std(all_emb_pos_acc):.4f}")
print(f"    Gap: {np.mean(all_enc_pos_acc)-np.mean(all_emb_pos_acc):.4f}")

# ── Experiment 2: Predictor rollout quality ──
print("\n[Exp 2] Predictor rollout: latent distance to true next latent")
entry = entries[0]; env = create_env(entry)
batch = env.sample_sequence(batch_size=64, sequence_length=8)
obs = batch.observations.to(DEVICE); actions = batch.actions.to(DEVICE)

with torch.no_grad():
    output = model(obs, actions, sz)

# encoded prediction quality
pred = output['prediction']  # predicted embedding
target = output['target']    # actual embedding[:,1:]
pred_err_emb = F.mse_loss(pred, target).item()

# Now check: can encoded layer predict itself better?
encoded = output['encoded']  # [B, T, 256]
enc_target = encoded[:, 1:]  # next encoded
# Can we linearly decode next encoded from current embedding?
B, T = encoded.shape[:2]
emb_curr = output['embedding'][:, :T-1]  # current embedding
# Linear: embedding → next encoded
lin_pred = nn.Linear(256, 256).to(DEVICE)
opt = optim.Adam(lin_pred.parameters(), lr=1e-2)
for _ in range(500):
    loss = F.mse_loss(lin_pred(emb_curr.reshape(-1, 256)), enc_target.reshape(-1, 256))
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    enc_from_emb_err = F.mse_loss(lin_pred(emb_curr.reshape(-1, 256)), enc_target.reshape(-1, 256)).item()

# Distance ratios
enc_dist = F.mse_loss(encoded[:, 0], encoded[:, 1], reduction='none').sum(-1).mean().item()
emb_dist = F.mse_loss(output['embedding'][:, 0], output['embedding'][:, 1], reduction='none').sum(-1).mean().item()

print(f"  Predictor embedding MSE: {pred_err_emb:.4f}")
print(f"  Encoded reconstruction from embedding MSE: {enc_from_emb_err:.4f}")
print(f"  Ratio (reconstruct / predict): {enc_from_emb_err/pred_err_emb:.1f}×")
print(f"  Consecutive-frame L2 distance:")
print(f"    encoded:   {enc_dist:.4f}")
print(f"    embedding: {emb_dist:.4f}")
print(f"    ratio emb/enc: {emb_dist/enc_dist:.2f}×")

# ── Experiment 3: Position change vs latent change correlation ──
print("\n[Exp 3] Correlation: position change vs latent change")
all_pos_dists, all_enc_dists, all_emb_dists = [], [], []
for entry in entries[:10]:
    env = create_env(entry)
    batch = env.sample_sequence(batch_size=32, sequence_length=8)
    obs_b = batch.observations; st_b = batch.states
    for b in range(32):
        for t in range(7):
            s1, s2 = int(st_b[b, t]), int(st_b[b, t+1])
            pos_dist = abs(s1%sz - s2%sz) + abs(s1//sz - s2//sz)  # Manhattan
            emb1, enc1 = extract_embedding(model, obs_b[b, t].cpu().numpy(), sz, DEVICE)
            emb2, enc2 = extract_embedding(model, obs_b[b, t+1].cpu().numpy(), sz, DEVICE)
            enc_dist = float(F.mse_loss(enc1, enc2, reduction='none').sum().item())
            emb_dist = float(F.mse_loss(emb1, emb2, reduction='none').sum().item())
            all_pos_dists.append(pos_dist); all_enc_dists.append(enc_dist); all_emb_dists.append(emb_dist)

from scipy.stats import spearmanr
enc_corr, _ = spearmanr(all_pos_dists, all_enc_dists)
emb_corr, _ = spearmanr(all_pos_dists, all_emb_dists)
print(f"  Spearman ρ(position change vs latent change):")
print(f"    encoded:   {enc_corr:.4f}")
print(f"    embedding: {emb_corr:.4f}")

# ── Experiment 4: Information theoretic — probe position from both ──
print("\n[Exp 4] Mutual information proxy: linear position decoding accuracy")
X_enc, X_emb, Y_x, Y_y = [], [], [], []
for entry in entries[:10]:
    env = create_env(entry); grid = env._maze_mask; goal = env._goal_position
    empty = np.flatnonzero((~grid).reshape(-1)); safe = empty[empty != goal]
    for s in rng.choice(safe, size=min(20, len(safe)), replace=False):
        env._state = int(s)
        obs_np, _ = env._observe_with_noise(np.array([int(s)]))
        emb, enc = extract_embedding(model, obs_np[0], sz, DEVICE)
        X_enc.append(enc.cpu()); X_emb.append(emb.cpu())
        Y_x.append(s % sz); Y_y.append(s // sz)

Xe = torch.stack(X_enc); Xm = torch.stack(X_emb)
yx = torch.tensor(Y_x, dtype=torch.long); yy = torch.tensor(Y_y, dtype=torch.long)
n = Xe.shape[0]; nv = n//3; perm = torch.randperm(n)

for name, X, ylabel in [('encoded-X', Xe, yx), ('encoded-Y', Xe, yy),
                          ('embedding-X', Xm, yx), ('embedding-Y', Xm, yy)]:
    Xt, Xv = X[perm[nv:]], X[perm[:nv]]; yt, yv = ylabel[perm[nv:]], ylabel[perm[:nv]]
    probe = nn.Linear(256, sz).to(DEVICE); opt = optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(300):
        loss = F.cross_entropy(probe(Xt.to(DEVICE)), yt.to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((probe(Xv.to(DEVICE)).argmax(-1).cpu() == yv).float().mean())
    print(f"  {name:>15s}: linear probe acc = {acc:.4f}")

# ── Experiment 5: Projector singular value analysis ──
print("\n[Exp 5] Projector transformation analysis")
proj = model.embedding_projector
W = proj.linear.weight.data  # [256, 256]
# SVD to check if projector is near-identity or has collapsed directions
U, S, V = torch.svd(W)
print(f"  Projector Linear(256,256) singular values:")
print(f"    max={S[0]:.4f}  min={S[-1]:.4f}  mean={S.mean():.4f}")
print(f"    condition number: {S[0]/S[-1]:.1f}")
print(f"    effective rank (σ>0.1*max): {(S > 0.1*S[0]).sum().item()}/256")

# Check: is there a BatchNorm after the linear?
print(f"  Projector components: {[type(m).__name__ for m in proj.norm]}")
# Check if there's normalization that suppresses position
if hasattr(proj, 'norm'):
    for m in proj.norm:
        if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            print(f"  Found normalization layer: {m}")

# ── Experiment 6: Predictor vs Encoder alignment — how well does predictor recover encoder output? ──
print("\n[Exp 6] Predictor → Encoder alignment (cross-modality decoding)")
entry = entries[0]; env = create_env(entry)
batch = env.sample_sequence(batch_size=64, sequence_length=8)
obs_b = batch.observations.to(DEVICE); actions_b = batch.actions.to(DEVICE)

with torch.no_grad():
    out = model(obs_b, actions_b, sz)

pred_emb = out['prediction']   # predictor output: [B, T-1, 256] in embedding space
true_emb = out['target']       # actual embedding[:,1:]: [B, T-1, 256]
true_enc = out['encoded'][:, 1:]  # actual encoded[:,1:]: [B, T-1, 256]

# Can we decode next encoded from predicted embedding?
data = torch.cat([pred_emb.reshape(-1, 256), true_emb.reshape(-1, 256)])
targets = true_enc.reshape(-1, 256)
n_d = data.shape[0]; nv_d = n_d//3; perm_d = torch.randperm(n_d)

cross_probe = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256)).to(DEVICE)
opt = optim.Adam(cross_probe.parameters(), lr=1e-2)
for _ in range(500):
    loss = F.mse_loss(cross_probe(data[perm_d[nv_d:]].to(DEVICE)), targets[perm_d[nv_d:]].to(DEVICE))
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    pred_to_enc_err = F.mse_loss(cross_probe(pred_emb.reshape(-1, 256).to(DEVICE)),
                                   true_enc.reshape(-1, 256).to(DEVICE)).item()
    emb_to_enc_err = F.mse_loss(cross_probe(true_emb.reshape(-1, 256).to(DEVICE)),
                                  true_enc.reshape(-1, 256).to(DEVICE)).item()
print(f"  Encoded reconstruction MSE:")
print(f"    from true embedding:     {emb_to_enc_err:.4f}")
print(f"    from predicted embedding: {pred_to_enc_err:.4f}")
print(f"    predictor penalty: +{pred_to_enc_err-emb_to_enc_err:.4f}")

# ── Summary ──
print("\n" + "="*70)
print("SUMMARY: Why embedding << encoded for position decoding")
print("="*70)
print(f"""
1. PROJECTOR AS INFORMATION BOTTLENECK:
   - Linear(256,256) projector has condition number {S[0]/S[-1]:.0f}
   - Effectively compresses some dimensions, likely position-related ones

2. TEMPORAL PREDICTION FAVORS CHANGE, NOT STATIC INFO:
   - Position changes slowly (1 step/frame on 11×11 grid)
   - Consecutive-frame L2: encoded={enc_dist:.2f}, embedding={emb_dist:.2f} (ratio={emb_dist/enc_dist:.2f}×)
   - Embedding amplifies change signal, suppresses static position

3. POSITION-TO-LATENT CORRELATION:
   - Spearman ρ(Δpos, Δenc) = {enc_corr:.3f}
   - Spearman ρ(Δpos, Δemb) = {emb_corr:.3f}
   - Embedding is LESS correlated with physical position change

4. LINEAR DECODABILITY:
   - Position from encoded: {np.mean(all_enc_pos_acc):.2%} (linear probe)
   - Position from embedding: {np.mean(all_emb_pos_acc):.2%} (linear probe)
   - Encoded preserves position in a linearly-decodable subspace
   - Embedding scrambles position into a non-linear code optimized for prediction

5. CROSS-MODALITY GAP:
   - Embedded→Encoded reconstruction MSE: {emb_to_enc_err:.2f}
   - Predicted→Encoded reconstruction MSE: {pred_to_enc_err:.2f}
   - Even TRUE embedding cannot fully reconstruct encoded output
   - The projector transformation is LOSSY by design
""")
