# LeWM Symbolic BFS Planning — 复现指南

本文档描述如何复现论文的核心实验：**在 Procgen Maze 上训练 LeWM backbone → lightweight MLP probe → Symbolic BFS Planning 达到 >80% SR**。

## 环境与依赖

```bash
# Python 3.10+, PyTorch 2.5+ (CUDA 12.4)
pip install torch numpy gymnasium pydantic omegaconf matplotlib
# Procgen Maze 环境
git clone git@github.com:yuhaoyan04/Procgen.git
cd Procgen && pip install -e .
cd ..
```

## 数据划分策略

### Train/Eval Split（严格 Hold-out Topology）

| Split | Manifest | Sizes | Entries | Topology Seeds |
|-------|----------|-------|---------|----------------|
| Train | `data/splits/unisize_train_manifest.jsonl` | 9,11,13,15,17,19,21 | 400/size | 90000-90416, 110000-110399, ... |
| Eval  | `data/splits/unisize_eval_manifest.jsonl` | 9,11,13,15,17,19,21,23,25 | 100/size | 190000+, 210000+, ... |

- **Train ∩ Eval topology overlap = 0**（严格 hold-out）
- Backbone 和 Probe 训练**仅使用 train manifest 数据**
- BFS 评估**仅使用 eval manifest 数据**
- Sizes 23,25 为 OOD（backbone 从未见过）

### Topology 种子命名规则

每个 `(maze_size, topology_seed)` 对产生唯一的迷宫拓扑（wall layout + goal position）。不同 maze_size 的相同 seed 产生不同的迷宫。

---

## 实验流程

### Step 1: 训练 256-dim Unisize LeWM Backbone

```bash
python scripts/train/train_dim256.py
# 输出: checkpoints/unisize_dim256.pt
# 配置: latent_dim=256, cnn_channels=(64,128,256), λ_rel=1.0, λ_abs=0.1, λ_goal=0.5
# 训练: 30k steps, batch=256, seq_len=8, lr=1e-3
# 数据: unisize_train_manifest (sizes 9-21, random walk trajectories)
# 时间: ~45 min (RTX 3090)
```

**关键配置**：
- Encoder: 3-layer CNN (stride=2) + SizeConditionedEncoder
- Aux losses (abs_pos, rel_pos, goal_pos) 加在 **encoded 层**（projector 之前）
- 训练 Loss = pred_loss + 0.09*sigreg + 0.1*abs + 1.0*rel + 0.5*goal

### Step 2: 训练 Per-size MLP Probe（使用 Spatial Features）

```bash
# 对每个 size 独立训练 probe（sz=9..21 for seen, sz=23,25 for OOD）
python scripts/probe/probe_optimal.py \
    --ckpt checkpoints/unisize_dim256.pt \
    --sizes 9,11,13,15,17,19,21,23,25 \
    --n-topos 80 --n-topos-val 20 --traj-per-topo 10 --epochs 50 \
    --device cuda --output results/optimal_probe.csv
# 输出: checkpoints/heads/canonical_lewm_rel1.0_persize_sz{sz}.pt
# 时间: ~2 min/size
```

**Probe 架构**：Spatial features (CNN pre-pooling output, 256×H'×W') → flatten → 4-layer MLP(512) + Dropout → sz-class logits

**为什么用 spatial features？** Projector（Linear+BN）会丢弃 position 信息（详见 `scripts/analysis/compare_enc_pred.py`）。Spatial features (CNN conv 输出 pre-pooling) 保留了 wall topology 和 agent/goal 的空间位置。

### Step 3: Symbolic BFS Planning

```bash
# Full-path BFS eval (train probe on train manifest, eval on eval manifest)
python scripts/eval/eval_full_bfs_correct.py
# 输出: 每 size 的 SR 和 posOK 指标
# 时间: ~5 min (所有 sizes)
```

**Planning 流程**：
1. 编码 obs → 提取 spatial features → MLP probe 预测 agent_x, agent_y, goal_x, goal_y
2. BFS 在 oracle occupancy grid 上从 (agent_x, agent_y) 规划到 (goal_x, goal_y) → 完整动作序列
3. **一次性执行全部动作**（不每步重新 decode）→ 避免误差累积

**为什么用 oracle occupancy？** 隔离 position 解码质量——SR 纯粹反映 probe 精度。实际部署时需添加 occupancy decoder。

---

## 预期结果

### Probe 准确率（Eval Manifest 上的 hold-out 评估）

| Size | Agent Acc | Goal Acc | 状态 |
|:---:|:---:|:---:|:---:|
| 9 | 1.00 | 0.94 | seen |
| 11 | 1.00 | 0.91 | seen |
| 13+ | 0.95+ | 0.85+ | seen |
| 23,25 | 0.90+ | 1.00(val) | OOD* |

> *OOD: probe 用独立拓扑训练，eval manifest 评估。seed-range gap 存在。

### Symbolic BFS SR（Oracle Occupancy）

| Size | SR | posOK | 状态 |
|:---:|:---:|:---:|:---:|
| 9-15 | **1.00** | 0.97-1.00 | seen ✅ |
| 17 | **0.97** | 0.90 | seen ✅ |
| 19 | **0.90** | 0.83 | seen ✅ |
| 21 | **0.80** | 0.63 | seen ✅ |
| 23 | 0.53 | 0.33 | OOD |
| 25 | 0.60 | 0.40 | OOD |

---

## 关键发现

1. **Projector Gap**：encoded 层 probing (acc>0.95) >> embedding 层 probing (acc~0.75)。Projector 训练目标仅为 prediction loss，position 信息（相邻帧几乎不变）被当作"噪声"丢弃。
2. **Spatial Features 是解药**：CNN pre-pooling 特征保留了 wall topology → probe 能达到 1.00 acc。
3. **Per-size Probe > Unified Head**：每个 size 独立训练小 MLP 优于共享大 MLP。
4. **Full-path BFS > Receding-horizon**：一次 decode + 完整路径 → SR ≈ posOK（位置准确率）。

## 项目结构

```
world_model/
├── hdwm/                    # 模型核心代码
│   ├── models/lewm.py       # CNN Encoder, Predictor
│   ├── models/shared.py     # Projector, Transformer
│   ├── envs/procgen_maze.py # Procgen Maze 环境
│   ├── config.py            # 配置定义
│   └── planning.py          # BFS, CEM 规划器
├── scripts/
│   ├── train/
│   │   ├── train_canonical_lewm.py  # 128-dim 训练
│   │   └── train_dim256.py          # 256-dim 训练
│   ├── probe/
│   │   ├── probe_optimal.py   # Per-size MLP probing
│   │   ├── probe_holdout.py   # Hold-out probing (layer comparison)
│   │   └── probe_spatial.py   # Spatial features probing
│   ├── eval/
│   │   ├── eval_full_bfs_correct.py  # 主实验: full-path BFS
│   │   ├── viz_bfs.py                # 可视化 GIF
│   │   └── sz99_ood.py               # Extreme OOD (sz=99)
│   └── analysis/
│       └── compare_enc_pred.py  # Encoder vs Predictor probing 分析
├── data/splits/              # Train/Eval 数据划分
│   ├── unisize_train_manifest.jsonl
│   └── unisize_eval_manifest.jsonl
├── configs/                  # 配置文件
├── results/                  # 实验报告
└── REPRODUCE.md              # 本文档
```
