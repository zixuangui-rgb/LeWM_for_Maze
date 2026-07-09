# LeWM 在 Procgen Maze 中导航可行性研究：逐步开发与实验计划

## 总体目标

系统性评估 LeWM 在 Procgen Maze 导航任务中的可行性，重点研究：

1. 标准 LeWM 的 latent 是否支持导航规划；
2. latent L2 distance + CEM/MPC 是否可行；
3. distance head、GCRL、QRL、candidate ranking 等方法是否能修复 latent planning；
4. encoder 架构改造是否能改善 topology representation；
5. topology-supervised LeWM 是否能形成可规划的 map-like latent；
6. LeWM 在固定 maze size 与 maze size OOD 下的泛化能力。

---

# 全局实验约束

## Maze size 设置

实验分为两类。

### 固定 size 实验

对于不涉及 maze size OOD 的实验，统一使用：

```text
maze_size = 11
```

这些实验包括：

```text
LeWM baseline
probing
latent L2 planning
distance head
GCRL
QRL
candidate ranking
CEM ablation
encoder architecture ablation
topology decoder
topology-supervised LeWM
```

### Maze size OOD 实验

仅在明确研究 size generalization 时使用：

```text
maze_size ∈ {7, 9, 11, 13, 15}
```

推荐设置：

```text
train sizes: 7, 9, 11
test ID sizes: 7, 9, 11 with unseen topology seeds
test OOD sizes: 13, 15 with unseen topology seeds
```

也可以增加 extrapolation setting：

```text
train sizes: 7, 9
test sizes: 11, 13, 15
```

所有 size-OOD 实验必须单独标注，不能和固定 11×11 实验混合比较。

---

## Train/Test topology split

环境必须严格分为 train 和 test。

要求：

```text
1. train level seeds 和 test level seeds 不重叠
2. train maze topology 和 test maze topology 不重叠
3. test 关卡不能出现在 train 中
4. 对每个 maze 计算 topology hash，检测重复
5. 所有模型训练只能使用 train split
6. 所有最终结果必须在 test split 上报告
```

Topology hash 建议定义为：

```text
topology_hash = SHA256(maze_size + wall_free_grid + start_cell + goal_cell)
```

也可以额外记录：

```text
layout_hash = SHA256(maze_size + wall_free_grid)
task_hash = SHA256(maze_size + wall_free_grid + start_cell + goal_cell)
```

其中：

```text
layout_hash 用于检测相同 maze 拓扑
task_hash 用于检测相同 start/goal 任务
```

---

## Probe 实验约束

Probe 必须使用训练好的 LeWM 权重。

要求：

```text
1. 加载已训练 LeWM checkpoint
2. freeze LeWM encoder / dynamics / decoder
3. 只训练 probe module
4. probe 不允许反向更新 LeWM
5. probe 分别评估 z_true 和 z_hat
```

Probe 输入包括：

```text
z_true = encoder(real observation)
z_hat = rollout latent from learned dynamics
pre-pooling feature, if accessible
spatial latent, if architecture supports it
```

Probe 目标包括：

```text
agent x coordinate
agent y coordinate
agent cell index
goal x coordinate
goal y coordinate
goal cell index
BFS distance to goal
oracle first action
valid action mask
local wall / occupancy
intersection / dead-end indicator
```

---

# Prompt 0：实验目录、registry 与全局配置

## 给 Claude Code 的 prompt

You are implementing a systematic feasibility study of LeWM navigation in Procgen Maze.

Before running any model training, set up the experiment infrastructure.

Create the following directory structure:

```text
configs/
configs/env/
configs/models/
configs/planners/
configs/probes/
configs/metrics/

scripts/
scripts/env/
scripts/train/
scripts/eval/
scripts/probe/
scripts/analysis/

results/
results/registry/
results/phase0_env/
results/phase1_lewm_baseline/
results/phase2_probe/
results/phase3_l2_cem/
results/phase4_metric_heads/
results/phase5_planner_ablation/
results/phase6_encoder_ablation/
results/phase7_topology_supervision/
results/phase8_size_ood/
results/final_report/

checkpoints/
data/
```

Create a global experiment registry:

```text
results/registry/experiment_registry.csv
```

with columns:

```text
experiment_id
date
phase
method
maze_size
size_ood_setting
train_seed_start
train_num_levels
test_seed_start
test_num_levels
train_topology_hash_file
test_topology_hash_file
checkpoint
encoder_architecture
latent_dim
uses_spatial_latent
uses_topology_supervision
probe_target
metric_head
planner
cem_horizon
cem_candidates
cem_iterations
random_seed
SR
SPL
mean_return
first_action_acc
neighbor_argmin_acc
bfs_spearman
agent_cell_acc
goal_cell_acc
occupancy_iou
notes
```

Also create a helper function:

```python
register_experiment(config, metrics, output_path)
```

Every future experiment must write one row into this registry.

Deliverables:

```text
results/registry/experiment_registry.csv
scripts/utils/experiment_registry.py
configs/base_config.yaml
```

Stop after infrastructure is complete. Do not train any model yet.

---

# Prompt 1：环境 split 与 topology hash 验证

## 给 Claude Code 的 prompt

Implement strict train/test splitting for Procgen Maze.

We need two settings.

Setting A: fixed-size experiments

```text
maze_size = 11
train split: train seeds only
validation split: validation seeds only
test split: test seeds only
```

Setting B: maze-size OOD experiments

```text
maze_size ∈ {7, 9, 11, 13, 15}
train sizes: 7, 9, 11
test ID sizes: 7, 9, 11 with unseen seeds/topologies
test OOD sizes: 13, 15 with unseen seeds/topologies
```

For every generated level, extract:

```text
maze_size
level_seed
wall_free_grid
agent_start_cell
goal_cell
layout_hash = SHA256(maze_size + wall_free_grid)
task_hash = SHA256(maze_size + wall_free_grid + start_cell + goal_cell)
```

Verify:

```text
1. no overlap between train and test level seeds
2. no overlap between train and test layout_hash
3. no overlap between train and test task_hash
4. no duplicate task_hash inside each split unless intentionally allowed
```

Save manifests:

```text
data/splits/fixed11_train_manifest.jsonl
data/splits/fixed11_val_manifest.jsonl
data/splits/fixed11_test_manifest.jsonl

data/splits/size_ood_train_manifest.jsonl
data/splits/size_ood_test_id_manifest.jsonl
data/splits/size_ood_test_ood_manifest.jsonl
```

Write reports:

```text
results/phase0_env/split_summary_fixed11.md
results/phase0_env/split_summary_size_ood.md
```

The report must include:

```text
number of levels
number of unique layout_hash
number of unique task_hash
overlap counts between train and test
maze size distribution
path length distribution
```

Do not proceed until all overlap counts are zero.

---

# Prompt 2：Pixel parser、BFS oracle 与环境 sanity check

## 给 Claude Code 的 prompt

Implement and validate oracle tools for Procgen Maze.

Build:

```python
parse_observation_to_grid(obs) -> wall_free_grid, agent_cell, goal_cell
build_graph_from_grid(wall_free_grid)
bfs_distance(grid, agent, goal)
bfs_value_map(grid, goal)
oracle_first_action(grid, agent, goal)
valid_action_mask(grid, agent)
is_dead_end(grid, cell)
is_intersection(grid, cell)
```

Validate on fixed 11×11 train and test levels.

Metrics:

```text
agent parse accuracy
goal parse accuracy
wall/free IoU
parsed BFS first-action match vs environment oracle
invalid oracle action rate
wall-pointing oracle action rate
tree property: E == V - 1
cycle count
unique shortest first action rate
```

Save:

```text
results/phase0_env/parser_bfs_oracle_metrics.csv
results/phase0_env/parser_bfs_oracle_report.md
```

Required pass criteria:

```text
agent parse accuracy = 100%
goal parse accuracy = 100%
wall/free IoU = 100%
parsed BFS first-action match = 100%
invalid oracle actions = 0
wall-pointing oracle actions = 0
tree property holds for all generated mazes
unique optimal first action for all non-goal states
```

If any check fails, stop and debug. Do not train LeWM yet.

---

# Prompt 3：训练标准 LeWM baseline

## 给 Claude Code 的 prompt

Train the original LeWM baseline on fixed 11×11 Procgen Maze using only the train split.

Important:

```text
Use only data from data/splits/fixed11_train_manifest.jsonl.
Do not use validation/test levels for training.
```

Model:

```text
original LeWM encoder
original pooled latent
original dynamics / transition model
original training losses
```

Save:

```text
checkpoints/lewm_original_fixed11/
results/phase1_lewm_baseline/lewm_original_training_log.csv
results/phase1_lewm_baseline/lewm_original_config.yaml
```

Evaluate reconstruction / prediction losses on:

```text
train
validation
test
```

Report:

```text
one-step latent prediction loss
multi-step latent prediction loss
reconstruction loss, if available
rollout error for horizons 1, 3, 5, 8, 16
```

Save:

```text
results/phase1_lewm_baseline/lewm_original_eval.csv
results/phase1_lewm_baseline/lewm_original_report.md
```

Register the experiment in:

```text
results/registry/experiment_registry.csv
```

Do not run planning yet.

---

# Prompt 4：Probe 已训练 LeWM latent

## 给 Claude Code 的 prompt

Run representation probing on the trained LeWM checkpoint.

Load:

```text
checkpoints/lewm_original_fixed11/
```

Freeze all LeWM parameters:

```text
encoder frozen
dynamics frozen
decoder frozen
```

Train only probe modules.

Probe inputs:

```text
z_true = encoder(real observation)
z_hat_h = learned rollout latent after h steps, h ∈ {1, 3, 5, 8}
pre-pooling feature, if accessible
```

Probe targets:

```text
agent x coordinate
agent y coordinate
agent cell index
goal x coordinate
goal y coordinate
goal cell index
BFS distance to goal
oracle first action
valid action mask
local occupancy
dead-end indicator
intersection indicator
```

Probe types:

```text
linear probe
2-layer MLP probe
small CNN probe for spatial features, if available
```

Metrics:

```text
x/y MAE
x/y Spearman
cell accuracy
BFS distance MAE
BFS distance Spearman
first-action accuracy
valid-action F1
occupancy IoU
dead-end accuracy
intersection accuracy
```

Evaluate on:

```text
train split
validation split
test split
```

Save:

```text
results/phase2_probe/lewm_original_probe_table.csv
results/phase2_probe/lewm_original_probe_report.md
```

The report must answer:

```text
1. Does z_true encode agent position?
2. Does z_true encode goal position?
3. Does z_true encode BFS distance?
4. Does z_true encode oracle first action?
5. Does z_hat preserve these signals after rollout?
6. Does pre-pooling feature preserve more topology than pooled latent?
```

Do not fine-tune LeWM during probing.

---

# Prompt 5：Latent L2 CEM planning baseline

## 给 Claude Code 的 prompt

Evaluate original LeWM with latent L2 CEM planning.

Use checkpoint:

```text
checkpoints/lewm_original_fixed11/
```

Planner:

```text
current latent z_t = encoder(obs_t)
goal latent z_g = encoder(goal observation or goal-conditioned observation)
sample action sequences with CEM
roll out learned dynamics to get z_hat_T
score candidate by L2(z_hat_T, z_g)
execute first action
```

Run ablations:

```text
horizon ∈ {1, 3, 5, 8, 12, 16, 24, 32}
num_candidates ∈ {64, 128, 256}
cem_iterations ∈ {1, 3, 5}
```

Evaluate on fixed 11×11 test split only.

Metrics:

```text
SR
SPL
mean return
average episode length
final BFS distance on failure
invalid action rate
stuck rate
dead-end entry rate
```

Also log candidate-level data:

```text
candidate action sequence
L2 score
true final BFS distance
true BFS progress
whether first action matches oracle
whether candidate reaches goal under true rollout
```

Save:

```text
results/phase3_l2_cem/l2_cem_ablation.csv
results/phase3_l2_cem/l2_cem_candidate_logs.parquet
results/phase3_l2_cem/l2_cem_report.md
```

Register all runs.

---

# Prompt 6：Oracle 四格实验

## 给 Claude Code 的 prompt

Run the oracle 4-way diagnosis.

Compare:

```text
1. learned LeWM dynamics + learned L2 metric
2. true environment dynamics + learned L2 metric
3. learned LeWM dynamics + oracle BFS metric
4. true environment dynamics + oracle BFS metric
```

Use the same CEM configuration across all four settings.

Recommended initial config:

```text
horizon = best horizon from Phase 3
num_candidates = 128
cem_iterations = 3
```

For each setting, evaluate:

```text
SR
SPL
mean return
neighbor accuracy
first-action accuracy
final BFS distance on failure
```

Save:

```text
results/phase3_l2_cem/oracle_4way_table.csv
results/phase3_l2_cem/oracle_4way_report.md
```

The report must answer:

```text
1. Is CEM planner itself capable when dynamics and metric are correct?
2. Is learned dynamics alone the bottleneck?
3. Is learned L2 metric alone the bottleneck?
4. Does failure emerge specifically from learned dynamics + learned metric interaction?
```

---

# Prompt 7：Candidate ranking 与 rollout-metric interaction audit

## 给 Claude Code 的 prompt

Use the candidate logs from latent L2 CEM to analyze why planning fails.

For each planning step and candidate action sequence, compute:

```text
z_hat_T = terminal latent from learned rollout
z_true_T = terminal latent from true environment rollout, encoded by LeWM
L2_hat = ||z_hat_T - z_goal||
L2_true = ||z_true_T - z_goal||
latent_error = ||z_hat_T - z_true_T||
true_final_bfs_distance
true_progress = bfs_before - bfs_after
first_action_matches_oracle
candidate_reaches_goal
```

Compute ranking metrics:

```text
Spearman(L2_hat, true_progress)
Spearman(L2_true, true_progress)
KendallTau(rank_by_L2_hat, rank_by_L2_true)
rank of oracle-best candidate under L2_hat
rank of oracle-best candidate under L2_true
selected-vs-oracle progress gap
top-k contains good candidate, k ∈ {1,5,10,20,40}
```

Save:

```text
results/phase3_l2_cem/rollout_metric_interaction_audit.csv
results/phase3_l2_cem/rank_flip_audit.csv
results/phase3_l2_cem/interaction_audit_report.md
```

The report must answer:

```text
1. Does L2 on imagined latent correlate with true BFS progress?
2. Does L2 on true rollout latent correlate better?
3. Are good candidates present but buried by L2_hat?
4. Is the failure candidate coverage or candidate selection?
```

---

# Prompt 8：Distance Head 实验

## 给 Claude Code 的 prompt

Train distance heads on top of the frozen LeWM latent.

Use checkpoint:

```text
checkpoints/lewm_original_fixed11/
```

Freeze LeWM.

Train heads:

```text
D1(z_current, z_goal) -> BFS distance
D2(z_hat_T, z_goal) -> true final BFS distance
D3(z_current, z_hat_T, z_goal, h) -> true candidate progress
```

Losses:

```text
MSE
Huber
bucketed cross entropy
pairwise ranking loss
```

Evaluate:

```text
BFS distance MAE
BFS distance Spearman
candidate ranking Spearman
oracle-best candidate rank
planning SR when used inside CEM
```

Save:

```text
results/phase4_metric_heads/distance_head_ablation.csv
results/phase4_metric_heads/distance_head_report.md
```

Important:

LeWM must remain frozen. Only distance heads are trained.

---

# Prompt 9：GCRL / Reachability Head 实验

## 给 Claude Code 的 prompt

Train goal-conditioned reachability heads on top of frozen LeWM latent.

Use checkpoint:

```text
checkpoints/lewm_original_fixed11/
```

Freeze LeWM.

Train:

```text
R(z_current, z_goal, h) = P(goal reachable within h steps)
R_hat(z_hat_T, z_goal, h_remaining) = P(goal reachable from imagined terminal latent)
```

Horizon buckets:

```text
h ∈ {1, 2, 4, 8, 16, 32, 64}
```

Training labels from oracle BFS:

```text
positive if BFS_distance <= h
negative otherwise
```

Include hard negatives:

```text
same maze wrong branch
wall-separated but visually close
dead-end states
states requiring backtracking
imagined latents with low L2 but bad true progress
```

Evaluate:

```text
AUC
accuracy
BFS Spearman
candidate ranking quality
planning SR with reachability score
```

Save:

```text
results/phase4_metric_heads/gcrl_reachability_ablation.csv
results/phase4_metric_heads/gcrl_report.md
```

---

# Prompt 10：QRL / Quasimetric 实验

## 给 Claude Code 的 prompt

Train QRL / quasimetric-style heads on frozen LeWM latent.

Use checkpoint:

```text
checkpoints/lewm_original_fixed11/
```

Freeze LeWM.

Implement variants:

```text
state-state quasimetric:
QDist(z_current, z_goal)

action-conditioned quasimetric:
QReach(z_current, action, z_goal, h)

imagined-latent quasimetric:
QDist(z_hat_T, z_goal)
```

Training targets:

```text
oracle BFS distance
oracle first-action improvement
reachability within horizon
true candidate progress
```

Losses:

```text
quasimetric loss
contrastive loss
pairwise ranking loss
temporal distance loss
```

Evaluate:

```text
BFS Spearman
first-action accuracy induced by Q
candidate ranking quality
planning SR
```

Save:

```text
results/phase4_metric_heads/qrl_ablation.csv
results/phase4_metric_heads/qrl_report.md
```

---

# Prompt 11：Planner tricks ablation

## 给 Claude Code 的 prompt

Evaluate planner modifications using the original LeWM latent.

Methods:

```text
vanilla CEM
balanced-first-action CEM
top-k voting
first-action voting
short-horizon MPC
stuck recovery
latent beam search
best-first latent search
```

Use scoring methods:

```text
L2
distance head
GCRL reachability
QRL score
candidate ranking score, if available
```

Metrics:

```text
SR
SPL
first-action accuracy
neighbor accuracy
final BFS distance
stuck rate
dead-end rate
candidate ranking metrics
```

Save:

```text
results/phase5_planner_ablation/planner_tricks_ablation.csv
results/phase5_planner_ablation/planner_tricks_report.md
```

The report must answer:

```text
Do planner tricks help when the underlying metric has weak or zero correlation with BFS progress?
```

---

# Prompt 12：Encoder architecture ablation

## 给 Claude Code 的 prompt

Train LeWM variants with different encoder architectures on fixed 11×11 train split.

Architectures:

```text
A. original pooled encoder
B. no-global-pooling encoder
C. CoordConv encoder
D. 2D positional encoding encoder
E. spatial latent tensor encoder
F. spatial token transformer encoder
G. conv encoder + transformer transition
```

For each architecture:

```text
train LeWM on train split only
save checkpoint
evaluate prediction loss
run frozen-probe suite
run L2 CEM planning only if probes are non-trivial
```

Probe metrics:

```text
agent cell accuracy
goal cell accuracy
BFS distance Spearman
first-action accuracy
valid-action F1
occupancy IoU
```

Save:

```text
results/phase6_encoder_ablation/encoder_probe_ablation.csv
results/phase6_encoder_ablation/encoder_planning_ablation.csv
results/phase6_encoder_ablation/encoder_ablation_report.md
```

Important:

Probe must freeze the trained LeWM weights. Only probe modules are trained.

---

# Prompt 13：Topology-supervised LeWM

## 给 Claude Code 的 prompt

Train topology-supervised LeWM variants on fixed 11×11 train split.

Start with the best encoder architecture from Phase 6.

Add topology decoder outputs:

```text
wall/free occupancy map
agent heatmap
goal heatmap
valid-action mask
BFS value map, optional
first-action policy, optional
```

Loss ablations:

```text
1. original LeWM loss only
2. + agent/goal heatmap
3. + occupancy map
4. + occupancy + agent/goal
5. + valid-action mask
6. + BFS value map
7. + first-action CE
8. + neighbor ranking loss
9. + imagined-latent topology consistency
10. all topology losses
```

For imagined-latent topology consistency:

```text
roll out z_hat for h ∈ {1, 3, 5}
decode topology from z_hat
supervise with true rollout labels
```

Evaluate:

```text
topology decoding metrics
probe metrics
L2 CEM planning
decoded-map + BFS planning
decoded-map + DiffBFS planning
first-action head planning
```

Save:

```text
results/phase7_topology_supervision/topology_supervised_lewm_ablation.csv
results/phase7_topology_supervision/topology_supervised_planning.csv
results/phase7_topology_supervision/topology_supervised_report.md
```

---

# Prompt 14：Decoded topology + BFS planning

## 给 Claude Code 的 prompt

Evaluate symbolic-style planning using predicted topology from LeWM latents.

Compare:

```text
1. oracle grid + BFS
2. hand-written pixel parser + BFS
3. raw observation learned parser + BFS
4. original LeWM latent + topology decoder + BFS
5. topology-supervised pooled LeWM + decoder + BFS
6. spatial LeWM + decoder + BFS
7. spatial LeWM + imagined consistency + decoder + BFS
```

At test time, learned methods may use only:

```text
raw observation
learned encoder
learned latent
learned topology decoder
fixed BFS / DiffBFS planner
```

Not allowed at test time:

```text
oracle grid
hand-written parser, except for explicit parser baseline
oracle BFS labels
oracle agent/goal position
```

Metrics:

```text
occupancy IoU
agent exact cell accuracy
goal exact cell accuracy
valid-action F1
decoded graph no-path rate
neighbor accuracy after BFS
SR
SPL
failure mode breakdown
```

Failure modes:

```text
wrong agent localization
wrong goal localization
false wall blocks true path
false free creates fake shortcut
no path in predicted graph
fallback planner failure
```

Save:

```text
results/phase7_topology_supervision/decoded_topology_bfs_eval.csv
results/phase7_topology_supervision/decoded_topology_failure_modes.csv
results/phase7_topology_supervision/decoded_topology_bfs_report.md
```

---

# Prompt 15：Maze size OOD 实验

## 给 Claude Code 的 prompt

Run maze size OOD experiments only after fixed 11×11 experiments are complete.

Use sizes:

```text
7, 9, 11, 13, 15
```

Recommended split:

```text
train sizes: 7, 9, 11
test ID: unseen topologies from 7, 9, 11
test OOD: unseen topologies from 13, 15
```

Methods to evaluate:

```text
best original LeWM + L2 CEM
best metric head method
best encoder architecture
best topology-supervised LeWM
best decoded-map + BFS method
oracle grid + BFS
```

Metrics:

```text
SR by maze size
SPL by maze size
BFS Spearman by maze size
topology decoding accuracy by maze size
failure mode by maze size
```

Save:

```text
results/phase8_size_ood/size_ood_results.csv
results/phase8_size_ood/size_ood_report.md
```

The report must answer:

```text
1. Does each method generalize from small mazes to larger mazes?
2. Does latent L2 planning degrade with size?
3. Does topology-supervised LeWM degrade less?
4. Does decoded-map BFS remain robust if topology prediction is accurate?
```

---

# Prompt 16：最终汇总报告

## 给 Claude Code 的 prompt

Generate the final feasibility report.

Inputs:

```text
results/registry/experiment_registry.csv
all phase reports
all metrics csv files
all failure mode files
```

Output:

```text
results/final_report/LeWM_Maze_Navigation_Feasibility_Report.md
results/final_report/all_experiment_summary.csv
results/final_report/key_figures/
```

The report should be structured as:

```text
1. Executive summary
2. Environment and split protocol
3. Oracle tools and sanity checks
4. Original LeWM baseline
5. Representation probing
6. Latent L2 CEM planning
7. Oracle 4-way diagnosis
8. Distance head / GCRL / QRL / ranking metrics
9. Planner tricks
10. Encoder architecture ablation
11. Topology-supervised LeWM
12. Decoded topology + BFS planning
13. Maze size OOD generalization
14. Failure mode analysis
15. Conclusion: feasibility of LeWM for Maze navigation
```

The final conclusion must explicitly answer:

```text
1. Is original LeWM + latent L2 CEM feasible?
2. Can post-hoc metric heads rescue latent planning?
3. Does changing encoder architecture improve topology representation?
4. Does topology supervision make LeWM more planning-compatible?
5. Is symbolic-style planning from decoded topology more promising than latent CEM?
6. What does this imply for future in-context symbolic world model research?
```

Produce a final comparison table:

| Category | Method | Fixed 11 SR | Fixed 11 SPL | Size-OOD SR | BFS Spearman | Agent Acc | Goal Acc | Occupancy IoU | Notes |
| -------- | ------ | ----------: | -----------: | ----------: | -----------: | --------: | -------: | ------------: | ----- |

Also generate plots:

```text
SR comparison
SPL comparison
probe results
metric correlation
oracle-best candidate rank
encoder ablation
topology decoder accuracy
size-OOD performance
failure mode breakdown
```
