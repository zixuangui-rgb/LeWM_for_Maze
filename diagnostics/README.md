# Maze-JEPA Diagnostics

这个目录是一套可复用的 Maze-JEPA 诊断基准。它不是为了替代现有训练脚本，而是为了在每个新 backbone / projector / metric head 出来以后，系统回答：

1. 信息到底保留在哪一层？
2. 信息是从 encoder 到 projector 丢失，还是在 predictor rollout 中丢失？
3. L2 / DistanceHead / QRL 的分数是否真的支持局部动作选择？
4. 失败 episode 主要来自 metric、predictor、循环、长路径，还是 OOD size？

核心原则是：**固定数据、固定诊断任务、固定输出格式**。后续所有新方法都跑同一套 diagnostics，才能形成科学可比的结论。

## 目录结构

```text
diagnostics/
  build_cache.py              # 抽取 layer-wise feature cache
  train_probes.py             # 训练 linear / MLP probes
  eval_metric_alignment.py    # 评估 L2 / DH / QRL 与 BFS/动作排序的一致性
  eval_predictor_rollout.py   # 评估 predictor rollout 随 horizon 的退化
  eval_failure_taxonomy.py    # 给失败 episode 分类
  generate_report.py          # 汇总 JSON，生成中文 Markdown 报告
  run_all.py                  # 一键跑完整诊断
  common.py                   # 公共工具
  configs/default.json        # 默认协议参数
```

输出统一写到：

```text
diagnostics_runs/<run_id>/
  feature_cache/
    features.pt
    manifest.json
  probe_checkpoints/
  metrics/
    probe_metrics.json
    metric_alignment.json
    predictor_rollout.json
    failure_taxonomy.json
  diagnostic_report.md
```

`diagnostics_runs/` 是实验产物目录，默认不需要提交到 Git。

## 一键运行

只诊断 LeWM backbone + latent L2：

```bash
python diagnostics/run_all.py \
  --model-ckpt checkpoints/backbones/unisize_dim256_setb_seqlen_ablation_20260708_seqlen2.pt \
  --run-id seqlen2_diagnostics \
  --device cuda
```

如果同时有 DistanceHead 和 QRL：

```bash
python diagnostics/run_all.py \
  --model-ckpt checkpoints/backbones/unisize_dim256_setb_seqlen_ablation_20260708_seqlen2.pt \
  --distance-head-ckpt checkpoints/metric_heads/distance_head_simple_setb_seqlen_ablation_20260708_seqlen2.pt \
  --qrl-ckpt checkpoints/metric_heads/qrl_v2_frozen_setb_seqlen_ablation_20260708_seqlen2.pt \
  --run-id seqlen2_diagnostics_with_heads \
  --device cuda
```

快速 smoke test 可以调小规模：

```bash
python diagnostics/run_all.py \
  --model-ckpt checkpoints/backbones/example.pt \
  --run-id smoke_test \
  --device cuda \
  --max-train-per-size 2 \
  --max-eval-per-size 2 \
  --states-per-maze 4 \
  --probe-epochs 1 \
  --stages cache,probes,metric,rollout,failure,report
```

## 分阶段运行

如果某个阶段失败或想单独调参，可以分开跑。

### 1. 构建 feature cache

```bash
python diagnostics/build_cache.py \
  --model-ckpt checkpoints/backbones/example.pt \
  --run-id my_run \
  --max-train-per-size 80 \
  --max-eval-per-size 100 \
  --states-per-maze 24 \
  --device cuda
```

默认抽取四层：

```text
spatial_flat   CNN conv feature 展平，保留强空间信息，但不同 size 维度不同
spatial_pool   CNN conv feature 全局池化后，固定维度
encoded        CNN pooling + size embedding + fuse 后的表征
embedding      projector 后的 256-d latent，规划真正使用的空间
```

### 2. 训练 layer-wise probes

```bash
python diagnostics/train_probes.py \
  --model-ckpt checkpoints/backbones/example.pt \
  --run-id my_run \
  --epochs 25 \
  --device cuda
```

每层会训练两类 probe：

```text
linear probe  判断信息是否线性可读
MLP probe     判断信息是否存在但需要非线性解码
```

诊断任务包括：

```text
agent_x / agent_y       是否知道自己在哪
goal_x / goal_y         是否知道目标在哪
valid_action            是否知道局部墙结构
bfs_distance_norm       是否编码了到 goal 的 geodesic 距离
optimal_action          是否能支持 BFS 最优下一步动作
```

### 3. 评估 metric alignment

```bash
python diagnostics/eval_metric_alignment.py \
  --model-ckpt checkpoints/backbones/example.pt \
  --distance-head-ckpt checkpoints/metric_heads/example_dh.pt \
  --qrl-ckpt checkpoints/metric_heads/example_qrl.pt \
  --run-id my_run \
  --device cuda
```

输出指标：

```text
Pearson / Spearman      score 与全局 BFS distance 的相关性
Local top-1             用 score 选动作时，是否选到 BFS 最优动作
Local pairwise          好动作是否排在坏动作前
Local margin            好动作和坏动作之间的 score 间隔
```

最重要的是 `Local top-1` 和 `Local pairwise`。如果全局相关性高但局部排序低，说明普通 distance regression 不能直接支持导航。

### 4. 评估 predictor rollout

```bash
python diagnostics/eval_predictor_rollout.py \
  --model-ckpt checkpoints/backbones/example.pt \
  --run-id my_run \
  --horizons 1,2,3,5,8,10 \
  --device cuda
```

输出指标：

```text
latent_mse      predicted latent 与 true future latent 的误差
cosine          predicted latent 与 true latent 的方向相似度
nn_exact        predicted latent 最近邻是否是真实 future state
nn_bfs_error    最近邻状态离真实 future state 的 BFS 距离
```

同时比较：

```text
teacher_forced  每步用真实 latent 刷新上下文
closed_loop     用预测 latent 继续 rollout
```

如果 teacher-forced 好但 closed-loop 差，说明主要是 rollout 累积误差。

### 5. 失败分类

```bash
python diagnostics/eval_failure_taxonomy.py \
  --model-ckpt checkpoints/backbones/example.pt \
  --run-id my_run \
  --scorer latent_l2 \
  --device cuda
```

可选 scorer：

```text
latent_l2
distance_head
qrl
```

失败标签：

```text
metric_wrong        真实 next latent 下就选错动作
predictor_wrong     model-free 能选对，但 predictor-greedy 选错
loop_or_cycle       episode 出现明显循环
validity_failure    出现撞墙/不动等非法动作
long_path           主要失败在长路径任务
ood_size            失败集中在 23/25 OOD size
unclassified        暂未归类
```

### 6. 生成报告

```bash
python diagnostics/generate_report.py \
  --run-id my_run
```

输出：

```text
diagnostics_runs/my_run/diagnostic_report.md
```

## 如何解释结果

### 情况 A：spatial 很强，embedding 明显变差

说明 CNN 里有 maze 信息，但 projector 把它压掉了。下一步优先做：

```text
embedding-level position/goal/valid-action aux
projector 结构改造
spatial-to-latent 信息保持约束
```

### 情况 B：position 很强，但 optimal action 很差

说明模型知道“我在哪、目标在哪”，但不知道“怎么绕墙过去”。下一步优先做：

```text
action-ranking head
local value/distance objective
geodesic metric learning
```

### 情况 C：global distance correlation 高，但 local top-1 低

说明 DistanceHead/QRL 学到了粗略距离，但没有学到 planner 最需要的局部动作排序。下一步优先做：

```text
score(z_next_action, z_goal) 的 action CE
local pairwise ranking
predictor-action ranking
```

### 情况 D：teacher-forced rollout 好，closed-loop rollout 差

说明一阶预测还可以，但长 rollout 偏离真实状态流形。下一步优先做：

```text
multi-step predictor loss
closed-loop consistency
uncertainty-aware planning
shorter-horizon receding planning
```

### 情况 E：OOD size 掉得最明显

说明问题是尺寸泛化。下一步优先做：

```text
fully convolutional value/distance map
coordinate-aware but size-normalized representations
spatial planner over feature map instead of global 256-d vector
```

## 后续新方法如何接入

任何新 backbone 最好仍然能提供：

```text
model.encoder(obs, size)
model.embedding_projector(encoded)
model.predictor(ctx_embedding, ctx_actions)
```

如果结构变了，只需要在 `diagnostics/common.py` 里新增一个 adapter，让 `extract_layers_batch()` 和 `encode_observations()` 能拿到对应层。

建议每个新方法都固定跑：

```bash
python diagnostics/run_all.py --model-ckpt <new_ckpt> --run-id <method_name>
```

然后把 `diagnostic_report.md` 和最终 navigation SR 一起比较。这样后续论文叙事就不是“分数涨了”，而是：

```text
新方法具体修复了 projector / metric / predictor / OOD 泛化中的哪一个瓶颈。
```
