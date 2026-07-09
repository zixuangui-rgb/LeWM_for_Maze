# Maze-JEPA Diagnostic Protocol

这份文档描述 diagnostics 的实验设计原则。它面向实现者和研究合作者，重点说明为什么这些诊断是必要的，以及如何避免“看起来很多指标，但无法指导下一步”的问题。

## 1. 背景判断

当前 Set B 报告显示：

```text
Symbolic BFS probe: 0.94-0.96
LeWM metric planning: 0.63-0.65
BC CNN: 0.781
BC-policy offline RL: 0.799
```

这说明 LeWM 并不是完全没有学到 maze 结构。更合理的判断是：

```text
空间结构存在于某些中间层，
但没有稳定进入最终用于 planning 的 embedding / metric / predictor。
```

因此下一步不是盲目加模型，而是系统定位瓶颈。

## 2. 科学问题

这套 diagnostics 要回答五个问题：

1. **Representation retention**
   spatial、encoded、embedding 哪一层保留了 agent、goal、wall、distance、action 信息？

2. **Projection loss**
   projector 是否把 CNN/encoded 中的空间拓扑信息压掉了？

3. **Metric alignment**
   latent L2 / DistanceHead / QRL 是否真的和 BFS distance 以及局部动作排序一致？

4. **Dynamics degradation**
   predictor rollout 是否随 horizon 变长偏离真实状态流形？

5. **Failure attribution**
   失败 episode 主要来自 metric、predictor、loop、validity、long path，还是 OOD size？

## 3. 固定层级

默认诊断四层：

| Layer | 含义 | 维度特性 | 用途 |
| --- | --- | --- | --- |
| `spatial_flat` | CNN conv 输出展平 | 随 size 变化 | 测空间信息上限 |
| `spatial_pool` | CNN conv 输出全局池化 | 固定维度 | 测可泛化空间摘要 |
| `encoded` | size-conditioned encoder 输出 | 固定维度 | 测 projector 前信息 |
| `embedding` | projector 后 latent | 固定 256-d | 测实际 planning 空间 |

如果以后有新架构，可以增加层，但不要删除这几个语义层。横向比较必须稳定。

## 4. 固定任务

每一层都评估：

| Task | 类型 | 科学含义 |
| --- | --- | --- |
| `agent_x/y` | 分类 | 是否知道自己在哪 |
| `goal_x/y` | 分类 | 是否知道目标在哪 |
| `valid_action` | 多标签 | 是否知道局部墙结构 |
| `bfs_distance_norm` | 回归 | 是否编码 geodesic distance |
| `optimal_action` | 分类/多最优 | 是否支持局部导航决策 |

`optimal_action` 是最重要的任务之一，因为最终导航不是预测位置，而是选择动作。

## 5. Linear Probe 与 MLP Probe

每个任务训练两类 probe：

| Probe | 解释 |
| --- | --- |
| Linear | 信息是否以简单、可直接利用的方式存在 |
| MLP | 信息是否存在但需要非线性解码 |

典型解释：

```text
Linear 差、MLP 好：信息存在，但表示不规整。
Linear 和 MLP 都差：这层很可能真的缺少该信息。
Linear 和 MLP 都好：该信息稳定且易用。
```

## 6. Per-size 与 Unified

诊断分两种 scope：

| Scope | 目的 |
| --- | --- |
| Per-size probe | 判断每个 seen size 内信息是否存在 |
| Unified probe | 判断一个共享 probe 是否能跨 size 泛化到 23/25 |

`spatial_flat` 因为维度随 size 变化，只做 per-size。`spatial_pool/encoded/embedding` 可以做 unified。

## 7. Metric Alignment

只看 BFS distance regression loss 不够。必须同时看：

| 指标 | 解释 |
| --- | --- |
| Pearson/Spearman | score 与全局 BFS distance 的相关性 |
| Local top-1 | 用 score 选动作是否选到 BFS 最优动作 |
| Local pairwise | 好动作是否排在坏动作前 |
| Local margin | 好坏动作 score 间隔是否足够大 |

最核心的是：

```text
Local top-1 / Local pairwise / Local margin
```

因为 planner 真正需要的是局部动作排序。

## 8. Predictor Rollout

predictor 诊断分两种：

| Mode | 含义 |
| --- | --- |
| Teacher-forced | 每一步用真实 latent 刷新上下文 |
| Closed-loop | 用预测 latent 继续预测 |

指标：

| 指标 | 解释 |
| --- | --- |
| Latent MSE | 预测 latent 与真实 latent 的误差 |
| Cosine | 方向相似度 |
| NN exact | 最近邻状态是否等于真实 future state |
| NN BFS error | 最近邻状态离真实 future state 的 BFS 距离 |

如果 closed-loop 比 teacher-forced 掉得快，说明 CEM / long rollout 很可能被动态误差限制。

## 9. Failure Taxonomy

失败分类不是为了精确归因到唯一原因，而是为了形成工程优先级。

| Tag | 解释 | 后续动作 |
| --- | --- | --- |
| `metric_wrong` | 真实 next latent 下就选错 | 改 metric/action-ranking |
| `predictor_wrong` | model-free 能对，predictor 错 | 改 predictor 或 predictor-aligned head |
| `loop_or_cycle` | 反复访问状态 | anti-loop planner / margin |
| `validity_failure` | 撞墙/不动 | valid-action head 或 action mask |
| `long_path` | 长路径任务失败 | multi-step planning/value map |
| `ood_size` | 23/25 上失败 | size-generalized spatial architecture |

## 10. 最终报告应该如何用于决策

推荐按这个顺序读：

1. 看 layer-wise probes，确定信息丢失层。
2. 看 metric alignment，确定 distance/score 是否适合选动作。
3. 看 predictor rollout，确定规划是否受动态误差限制。
4. 看 failure taxonomy，确定下一步改哪个模块最划算。

决策规则：

```text
embedding 丢信息 -> 改 projector / embedding aux
optimal action probe 差 -> 做 action-ranking objective
local top-1 差 -> 改 metric head
closed-loop rollout 差 -> 改 predictor / 缩短 planning horizon
OOD size 差 -> 做 spatial value map / fully convolutional planner
```

## 11. 研究记录要求

每个新方法至少保存：

```text
diagnostics_runs/<run_id>/diagnostic_report.md
diagnostics_runs/<run_id>/metrics/*.json
最终 navigation SR/SPL
checkpoint config
```

论文或组会中不要只汇报 SR，也要汇报它修复了哪个诊断指标。这样才能说明方法不是偶然刷分，而是在机制上解决了某个瓶颈。
