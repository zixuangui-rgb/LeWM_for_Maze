# Planning Repair Design

## 目标

本轮实验要回答的问题不是“再换一个 scorer 是否更高”，而是：

1. 能否让 post-projector `embedding` 保留 maze 规划所需的几何和局部拓扑信息？
2. 能否把局部动作排序从 `Local top-1 ~= 0.60` 往上推？
3. 能否降低 closed-loop rollout 漂移，使 latent planner 不再离开可导航流形？
4. 如果单向量 embedding 仍然卡住，是否需要切到 spatial patch-token / VIN planner 路线？

## 设计依据

### 诊断事实

当前 `seqlen4` backbone 的关键诊断：

| 墙 | 关键数字 | 含义 |
|---|---:|---|
| 表征/动作墙 | `embedding optimal_action = 0.341` | 几乎随机，embedding 不知道“往哪走” |
| metric 墙 | `Local top-1 = 0.588-0.598` | L2/DH/QRL 都不能可靠排局部动作 |
| projector 信息墙 | `valid_action 0.676 -> 0.406` | 局部墙结构从 spatial 到 embedding 丢失 |
| rollout 漂移墙 | `closed-loop h=10 nn_bfs_error = 9.71` | 多步想象离真实 future 很远 |
| failure taxonomy | `loop_or_cycle 0.387, metric_wrong 0.364` | 循环和局部排序错是主要失败 |

### 文献对照

- RC-aux: latent world model 可以 short-horizon predictive 但 not plannable；需要 multi-horizon prediction 和 budget-conditioned reachability supervision。
- P-JEPA / auxiliary tasks for JEPA: aux task 与 latent dynamics 联合训练能锚定 representation 应保留的等价类。
- Projection head as information bottleneck: projector 会过滤掉与 SSL 目标无关的信息，这解释了 `encoded -> embedding` 的信息丢失。
- DINO-WM: 保留 spatial patch features 做 world model 和 planning，避免过早 pool 成单向量。
- Fast-LeWM: 用 action-prefix prediction 减少递归 rollout 误差。
- Value-guided JEPA planning: value / quasi-distance 结构应进入训练目标，而不是 frozen embedding 上事后 bolt-on。

参考链接：

- RC-aux: <https://arxiv.org/abs/2605.07278>
- JEPA-WM physical planning: <https://arxiv.org/abs/2512.24497>
- Projection head as information bottleneck: <https://arxiv.org/abs/2503.00507>
- DINO-WM: <https://arxiv.org/abs/2411.04983>
- P-JEPA / auxiliary tasks for JEPA: <https://arxiv.org/abs/2509.12249>
- Value-guided JEPA planning: <https://arxiv.org/abs/2601.00844>
- LeWorldModel: <https://arxiv.org/abs/2603.19312>
- Fast LeWorldModel: <https://arxiv.org/abs/2606.26217>

## 实验矩阵

### P0: Short-horizon receding CEM

目的：先验证 rollout 漂移是否是可被 planner 工程缓解的。

实现：`eval_b2_receding.py`

变量：

| 变量 | 值 |
|---|---|
| horizon | 3, 5, 8, 12 |
| scorer | latent_l2，可选 distance_head / aux_bfs |
| replan | 每步 replan |

判据：

- 如果 horizon 3/5 显著高于 12，说明长 rollout 是真实瓶颈。
- 如果 SR 涨但 diagnostics 中 `embedding optimal_action` 和 `Local top-1` 不动，只能说明 planner 绕过了 Wall 3，不能说明表征修复。

### P1: Embedding-level auxiliary supervision

目的：修 projector 信息墙。

实现：`train_planning_aligned.py`

Aux heads 直接加在 post-projector embedding 上：

| Head | 标签 | 对应诊断 |
|---|---|---|
| agent xy | normalized `(x,y)` | agent RMSE |
| goal xy | normalized goal `(x,y)` | goal RMSE |
| valid-action mask | 4 个移动动作是否撞墙 | valid_action |
| BFS distance norm | 到 goal 的 normalized BFS distance | bfs_distance_norm |
| budget reachability | distance <= {1,3,5,8,12} | reachability / RC-aux |

判据：

- `embedding goal_y RMSE` 明显下降；
- `embedding valid_action` 从 `0.406` 往 `0.55+` 移动；
- `embedding optimal_action` 至少从 `0.341` 往 `0.45+` 移动。

### P1.5: Listwise action ranking

目的：直接修局部动作排序，而不是只回归全局距离。

实现：`EmbeddingAuxHeads.action_logits` + `soft_target_cross_entropy`

标签：所有 BFS 最优动作共享概率质量，处理 tie shortest path。

判据：

- `Local top-1` 从 `0.60` 往 `0.65+` 移动；
- `metric_wrong` 下降；
- `eval_aux_action_head.py` 的 model-free greedy SR 上升。

### P2: Action-prefix prediction

目的：修 rollout 漂移。

实现：`ActionPrefixPredictor`

机制：

- 输入当前 embedding 与 action prefix；
- 直接预测每个 prefix horizon 的 latent；
- 避免 test-time 对同一个 one-step predictor 递归展开。

判据：

- diagnostics 中 closed-loop `nn_bfs_error` 下降；
- `predictor_wrong / long_path / loop_or_cycle` 下降；
- `eval_prefix_planner.py` 的 SR 优于普通 horizon-12 recursive CEM。

### P4: Spatial patch-token planner

本目录暂不实现 P4 的完整 VIN / Neural-A* 版本，因为它是架构级大改。P4 应在 P1-P2 的诊断结果明确后再做：

- 如果 embedding aux 能显著修复 RMSE/valid/action，但 SR 仍不高，优先做更强 planner；
- 如果 embedding aux 也修不动，则说明单向量 embedding 是根本瓶颈，应转向 spatial patch-token。

P4 应分两档：

1. A2-light: spatial patch-token JEPA predictor/planner，仍属于 latent world model。
2. A2-strong: VIN / Neural-A* over spatial map，作为结构化规划上界。

## 为什么不优先做的方向

| 方向 | 不优先原因 |
|---|---|
| frozen embedding 上继续换 DH/QRL | 三个 scorer 已并列 `Local top-1 ~= 0.60` |
| pixel reconstruction | LeWM ablation 中不支持；maze 输入本身已低维 one-hot |
| 更大 predictor | teacher-forced 单步已好，主要是 closed-loop exposure bias |
| size generalization 作首要修复 | OOD 放大弱点，但当前 seen 也被三堵墙卡住 |
| bisimulation 主目标 | 可能把 reward 相同但拓扑位置不同的状态塌成一个 |
| POMDP memory | 当前 5 通道 grid 是 fully observable |
| 去掉 L2 norm | 复现配置已无 `latent_l2_norm` |

## 成功标准

最低成功标准：

- P0 证明短 horizon 能减轻 rollout failure；
- P1 证明 embedding 信息墙可被 aux 修复；
- P1.5 让 `Local top-1` 明显超过 `0.60`；
- P2 降低 closed-loop drift。

中等成功标准：

- latent planning SR 稳定进入 `0.75-0.85`；
- failure taxonomy 中 `metric_wrong` 和 `predictor_wrong` 同时下降。

高成功标准：

- P1/P2 后仍保持 topology holdout 与 OOD size 泛化；
- spatial patch-token 路线能接近 symbolic BFS 上界。
