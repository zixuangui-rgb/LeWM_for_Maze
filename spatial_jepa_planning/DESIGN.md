# Spatial-JEPA Iterative Planning: Scientific Design

## 1. 已有证据

此前实验形成了相当清楚的因果链：

| 证据 | 结果 | 含义 |
|---|---:|---|
| Current embedding optimal-action | 约 0.32-0.34 | pooled embedding 几乎不含可靠动作信息 |
| L2/DH/QRL Local top-1 | 约 0.58-0.60 | 换 scalar scorer 没修复局部排序 |
| P1 valid-action | 0.388 -> 0.850 | projector 信息可以被 auxiliary supervision 修复 |
| P1 latent-L2 CEM | 约 0.636 | 信息恢复没有自动转化为 planning SR |
| Prefix-only h10 drift | 9.706 -> 7.120 | rollout 可改善，但 planner 仍没有突破 |
| FCVP | SR 0.629，local top-1 0.690 | spatial field + global value regression 仍不够 |
| Old hardcoded VI | SR 0.957 | 迭代传播接近 oracle，但旧实现仍有非理论失败 |
| New exact BFS / VI K256 | SR 0.981111 | 等于 `max_steps=128` 下的精确上限 |

最稳健的阶段性结论不是“所有 feedforward 模型有 0.63 理论上限”，而是：

> 当前测试过的 pooled metric 和单次 feedforward value regression 都没有学出稳定的局部 shortest-path action ordering；精确局部更新反复传播可以解决该任务。

## 2. 研究假设

### H1：目标函数假设

FCVP 的问题主要是 global scalar regression。全图 tie-aware action CE、Bellman consistency 和 action-gap loss 可以提升 local top-1 和 SR。

对照：R0 -> R1 -> R2。

### H2：算法假设

在相同 raw-grid input 和相同 planning losses 下，共享权重的局部 recurrent update 比 capacity-matched feedforward blocks 更接近 value iteration。

对照：R2、full-receptive-field dilated R2D vs R3/R4。R2D 使用 dilation `1/2/4/8`，在不增加参数量的情况下覆盖完整训练 maze，从而排除“recurrent 只是感受野更大”的替代解释。

### H3：test-time compute 假设

如果 planner 学到可重复算法而不是固定深度模式，训练尺寸外的 maze 应在增加 K 时继续改善，尤其是 size 23/25 和长 shortest-path bins。

证据：SR、OOD SR、local top-1 随 K 的曲线，以及 hidden/policy convergence。

### H4：空间表征假设

Full-resolution Spatial-JEPA 能保留 wall/goal/local transition 信息，避免现有 stride-8 pooled backbone 的 projector bottleneck。

证据：decoded-map BFS、wall IoU、agent/goal accuracy、valid-action 和 JEPA recurrent performance。

### H5：梯度干扰假设

Dynamics prediction 与 planning supervision 如果全部压在同一 projector 上会重复 P2 full 的退化。分开的 dynamics/planning projectors 和 staged training 会优于直接 joint training。

对照：J1 frozen、J2 last-block、J3 joint；J3 记录 shared encoder 的 gradient cosine。

## 3. 因果矩阵

| Representation | Exact algorithm | Learned feedforward | Learned recurrent |
|---|---|---|---|
| Raw grid | exact BFS / oracle VI | R0-R2 | R3-R4 |
| Spatial-JEPA | decoded-map BFS | J0 | J1-J3 |

解释规则：

- raw recurrent 高、JEPA recurrent 低：representation bottleneck；
- decoded-map BFS 高、JEPA recurrent 低：learned algorithm bottleneck；
- frozen 高、joint 低：gradient interference；
- R1 高、R4 无额外收益：主要是 action objective，不应声称 recurrence 是贡献；
- K 增加只改善 OOD/long-path：支持 algorithmic extrapolation；
- K 超过训练范围后退化：overthinking 或 limit cycle。

## 4. Spatial-JEPA

输入为 `[B,T,H,W,5]` one-hot observation。encoder 全程 stride 1，输出：

```text
z = Encoder(o) in R^(H x W x C)
z_dyn  = DynamicsProjector(z)
z_plan = PlanningProjector(z)
```

### Dynamics branch

```text
z_hat_(t+1) = Predictor(z_dyn_t, action_t)
target_(t+1) = EMAEncoderTarget(o_(t+1))
L_pred = SmoothL1(z_hat_(t+1), stopgrad(target_(t+1)))
```

action 以 one-hot planes 广播到全图。target encoder 用 EMA 更新，避免 online/target 同时追逐。

### Planning branch

从 `z_plan` 解码：

- wall mask；
- agent cell；
- goal cell；
- 每格四方向 valid-action field。

agent/goal 使用 spatial cross entropy，不使用极度不平衡的逐像素 BCE。

### Collapse control

主模型保留与旧 LeWM 同一类 SIGReg，但作用在 `time x spatial-token` 分布上。为避免 `T x H x W x batch x 1024 projections x 17 knots` 产生不可接受的中间张量，每个 batch 在 time×space 轴确定性抽取最多 64 个 tokens，仍保留旧设置的 1024 个 random projections。可选 `spatial_info_sigreg_var` 再加入 per-channel variance floor 和 off-diagonal covariance；该 variant 默认关闭，只在主模型仍出现 token collapse 时启用，避免第一轮同时改变过多因素。

## 5. Iterative planner

planner 输入 raw grid 或 `z_plan`，先编码为 immutable recall feature：

```text
r = RecallEncoder(input)
h_0 = Init(r)
h_(k+1) = ConvGRU(h_k, r)
```

每一轮都重新注入 `r`，避免 recurrent state 忘记墙体和 goal。所有迭代共享同一个 cell 参数。

readout 输出：

- non-negative cost field `V`；
- 4-action policy logits；
- 4-action validity logits。

### 迭代监督

对于 iterative model，K 次局部传播只对 `BFS distance <= K` 的位置施加 value/action/Bellman/gap supervision；valid-action 是局部任务，始终监督全图。这防止模型在 K=8 时被迫“瞬间”预测距离 100 的完整解。

R4 使用 progressive/random K：

```text
K_train in {8, 16, 32, 64, 128}
K_test  in {8, 16, 32, 64, 128, 256}
```

## 6. Losses

### Tie-aware action CE

若多个动作都位于 shortest path 上，target probability 在全部最优动作之间平均分配。不会把等价最短路误标为错误。

### Value loss

`V` 预测 raw BFS cost。为避免大 maze 产生过大梯度，同时避免简单除以 128 后梯度几乎消失：

```text
L_value = distance_scale * Huber(V / distance_scale, d_BFS / distance_scale)
```

### Bellman consistency

```text
target(s) = 0                                      if s = goal
target(s) = 1 + min_(a valid) stopgrad(V(T(s,a))) otherwise
L_Bellman = Huber(V(s), target(s))
```

Bellman residual 保留“一步代价为 1”的原始单位。对 K-budgeted iterative
output，backup 还要求邻居满足 `d_BFS(T(s,a)) <= K-1`，避免尚未传播到的
frontier 外低值污染边界 target。

### Action gap

```text
max_logit(optimal) >= max_logit(valid suboptimal) + margin
```

这直接针对旧 scorer margin 太小、近似误差容易翻转动作的问题。

## 7. Variant matrix

| Variant | Input | Planner | 关键变化 | Primary K | 因果作用 |
|---|---|---|---|---:|---|
| R0 | raw | FF | value only | 4 | FCVP mechanism control |
| R1 | raw | FF | all-state action CE | 4 | objective effect |
| R2 | raw | FF | value+CE+valid+Bellman+gap | 4 | matched-loss FF control |
| R2D | raw | dilated FF | R2 losses，dilation 1/2/4/8 | 4 | full-receptive-field control |
| R3 | raw | recurrent | fixed K=64 | 64 | fixed recurrence |
| R4 | raw | recurrent | progressive/random K | 128 | main algorithm hypothesis |
| J0 | Spatial-JEPA | FF | frozen representation | 4 | representation + FF |
| J1 | Spatial-JEPA | recurrent | frozen representation | 128 | clean representation transfer |
| J2 | Spatial-JEPA | recurrent | last block 0.1x LR + map loss | 128 | staged adaptation |
| J3 | Spatial-JEPA | recurrent | joint JEPA/planning | 128 | interference test |

在默认 `hidden_dim=64` 下，raw R2/R2D feedforward planner 均为 299,529 个参数，R4 recurrent planner 为 303,113 个参数，参数量差约 1.2%。实际 trainable parameter count 仍会写入 checkpoint；正式报告还应给出迭代次数和训练 FLOPs，不能把额外计算量隐去。

## 8. Training stages

### Stage A：representation

先训练 Spatial-JEPA。不得看 eval900 调 loss。representation gate：

- decoded wall IoU、agent/goal accuracy 明显高于随机；
- decoded-map BFS 能稳定运行；
- token/channel variance 不出现数值 collapse；
- prediction loss 与 map losses 均稳定。

理想 gate 是 decoded-map BFS `SR@128 >= 0.90`。若远低于该值，优先修表征，不进入“JEPA planner 不会推理”的结论。

### Stage B：raw planner

先完成 R0-R4。只有 raw recurrent 相比 full-receptive-field R2D 有稳定提升，才说明当前 recurrent architecture 值得接到 JEPA。

最低算法 gate：

- R4 - R2D 的 paired delta SR >= 0.03；
- 95% CI 不包含 0；
- local top-1 同向提升；
- K 增加时 OOD/long-path 不退化。

### Stage C：Spatial-JEPA integration

依次 J0、J1、J2、J3。不要首先跑 joint。J2 只解冻最后一个 encoder block、planning projector 和 map decoder，representation 参数使用 planner LR 的 0.1 倍，并保留 wall/agent/goal/valid map loss。J3 需要检查：

- representation/planning gradient cosine；
- decoded-map performance 是否下降；
- planner SR 是否优于 J1/J2；
- 是否复现 P2 full 式组合退化。

## 9. Metrics

Primary：

- full900 `SR@128`、SPL；
- seen/OOD SR；
- diagnostic-aligned Local top-1/margin；
- per-size；
- K-scaling curve。

Secondary：

- all-cell local top-1；
- value Pearson/R2；
- wall IoU、agent/goal accuracy；
- invalid、loop/cycle；
- gradient cosine/norm；
- decoded reachability rate。

必须按 shortest-path length 分桶补充分析，因为 size 与实际推理步数不是同一变量。

## 10. 后续而非本轮主实验

只有当 J1/J2 已明显改善 local ordering 后，再加入：

- soft visit-count memory；
- directed-edge memory；
- adaptive halting；
- texture/color perturbation；
- size 27/29/31；
- multi-task JEPA。

记忆只能作为 loop symptom control，不能替代正确的 value propagation。

## 11. 文献依据

- DINO-WM: <https://arxiv.org/abs/2411.04983>
- RC-aux: <https://arxiv.org/abs/2605.07278>
- Fast LeWorldModel: <https://arxiv.org/abs/2606.26217>
- Value Iteration Networks: <https://arxiv.org/abs/1602.02867>
- Gated Path Planning Networks: <https://arxiv.org/abs/1806.06408>
- XLVIN: <https://arxiv.org/abs/2010.13146>
- Neural A*: <https://proceedings.mlr.press/v139/yonetani21a.html>
- Easy-to-hard recurrent reasoning: <https://arxiv.org/abs/2106.04537>
- Thinking without overthinking: <https://arxiv.org/abs/2202.05826>
- Logical extrapolation in mazes: <https://arxiv.org/abs/2410.03020>
- Increasing the action gap: <https://arxiv.org/abs/1512.04860>
- GradNorm: <https://proceedings.mlr.press/v80/chen18a.html>
- CAGrad: <https://arxiv.org/abs/2110.14048>
