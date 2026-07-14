# 方法、结构与训练定义

## 1. 共同 Vector-JEPA 接口

所有非 oracle 方法都从原 `Unisize256` checkpoint 开始：

```text
RGB observation
-> encoder
-> embedding_projector
-> pooled 256-d vector z_t
-> action-conditioned predictor
-> imagined vectors z_t+1 ... z_t+H
-> scorer / proposal / search
-> first action
-> optional Corrected-v1 executor
```

planner 不读取真实坐标、wall map、BFS 距离或 spatial feature map。BFS 只允许作为 train-only head target、validation 后验标签和显式 oracle 变量。默认 `history=3`、`horizon=12`，`1x=768` predictor transitions/decision，硬预算为 `0.5x/1x/4x/16x`。

Track F 冻结 encoder、projector、predictor 的结构和参数，只训练 planner heads。Track J 保持三者结构不变，但允许参数联合更新。二者都属于本项目定义下的 Vector-JEPA；报告中必须明确 Track F 或 Track J。

## 2. B0：严格历史锚点

`b0_legacy_l2_cem` 不是重写版近似，而是直接调用旧 `hdwm.planning.cem_plan`：

```text
history=3, horizon=12
num_candidates=64, num_elites=8
cem_iters=1, momentum=0.1
terminal pooled-latent squared L2
legacy_warmup_v1
replan every environment step
max_steps=128
```

B0 没有 planner-head seed。新 adapter 与旧 `_latent_rollout_cost`、CEM sequence/cost、真实 maze 首动作均有逐值回归测试。

## 3. P1：Oracle Ladder

Oracle 不进入主表，只回答“还有多少可恢复空间”。

| Rung | 唯一打开的 oracle | 诊断问题 |
|---|---|---|
| O0 | 无额外 oracle | Corrected-v1 anchor |
| O1_PROP | 注入一条 BFS-compatible prefix | candidate coverage 上限 |
| O2_SELECT | 真实候选终点的 BFS progress 选优 | selection/backup 最大损失 |
| O3_DYN | 真实环境 transition 产生终点观测，再做 latent-L2 | rollout drift 最大损失 |
| O4_VALUE | imagined terminal 最近邻解码为真实 cell，再读 BFS value | distance/value 最大损失 |
| O5_JOIN | 双向候选使用真实状态 equality/separation | learned join 上限 |
| O6_VALID_FUTURE | 每个 imagined step 只采样真实合法动作 | future feasibility 上限 |

O5 使用双向搜索的匹配预算；禁止把它与 1x O0 的差直接写成单模块收益。所有 oracle environment queries 单独计数。

## 4. P2：搜索 backbone

P2 在完全相同的 latent-L2 scorer、uniform proposal、task、search seed 和 predictor-transition cap 下比较：

| Planner | 主要机制 |
|---|---|
| categorical CEM | 完整 action sequence 的 categorical distribution 更新 |
| iCEM | elite reuse 和跨 replanning warm start |
| beam | 按 prefix 深度保留多样候选 |
| best-first | 以统一 scorer 为优先级展开 latent prefix |
| MCTS | PUCT selection、leaf value、backup |

每种搜索器都运行四档预算。`legacy_cem` 只占 categorical-CEM 的历史 1x
cell；其余 cell 使用有完整 compute ledger 的新实现。P2 选择只允许在 1x/4x
中进行。赢家总是映射到该搜索器已登记的 4x cell；P3 的全部方法随后动态
继承这份冻结 planner spec，而不是把 best-first 写死。

## 5. P3：完整 `2^4` 因子设计

搜索 backbone 是 P2 冻结赢家的 4x 版本。除搜索器外，16 个 cell 只改变
四个二元因子；编码为 `v?r?p?m?`：

| 因子 | 0 | 1 |
|---|---|---|
| V | 无 verifier | action-consistency verifier，权重 0.3 |
| R | 仅 latent-L2 | distributional reachability，权重 1.0 |
| P | uniform proposal | 0.50 uniform + 0.25 retrieval + 0.25 learned |
| M | 无 memory | precision-gated transposition memory |

必须报告四个主效应、六个二阶交互和全部 16 个 cell，不能只报告最好 cell。

五个机制对照是 verifier action-label shuffle、transition-pair shuffle、随机未训练 verifier、candidate-association shuffle 和 proposal-only。它们用于判断增益来自正确监督关系还是参数量/额外计算。

## 6. Planner heads

所有 head 输入都只包含 pooled vectors 及其代数组合。默认 `latent_dim=256`、`hidden_dim=512`。

### ActionConsistencyVerifier

输入 `[z_t, z_next, z_next-z_t, z_t*z_next]`，三层 MLP 输出四动作 logits，训练目标是 `q(a | z_t,z_t+1)` 的交叉熵。规划时把候选 transition 的 action NLL 加入 scorer。

### DistributionalReachability

输入当前/goal pair features，输出八个单调 CDF 值：

```text
P(D <= b | z, z_goal), b in {1,2,4,8,16,32,64,128}
```

正增量参数化保证随 budget 单调。监督是 train topology 上的 directed BFS distance。它不是简单 latent L2，也不是无方向 scalar distance；正式报告必须给每个 bin 的 Brier、ECE 和 AUROC。

### StateJoinHead

pair-feature MLP 输出 same-state/join 概率。Validation 校准阈值；hard pruning 只有在 precision `>=0.95` 时允许，否则 evaluator fail closed。未达门槛时不能降低阈值救结果。

### AutoregressiveProposal

goal-conditioned GRU 逐步生成 action chunk。条件来自 source/goal pair features，teacher target 是 train topology 上的 BFS-optimal chunk。它只负责候选分布，最终仍由统一 search/scorer 选择。

### DiscreteDenoisingProposal

两层、8-head Transformer 对 mask/noisy action tokens 做离散去噪，默认 8 个 denoising steps。它与 uniform、retrieval、proposal-only 对照配套运行。

### VectorDTSHead

共享 pair-feature trunk 后输出四动作 expansion policy、八-bin value logits 和非负 uncertainty。它服务于 Vector-DTS 搜索；direct/random/fixed-breadth/learned expansion 四个版本用于拆分收益。

### CounterexampleRanker

输入 source/terminal pair features 和候选 action histogram，输出完整 trajectory score。训练 pair 必须共享 root、goal、horizon 和 candidate budget：

```text
L_rank = -log sigmoid(score(good) - score(false_optimistic))
```

## 7. P4：激进方法与匹配对照

P4 有三族，共 10 个配置：

1. Vector-DTS：learned、direct、random expansion、fixed breadth。
2. Bidirectional：learned bidirectional 与 forward-only；真实 join 位于 O5，不进入主表。
3. Denoising iCEM：learned denoising、uniform、retrieval、proposal-only。

P4 方法彼此不先组合。每个处理组都必须与同预算、同 task、同 seed 的对应对照比较。

## 8. P5/P6 继承链

`p5_track_f_all_hard_memory` 是历史兼容名称，不代表四个组件必然全部入选。
P5 决策会对 verifier、reachability、proposal、memory 各检查六项证据：机制
指标改善、overall 非劣、size-19/21 非劣、等计算、负对照通过、方向一致。
通过全部六项的组件组成 `selected_components`，并精确映射到一个 P3 factorial
cell。P4 的 Vector-DTS、bidirectional、denoising 三个 radical 使用同样六项
gate，最多选择一个；若多个通过，则按 corrected overall SR、size-19/21 SR、
名称的预注册规则确定。若组件和 radical 均为空，实验按协议关闭。

P5 本身不训练。`assemble_p5.py` 从所选 P3 calibrated checkpoint 和可选 P4
radical checkpoint 逐 tensor 合并所需 heads，并记录每个 head 的 owner。P3
source 优先拥有同名 head；radical 只补充 effective P5 仍需要而 P3 未提供的
head。确认冻结会重新加载父工件、核对 SHA256，并逐 tensor 验证组装结果。
planner/control 可由 radical 替换；scorer 和 memory 仍由所选 P3 cell 决定；
denoising radical 还会替换 proposal。新增 optimizer steps 始终为 0。

P6 从 P5 开始，只新增 ranker 参数：

```text
round 1 <- mining fold M1
round 2 <- mining fold M2
round 3 <- mining fold M3
```

fold 由 task hash 决定，topology-disjoint；ranker 在进入三轮采矿前已有匹配的
30,000-step 初始训练，每轮再恰好运行 20,000 optimizer steps，不追加 round 4。
每轮 hard dataset 必须保留 root、goal、candidate actions、false-optimism 标签、
source checkpoint 和前一轮 checkpoint 的完整哈希链。`p6_control_random_negative_ranker`
使用相同结构、初始步数、三轮步数、seed 和 root，只替换 negative action
sequence，不能改变正例或任务分布。

## 9. P7：语义对照与 Track J

`p7_control_action_aligned_frozen` 直接复用 P6 round-3 checkpoint，所有参数不变，只把 rollout 切换为 `action_aligned_v2`。它不生成 train、calibrate、retrieval 或 mining job。

`p7_track_j_joint_all` 是配置模板。Schema 在运行前把它展开为 54 个 cell，
每个 cell 都从同一 P6 hard-negative round-3 parent 初始化，在
`action_aligned_v2` 下联合更新 world model 与 planner heads。Track J 使用长度
`T=8` 的真实轨迹；JEPA prediction/SIGReg/位置损失和 planner heads 在同一训练
过程中更新，ranker negatives 只来自已冻结并验证的 P6 三轮 hard datasets。
目标为：

```text
L_joint = L_prediction
        + 0.09 * L_SIGReg
        + absolute/relative/goal-position losses
        + 0.3 * L_planner
```

网格为：

```text
planner LR              in {1e-4, 3e-4}
backbone LR multiplier  in {0.01, 0.03, 0.1}
planner-loss weight     in {0.1, 0.3, 1.0}
SIGReg multiplier       in {0.5, 1.0, 2.0}
```

每个 cell 都运行恰好 30,000 steps，采用 5% linear warmup 后 cosine decay 到
初始 LR 的 10%。模块初始化和 stochastic stream 按组件隔离，使 factorial 差异
不会因启用另一 head 而改变无关 head 的初值、batch 顺序、denoising mask 或
random-negative stream。

P7 只在 54 个 cell 全部完成后冻结选择。候选必须让全部
`20 backbones x 2 planner seeds=40` 个 final-step checkpoint 通过 10% JEPA
stability gate；在稳定候选中先保留 corrected validation SR 距稳定最大值不超过
0.01 的 cell，再最小化最大/平均 JEPA 恶化，随后依次偏好 size-19/21 SR 和
方法名。若没有候选满足稳定性，Track J 分支关闭，P8 与确认仍可只运行 Track F。

Track J final-step checkpoint 必须重新计算与 source 完全匹配、同为 `T=8`
且按 validation sizes round-robin 的 JEPA objective。相对恶化超过 10% 时该
cell 不可成为 P7 赢家，不能换回更早 checkpoint。

## 10. P8：只改预算的最终前沿

P8 为 P5、P6 和 **P7 冻结赢家**各定义 `0.5x/1x/16x` alias，并把各自原
`4x` 结果作为第四点。若 P7 没有稳定赢家，Track J 三个 alias fail closed，
P8 只形成两条 Track F frontier。Schema 强制 alias 的 track、scorer、proposal、
memory、control、rollout semantics 和 planner 其他字段与 source 完全一致。

P8 alias 解析到 source 的同一个 checkpoint SHA256：

```text
p8_p5_* -> P5 calibrated checkpoint
p8_p6_* -> P6 round-3 checkpoint
p8_p7_* -> P7 冻结赢家的 calibrated Track J checkpoint
```

因此 P8 的任何差异都只能来自 planner transition cap，不能来自再次训练、重新校准或新 retrieval bank。

## 11. 校准、checkpoint 与随机性

- 所有梯度样本来自 train manifest；validation 不执行 optimizer step。
- Amendment 001 规定所有训练方法统一使用 final optimizer step，不选择中间最好 checkpoint。
- planner-head seed 是真实独立初始化；search seed 只控制随机搜索；task/time step 不是独立重复。
- 描述性聚合先平均 search seed，再平均 planner seed；确认性 nested bootstrap
  先以 backbone 为最高层重采样，再在每个被抽中的 backbone 内重采样 planner
  seeds，最后在 maze-size strata 内重采样 task。exact sign-flip 仍在 20 个
  独立 backbone 差值上进行。
- 每个 checkpoint 保存 analysis spec、training spec、source/parent SHA256、代码 fingerprint、Git dirty 状态和校准指标。
