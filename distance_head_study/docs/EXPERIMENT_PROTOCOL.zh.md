# Procgen Maze DistanceHead 规范实验协议

协议 ID：`procgen-maze-distance-head-staged-v1`

状态：代码实现版。服务器正式运行前必须通过 `audit_protocol` 并使用仓库内已提交的
`configs/protocol_lock.json`。本文件与 schema/config 冲突时，程序会 fail closed；
不得通过改代码绕过。

## 1. 科学问题

在不改变 LeWM encoder/projector 结构、Set B topology split、action mapping、planner
预算和 evaluator 的条件下，研究 goal-conditioned DistanceHead 的性能上限：

```text
o_s -> Encoder -> spatial -> Projector -> z_s
o_g -> Encoder -> spatial -> Projector -> z_g
(z_s, z_g) -> DistanceHead -> learned cost
z_history + action sequence -> Predictor -> predicted latent trajectory
predicted trajectory + learned cost -> search -> action
```

核心因果链是：

```text
distance accuracy
-> local action ordering
-> predicted/candidate ordering
-> closed-loop behavior
-> seen/OOD SR and SPL
```

任何中间 probe 改善都不能单独替代最终 SR；任何 SR 改善也必须用机制诊断判断它来自
head、predictor、search 还是 corrected assistance。

## 2. 数据合同

| Split | 数量 | Size | 用途 | 可否选模型 |
|---|---:|---|---|---|
| `D_train` | 2800 | `9-21` odd | backbone/head 训练 | 是 |
| `D_cal` | 140 | `9-21` odd | train-topology 数值与 gradient calibration | 不按 SR 选择 |
| `D_screen` | 140 | `9-21` odd | Seed-1 机制筛选 | 是，探索性 |
| `D_select` | 210 | `9-21` odd | Seed-3 一次性 shortlist 复现 | 仅一次 |
| `D_dev_legacy` | 900 | `9-25` odd | 历史 parity | 否 |
| `D_confirm` | 900 | `9-25` odd | fresh Seed-10 最终确认 | 否 |
| `D_stress` | 150 | `27/29/31` | primary closure 后 size 边界 | 否 |

硬约束：

- `23/25` 不得出现在 `D_screen/D_select`；
- `train/screen/select/confirm/stress` 的 `layout_hash` 和 `task_hash` 两两不重合；
- `D_cal` 是明确标记的 train-topology subset，不得称作 topology generalization；
- manifest 在运行前全部生成、提交、逐字节再生检查；
- evaluator 读取 manifest 固定的 start/goal，不得 reset 后换题；
- training cache 中 observation 的 goal marker、goal latent 与 BFS label 必须是同一个
  manifest goal。
- 固定 128-step cap 下，`D_confirm` 有 12 个、`D_stress` 有 33 个任务的 oracle BFS
  shortest path 超过 128。这些 task 不删除、不重采样，统一标为 `step_cap_ineligible`；
  因为比较逐 task 配对，它们影响绝对 SR 天花板，但不能按方法选择性处理。

## 3. Seed 与证据等级

| Tier | Backbone | Head | Evidence status |
|---|---|---|---|
| Seed-1 | `42` | `0/1/2` | `exploratory_single_backbone` |
| Seed-3 | `42/43/44` | 每 backbone `0/1` | `replicated_development` |
| Seed-10 | ordered fresh prefix，默认 `1001-1010` | 每 backbone `0` | `confirmatory` |

- 历史 seed registry 是 `42-61`；fresh list 从 `1001` 连续开始；
- head seed 是 backbone 内嵌套方差，不能当独立模型重复；
- Seed-1 未通过并锁定 shortlist 前不能释放 Seed-3；
- D_select 未完成、finalist/n 未锁定前不能释放 Seed-10；
- positive route 只有 Seed-3 expansion gate 通过才释放 Seed-10；
- negative route 必须先完成 method-family closure，仍确认两个不同机制 finalist；
- 任何 tier 训练都必须有签名 `seed_release`，不能按表现跳 seed。

## 4. 固定 planner/evaluator

```text
max_steps            128
history_size         3
horizon              12
num_candidates       64
num_elites            8
cem_iters              1
momentum             0.1
actions              1,2,3,4
model action vocab   0,1,2,3,4
rollout semantics    legacy_warmup_v1
reference budget     768 predictor transitions/decision
```

`b_l2_cem` 和 terminal DistanceHead CEM 调用原 `hdwm.planning.cem_plan`。扩展
path/iCEM/Beam/Best-first 使用 `vector_jepa_planner_frontier`，但保持相同 world-model
rollout semantics 与 transition cap。`cem_iters=1` 时，instrumented categorical CEM 与
原实现使用同一 seed 产生逐 candidate 完全一致的 population；回归测试会比较最终序列与
cost，避免把 planner 实现差异误当 cost 差异。

`768` 是每次决策统一的 predictor-transition **硬上限**。iCEM 精确用满；Beam 与
Best-first 只能扩展完整 action branch，余量不足一个完整 branch 时允许少量未用预算，
但绝不超限，也不使用无意义 no-op 补齐。每个 task 记录实际 transitions，比较时同时报告
性能、predictor-transition compute proxy 与每决策 wall-clock。Transition 数不是总 FLOPs：
它不包含 encoder、head 与 search bookkeeping，因此不能单独声称完整系统更快。

这里必须区分两个动作集合：环境与 planner 只允许四个移动动作 `1/2/3/4`，所以局部
optimal-action 随机基准是 `0.25`；LeWM predictor 的训练词表是五类 `0..4`，其中 `0` 是
stay/context。历史 corrected fallback 会一次计算五个 next-latent，再只在允许的移动动作中
选择，因此每次 assistance 的真实额外 compute 是 5 transitions，而不是 4。

Action protocols：

- `corrected_v1`：真实当前 state 上过滤撞墙动作、尽量避免 immediate backtrack；若
  CEM first action 不合法，使用同一 frozen predictor/scorer 做 one-step fallback；
- `unmasked`：直接执行 planner first action，不使用 validity correction；
- 每个 task 保存 assistance、proposed invalid/backtrack、executed invalid、repeat、loop、
  search transitions 与 fallback transitions；正式 compute 为二者之和除以决策步数；
- corrected 提升只支持 assisted Vector-JEPA claim。

## 5. 训练合同

Frozen head 主线：

```text
steps                 30,000
effective batch       512
microbatch            128
topologies/batch        8
sources/topology       64
head LR              1e-3
weight decay         1e-5
grad clip              1.0
warmup                 5%
final LR ratio         0.1
checkpoint             final step only
```

Sampler 每个 batch 只选一个 maze size，再在该 size 内 topology-balanced 抽样。相同
backbone、step 的所有方法读取同一 stateless schedule；head seed 只改变初始化，不改变
样本。

TRM 每个 optimizer step 共取 16 个 contexts，并均分到四个 microbatches。由于训练 batch
按 topology 连续分组，context row 使用固定等距索引覆盖每个 microbatch 内的 topology
block，不能总取前几行。Calibration 和 trajectory diagnostics 使用同一确定性规则。

Auxiliary loss weights 在 `D_cal` 上按相对 absolute-loss gradient norm 标定一次：

```text
target auxiliary gradient ratio = 0.5
weight multiplier clip          = [0.1, 10]
```

标定结果进入 `training_spec_sha256` 和 final checkpoint。不能对每个 seed 根据 SR 调权。

Joint route：

- `predictor`、`projector+predictor`、`full` 分别有 matched continuation control；
- treatment/control 从相同 `B-DH-CEM` checkpoint 开始；
- 同样步数、batch、optimizer、sample stream；
- backbone LR 固定 `1e-4`；
- 原 JEPA prediction/SIGReg/position/goal losses权重完全保留；
- matched control 与 treatment 都训练同一个 head、使用完全相同的 distance/predicted
  objective、batch、loss weights 和 optimizer group；唯一干预是 treatment 允许 distance
  gradient 穿过 latent 进入指定 backbone scope，control 在 latent 处 stop-gradient；
- joint diagnostics 必须从 checkpoint 内的更新模型重编码 source/goal/history/next/path
  observation，旧 frozen cache latent 只提供索引和 BFS label；
- `j3_rcaux_reach` 改变 horizon input/output shape，不能 strict-load 整个旧 head；它只
  兼容加载 shape 相同的 shared trunk/scalar scoring 层，新增输入权重与 reachability
  层随机初始化，实际 loaded key list 和 parent checkpoint hash 写入 checkpoint；
- 不允许只报告 treatment 而不跑 control。

## 6. Target、输出与 loss

Scalar target：`legacy_log_norm`、`log1p`、`raw`、`global_norm`。所有 raw BFS metric
必须 inverse-transform；planner 排序可以使用单调 transformed score。Ordinal、
distributional、quantile 输出直接产生 raw-step expected cost。

主要目标：

- absolute Huber/MAE/MSE；
- tie-aware all-valid-action listwise；
- one-step delta、Bellman、discrete Eikonal；
- shortest-path multistep 与 triangle；
- shortest-path prefix 到达 goal 后保持吸收态，不允许再走离 goal 污染长 horizon label；
- action-aligned one-step true-next 与 predicted-next listwise/consistency；对
  horizon-conditioned head，这两类局部动作查询的 horizon 输入都必须显式为 `1`，不能沿用
  trajectory scorer 的默认 `12`；
- horizon-matched candidate trajectory ranking；
- `legacy_warmup_v1` 下 12 个 rollout slots 只包含 11 个改变 terminal 的 candidate actions；
  因而训练 action-horizon grid 是 `1/3/5/8/11`，对应 scorer rollout-slot 输入
  `2/4/6/9/12`，不得把 action horizon 误写为 12；
- multi-budget reachability 与 monotonicity；
- uncertainty、successor contrastive、directed quasimetric。

`D3-TRM-SHUFFLE` 必须保留。没有 shuffled-label negative control，不能把 trajectory
增益归因于 horizon-matched supervision。

## 7. 诊断与 oracle

每个 development run 产生：

- raw BFS MAE/RMSE/bias/Spearman/calibration；对 horizon-conditioned head 固定以
  `h=12` 查询当前状态，训练时则在预注册 horizon grid 上对同一个 absolute BFS target
  重复监督，因此该指标不改变 target 含义；
- true-next Local top-1/regret/margin；
- predicted-next Local top-1/regret/margin；
- h12 fixed-bank candidate Spearman/regret；
- true-dynamics endpoint scorer 对照；
- fixed-bank 全部 candidate 的 nearest-latent BFS drift；
- by-size breakdown。

这里的 `predicted-next` 是 action-aligned 诊断：从相同当前 latent 分别施加四个移动动作，
并以 `h=1` 查询 scorer。它与历史 `b_dh_predictor_greedy` 的 corrected-v1 fallback
接口不同；后者只用于历史 parity，不能替代前者来证明局部动作排序能力。

Oracle 只路由研究，不进入 learned 排名：

- `O-DYN`：true rollout endpoint + learned head；
- `O-SCORE`：true rollout endpoint + true BFS score；
- `O-BFS1`：真实 one-step BFS oracle。

## 8. 阶段 gate

Seed-1 ordinary gate：

- 三个 head seeds 方向记录完整；
- predicted local top-1 `+0.05`，或 candidate regret 至少下降 `20%`；
- corrected SR 相对 `B-DH-CEM` 至少 `+0.04`；
- unmasked SR、SPL 均不下降超过 `0.02`；
- negative control、compute、assistance 完整。

Strong fast lane 将 corrected SR 门槛提高到 `+0.06`，且三个 head runs 都优于配对
baseline。它只允许跳过 reserve，不构成最终结论。

Seed-3 expansion gate：

- 每 backbone 先平均 head `0/1`；
- 三 backbone mean corrected delta 至少 `+0.04`；
- 至少 `2/3` backbone 为正；
- 任一 backbone 不低于 `-0.02`；
- ranking gate 和 secondary non-inferiority 同时通过。

若正路线 Seed-3 gate 失败但要形成广泛负结论，不能覆盖已打开 `D_select` 使用的
`shortlist_lock/finalist_lock`，也不能往同一 split 补新候选。必须在 Seed-1 `D_screen`
完成预注册 reserve closure，写入独立 `negative_shortlist_lock`；该 lock 同时绑定失败的
原 Seed-3 finalist、完整 closure ranking 和两个机制 strata。它只决定 fresh confirmation
中要检验哪两个方法，不把 `D_screen` 结果当最终证据。

## 9. Confirmatory analysis

- 默认至少 10 个 fresh backbones；若 baseline-only power 要求更多，只能按 ordered
  list 扩大；
- power 不得读取 candidate/finalist effect；
- power 输入固定为 `D_dev_legacy` 上 seeds `42/43/44`、head seed 0 的
  `B-DH-CEM/corrected_v1` task rows；Seed-3 DAG 自动生成这三组 baseline-only 输入；
- primary：corrected overall SR 与 corrected size-23/25 OOD SR；
- contrasts：final vs `B-DH-CEM`，以及 final vs `B-L2` anchor；
- crossed paired bootstrap：每次先重采样 backbone，再按 maze size 重采样一套共享 task
  indices，并把同一 task draw 应用于所有入选 backbones；至少 10,000 replicates，用于
  effect interval。不能把所有 backbones 共用的 task 错当成各 backbone 内独立嵌套样本；
- bootstrap seed schedule 必须按锁定 base seed 逐项重生成一致，而不只检查文件签名；
- superiority p-value 使用 backbone 为独立单位的单侧 sign-flip test；`n<=16` 精确枚举，
  更大 n 使用锁定 seed 的 Monte Carlo；
- Holm correction 覆盖全部 primary contrasts；负路线两个 finalist 的 8 个 contrasts
  作为同一个 family 做 Bonferroni；
- positive MEI：overall `+0.04`、OOD `+0.05`；
- secondary：unmasked、SPL、loop、invalid、assistance、compute；不能挽救 primary failure。

Positive claim 需 MEI、adjusted superiority 与 secondary safety 全通过。若只超过
`B-DH-CEM` 而未超过 `B-L2`，不能称为新的最强 Vector-JEPA。

Negative claim 需：method-family closure 完成、两个机制不同 finalists 均进入 confirm、
familywise one-sided upper bound 对两个 endpoint 都排除 MEI。否则结论是 `null`，不是
“证明无效”。

## 10. 不允许的改动

- 用 `23/25` 或 legacy full-900 选模型；
- 改 candidate budget 后与原 baseline 直接归因；
- 对好看的方法多训练、对差的方法早停；
- 从多个 head seed 事后挑最佳；
- 根据 D_select effect 调 confirmation n；
- 在打开 D_confirm 后新增方法、改权重、换 checkpoint；
- 把 oracle/test BFS 结果混入 learned ranking；
- 用 aggregate JSON 代替 per-task rows；
- 覆盖正式 artifact；
- 在 dirty worktree 做 formal run。

`--diagnostic-limit`、diagnostic batch override 等 smoke 参数只改变输出目录和样本量，
不绕过 seed release、shortlist、confirm-open 或 closure gate。Job 进程返回 0 也不自动算
成功；cache/checkpoint/diagnostic/result/confirm-open 必须先通过对应内容级 validator，才会
写 completion seal。所有阶段 lock 扁平保存上游 evidence hashes，最终 closure 可直接核验
完整证据链。

完整方法动机见 [研究总纲](DISTANCE_HEAD_RESEARCH_PLAN.zh.md)，高效率路径与计算估计见
[执行总纲](EFFICIENT_EXECUTION_PROTOCOL.zh.md)。
