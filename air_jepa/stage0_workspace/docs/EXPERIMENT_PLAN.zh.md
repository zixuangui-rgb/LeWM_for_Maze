# AIR-0：冻结 Spatial-JEPA 的迭代工作空间可行性实验

## 0. 文档状态

- 实验 ID：`procgen-maze-air0-workspace-v1`
- 方法版本：`AIR0-v1`
- 证据等级：`exploratory_architecture_feasibility`
- 主任务：Procgen Maze topology hold-out 与 size OOD 泛化
- 主执行协议：`unmasked`、每步重新观测并规划、`max_steps=128`
- 独立训练重复：3 个完整 system seeds，`42/43/44`
- 主评测：每个 seed 的新 `AIR_dev` full-900
- 计划状态：在第一次正式训练前冻结；冻结后任何科学变化都必须升级实验 ID

本实验是激进架构重构路线的第一个正式决策单元。它不以论文级最终确认
为目标，而是判断共享循环 Token Workspace 是否值得成为下一代 AIR-JEPA 的
模型主干。实验必须完整执行预先锁定的训练、full-900 主评测、全 K 曲线、
机制干预和审计；中途看到的性能不得改变本阶段矩阵。

本文中的“快速初步结论”通过预先规定的分层读出实现，而不是缩减任务、减少
seed、按分数早停或临时修改方法。快速读出只允许提前讨论和准备下一阶段，不能
删除、增加或重跑本阶段的科学 cell。

## 1. 背景与决策依据

现有同协议的重要锚点为：

| 方法 | Overall SR | OOD SR | 解释 |
|---|---:|---:|---|
| `j0_spatial_feedforward` | 0.623 | 0.248 | Frozen Spatial-JEPA + 非迭代 planner |
| `j1_spatial_iterative_frozen` | 0.936 | 0.805 | Frozen Spatial-JEPA + ConvGRU 迭代 planner |
| `r4_raw_iterative_progressive` | 0.949 | 0.844 | Raw map + ConvGRU 迭代 planner |
| exact BFS / VI oracle | 0.979 | - | `max_steps=128` 下的环境上限 |

这些结果已经支持三个判断：

1. Spatial-JEPA 表征在该任务中可以支持高质量规划；
2. 固定深度 feedforward planner 在长路径和未见尺寸上明显不足；
3. 冻结 encoder 已能达到 0.936，立即联合训练 encoder 不是第一优先级。

因此 AIR-0 不再把 `a1_reach=0.714` 或早期 pooled-vector CEM 当作主要对手。
AIR-0 的直接强基线是使用同一 Spatial-JEPA 表征的 `j1`。本阶段需要回答的是：

> 在冻结相同 Spatial-JEPA encoder、保持数据与评测协议一致时，一个把空间状态、
> action-conditioned future、cost/energy 和共享循环推理放入同一工作空间的模型，
> 能否接近现有强迭代 planner，并且表现出可验证的 future 使用和 test-time
> compute scaling？

## 2. 本阶段可以和不可以回答的问题

### 2.1 可以回答

- 新 Token Workspace 主干是否具有继续开发的绝对性能基础；
- 在相同冻结 Spatial-JEPA 表征下，它是否接近或达到 `j1`；
- 更多共享迭代步数是否改善长路径与 size OOD；
- action-conditioned future 监督是否被模型真正使用；
- 剩余瓶颈更接近 future prediction、energy ranking、迭代传播还是执行循环；
- 下一阶段应优先加入 memory、adaptive halting，还是先修复核心数据流。

### 2.2 不可以回答

- AIR-JEPA 已达到论文级最终优越性；
- 三个 seeds 足以证明跨随机初始化的普遍规律；
- 在 texture、背景、跨任务或跨环境上已经泛化；
- weight sharing 在严格等 FLOPs 下优于所有 feedforward Transformer；
- memory、动态停止、层级 subgoal 或联合训练 encoder 的效果；
- BFS 训练监督是否具有更高 sample efficiency；
- corrected 成绩代表模型的无辅助能力。

上述问题只能在 AIR-0 通过后，由后续独立实验处理。

## 3. 研究问题、估计量与独立重复

### 3.1 研究问题

**RQ1：架构可行性**

`AIR0-jepa` 在 `AIR_dev` full-900、unmasked、K128 上能否达到 overall SR 0.90，
并相对 matched `j1-receding` 满足 0.03 的非劣边界？

**RQ2：JEPA future 是否有实际作用**

在相同模型结构、训练样本、参数量和 K schedule 下，加入 next-latent 与
cost-to-go 监督后，`AIR0-jepa` 是否不劣于 action-only 的 `AIR0-direct`，且
future 干预能够改变动作排序？

**RQ3：迭代计算是否产生预期行为**

对同一个 checkpoint 增加 K，是否主要改善 OOD 与长路径；同时 K128 的局部动作
排序是否达到足以支持该导航结果的水平，而非只改变输出数值尺度？

**RQ4：若未通过，失败位于哪里**

失败究竟来自表示不足、future prediction、predicted-future 到 energy 的转化、
循环推理传播，还是闭环执行？

### 3.2 处理单位与独立重复

- 科学处理发生在完整训练运行层面；一个 `system seed × method` 是一个独立训练
  单位。
- 本阶段有 3 个独立 system seeds，而不是 2700 个独立重复。
- 同一 seed 下的 900 个 task 是对该训练模型的重复测量；它们用于配对提高精度，
  不能当作 900 个独立训练重复。
- 同一 checkpoint 的多个 K 是 repeated measures，不得按独立模型处理。
- 统计分析必须保留 `seed` 和 `task_id` 的交叉结构。

## 4. 固定模型结构

### 4.1 输入与冻结表征

正式方法统一使用已有 `spatial_info_sigreg` 表征：

- 每个 system seed 使用与其 seed 对应的正式 encoder checkpoint；
- encoder 与既有 non-pooled planning projector 在 AIR-0 中完全冻结；
- AIR 与 matched J1 都使用该 planning projector 输出的 full-resolution
  `H × W × 64` spatial latent；
- 不允许全局 pooling 后再进入 Reasoner；
- 不向模型输入 agent/goal 坐标、symbolic wall grid、BFS 距离或真实有效动作；
- 目标 encoder 与在线 encoder 相同且全程 `detach`，因为 encoder 已冻结。

### 4.2 AIR0 Workspace

每个样本包含：

- mutable spatial state field：`S^k ∈ R^(H×W×64)`；
- 1 个 goal/workspace token：`G^k ∈ R^64`；
- 4 个 action tokens：`A^k ∈ R^(4×64)`；
- 4 个 future summary tokens：`U^k ∈ R^(4×64)`；
- 4 个 lower-is-better energy outputs：`E^k ∈ R^4`。

`G^0` 必须通过学习查询从 spatial latent 中提取，不能用真实 goal 坐标初始化。
action tokens 只编码固定的 `[UP, DOWN, LEFT, RIGHT]` 动作身份。模型不得读取
测试环境的动作有效性列表。

### 4.3 共享 Reasoner Block

AIR0 只使用一个参数共享的 Reasoner Block，循环执行 K 次：

1. spatial tokens 通过四邻域加 self 的局部相对注意力交换信息；
2. workspace tokens 在自身之间执行 self-attention；
3. workspace tokens 与 spatial tokens 双向 cross-attention；
4. pre-norm、gated residual 和 FFN 更新 `S/G/A/U`；
5. 所有 K 共享同一组权重。

固定结构约束：

- hidden dimension：64；
- attention heads：4；
- FFN expansion：2；
- dropout：0；
- 空间位置只允许 size-agnostic 的二维相对偏置；
- 不允许按最大 maze size 学习绝对位置表；
- padding cell 必须显式 mask；
- K 改变时参数量不变；
- 正式训练前必须记录总参数量和 size-21/25 的确定性 analytical MACs；硬件
  wall-clock 另行实测，不能与 MAC 混称。

如果实现无法在 H800 上满足显存约束，可以统一使用等价的 chunked/local
attention 实现，但不能改变邻接范围、hidden dimension、token 数或有效计算图。

### 4.4 Action-conditioned future 与 energy

在需要监督或决策的迭代点，使用共享 action-conditioned decoder 从
`S^k/G^k/A^k/U^k` 产生四个 spatial future fields：

```text
F_a^k ∈ R^(H×W×64),  a ∈ {UP, DOWN, LEFT, RIGHT}
```

四个动作共享 decoder 参数，仅 action token 不同。`U_a^k` 只进入 future decoder，
不得直接进入 energy head。energy head 用 goal/action tokens 构造 attention query，
但最终 distance classifier 只能读取从对应 `F_a^k` 池化出的 value，不得把 query
token 直接拼入 classifier，也不能建立从原始 `S^0` 到 energy 的旁路。该限制使
future permutation 和 true-future replacement 具有可解释性。

测试时选择：

```text
a_t = argmin_a E_a^K
```

模型执行一步，读取新观测，再从 `S^0` 开始运行新的 K 轮推理。AIR-0 不跨环境
步保留 memory。

## 5. 方法与对照矩阵

### 5.1 正式方法

| ID | Encoder | Planner/Reasoner | 训练目标 | 执行方式 | 身份 |
|---|---|---|---|---|---|
| `oracle_bfs` | symbolic | exact BFS | 无 | unmasked equivalent | evaluator gate |
| `j0-static` | frozen spatial JEPA | 既有 feedforward field | 既有正式 checkpoint | static field | 下界锚点 |
| `j1-static` | frozen spatial JEPA | 既有 ConvGRU K128 | 既有正式 checkpoint | 每 task 一次 field | 历史桥接 |
| `j1-receding` | frozen spatial JEPA | 同一既有 ConvGRU | 既有正式 checkpoint | 每步重算 | matched 强基线 |
| `AIR0-direct` | frozen spatial JEPA | AIR0 shared Reasoner | tie-aware action CE | 每步重算 | action-only control |
| `AIR0-jepa` | frozen spatial JEPA | 完全相同 AIR0 Reasoner | action + future + cost | 每步重算 | treatment |

`AIR0-direct` 与 `AIR0-jepa` 必须逐字段相同，唯一允许的科学差异是以下 loss
权重：

| Loss | `AIR0-direct` | `AIR0-jepa` |
|---|---:|---:|
| tie-aware action | 1.0 | 1.0 |
| future latent | 0.0 | 1.0 |
| distributional cost | 0.0 | 0.5 |

不能为两种 AIR 方法分别调整 learning rate、K schedule、训练步数、batch size、
初始化、decoder 或 checkpoint 选择规则。

因此 `AIR0-jepa - AIR0-direct` 估计的是“future latent + distributional cost”
联合训练包的效果，不能在本实验中把差异单独归因于其中某一项。两者的进一步
拆分只有在 AIR-0 通过且使用独立数据角色时才有意义。

可执行 protocol matrix 共 135 个科学 cells：6 个正式训练、6 个历史 bridge、
69 个 AIR_dev unmasked、15 个 AIR_dev corrected、4 个 early-context、30 个
early future-intervention、1 个 early diagnostic、3 个 full diagnostic 和 1 个 BFS
evaluator oracle。另有 protocol audit、2 个 smoke train、benchmark、bridge audit 与
3 个 release 共 8 个编排/门禁 job，因此完整 DAG 固定为 143 jobs。两类计数必须
同时由代码重建并匹配，不能只检查总数。

### 5.2 为什么同时保留 static 与 receding J1

- `j1-static` 先在旧 confirmatory manifest 上复跑或逐行验证历史结果，再在新的
  AIR_dev 上评测；前者检查 evaluator/bridge 语义，后者才进入本阶段结果；
- `j1-receding` 与 AIR0 一样每步读取当前观测，是主要架构比较对象；
- static/receding 的差异是执行协议诊断，不能混成同一个 baseline。

桥接不得改写 source checkpoint 的训练 metadata 或 tensor。它必须保存 source
checkpoint hash、旧 evaluator 结果 hash、新 evaluator code hash，以及旧任务上的
逐 task parity 差异。parity 不通过时禁止打开 AIR_dev。

## 6. 训练目标与标签

### 6.1 训练状态采样

每个 optimizer step：

1. 从 2800 个 train topology 中均匀采样 map；
2. 从该 map 的可达、非 goal free states 中均匀采样当前状态；
3. 按环境真实转移语义生成四个动作的 successor；
4. 编码当前观测与四个 successor 观测；
5. 计算四个 successor 到 goal 的 BFS distance 标签。

无效动作必须沿用环境本身的 no-move 转移，不允许人为删除。它的 cost 由真实
successor 自然确定，不增加测试时可用的 oracle mask。

同一 seed 下两种 AIR 方法使用相同的 map、state、action-target 与 K RNG 流。
各 RNG 流必须独立命名，防止某种 loss 多抽样一次后改变后续训练序列。

### 6.2 Tie-aware action loss

令 `d_a = d_BFS(T(s,a), goal)`。所有达到最小 `d_a` 的动作构成最优集合，目标
概率在这些动作上均匀分配。不得用动作编号顺序强行选一个唯一标签。

energy 作为 logits 时使用 `-E_a`：

```text
L_action = CE(soft_optimal_action_target, -E)
```

### 6.3 Future latent loss

直接比较预测 spatial field 与冻结 encoder 的真实 successor field。为了避免
“地图大部分没变，所以复制当前 latent 也很准”的假象，future loss 固定为：

```text
L_future = 0.5 * normalized_field_error(F_a, Z_next_a)
         + 0.5 * normalized_delta_error(F_a - Z_now,
                                         Z_next_a - Z_now)
```

两个误差都按非 padding cell 平均，并以 detached target variance 和 epsilon
归一化。报告必须同时给出原始误差、delta 误差和 copy-current baseline；禁止只
报告全图 cosine。

### 6.4 Distributional cost loss

- 目标为 successor 到 goal 的 BFS distance；
- bins 固定为 `0..128`，超过 128 的值截断到 128；
- 使用 129 类交叉熵并除以 `log(129)` 归一化；
- scalar energy 由预测分布的期望得到；
- action CE 与 cost loss 使用同一四个 candidate futures。

这是与 `max_steps=128` 对齐的**有意 bounded-cost 定义**，不是对任意长路径绝对 BFS
距离的无截断估计。训练与评测 topology 中确实可能出现 `d_BFS>128`；action CE 的
最优集合仍由未截断的真实 candidate distances 计算，而 distance calibration 必须报告
target clipped rate。任何结论都不得声称该 head 区分了 128 以上的绝对距离。

### 6.5 Deep supervision

- `K_train = {4,8,16,32,64,128}`；
- 30,000 steps 平分为 6 个连续的 5,000-step phase；phase `i` 使用独立 K RNG
  从 `K_train` 的前 `i` 个值中均匀采样，与既有 `j1/r4` progressive 规则一致；
- 每次运行的最终 K 必定输出 action/cost 监督；当 `K>=16` 时，另在
  `k=16,32,48,...` 且小于 K 的位置输出中间监督；
- 各中间点权重与 `k/K` 成正比，归一化后总和为 1；
- future field loss 只在该 step 的最终 K 计算，避免训练计算无边界增长；
- 不使用验证集选择 K 或 checkpoint。

## 7. 数据角色与拓扑隔离

### 7.1 固定数据角色

| Role | 数量 | Sizes | 是否可看性能 | 用途 |
|---|---:|---|---|---|
| Train | 2800 | 9-21，每 size 400 | 是 | 两个 AIR 模型训练 |
| `AIR_preflight` | 140 | 9-21，每 size 20，train topology states | 是 | 数值与性能 benchmark，不做方法选择 |
| `AIR_dev` | 900 | 9-25，每 size 100 | 本阶段按权限分层开放 | AIR-0 全部正式结论 |
| `AIR_select` | 900 | 9-25，每 size 100 | 本阶段禁止 | 下一阶段独立选择集 |
| `AIR_final` | 900 | 9-25，每 size 100 | 架构冻结前禁止 | 最终确认集 |

`AIR_dev` 中 sizes 9-21 共 700 个 seen tasks，sizes 23/25 共 200 个 OOD tasks。

### 7.2 生成与封存

- 三个 900-task manifest 必须在第一次 AIR 正式训练前一次性确定性生成；
- 生成种子、任务行、canonical topology/layout/task hashes 和 SHA256 全部提交；
- Train、旧 development、旧 confirmatory、AIR_dev、AIR_select、AIR_final 之间的
  topology/layout/task overlap 必须全部为 0；
- `AIR_select` 与 `AIR_final` evaluator 默认拒绝执行，必须由未来独立 release
  文件解锁；
- 任何 manifest 生成失败只能修生成器并重新冻结全部新 manifest，不能人工挑图；
- AIR-0 方法、loss、K、seeds 和门槛不能根据 AIR_dev 结果修改后在同一 ID 下重跑。

### 7.3 Early-look 子集

`AIR_dev_early210` 是 AIR_dev 的预先固定分层子集：

- seen 9-21：每 size 20，共 140；
- OOD 23/25：每 size 35，共 70；
- 子集选择只由 task hash 的固定排序决定；
- 不允许根据 maze 难度、BFS 长度或初次结果替换任务。

该子集只用于最快的方向提示。完整 full-900 必须随后无条件运行。

## 8. 训练协议锁

两个 AIR 方法统一固定：

| 字段 | 固定值 |
|---|---|
| optimizer steps | 30,000 |
| map-state batch | 8 |
| optimizer | AdamW |
| learning rate | `1e-3` |
| betas / epsilon | `(0.9,0.999)` / `1e-8` |
| weight decay | `0` |
| scheduler | cosine |
| gradient clip | `1.0` |
| hidden dimension | 64 |
| seeds | `42,43,44` |
| deterministic algorithms | true |
| dtype / AMP | float32 / disabled |
| checkpoint | final step only |
| early stopping | 禁止 |
| encoder updates | 0 |
| overwrite existing artifact | 禁止 |

每 500 steps 必须记录：

- total/action/future/cost losses；
- field 与 delta future errors；
- copy-current baseline；
- gradient norm；
- 当前 500-step window 和累计实际 K 分布；
- 每个 window 的 elapsed seconds、steps/s、peak GPU memory 与 non-finite 检查；
- 每个 loss 分支对 shared Reasoner 的 gradient norm。

如果出现 NaN、Inf、OOM 或实现错误，受影响的正式 cell 标记为 technical invalid，
修复后必须升级 code hash 并从 step 0 重跑同一 method family 的所有受影响 cells。
低 SR、收敛慢或不符合预期不是 technical invalid，也不是追加 steps 的理由。

## 9. 评测协议与完整矩阵

### 9.1 Primary absolute-ability protocol

| 字段 | 固定值 |
|---|---|
| split | `AIR_dev` full-900 |
| max steps | 128 |
| action order | `[UP, DOWN, LEFT, RIGHT] = [1,2,3,4]` |
| action selection | unmasked |
| observation cadence | AIR 与 matched J1 每步重新读取观测 |
| primary K | 128 |
| eval RNG | 固定，不随 checkpoint 改变 |
| BFS/map/validity access | 模型不可访问 |

### 9.2 无折扣完整 K 曲线

对 `j1-receding`、`AIR0-direct` 和 `AIR0-jepa` 的每个 seed，在 full-900 上全部运行：

```text
K_test = {1, 4, 8, 16, 32, 64, 128}
```

K128 必须先运行以尽快产生 primary provisional result，其余 K 随后继续。不能从
K 曲线选择最佳 K 改写主结果，K128 始终是 primary。

`j0-static` 只运行其锁定 feedforward depth；`j1-static` 运行 K128 作为历史桥接。

### 9.3 Assistance diagnostic

所有 K128 方法额外运行 `corrected`：使用真实墙体过滤 no-move action，并避免
immediate backtracking。它只量化 assistance gap，不能进入主表或晋级判定。

AIR-0 不加入 learned validity head，因此不临时构造不对称的 `model_valid`
比较。既有 J1 的 model-valid 数值只能列为历史附录。

### 9.4 Future 因果诊断

对每个 `AIR0-jepa` seed 固定运行：

1. **copy-current**：用当前 spatial latent 代替预测 future；
2. **true-future replacement**：用冻结 encoder 编码的真实 successor latent 进入
   同一个 energy head；
3. **future permutation**：在四个动作之间按固定 permutation 交换 predicted
   future，action token 和 energy readout 不交换；
4. **future zeroing**：将 future field 置为训练 target 均值；
5. **energy-only local evaluation**：分别在 true/predicted future 上计算 tie-aware
   local top-1、regret、margin 与排序一致率。

导航级 true-future/permutation/zeroing 在 `AIR_dev_early210` 上运行；local
diagnostics 在 full-900 每个 maze 固定采样 24 个满足条件的 free states。
这些干预不能用于选择 checkpoint 或修改正式预测路径。

Future collapse 固定定义为满足任一条件：

- prediction token/cell variance 小于对应 target variance 的 10%；
- 四动作 candidate pairwise distance 小于真实 successor pairwise distance 的 10%；
- normalized delta error 不优于 copy-current，且 future permutation 对 local
  top-1 的绝对影响小于 0.01。

“优于 copy-current 30%”固定计算为：

```text
(copy_delta_error - model_delta_error) / copy_delta_error >= 0.30
```

若分母小于 analysis lock 中的数值 epsilon，该样本不进入 ratio，而是单独报告。

### 9.5 Compute accounting

MAC 口径固定为推理路径中的 Conv/Linear/attention multiply-accumulate；归一化与
逐元素非线性不计入。Spatial-JEPA representation 的 Conv MAC 对所有 learned 方法
相同但仍计入总量。L0 performance-blind benchmark 在任何正式训练开始前用 seed-42
的结构计算并签名锁定 `K_compute_match`；L3 用三个 seed 的 final checkpoint 独立复算，
若与 L0 不一致则拒绝 release。

报告必须包含：

- encoder、adapter、Reasoner、future decoder、energy head 参数量；
- size 21/25 下每个 K 的 MACs；
- 每 task 与每成功 episode 的 wall-clock；
- peak GPU memory；
- quality-vs-K 与 quality-vs-MAC 曲线。

在第一次正式训练前，根据静态 MACs 定义：

```text
K_compute_match = 最大的 K_test，使 AIR0 总 MACs <= 1.05 × j1-receding@K128
```

该 K 只用于次级 compute-matched 比较，不能按性能选择。

## 10. 快速读出与完整运行并行设计

### 10.1 原则

- 完整 job DAG 在第一次正式性能读出前生成、hash 并锁定；
- 解锁 quicklook 不会取消或改写任何后续 job；
- formal runner 在 quicklook 结束后自动继续，不等待人工批准；
- 分数不允许触发本阶段早停；
- 只有 technical invalidity 可以暂停队列；
- 初步结果可用于讨论和准备下一阶段代码，但下一阶段协议只能在 AIR-0 final
  closure 后冻结。

L1/L2 不报告确认性 p-value，也不触发样本量、方法、终点或分析变化。最终 planned
comparisons 只在 L3 按锁定规则计算一次。由于中间读出不改变本阶段设计，L3 不做
alpha spending；如果实际运行中发生任何基于分数的适应，原 L3 推断资格立即失效，
必须升级实验版本并在独立数据上重新定义分析。

### 10.2 L0：性能盲的 preflight

预计：服务器启动后 2-4 小时。

执行：

- protocol、manifest、checkpoint、code/runtime hash 审计；
- 签名记录四张同构 H800 和 128-batch paired-stream audit；L1/L2/L3 缺少该证据时拒绝生成；
- 1000-step AIR0-direct/AIR0-jepa smoke training；
- 50 个 train-topology task 的 K128 throughput benchmark；
- 5 次真实 batch8、deep supervision 的 K128 forward/backward，验证最坏训练显存路径；
- 在不读取性能分数的前提下锁定 size-21/25 MAC 与 `K_compute_match`；
- forward/backward、四 action successor、tie labels、future permutation 单元检查；
- ETA 和显存报告。

L0 不打开 AIR_dev，不能产生方法优劣判断。preflight 通过后正式配置不再修改。

### 10.3 L1：最快方向提示

预计：10-18 小时。

条件：seed42 的 AIR0-direct 与 AIR0-jepa final-step checkpoint 均完成，并且
`j1-receding` seed42 baseline 已通过桥接。

读出：

- `AIR_dev_early210`，K16/K128，unmasked；
- seed42 的 overall/seen/OOD SR 与 paired task delta；
- future collapse、copy-current、true-future、permutation diagnostics；
- 训练稳定性和 K scaling 初步信号。

L1 只允许标记：

| 标签 | 预先定义的方向信号 |
|---|---|
| `early_green` | AIR0-jepa SR >= 0.88，vs J1 delta >= -0.05，K128-K16 >= +0.03，且 future 未坍缩 |
| `early_red` | AIR0-jepa SR < 0.75，或 future collapse，或 K128 不优于 K16 且 future permutation 的 local top-1 drop <= 0 |
| `early_yellow` | 其他情况 |

单 seed 不估计训练随机性。即使标为 red，剩余 2 seeds 和完整 full-900 仍继续。

### 10.4 L2：三 seed primary provisional result

预计：24-40 小时。

条件：3 seeds 的以下 K128 full-900 unmasked 全部完成：

- `j1-receding`；
- `AIR0-direct`；
- `AIR0-jepa`。

读出：

- 三 seed overall/seen/OOD SR、SPL；
- 与 J1、direct 的 crossed paired bootstrap；
- per-size 与 shortest-path bins；
- invalid、loop/cycle 与 timeout；
- K128 local ranking 和 future 诊断。

L2 已足够让研究讨论快速转向下一阶段的候选方向，但报告标题必须包含
`PRIMARY_PROVISIONAL`。此时不得：

- 宣布 AIR-0 final pass/fail；
- 解锁 AIR_select 或 AIR_final；
- 取消剩余 K、corrected、干预或审计；
- 在 AIR_dev 上训练修正版并沿用 AIR0-v1 名称。

### 10.5 L3：完整闭环

预计：48-96 小时；若 full-900 全 K 评测吞吐较低，允许延长到 6 天。

必须完成：

- 3 seeds × 3 recurrent methods × 7 K × full-900 unmasked；
- static bridges、oracle 和 K128 corrected；
- 全部 future 干预与 local diagnostics；
- 参数/MAC/runtime 报告；
- manifest/checkpoint/result/task-row 完整性审计；
- 预注册统计与最终红/黄/绿决策。

只有 L3 可以关闭本阶段并冻结下一阶段设计。

## 11. 指标

### 11.1 Primary endpoints

- `AIR0-jepa` K128 full-900 unmasked overall SR；
- paired `AIR0-jepa - j1-receding` overall SR delta。

### 11.2 Key secondary endpoints

- OOD size 23/25 SR；
- seen size 9-21 SR；
- SPL 与 eligible SR；
- paired `AIR0-jepa - AIR0-direct` SR delta；
- K128-K16 的 overall/OOD/long-path delta；
- local top-1、regret、margin；
- invalid、loop/cycle、timeout；
- true-future replacement gap；
- future permutation/zeroing degradation。

### 11.3 Future prediction endpoints

- normalized full-field error；
- normalized delta-field error；
- improvement over copy-current；
- per-action candidate separation；
- predicted/true-future energy rank agreement；
- distance distribution 的 expected-distance MAE/RMSE/Spearman、精确类别准确率与
  top-class ECE；
- target/prediction future-field variance ratio。

Distance 指标统一以 `min(d_BFS, 128)` 为标签：MAE/RMSE 使用 129 类分布的期望，
Spearman 在所有确定性抽样 state-action 对上计算；ECE 固定使用 15 个 `[0,1]`
等宽 confidence bins，置信度为预测分布最大概率，正确性为 argmax 类别是否等于
截断后的 BFS 类别。报告必须同时给出被截断 target 的比例，不能把该 ECE 描述成
任意距离容差下的“近似正确率”。

全图 cosine 只能作为描述性补充，不能单独证明 next-state prediction 成功。

### 11.4 Failure taxonomy 与动作级诊断

episode 的唯一最终类别固定为 `success`、`loop_or_cycle`、
`invalid_action_stall` 或 `step_cap_or_unresolved`。其中 loop/cycle 定义为同一 state
在一个 episode 中至少访问 4 次；invalid stall 定义为 no-move 次数至少为
`max(4, path_length/2)`。未成功 episode 必须恰好耗尽 128 steps。

另外逐 episode 保存以下动作计数，但它们不是互斥的最终失败类别：

- invalid/no-move 与 BFS distance flat action；
- immediate backtrack：移动后回到上一个不同 state；
- distance decrease/increase action；
- dead-end recovery opportunity：当前非 goal cell 只有一个可移动 successor；
- dead-end recovery failure：在上述 opportunity 上选择 no-move，而非唯一出口。

`shortest path >128` 是 structural censoring 分层；`prediction_flip` 和
`energy_wrong_with_true_future` 是 local mechanism label；`compute_insufficient` 只能由
同 checkpoint 的 K 曲线描述。三者不得伪装成逐 episode 互斥失败原因。

## 12. 统计分析

### 12.1 配对与重采样

- 所有方法必须按完全相同的 `task_id` 对齐；
- bootstrap 每次先对 3 个 system seeds 重采样，再在每个 maze size 内对共同的
  task IDs 重采样；
- 同一重采样索引用于所有配对方法和所有 K；
- 20,000 次确定性 bootstrap，bootstrap seed 写入 analysis lock；
- 报告 mean、每 seed 值、paired delta 和 interval；
- 禁止把 2700 个 seed-task rows 当作相互独立后做普通二项检验。

### 12.2 Planned comparison family

四个 planned comparisons 组成一个 family：

1. `AIR0-jepa - j1-receding`，非劣界 `-0.03 SR`；
2. `AIR0-jepa - AIR0-direct`，非劣界 `-0.01 SR`；
3. `AIR0-jepa K128 - K16`，OOD tasks；
4. `AIR0-jepa K128 - K16`，最短路 33-128 tasks。

使用 family-wise alpha 0.05 的 Bonferroni simultaneous percentile intervals。
L2 虽只开放前两个比较，仍预留完整 family size 4，不因中间读出缩小校正。
由于只有 3 个训练 seeds，区间只能作为本阶段的严格探索性证据，不能升级为
跨训练随机性的论文级最终确认。

### 12.3 K scaling

同一 checkpoint 的 K 曲线报告：

- `SR(K128)-SR(K16)`；
- log2(K) 与 SR 的 Spearman；
- OOD 与 shortest-path 33-64/65-128 分层；
- performance per GMAC。

不得因某个非 primary K 的结果更高而把它改称主模型。

## 13. 最终决策门

### 13.1 Green：AIR 主干通过

必须同时满足：

1. `AIR0-jepa` mean overall SR >= 0.90；
2. 每个 seed overall SR >= 0.87；
3. mean OOD SR >= 0.75，且每个 seed OOD SR >= 0.68；
4. `AIR0-jepa - j1-receding` simultaneous CI lower bound > -0.03；
5. `AIR0-jepa - AIR0-direct` simultaneous CI lower bound > -0.01；
6. K128 相对 K16 在 OOD 或 33-128 长路径上的 mean SR delta >= +0.05；
7. future delta error 优于 copy-current 至少 30%；
8. future permutation 使 early210 local top-1 下降至少 0.10，或 SR 下降至少 0.05；
9. 无 collapse、hash mismatch、数据泄漏或 protocol violation。

Green 只能支持：AIR0 shared workspace 是值得继续建设的架构主干。它不等于已经
证明最终优于 J1。

### 13.2 Yellow：只允许一次有边界的核心修正

满足任一典型情况：

- overall SR 在 0.80-0.90；
- point estimate 接近 J1，但 interval 因 3 seeds 过宽未通过；
- K scaling 清楚，future 也被使用，但 energy 转化仍弱；
- AIR0-direct 很强，而 AIR0-jepa 没有额外泛化收益；
- true-future replacement 显著修复 SR，说明 future predictor 是主要瓶颈。

Yellow 后只允许设计一个“核心接口修正”阶段，例如 future target、energy readout、
局部传播或训练稳定性。不能同时加入 memory、halting、层级 planner 和 joint
encoder，使原因再次混杂。

### 13.3 Red：停止 AIR0-v1 实现

满足任一典型情况：

- mean overall SR < 0.80；
- K128 相对 K16 无改善，且 long-path/OOD 也无改善；
- future collapse 或模型对 future permutation 基本不敏感；
- AIR0-jepa 明显差于结构相同的 AIR0-direct；
- true-future replacement 仍不能改善 energy ranking；
- 三 seeds 呈现不稳定的架构级失败。

Red 时不得用增加 memory 或解冻 encoder 掩盖核心失败。下一步应重新设计数据流，
或回到已经验证的 J1/GJVI 主干。

## 14. 结果到下一阶段的映射

| AIR-0 结果特征 | 下一阶段优先方向 | 暂不做 |
|---|---|---|
| 高 SR，主要剩余 loop/backtrack | persistent memory | 改 encoder |
| K 越大越好，但简单任务计算浪费 | adaptive halting | 新 loss 大扫荡 |
| Seen 高、OOD 低，K scaling 存在 | size curriculum、相对位置与传播稳定性 | memory |
| true-future 强、predicted-future 弱 | future dynamics/target redesign | 搜索器堆叠 |
| future 准、energy 排序错误 | cost/Bellman/energy interface | joint encoder |
| direct 与 jepa 一样好 | 强化 future 因果瓶颈与跨任务检验 | 宣称 JEPA reasoning |
| AIR 明确超过 J1 且机制通过 | memory 或 halting 的单因素扩展 | 同时加入全部模块 |

最终方向只能依据 L3 closure，而不是 L1 的单 seed signal。

## 15. 四卡执行顺序与预计时间

四张 H800 必须运行同一 Git commit、container、PyTorch/CUDA/cuDNN 与 deterministic
设置。seed 是 blocking factor，AIR0-direct/AIR0-jepa 在 seed 内配对。

推荐队列：

1. GPU0/1 并行训练 seed42 direct/jepa；GPU2/3 并行训练 seed43 direct/jepa；
2. seed42 完成后立即释放 L1，同时空闲 GPU 启动 seed44 pair；
3. 各 checkpoint 完成后优先运行 K128 full-900；
4. 三 seed K128 完成后释放 L2；
5. 后台继续 K1/4/8/16/32/64、static bridges、corrected 与 future diagnostics；
6. 全部 job 和审计完成后生成 L3。

签名 job plan 保存计划命令、依赖、priority 与 GPU assignment；每个 job 的实际 PID、
开始时间、耗时、返回码、日志路径和输出 hash 写入独立 job-status JSON。若 GPU 完全同构，
调度顺序不会改变最终配对估计；仍需保留这些记录以排查温度、负载或 runtime 漂移。
direct/jepa 到物理 GPU 的映射必须在 seeds 间交叉平衡，不能让某个方法始终固定在
同一张卡上。

预计时间：

- L0：2-4 小时；
- L1：10-18 小时；
- L2：24-40 小时；
- L3：48-96 小时，保守不超过 6 天；
- 预计总消耗：约 80-170 H800 GPU-hours，实测 benchmark 后更新 ETA，不能据此
  删除正式工作。

## 16. Artifact、hash 与审计合同

checkpoint 与 result 共同形成以下证据合同；并非每一种 artifact 都重复保存所有字段：

- experiment/method/config、protocol/package/source 与 code hashes；source analysis-spec
  hash 由 `source_lock.json` 保存并绑定；
- Git commit、dirty flag、code fingerprint；
- runtime、GPU、CUDA、cuDNN、PyTorch；
- train/eval manifest paths 与 SHA256；
- source encoder checkpoint path/hash；
- seed、RNG stream seeds、sample counts；
- optimizer steps、loss weights、K schedule；
- parameter count、MACs、elapsed seconds、peak memory；
- 每 task 的 task_id、size、shortest path、success、steps、failure reason；
- 是否属于 `EARLY_SIGNAL`、`PRIMARY_PROVISIONAL` 或 `FINAL_CLOSURE`。

严格汇总必须拒绝：

- task 缺失、重复或 task hash 不匹配；
- checkpoint/config/code/runtime 混用；
- 脏 worktree 训练或评测；
- 非有限 loss、gradient、logit 或 metric；
- 使用非 final-step checkpoint；
- AIR 方法间 sample/K RNG 不配对；
- 未授权访问 AIR_select/AIR_final；
- 用 corrected 替换 unmasked 主结果；
- formal artifact 被覆盖而无 replacement record。

允许替代运行的唯一原因是客观 technical invalidity。AIR0-v1 不支持选择性局部重跑；
必须按 `REPLACEMENT_PROTOCOL.zh.md` 封存整个失败 attempt、记录旧文件 hash、失败原因
和审批，再从 L0 启动完整新 attempt。结果偏低不允许重跑。

## 17. 明确禁止的临时变化

AIR0-v1 正式启动后禁止：

- 根据 L1/L2 修改 loss 权重、K、hidden size、训练步数或 seed；
- 只给表现较差的方法追加训练；
- 用 best checkpoint 代替 final checkpoint；
- 增加 learned validity、memory、halting、MCTS、CEM 或 subgoal；
- 解冻 encoder 或更换 encoder checkpoint；
- 删除某个表现不佳的 seed；
- 根据 task 难度替换 AIR_dev rows；
- 将 single-seed task bootstrap 写成跨 seed 结论；
- 把 `j1-static` 与 `j1-receding` 合并；
- 把 true-future/corrected diagnostic 写成模型绝对能力。

任何确有必要的科学变化必须创建 `AIR0-v2` 或下一阶段 ID，并使用尚未打开的
独立数据角色；不能覆盖 AIR0-v1。

## 18. 完成定义

AIR-0 只有满足以下条件才算完成：

- [ ] 文档、config、job DAG、manifest 与 analysis lock 已在首个正式 run 前提交；
- [ ] L0 全部协议和数值检查通过；
- [ ] 6 个 AIR final-step checkpoints 齐全；
- [ ] 3 seeds 的 j0/j1 baseline bridge 合法；
- [ ] full-900 K128 primary rows 完整；
- [ ] full-900 七点 K 曲线完整；
- [ ] corrected、future interventions 与 local diagnostics 完整；
- [ ] task/seed crossed paired statistics 已生成；
- [ ] 参数、MAC、runtime 与 failure taxonomy 已生成；
- [ ] 所有文件通过 hash、schema、finite、row-count、overlap 审计；
- [ ] final report 明确列出可得和不可得结论；
- [ ] 按 Green/Yellow/Red 规则给出唯一 closure decision；
- [ ] AIR_select 与 AIR_final 在整个阶段保持未释放。

即使 L1 或 L2 已经显示明显方向，也不能跳过剩余项目。完整运行和最终审计是
快速反应可以被信任的前提。

## 19. 工程师最终交付清单

1. `L0_preflight.json` 与 ETA；
2. `L1_early_signal.md/json`；
3. `L2_primary_provisional.md/json`；
4. `L3_final_closure.md/json`；
5. 所有 checkpoint hashes 与训练曲线；
6. full-900 per-task rows；
7. per-size、path-bin、failure、K 与 compute tables；
8. future prediction、true-future、permutation、copy-current diagnostics；
9. crossed bootstrap 原始 draws 或可复现 seed/spec；
10. protocol audit、replacement ledger 和完整 job status；
11. 一段严格限定的结论：通过了什么、没有证明什么、下一阶段只能优先改什么。

## 20. 可执行实现映射

本节把科学计划映射到代码，避免执行者自行解释：

| 计划元素 | 唯一实现 |
|---|---|
| current + 四 successor 与 BFS labels | `data.py::sample_training_batch` |
| shared recurrent workspace | `models.py::AIRWorkspaceModel` |
| tie action、future、distributional cost | `losses.py::air_loss` |
| progressive K 与 paired streams | `data.py`、`train.py` |
| source tensor identity | `checkpoints.py`、`lock_sources.py` |
| J0/J1 historical parity | `evaluate.py`、`audit_bridges.py` |
| unmasked/corrected/future interventions | `evaluate.py` |
| full-900 local causal diagnosis | `diagnose.py` |
| complete score-independent matrix | `plan_jobs.py` 生成的 143-job DAG |
| non-adaptive four-GPU execution | `run_jobs.py` |
| L1/L2/L3 statistics and gates | `summarize.py` |

正式配置中的关键数字还在 `schemas.py` 中二次硬锁；仅编辑 JSON 不能悄悄改变
hidden size、loss、seed、训练预算、K、bootstrap 或 early subset。代码、文档、manifest
与依赖文件共同进入 `package_lock.json`。工程执行顺序以
`ENGINEER_RUNBOOK.zh.md` 为准。
