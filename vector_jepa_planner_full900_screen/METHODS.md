# 方法与训练定义

## 共同Vector-JEPA接口

所有方法只接收：当前及历史观测经过冻结encoder/projector得到的pooled vector、
goal vector、动作历史和真实执行动作历史。预测未来只能调用冻结action-conditioned
predictor。测试时禁止访问坐标、地图、BFS distance或spatial feature map。

所有新head的输入均为一个或多个256维pooled vectors和动作序列，因此仍属于本项目
约定的严格Vector-JEPA方法。

## B0与Q1

### `b0_legacy_l2_cem`

逐字复用历史 `hdwm.planning.cem_plan`：64条长度12候选、8 elites、1次CEM、
momentum 0.1，terminal squared latent-L2打分。该方法不接受新scorer和proposal。

### `q1_control_categorical_cem_1x`

使用可插拔planner接口重新实现同一categorical CEM。seed42必须与B0逐任务相同。
它不是科学候选，只是当B0最优时，为Q2提供能真正消费新scorer/proposal的等价父
方法。

### `q1_icem_1x`

在总768 transitions内复用elite并在receding-horizon步骤间warm-start候选分布。
其唯一实验变量为候选更新和复用策略。

### `q1_beam_1x`

按latent scorer保留固定宽度的前缀，逐层展开到深度12，达到预算立即停止。

### `q1_best_first_1x`

以latent scorer作为priority，从得分最优的未展开节点继续搜索。无memory版本是
Memory实验的直接父控制。

### `q1_mcts_1x`

在单向pooled-vector latent树上执行selection/expansion/evaluation。没有真实状态
join、地图或oracle rollout；达到768 predictor transitions即停止。

## Q2A：单接口方法

### `q2a_reachability`

训练Distributional Reachability head，预测当前latent到goal latent的BFS距离落在
`1/2/4/8/16/32/64/128`阈值内的概率。训练标签来自训练maze的BFS，测试时只调用
head。它与latent-L2共同打分，不替换backbone。

训练：30,000 steps、batch 512、AdamW、LR 3e-4、weight decay 1e-4、5% warmup、
cosine衰减到0.1倍、grad clip 1.0、final-step checkpoint。

### `q2a_verifier`

训练Action Consistency Verifier，根据source/successor latent判断哪一个动作产生该
转移。规划时作为候选一致性惩罚，权重0.3。训练配置与Reachability相同。

### `q2a_autoregressive_proposal`

训练长度12的自回归动作proposal，标签为训练maze上的BFS最优动作chunk。搜索候选
由50% uniform和50% learned proposal组成，防止proposal退化时完全失去覆盖。

训练：60,000 steps、batch 256、AdamW、LR 1e-4，其余schedule与共同配置一致。

### `q2a_transposition_memory`

训练StateJoin head判断imagined latent是否对应同一状态。Best-first维护
transposition table，对疑似重复节点施加soft priority penalty，不做hard prune。
join threshold只在700-task validation上校准。直接控制为无memory Best-first。

训练：30,000 steps、batch 512、LR 3e-4。Memory不接到其他Q1搜索器，因为这些
实现没有读取transposition table；那样的实验变量不会真实进入决策。

## Q2B：激进替代

### `q2b_vector_dts`

Vector-DTS head给出root policy和distance-bin value；同一个系统还使用distributional
reachability scorer。训练100,000 steps、batch 64、LR 1e-4。正式晋级对照
`q2b_control_dts_uniform_expansion`加载相同种类的head、value和scorer，也运行同一
768-transition MCTS，只把learned policy prior换成uniform prior。因此晋级比较隔离
的是learned expansion policy，而不是“有无搜索”或“有无参数”。比较前必须通过
checkpoint exact-parity gate：共享head逐张量相同，训练损失（排除耗时）、校准指标、
源backbone与数据哈希相同。Direct-DTS也进入这项参数一致性审计。

`q2b_control_dts_direct`保留为search-disabled直接动作头诊断。它不消耗同等predictor
预算，所以只报告原始结果，不作为晋级门槛中的机制对照。

### `q2b_bidirectional`

从当前latent和goal latent两端生成候选，并由StateJoin判断两端是否可连接。使用
与forward-only控制相同的verifier、join、memory设置。每端固定48条长度6候选，
先消耗576 transitions；剩余192 transitions最多重排16条拼接路径，避免两端采样
耗尽全部1x预算。该对照比较的是完整双向搜索系统与完整forward-only系统，不能把
差值进一步拆成某一个内部算子的纯效应。两边共享head同样必须通过逐张量exact-parity
gate；实际每decision compute仍按结果单独报告，不宣称完全相等。

### `q2b_denoising_icem`

把动作序列视为离散token，从mask/noisy sequence迭代去噪。候选由25% uniform和
75% denoising proposal组成；该比例作用于iCEM每轮预留的proposal注入槽，而不是
全部64条候选。训练100,000 steps、batch 256、LR 1e-4。直接控制为完全相同预算的
uniform-iCEM。

## Q2C：Hard-negative ranker

先训练30,000 steps的trajectory ranker，再按task hash把训练topology固定分成三
fold。第r轮只在fold r上运行当前planner，收集以下false-optimistic候选：撞墙、
2-8步cycle、或存在进步候选但选中序列没有BFS进步。用BFS最优chunk和失败chunk
做pairwise ranking，每轮更新20,000 steps。

Random-negative控制使用相同的固定fold规则、随机种子层级、轮数和更新步数，但
负例是随机动作chunk。每种训练规则都用自己当前的ranker继续挖掘，因此第2/3轮的
实际触发样本可以因前轮处理而分化；这里检验的是“自适应hard-negative流程”相对
“自适应random-negative流程”的算法级效应，不是固定同一语料上只替换负例标签的
纯语义消融。两者都禁止读取900个screen任务。

## Calibration

所有head在独立700-task validation上运行32个固定随机batch。校准只决定join
threshold并记录verifier accuracy、reachability calibration、proposal teacher-
forced accuracy和DTS root-action accuracy；不得根据这些数值换模型、追加训练或
改变权重。

## 不进行的方法组合

本协议不运行 Reachability+Verifier、Proposal+Memory 或“全模块”系统。单方法
筛选结束后也不重新打开组合搜索；Q4唯一任务是补齐胜者的重复，不是继续发明配置。
