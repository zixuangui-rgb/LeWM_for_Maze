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

Vector-DTS head同时给出root policy、distance-bin value和uncertainty，由这些量决定
树节点扩展。训练100,000 steps、batch 64、LR 1e-4。直接控制加载同类head，但
使用direct expansion，隔离“学会扩展”而不是“多了参数”。

### `q2b_bidirectional`

从当前latent和goal latent两端生成候选，并由StateJoin判断两端是否可连接。使用
与forward-only控制相同的verifier、join、memory设置；两者主要差异为是否双向。

### `q2b_denoising_icem`

把动作序列视为离散token，从mask/noisy sequence迭代去噪。候选由25% uniform和
75% denoising proposal组成。训练100,000 steps、batch 256、LR 1e-4。直接控制
为完全相同预算的uniform-iCEM。

## Q2C：Hard-negative ranker

先训练30,000 steps的trajectory ranker，再按task hash把训练topology固定分成三
fold。第r轮只在fold r上运行当前planner，收集以下false-optimistic候选：撞墙、
2-8步cycle、或存在进步候选但选中序列没有BFS进步。用BFS最优chunk和失败chunk
做pairwise ranking，每轮更新20,000 steps。

Random-negative控制使用完全相同的fold、触发轨迹、轮数和更新步数，但负例是随机
动作chunk。两者都禁止读取900个screen任务。

## Calibration

所有head在独立700-task validation上运行32个固定随机batch。校准只决定join
threshold并记录verifier accuracy、reachability calibration、proposal teacher-
forced accuracy和DTS root-action accuracy；不得根据这些数值换模型、追加训练或
改变权重。

## 不进行的方法组合

本协议不运行 Reachability+Verifier、Proposal+Memory 或“全模块”系统。单方法
筛选结束后也不重新打开组合搜索；Q4唯一任务是补齐胜者的重复，不是继续发明配置。
