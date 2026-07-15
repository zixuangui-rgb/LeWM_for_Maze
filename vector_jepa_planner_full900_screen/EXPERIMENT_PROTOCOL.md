# Full-900 Vector-JEPA Planner筛选实验协议

协议ID：`vector-jepa-planner-full900-screen-v1`

## 1. 研究目的

本实验回答：在不改变历史Vector-JEPA的encoder、projector、predictor结构和参数
的条件下，仅改变planner搜索、scorer、proposal、memory或planner head，能否在
Procgen Maze的历史full-900上稳定提高导航成功率？

本实验不是超参数优化，也不是论文最终确认。它的产物是：

- 对12个主要方法族做覆盖完整尺寸分布的统一筛选；
- 最多保留两个候选做3-backbone复核；
- 最多保留一个候选补齐与旧报告一致的10-backbone结果；
- 无论结果方向如何，Q4后永久关闭本实验族。

## 2. 研究对象与不可变边界

共同backbone为 `lewm_l2_cem_seqlen2`：

| 项目 | 锁定值 |
|---|---|
| 架构 | `unisize256_sizecond_cnn_projector_transformer` |
| latent维度 | 256 |
| JEPA训练步数 | 30,000 |
| batch size | 256 |
| sequence length | 2 |
| prediction/SIGReg/position loss | 与final_closure完全相同 |
| checkpoint选择 | final step |
| backbone seeds | 42-51 |
| planner history | 3 |
| rollout semantics | `legacy_warmup_v1` |
| horizon | 12 |
| max environment steps | 128 |
| action IDs | 1、2、3、4 |
| replanning | 每个真实环境步重新规划 |

禁止改变encoder/projector/predictor结构，禁止更新其参数，禁止加入spatial latent、
真实地图、坐标、在线BFS、真实墙体特征或oracle join。

BFS只可用于训练标签、离线反例标注和事后分析，不可在测试时选择动作。

## 3. 数据角色

| 数据 | 数量 | 用途 | 是否可影响梯度/选择 |
|---|---:|---|---|
| `unisize_train_manifest` | 2800 | head训练、反例挖掘 | 只影响训练 |
| `vector_jepa_frontier_validation` | 700 | calibration | 不做候选选择 |
| `unisize_eval_manifest` | 900 | 本实验所有方法评测与选择 | 可以，明确标记development |
| `vector_jepa_frontier_confirmatory` | 900 | 本实验禁止打开 | 不可以 |

full-900包含尺寸9、11、13、15、17、19、21、23、25，每个尺寸100任务；seen为
9-21共700任务，OOD为23/25共200任务。四个manifest必须task hash、topology hash
两两无交集。

历史full-900已经被观察，因此任何结果都不得称为fresh confirmation。选择完成后
若要形成论文确认性结论，必须另建协议和未观察holdout。

## 4. Action protocol

每个方法、checkpoint和任务必须运行两次：

- `corrected_v1`：允许历史协议中的真实合法移动、立即回退过滤和冻结one-step
  latent-L2 fallback；
- `unmasked`：planner动作原样执行，不访问墙体有效性。

二者使用同一checkpoint、task顺序和evaluation seed。两个结果分开排名，不允许
平均或混成一个分数。`corrected_v1`是父方法和最终胜者的主选择指标，`unmasked`
是必须满足非退化约束的自主能力诊断。

## 5. 计算预算

`1x`严格定义为每次环境决策最多768次predictor latent transitions，对应旧基线
`64 candidates x horizon 12 x 1 iteration`。所有搜索方法共享该硬上限。

Corrected-v1 fallback产生的5个one-step transitions单独记入 `B_assist`，不计入
`B_plan`。proposal、verifier、reachability、join、ranker和DTS forward calls分别
记账。主公平口径是predictor transition budget；参数量、head calls和wall time作为
二级代价报告。

## 6. 实验单位与随机性层级

- 最高层独立重复：backbone training seed。
- planner-head seed嵌套在backbone内。
- task是同一checkpoint下的配对测量，不是独立训练重复。
- 搜索随机性使用历史 `task_seed(evaluation_seed, task_index, step)`，evaluation
  seed固定为42；不新增search-seed伪重复。

种子递进：

| 层级 | Backbone | Planner head | 目的 |
|---|---|---|---|
| Screen | 42 | 104729 | 所有候选full-900 |
| Expansion | 42-44 | 104729 | 最多两个候选复核 |
| Final | 42-51 | 104729、130363 | 唯一胜者最终对齐 |

无head方法没有planner seed，不得复制结果制造伪重复。最终统计先在每个backbone
内部平均两个planner-head seed，再把10个backbone作为独立重复。

## 7. 阶段设计

### Q0：B0实现一致性

seed42上分别用旧 `LeWMCEMController` 和新 `LegacyCEMPlanner` 运行完整900任务，
corrected/unmasked均运行。逐任务比较task ID、成功、路径长度、invalid、loop、最终
BFS distance等字段，并逐步比较实际执行动作序列。任一差异即停止。
两个parity工件的SHA256写入Q1 decision；以后任何阶段都会重新验证，不能绕过Q0
直接冻结父方法。

### Q1：只改变搜索

共同条件为latent-L2、uniform proposal、无memory、无新head、1x预算。候选为
iCEM、Beam、Best-first和MCTS。额外运行instrumented categorical-CEM bridge；
bridge必须与历史B0逐任务一致，但它允许后续scorer/proposal接入。

Q1父方法在seed42 corrected full-900按以下顺序确定：overall SR、OOD SR、unmasked
SR、较少planner serial calls、方法名。即使没有方法通过晋级门槛，也冻结得分最高
的scorer-compatible父方法供Q2使用；父方法选择和候选晋级是两个不同决策。

### Q2A：独立planner接口

Reachability、Verifier、AR Proposal各自在Q1父方法上独立加入。Memory只在
Best-first上测试，因为现有其他搜索器不消费transposition table；其直接控制为
同一个无memory Best-first。四项互不累计。

### Q2B：激进替代

- Vector-DTS vs direct-DTS expansion；
- Bidirectional vs 共享verifier/join语义的forward-only Best-first；
- Denoising-iCEM vs uniform-iCEM。

每一对共享除目标机制外的backbone、训练数据、预算和评测任务。

### Q2C：反例排序

Q1父方法上训练ranker，然后只在2800个训练topology中完成三轮false-optimism
挖掘和更新。匹配控制执行相同轮数、相同fold、相同步数，但负例替换为随机动作
序列。full-900结果绝不进入训练。

### Q3与Q4

Q1/Q2最多晋级两个候选，在backbones43/44补齐full-900。通过3-backbone门槛后
只冻结一个胜者。Q4为胜者、B0和胜者的直接控制补齐backbones42-51；有head的
方法再补第二planner seed。

## 8. Seed42晋级规则

分别建立corrected和unmasked榜单。某候选在一个榜单晋级必须同时满足：

1. 相对B0 overall SR增量至少0.03；
2. maze-size分层配对task bootstrap的Bonferroni同时区间下界大于0；
3. 另一action protocol的SR下降不超过0.03；
4. 对应protocol的OOD SR下降不超过0.03；
5. Q2候选相对直接控制的SR增量至少0.02，且Bonferroni同时区间下界大于0。

同时区间的比较家族在运行前固定为48项：12个可晋级候选乘2个action protocols，
再乘2类对照（B0系统效应、直接控制机制效应）。family-wise alpha为0.05，因此每项
bootstrap区间使用alpha=`0.05/48`。即使Q1不以机制效应作为门槛，也保留在同一比较
家族中，不根据结果缩小分母。

corrected榜第一和unmasked榜第一进入shortlist；若为同一方法则只保留一个。最多
两个方法。

## 9. 三backbone最终门槛

候选在一个protocol上通过必须满足：

- backbones42-44平均SR增量至少0.03；
- 至少2/3 backbone方向为正；
- 另一protocol平均下降不超过0.03；
- OOD平均下降不超过0.03；
- Q2候选相对直接控制平均增量至少0.02。

优先选择通过corrected门槛且平均增量最大的候选；若无，则选择通过unmasked门槛
且平均增量最大的候选。完全并列按方法名固定排序，不允许人工裁决。

## 10. 统计与报告

主终点为SR；SPL、Seen/OOD SR、逐尺寸SR、invalid、loop/cycle、assistance rate、
compute为二级结果。Q4报告的是经过选择后的描述性、逐点95%区间，不是确认性
p值。区间使用交叉配对bootstrap：有放回采样backbone，同时在每个maze size内
对共同task panel做一次配对采样；同一draw抽中的task索引在所有抽中backbone间
保持一致。planner seeds先在backbone内平均。该设计对应“同一批任务被所有
backbone重复测量”，不得误写成每个backbone拥有独立task的嵌套设计。

低分、异常方向或不符合预期都不是重跑理由。允许的基础设施重跑必须保留原失败
文件、原因和哈希，且不得在看到分数后改变方法、步数、种子或门槛。
