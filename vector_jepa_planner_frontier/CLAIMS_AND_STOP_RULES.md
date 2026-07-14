# 结论边界、晋级与停止规则

## 1. 确认性问题

确认集只检验 P8 冻结后的方法，不重新选择模型。

### H1：Track F 相对 B0

- overall Corrected-v1 SR；
- size `23/25` OOD Corrected-v1 SR。

### H2：Track J 相对 Track F

只有 Track J 通过 JEPA stability 和 validation noninferiority gate 时才存在：

- overall Corrected-v1 SR；
- size `23/25` OOD Corrected-v1 SR。

因此主 family 为 K=2 或 K=4。每个 contrast 的 CI 使用 `alpha=0.05/K` 的同时 percentile bootstrap；p-value 在 backbone 层做 exact two-sided sign-flip，并用 Holm 控制 familywise error。Minimum effect of interest 是 SR `0.05`。

`unmasked`、SPL、loop、dead-end recovery、candidate mechanism、compute 和结构分层是关键次要分析，但不是替换失败主终点的备用显著性入口。

## 2. 晋级规则

### P2

只在 1x/4x 的 Corrected-v1 validation 中选搜索 backbone。overall SR 优先；
差值小于 `0.01` 时看 size-19/21；仍接近时选 predictor serial calls 更少者。
赢家映射到该搜索器的 4x cell，并由 P3 全矩阵继承。该选择改变搜索器但不
改变 P3 的训练数据、因子定义、rollout semantics 或预算。

### P3/P4 -> P5

Verifier、reachability、proposal、memory 分别通过以下六项时才进入 P5：机制
改善、overall 非劣、size-19/21 非劣、等计算、负对照、方向一致。P5 采用全部
通过者，未通过的组件不进入。P4 三个 radical 也逐一使用同样 gate；最多选择
一个，多个通过时使用预注册的 overall SR、size-19/21 SR、名称 tie-break。
组件和 radical 均无通过者时关闭当前实验。Reviewer 只能填写带表格/行号的
证据判断；程序会验证证据完整性和选择规则，但不会替代科学判断本身。

### P6

无论表现如何只运行 M1/M2/M3 三轮。hard-negative 方法必须优于 random-negative 对照的对应机制指标，才能把收益归因于 planner-specific false optimism。

### P7

Track J 先完整运行 54-cell 网格。一个 cell 只有在全部 40 个 final-step
checkpoint 的匹配 JEPA validation objective 相对 source 恶化不超过 10% 时
才有资格参选；稳定 cell 内按“SR 距最大值不超过 0.01，再最小化最大/平均
JEPA 恶化，再看 size-19/21”冻结唯一赢家。若无稳定 cell，Track J 分支关闭。
此外，P8 选择后的 Track J SR 不得比 Track F 低超过 `0.01`。任一条件失败时
H2 删除，H1 仍可按 K=2 进入确认。

### P8

先在共同 4x 下锁定 P5/P6 family，再选距离该 family 最高 SR 不超过 `0.01` 的最小预算。Track J 独立使用相同预算规则。不得因为 16x 分数看起来更好就忽略预注册近优最小预算规则。

### 功效

若 validation pilot 得到的 `required_backbones > 20`，当前资源不足以支持确认性主张。必须在打开确认集前标记 `exploratory_only` 并停止；不能先看 900 个 confirmatory tasks 再决定是否降级。

## 3. 允许的结论

证据支持时可以写：

- 在冻结旧 encoder/projector/predictor 参数、相同 Corrected-v1 和给定 planner budget 下，某 Track F planner 相对 B0 改善 overall/OOD SR；
- 在 unmasked 配对结果中，planner 自身而非 current-state assistance 保留了多少收益；
- candidate coverage、selection regret、false optimism、rollout drift、future feasibility 或 join 中哪一项是主要可恢复损失；
- distributional directed reachability 是否比 terminal latent-L2 改善候选排序和规划；
- memory 是否降低 loop/revisit 并提高 dead-end recovery；
- 三轮 planner-specific hard negatives 是否相对 random negatives 有增益；
- P8 中性能是否随实际 predictor transitions 饱和，最小近优预算是多少；
- 预注册 54-cell Track J 网格中是否存在满足 JEPA stability 的近优配置，以及
  其完整 package 相对 Track F 是否有额外确认性收益；
- 收益在 seen-size topology hold-out 和 size `23/25` OOD 上是否一致。

## 4. P7 的特殊归因边界

P6 与 Track J 同时存在 rollout semantics 和参数更新差异。只有借助 validation 的 `p7_control_action_aligned_frozen` 才能拆分：

- P6 -> frozen aligned control：action alignment；
- frozen aligned control -> Track J：同语义下的联合参数更新；
- P6 -> Track J：完整 package 总效应。

确认性 H2 比较的是完整 package，不单独证明 joint training。除非 frozen aligned control 也获得独立确认性设计，否则论文不得把 H2 的全部增益归因给参数联合训练。

## 5. 禁止的结论

- 把 Corrected-v1 的合法动作过滤称为纯自主 JEPA 能力；
- 把 Track J 写成 frozen representation 或 planner-only；
- 把 action alignment、搜索、head 训练的混合差异归因给单一模块；
- 把 oracle SR、BFS online query、真实 cell decoding 放进正式方法主表；
- 把 spatial latent、坐标、wall map 或真实状态 equality 混称 pooled-vector JEPA；
- 用 task/time-step 作为独立重复，或复制 B0 形成 planner-seed 重复；
- 挑最好 seed/checkpoint、删失败 run、追加 round 4、只延长表现好的方法；
- 用 validation 的 size-19/21 代称真正的 OOD `23/25`；
- 在打开 confirmatory 后改变方法、budget、seed、K、阈值或报告口径；
- 看到主检验失败后改用 unmasked/SPL/某个 size 做新的“主要结论”。

## 6. 永久关闭条件

满足以下工件后，本实验族无论正负都关闭：

1. P1 oracle attribution；
2. P2 等预算搜索前沿；
3. P3 全 16 cells、主效应与二阶交互；
4. P4 matched controls；
5. P5/P6/P7 继承和机制对照；
6. P8 四档 compute frontier 与冻结选择；
7. paired corrected/unmasked、candidate diagnostics、结构分层和完整 compute ledger；
8. 功效允许时的一次性 opaque confirmation、全家族解盲与 K2/K4 主检验。

若 stage gate 使后续确认无法进行，该 gate 本身也是预注册负结论，当前分支关闭。不能为了得到正结果无限增加 planner、数据、head 或预算；新想法必须使用新 protocol ID 和未打开的数据。
