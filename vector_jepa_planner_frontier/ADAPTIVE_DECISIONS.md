# 动态决策与派生方法规范

本实验预注册的是“规则 + 候选集合”，不是提前假定赢家。所有 validation
决策必须在 confirmatory manifest 打开前冻结，且后续阶段从原始工件重算。

## 1. 模板与有效方法

`configs/default.json` 登记 65 个模板。加载配置时，schema 把唯一的 Track J
模板展开为完整 `2x3x3x3=54` 网格，得到 118 个有效方法：

| Stage | 有效方法数 | 是否依赖前序决策 |
|---|---:|---|
| P2 | 20 | 否 |
| P3 | 21 | 是，继承 P2 planner |
| P4 | 10 | 否 |
| P5 | 1 | 是，继承 P2 和 P5 advancement |
| P6 | 2 | 是，继承 effective P5 |
| P7 | 55 | 是，继承 effective P5/P6 |
| P8 | 9 | 是，继承 P7 winner；Track J 可关闭 |

`effective_method_sha256` 对完整派生 MethodConfig 计算，正式 checkpoint/result
同时保存所有相关 decision SHA256。模板同名不意味着不同运行可共享工件；
只有 effective spec、source hashes 和随机性身份均一致才可复用。

## 2. P2：搜索器冻结

候选是五种搜索器的 1x/4x cells。顺序规则为 corrected validation overall SR、
`0.01` 内的 size-19/21 SR、`0.01` 内更少 predictor serial calls、方法名字典序。
赢家映射到同搜索器的 4x spec，P3 的 21 个方法全部继承它。P3 其他字段不变。

冻结文件：`vector_jepa_planner_frontier_runs/decisions/p2_selection.json`。

## 3. P5：可解释组件与 radical 晋级

具名 reviewer 对四个 factorial 组件和三个 radical 分别填写六项 gate，并给出
表格/行号证据。组件只有六项全真才进入 `selected_components`；该集合精确
映射到一个 P3 `v?r?p?m?` cell。

radical 最多选择一个。只在六项全真的 radical 中，依次按 corrected overall
SR 距最大值 `<=0.01`、size-19/21 SR 距剩余最大值 `<=0.01`、名称字典序选择。
若无 radical 通过则为 `null`。若组件与 radical 都为空，实验关闭。

P5 不优化参数。组装顺序固定为 selected P3 source 在前、可选 radical 在后；
同名 head 归 P3，radical 只补齐缺失的 required head。每个 parent、head owner
和 tensor equality 都进入冻结验证。这一优先级防止把同一 head 的两个版本
事后拼成没有被单独评估过的混合体。

冻结文件：`vector_jepa_planner_frontier_runs/decisions/p5_advancement.json`。

## 4. P7：联合训练网格冻结

54 个 cell 的四维网格固定为：

```text
planner LR             2 levels: 1e-4, 3e-4
backbone LR multiplier 3 levels: 0.01, 0.03, 0.1
planner loss weight    3 levels: 0.1, 0.3, 1.0
SIGReg multiplier      3 levels: 0.5, 1.0, 2.0
```

所有 cell 共享 P6 hard round-3 parent、30k final steps、T=8 trajectory protocol、
训练/验证 size schedule 和 P6 三轮 hard-negative datasets。每个 cell 有
20 backbone x 2 planner seeds = 40 checkpoints。

先删除任何有一个 checkpoint 超过 10% matched JEPA objective 恶化的 cell。
剩余 cell 中，保留 corrected validation SR 距稳定最大值 `<=0.01` 的候选，
再依次最小化最大 JEPA 恶化、平均 JEPA 恶化，偏好更高 size-19/21 SR，最后
按名称确定。无稳定 cell 时写入失败记录，不人工替补。

冻结文件：`vector_jepa_planner_frontier_runs/decisions/p7_selection.json`。

## 5. P8：family、预算和确认集 family

Track F 先在共同 4x 预算比较 effective P5 与 P6；SR 距最大值 `<=0.01` 时选
实际 plan transitions 更少者。随后在赢家的 `0.5/1/4/16x` 四点中选择距离
最高 SR `<=0.01` 的最小 multiplier。P7 有赢家时，对 Track J 同样选择预算。

Track J 只有在 40-checkpoint stability 再验证通过，且所选预算 SR 不比 Track F
低超过 `0.01` 时进入确认集。否则确认 family 为 B0 + Track F，K=2；通过时
为 B0 + Track F + Track J，K=4。K 表示主 contrasts 数，不是方法数。

冻结文件：`vector_jepa_planner_frontier_runs/decisions/p8_selection.json`。

## 6. 人工判断与机器判断边界

唯一人工科学判断点是 P5 evidence 对六项 gate 的解释。机器负责验证 reviewer、
证据引用、布尔类型、通过集合、radical tie-break、summary hashes 和确认集未打开。
P2、P7、P8 的数值选择完全由代码重算。任何更改阈值、候选集合、tie-break 或
人工 evidence 标准都需要新 protocol/amendment，不能覆盖既有 decision JSON。
