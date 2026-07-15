# 结果与决策工件Schema

## Component checkpoint

为了复用已审查的frontier loader，顶层artifact format保留
`vector_jepa_planner_frontier` family和format version；同时写入
`study_experiment_family=vector_jepa_planner_full900_screen`。

必需字段：method、backbone seed、planner seed、analysis spec、training spec、
source checkpoint SHA256、head config/state dict、训练摘要、validation metrics、
quick protocol metadata和code fingerprint。

合法stage顺序：

```text
component_training
-> component_calibration
-> counterexample_training_round 1/2/3（仅Q2C）
```

## Evaluation JSON

每个文件唯一对应：method、backbone seed、planner seed和action protocol。必需字段：

- `metadata`：protocol、method spec、代码/配置哈希和嵌套seed；
- `manifest`：full-900路径、SHA256和count=900；
- `provenance`：source/component checkpoint SHA256和参数量；
- `summary`：overall/seen/OOD/by-size/by-path-bin；
- `tasks`：严格900条，按manifest顺序；
- 每个task包含完整 `decision_traces` 和分开的plan/assist compute；
- `candidate_replay.enabled=false`。

本快筛不执行论文包的10%候选事后重放，因为它会把每次评测近似再跑一次。完整
decision trace、路径、失败类别和compute仍然保留；取消重放不能改变动作或主指标。

## Decision artifacts

- `q1_parent.json`：scorer-compatible父方法及完整排名；
- `shortlist.json`：12候选的system/mechanism配对效应、CI和两条榜单；
- `final_winner.json`：3-backbone方向一致性及唯一胜者；
- `closure.json`：summary/report哈希和永久关闭状态。

所有decision artifact包含 `quick_spec_sha256`，写入后不可覆盖。Q2 effective method
还把Q1 decision SHA256写入自身method spec和training spec。

每个decision还保存其全部输入result SHA256；Q1绑定两个Q0 parity SHA，shortlist
额外绑定Q1 decision SHA，final winner额外绑定shortlist SHA。后续阶段会重新计算
这些哈希，任何结果替换、截断或decision改写都会失败关闭。

## Run schedule

每个Q阶段的CSV记录protocol、quick spec、顺序、job、输出路径和command SHA256。
相同stage再次生成必须逐字一致。

## Final summary

`summary.json`中每个method/action protocol包含：

- 10-backbone mean/SD；
- planner seeds已在backbone内平均的task结果；
- winner-B0和winner-direct-control的overall/seen/OOD配对效应；
- backbone-by-task交叉、maze-size分层的描述性95% bootstrap区间；
- 每backbone效应和正方向backbone数。

`REPORT.md`是同一工件的人类可读摘要，不是独立统计来源。
