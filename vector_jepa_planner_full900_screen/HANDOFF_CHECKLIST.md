# 工程师交接清单

## 接收时

- [ ] 当前branch/commit与交接记录一致。
- [ ] `lock_protocol --check`通过。
- [ ] `audit_protocol --require-checkpoints`通过。
- [ ] 10个LeWM checkpoint均存在且不是复制seed。
- [ ] 四个manifest hash和零overlap通过。
- [ ] 工作树受监控路径clean。

## 每个阶段

- [ ] 先运行`--dry-run`并保存schedule。
- [ ] stage CSV未发生变化。
- [ ] 使用固定action protocols和全部900任务。
- [ ] 未手工修改method、seed、steps、weights或budget。
- [ ] 中断恢复只使用`--resume-missing`。
- [ ] 所有低分和失败seed均保留。

## 决策点

- [ ] Q0两个parity均为pass，且`executed_actions_compared=true`。
- [ ] Q1 parent由脚本冻结，不人工选择。
- [ ] Q2A/Q2B/Q2C全部完成后才冻结shortlist。
- [ ] Shortlist最多两个，corrected/unmasked不混分。
- [ ] Q3完成后由脚本冻结零或一个winner。
- [ ] Q4只运行winner、B0和直接控制。

## 交付时

- [ ] `summary.json`与`REPORT.md`同时存在。
- [ ] closure SHA256与summary/report一致。
- [ ] 10-backbone结果齐全。
- [ ] 有head方法的两个planner seeds齐全。
- [ ] planner seeds在backbone内平均，没有按20个独立run报告。
- [ ] 最终区间按共同task panel的crossed bootstrap计算，不称为嵌套独立task。
- [ ] Seen/OOD、corrected/unmasked和assistance分别报告。
- [ ] 最终文字明确development/exploratory状态。
- [ ] 没有在结果后追加配置或重新开启组合搜索。
