# Protocol History

## v1

- 取消180-task子集，所有候选使用历史full-900。
- 复用历史backbones42-51，不改变representation/dynamics。
- 将方法覆盖限制为12个科学上不同的方法族和必要匹配控制。
- 取消多预算、组合、20-backbone、双search-seed和joint-training网格。
- 增加Q0旧新controller full-900 parity。
- 增加1->3->10 backbone递进和最终嵌套planner seed统计。
- Q0 parity提升为完整executed-action序列一致。
- seed42门槛固定48项Bonferroni比较家族。
- 最终描述性区间按共同task panel的backbone-by-task交叉bootstrap计算。
- decision工件绑定全部输入result和上游decision SHA256。
- 明确本实验为development screening，不是fresh confirmation。

锁定后没有协议修订。任何未来修订必须新建protocol ID，不修改v1工件。
