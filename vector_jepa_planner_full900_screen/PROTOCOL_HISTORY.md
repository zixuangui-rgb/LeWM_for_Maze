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

## v1.1 pre-run strict audit (2026-07-15)

本次修订发生在任何正式run、checkpoint、schedule或decision产生之前，因此保留同一
协议ID，并在 `amendments.jsonl` 中留下可审计记录：

- 为Vector-DTS增加同预算uniform-expansion MCTS正式控制；Direct-DTS降为不参与
  晋级的描述性诊断；
- Bidirectional每端候选从64锁为48，给拼接路径预留192 predictor transitions；
- seed42的Bonferroni极端尾部改为精确枚举经验bootstrap分布，消除有限Monte Carlo
  分位数误差；
- 控制差异审计从顶层字段收紧到叶子字段；四个manifest增加内部重复和完整尺寸计数；
- Q1 categorical-CEM bridge改为复用Q0完整任务字段集合并继续逐动作核对；
- Q0 parity增加规范路径、当前source checkpoint SHA、manifest顺序和reference summary
  回算；
- DTS与Bidirectional匹配组增加共享head逐张量、非计时训练摘要和校准指标exact
  parity gate，并在递进seed阶段重验；
- result加载增加源backbone与组件checkpoint的当前路径/SHA回查，阻止权重替换后
  沿用旧评测；
- 反例断点恢复增加逐record训练split、fold、动作、来源和mining预算校验；
- 最终报告补齐逐尺寸、SD和compute，并支持closure前中断的同内容恢复；
- `uv.lock`与`pyproject.toml`一并纳入新协议锁和代码指纹；协议锁只允许在无正式
  工件时通过显式`--replace-before-run`更新。

v1.1重新锁定后，任何涉及方法、门槛、seed、数据或预算的修订都必须新建protocol
ID，不得修改本工件。
