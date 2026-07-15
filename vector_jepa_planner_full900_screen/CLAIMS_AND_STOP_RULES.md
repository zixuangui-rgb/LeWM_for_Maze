# 允许结论与停止规则

## 可以得出的结论

Q1通过时，可以说：在同一个冻结Vector-JEPA、同一个latent-L2 scorer、相同任务和
1x predictor预算下，某搜索策略相对历史CEM产生了配对SR增益。

Q2A通过时，可以说：在冻结representation/dynamics和固定父搜索器下，加入指定
planner接口的完整系统优于父控制。若同时通过直接机制控制，可以进一步说增益不能
仅由父搜索器解释。

Vector-DTS通过时，可以说：在相同DTS heads、value/scorer和1x搜索上限下，learned
expansion policy相对uniform expansion提高了开发集SR。search-disabled direct-DTS
只提供描述性锚点，不支持同预算机制结论。

Bidirectional和Denoising通过时，可以说：预先固定的完整算法系统相对其控制提高了
开发集SR；不能把系统差值继续归因于单一内部算子。

Q2C通过时，可以说：自适应hard-negative训练流程相对同轮数的自适应random-negative
流程提高了开发集SR。由于后续轮次的触发语料可以随前轮模型分化，不得声称这是在
完全相同固定语料上仅替换负例标签的纯语义效应。

Q4可以报告10个backbone上的均值、SD、Seen/OOD和交叉配对bootstrap区间，并判断
效应是否跨backbone稳定。该区间必须标注为post-selection描述性区间。

## 不可以得出的结论

- 不得称为新的confirmatory或论文最终测试；
- 不得称为对未知test set的无偏泛化估计；
- 不得把corrected-v1增益写成纯自主能力；
- 不得声称encoder/projector表征得到改善，因为它们完全冻结；
- 不得把task-level样本量900写成训练重复数900；
- 不得把两个planner seed写成两个独立backbone；
- 不得把“最多768 transitions”写成每个方法实际使用完全相同compute；
- 不得把Q2系统增益全部归因于head语义，除非其匹配控制也通过；
- 不得将本实验与spatial-JEPA、GJVI或BC的数字作单因素因果比较；
- 不得声称穷尽所有planner，仅可说覆盖协议预注册的12个主要方法族。

## 强制停止

1. Q0任一action protocol不满足900任务parity：立即停止，修复只能建立新锁。
2. Seed42没有方法通过门槛：shortlist为空，实验关闭。
3. Q3没有方法通过3-backbone门槛：winner为空，实验关闭。
4. Q4完成：无论结果高低，生成closure并永久关闭。

## 明确禁止的追加

- 看分数后增加训练步数、seed、候选数或预算；
- 看分数后调整scorer权重、threshold或proposal比例；
- 把多个Q2A胜者组合后继续筛选；
- 追加0.5x/4x/16x预算曲线；
- 在full-900失败任务上继续训练；
- 删除差seed、重跑低分seed或更换checkpoint；
- 打开本协议声明禁止的confirmatory manifest。

这些方向可能有研究价值，但必须成为新协议，而不是当前实验的隐性延长。
