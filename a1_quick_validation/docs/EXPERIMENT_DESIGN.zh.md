# A1 DistanceHead 快速验证实验设计

## 1. 文档状态

- 实验 ID：`procgen-maze-a1-quick-validation-v1`
- 证据等级：`exploratory_fast_validation`
- 研究对象：冻结 Vector-JEPA 表征与 dynamics，仅训练 DistanceHead
- 首要目标：快速判断三个机制方向是否值得扩大，而不是完成论文级确认
- 禁止声明：不能用本实验单独证明跨 backbone 稳定性、统计显著的总体优越性或最终泛化上限

本设计在运行前锁定。方法、数据、seed、训练步数、planner 预算、筛选阈值和停止规则都不得根据中间结果修改。

## 2. 背景和出发点

上一轮 DistanceHead 快筛得到的 SR 为：

| 方法 | SR |
|---|---:|
| `b_dh_cem` | 0.671 |
| `a1` / `a1_log` | 0.707 |
| `c1` | 0.695 |
| `b2` | 0.693 |
| `b1` / `b5` / `c2` | 0.690 |
| `a3` / `d4` | 0.681 |
| `d1` | 0.676 |
| `a2` / `b3` / `d3` | 0.674 |
| `d2` | 0.669 |

这些数值说明 `a1_log` 是当前 provisional winner，但不能直接说明 Bellman、predicted-listwise 或 reachability 本身无效。旧方法往往继承不同的父节点或同时混入其他改动，因而机制归因不够干净。本实验把三个方向重新挂到同一个 `a1_log` 父节点上，只改变声明字段。

## 3. 核心问题

### RQ1：真实局部几何

在 A1 上加入 Bellman consistency，能否提高 true-next latent 上的最佳动作排序，并转化为 SR？

### RQ2：预测域对齐

在 A1 上直接监督 predictor 产生的下一状态排序，能否改善 planner 实际看到的 predicted-latent candidate ranking？

### RQ3：剩余步数可达性

在有 horizon 输入的匹配控制之上加入多预算 reachability 辅助监督，能否学出校准、单调的可达性信号，并通过共享 DistanceHead trunk 改善现有 terminal-distance 规划的 SR？

### RQ4：转化瓶颈

若机制诊断改善而 SR 不改善，问题更可能位于 scorer-to-planner 转化；若机制诊断本身不改善，则不能把失败归因于 planner。

## 4. 固定因果结构

### 4.1 参考方法

| 方法 | 作用 | 与旧实验关系 |
|---|---|---|
| `b_l2_cem` | pooled latent L2 + 同一 CEM | 精确参考锚点 |
| `b_dh_cem` | legacy normalization DistanceHead | 精确参考锚点 |
| `a1_log` | `b_dh_cem` 仅将 target 改为 `log1p` | 当前 provisional winner |

这三个方法的 resolved method 对象和 method hash 必须与原 `distance_head_study` 完全一致。

这里的“原实验”特指协议修正后的 `distance_head_study`。更早的临时复现脚本只作为方法来源证据；二者的继承项和必要修复详见 `CONSISTENCY_AUDIT.zh.md`。

### 4.2 新处理和匹配控制

| 方法 | 直接父节点 | 唯一科学变化 | 身份 |
|---|---|---|---|
| `a1_bellman` | `a1_log` | `objectives.bellman: 0 -> 1` | 候选 |
| `a1_predicted` | `a1_log` | `objectives.predicted_listwise: 0 -> 1` | 候选 |
| `a1_hcond` | `a1_log` | `head.horizon_conditioned: false -> true` | 匹配控制 |
| `a1_reach` | `a1_hcond` | 增加 multitask reachability 输出和监督 | 候选 |

`a1_reach` 需要一个 reachability 输出通道和对应损失，这两项共同构成不可拆分的 treatment；`a1_hcond` 专门排除“只是多给了 horizon 输入”的解释。

`a1_reach` 的 planner 仍与 `a1_log` 完全相同，使用 `terminal_distance` 标量；reachability logits 只进入训练辅助损失和 diagnostics，不直接进入 CEM cost。因此本实验检验的是“reachability 辅助监督能否改善现有距离 scorer”，不能据此声称“以 reachability 作为 planner cost”已经被验证。直接消费 reachability logits 属于另一个 planner treatment，不得在本协议中临时加入。

### 4.3 明确不做的事情

- 不改 encoder、embedding projector 或 predictor 的结构；
- 不更新 JEPA backbone 参数；
- 不增加 CEM candidates、iterations 或 horizon；
- 不加入 memory、MCTS、beam search、best-first 等 planner 处理；
- 不测试 multistep、quasimetric、joint training 或新 backbone；
- 不根据训练曲线早停或选择 best checkpoint；
- 不用测试集 BFS 距离选择动作或超参数。

因此，本实验若产生 SR 差异，可归因于 DistanceHead treatment，而不是表示学习或搜索预算变化。

## 5. 数据和拓扑隔离

### 5.1 固定数据角色

| 角色 | 数量 | 尺寸 | 用途 |
|---|---:|---|---|
| train | 2800 | 9–21 奇数 | head 训练 |
| D_cal | 140 | 9–21 奇数，来自训练拓扑 | loss 权重校准 |
| D_screen | 140 | 9–21 奇数 | Q1 快筛 |
| D_select | 210 | 9–21 奇数 | Q2 独立开发复核 |
| legacy full-900 | 900 | 9–25 奇数 | Q3 最终探索性复核 |

沿用原 manifest 的 layout/task hash。train、D_screen、D_select 和 full-900 间不得出现 topology/task 泄漏；D_cal 只能是训练拓扑子集。

### 5.2 为什么 Q1 不直接跑 900

D_screen 不是为了给最终 SR，而是快速淘汰不能改善自身目标机制的处理。完整 900 只留给已经在独立 D_select 上跨两个 head seed 通过门槛的一个 winner。这样减少了无效 CEM 评估，同时不改变任何已运行 cell 的训练和评估定义。

### 5.3 Q3 为什么仍非确认性

Q3 的 winner 已由 Q1/Q2 选出，而且仍只使用历史 backbone 42。full-900 能检验跨尺寸 23/25 的表现和 paired task 差异，但不能替代 fresh multi-backbone confirmation。

## 6. 训练和规划锁

### 6.1 Head 训练

- steps：30,000；
- effective batch：512；
- microbatch：128；
- pairs per topology：64；
- optimizer：AdamW；
- learning rate：`1e-3`；
- weight decay：`1e-5`；
- gradient clip：1.0；
- deterministic：true；
- checkpoint：只能使用 final step；
- backbone：seed 42，完全冻结；
- head seeds：Q1 为 0，Q2 为 0/1。

所有方法使用相同 sample schedule、candidate bank、训练步数和初始化 seed 生成规则。不得因为某个方法收敛较慢而追加 steps。

### 6.2 Planner

- receding CEM，每个环境步重新规划；
- horizon：12；
- candidates：64；
- elites：8；
- CEM iterations：1；
- momentum：0.1；
- max episode steps：128；
- rollout semantics：`legacy_warmup_v1`；
- corrected reference transitions：768。

Q1 只跑 `corrected_v1`；Q2/Q3 同时跑 `corrected_v1` 和 `unmasked`。后者是安全性/协议敏感性终点，不参与临时改规则。

## 7. 参考产物复用

为了不重复训练 `b_dh_cem` 和 `a1_log`，允许复用原实验 final-step checkpoint，但不是无条件复制。

### 7.1 缓存重绑定

缓存 shard 张量不修改、不复制。新 index 只有在以下条件全部成立时才能指向旧 shard：

1. 原 index 通过原协议锁验证；
2. 每个 shard 文件仍存在且 SHA256 未变；
3. 新旧 manifest 路径和 hash 一致；
4. 新旧 backbone 路径和 hash 一致；
5. record 顺序、task hash 和 shard hash 一致；
6. 新 index 写入原 index、原协议锁和“未复制张量”的来源记录。

若原 cache 不存在，可用 quick config 原生重建；原生 cache 同样接受完整 binding 和 shard hash 验证。

### 7.2 参考 checkpoint 重绑定

只有以下条件全部成立，参考 head 参数才可重绑定到 quick lock：

1. 新旧 resolved method 和 method hash 完全一致；
2. source checkpoint 是 formal final-step 30,000；
3. source protocol/analysis lock 正确；
4. backbone hash、seed 和 head seed 正确；
5. train/cal cache 的 task hash 与 tensor hash 序列一致；
6. source/quick candidate action tensors 逐元素相等；
7. head state tensor hash 在写入前后相同；
8. quick training-spec hash根据新绑定重新计算；
9. checkpoint 写入完整 source path/hash 和“未重训”声明。

任一检查失败，都必须重建相应输入或重训，不能加 bypass flag。

## 8. 分阶段流程

## Q0：协议与输入预检

Q0 不看性能。它完成：

1. 验证原协议锁、quick 内层锁、package 外层锁；
2. 验证方法 resolved diff；
3. 验证 manifests 和 topology hold-out；
4. 重绑定或验证 train/cal/screen/select cache；
5. 生成与 quick lock 绑定的 candidate bank；
6. 释放 seed1；
7. 导入 `b_dh_cem`、`a1_log` 的 head0 reference checkpoint。

Q0 任一失败，禁止运行 Q1。

quick seed release 必须严格等于 profile：seed1 只包含 backbone42/head0；seed3 只包含 backbone42/head0,1。不得继承上游大实验中更宽的 head/backbone seed 池。

## Q1：D_screen 单 seed 机制快筛

矩阵：backbone 42、head 0、D_screen 140、`corrected_v1`，七个方法全部评估并诊断。

### Bellman 晋级门

相对 `a1_log`：

- SR delta 不低于 -0.01；并且
- true-latent local top-1 提升至少 0.03，或 true-latent regret 相对下降至少 15%。

### Predicted-listwise 晋级门

相对 `a1_log`：

- SR delta 不低于 -0.01；并且
- predicted-latent local top-1 提升至少 0.03；并且
- predicted-latent regret 相对下降至少 15%。

### Reachability 晋级门

必须同时满足：

- macro AUROC ≥ 0.65；
- macro Brier ≤ 0.25；
- macro ECE10 ≤ 0.15；
- monotonic violation ≤ 0.05；
- SR 相对 `a1_log` 不低于 -0.01；
- SR 相对 `a1_hcond` 不低于 -0.01。

`a1_hcond` 永远不作为新机制晋级。通过者先按 SR delta 排序，再按预注册方法顺序打破完全相同的 tie，最多晋级两个新方法。若无方法通过，实验停止，不打开 D_select。

## Q2：D_select 双 head seed 复核

矩阵：backbone 42、head 0/1、D_select 210、`corrected_v1 + unmasked`。方法为 `b_dh_cem`、`a1_log` 和 Q1 最多两个候选。

候选必须同时满足：

- 两个 head seed 的 corrected SR delta 相对 A1 的平均值 ≥ 0.02；
- 每个 head seed 的 corrected SR delta 均 ≥ 0；
- corrected SPL 平均 delta ≥ -0.02；
- unmasked SR 平均 delta ≥ -0.02；
- 两个 head seed 均保持对应的机制诊断通过。

通过者按 corrected SR 平均 delta、unmasked SR delta、预注册方法顺序排序，只锁一个 winner。若没有 winner，实验停止，不跑 full-900。

## Q3：legacy full-900 最终探索性闭环

winner 在查看 full-900 前锁定。矩阵为 backbone 42、head 0、900 tasks、两种动作协议，对比：

- `b_l2_cem`；
- `b_dh_cem`；
- `a1_log`；
- Q2 winner。

成功门：

- corrected overall SR delta 相对 A1 ≥ 0.02；
- corrected SPL delta ≥ -0.02；
- unmasked SR delta ≥ -0.02；
- seen 和 OOD SR delta 均不为负。

无论通过与否，Q3 后实验关闭。若通过，结论只能是“在 backbone42 上得到可复核的 full-900 提升信号”；若失败，结论是“该机制没有越过预注册的单-backbone转化门”。

## 9. 指标和统计

### 9.1 主终点

- Q1/Q2：paired task corrected SR delta；
- Q3：paired task corrected overall SR delta。

### 9.2 次终点

- SPL；
- unmasked SR；
- seen/OOD SR；
- loop/cycle rate、failure mode；
- true/predicted local top-1、regret、margin；
- candidate order Spearman/regret；
- absolute BFS distance MAE/RMSE/Spearman；
- reachability AUROC/Brier/ECE/单调违例；
- closed-loop drift。

### 9.3 配对原则

所有 delta 必须通过 `task_id` 一一对齐后计算。任务集合、manifest hash 或 row 数不一致时直接失败，不允许用两个独立均值相减代替 paired comparison。

### 9.4 不确定性

每个 paired SR/SPL delta 使用原协议固定的 10,000 个 bootstrap replicate seeds，并在每个 maze size 内分别进行有放回重采样，报告 95% percentile CI。每个 replicate 保持各尺寸原有任务数，因此不会让尺寸构成的抽样波动改变预设权重。快速晋级仍按预注册 effect-size 门槛，不把单次 CI 当作论文级显著性证明。

## 10. 防泄漏和防选择偏差

- Q1 只能读 D_screen；
- Q2 只能在 Q1 shortlist 锁定后读 D_select；
- Q3 只能在 Q2 winner 锁定后读 full-900；
- Q2 不得添加 Q1 未晋级的方法；
- Q3 不得根据 full-900 换 winner；
- OOD size 23/25 不参与 Q1/Q2 选择；
- 测试 BFS 只能进入 diagnostics，不进入 planner score；
- 所有 decision 都绑定输入文件 hash、protocol lock 和 package lock；
- 已存在的 immutable decision 不得覆盖。

## 11. 停止规则

1. Q0 失败：修复输入或环境，不能跳过。
2. Q1 无候选通过：停止，报告三个机制均未形成最低可转化信号。
3. Q2 无候选通过：停止，报告 Q1 信号未跨 head seed 和独立 split 复现。
4. Q3 无论正负：实验结束，不在本包内继续加方法。

任何后续新想法都必须进入另一个新协议，而不是修改本实验。

## 12. 可支持的结论

### 情形 A：机制和 SR 同时改善

可以说该 head treatment 在固定 Vector-JEPA 和固定 planner 下产生了可转化信号，并值得进入 fresh multi-backbone confirmation。

### 情形 B：机制改善，SR 不改善

可以说 head 的目标能力已经改善，但现有 terminal-distance scorer/CEM 没有把它转化为成功率；这是另立协议研究 planner/scorer coupling 的依据。对 `a1_reach` 尤其不能把该结果表述成“reachability planner 已失败”，因为本实验没有让 planner 直接消费 reachability logits。

### 情形 C：机制不改善

不能把失败归咎于 planner。应认为该 auxiliary objective 在当前冻结 embedding 上没有学出预期结构。

### 情形 D：Q1 改善，Q2 失败

最可能是单 seed 或小 split 波动，不能继续扩大或组合。

## 13. 资源和预期时长

四张 H800 并行时：

- Q0：以 I/O/hash 校验为主；
- Q1：四个新 head 的完整 30k 训练，加七个 140-task CEM；
- Q2：最多两个 head1 训练，加最多四方法 × 两 seed × 两协议的 210-task 评估；
- Q3：仅一个 winner 晋级时运行 4 方法 × 2 协议 × 900 tasks。

实际时长取决于原 cache/checkpoint 是否齐全和单 episode CEM 吞吐。设计目标是约 24–48 小时内得到主要方向结论；若 Q1/Q2 失败会更早停止。

## 14. 最终报告最少内容

1. package/protocol lock hash；
2. 所有运行 checkpoint、cache、manifest hash；
3. Q1 全方法 paired 表和机制表；
4. Q1 shortlist；
5. Q2 每个 head seed、协议的 paired rows 汇总；
6. Q2 winner 或停止原因；
7. 若运行 Q3，报告 seen/OOD/by-size、SPL、unmasked 和 failure modes；
8. 明确写出单-backbone探索性边界；
9. 保留所有原始 rows、diagnostics、logs、plans 和 completion seals。
