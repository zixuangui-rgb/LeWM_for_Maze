# 与最初复现的一致性审计

## 1. 审计结论边界

本实验的正式直接对照是锁定后的 `distance_head_study`，不是仓库中最早的临时脚本产物。最初复现代码包含已确认的 checkpoint 选择和 planner/evaluator 问题，因此只能作为来源追溯，不能重新成为正式 comparator。

一致性分为四层：

| 层 | 位置 | 本实验中的作用 |
|---|---|---|
| 历史原始复现 | `scripts/*`、`results/FINAL_REPORT.md` | 证明方法来源和最初参数，不直接提供正式数值对照 |
| 修正后的 Vector-JEPA | `final_closure` | 提供 seqlen2 backbone、固定任务和 corrected planner 语义 |
| 修正后的 DistanceHead | `distance_head_study` | 提供本实验的直接 `b_dh_cem`、`a1_log` comparator |
| 快速处理实验 | `a1_quick_validation` | 仅增加三个候选机制和一个匹配控制 |

机器可读来源锁位于 `configs/reproduction_contract.json`。正式 gateway 每次启动都会验证其中所有历史脚本、manifest、上游配置和协议锁的 SHA256；任一来源变化都会失败。

## 2. 逐项继承关系

| 科学字段 | 最初复现 | 当前正式值 | 状态 |
|---|---|---|---|
| 数据集 | Set B train 2800 / eval 900 | 同一 train/full-900 manifest hash | 精确继承 |
| topology hold-out | train 与 eval 拓扑隔离 | train/D_screen/D_select/full-900 隔离 | 保持并细化用途 |
| JEPA backbone | `lewm_l2_cem_seqlen2` | 同一 final-step seed42 checkpoint | 精确锚定，服务器验 hash |
| latent dim | 256 | 256 | 精确继承 |
| DistanceHead | concat MLP 512/256/128 | `historical_concat` 512/256/128 | 精确继承核心架构 |
| 原始 target | `log_norm` | `legacy_log_norm` | 同一变换的规范化命名 |
| 原始 loss | `smooth_l1` | Huber beta=1 | 同一损失的规范化命名 |
| head steps/batch/pairs | 30000/512/64 | 30000/512/64 | 精确继承 |
| AdamW lr/wd | 1e-3/1e-5 | 1e-3/1e-5 | 精确继承 |
| CEM | H12/C64/E8/I1/M0.1 | H12/C64/E8/I1/M0.1 | 精确继承修正后预算 |
| episode cap | 128 | 128 | 精确继承 |
| action space | moving actions 1-4 | model actions 1-4 | 精确继承 |

三个 reference method 的 resolved object 和 method hash 必须与 `distance_head_study` 完全相同。quick 的 `splits/seeds/planner/training/analysis` 五个配置 section 也必须逐字节语义相等。

## 3. 明确且必要的非同一项

以下差异不是 treatment，也不能被描述成“与最初临时脚本逐行相同”：

1. 最初 DistanceHead 按已经看过的 eval-seen loss 选择 best checkpoint；正式协议固定使用 step 30000，防止 full-900 反向参与模型选择。
2. 最初脚本采用运行时 lazy sampling；正式协议使用带 manifest/backbone/shard hash 的缓存和无状态 sample schedule，以保证所有方法收到可复现的数据流。
3. 最初 head scheduler 是无 warmup 的 cosine-to-zero；正式直接 comparator 继承 `distance_head_study` 的 5% warmup 与 cosine-to-0.1x。它属于上游已锁定修正，所有 quick 方法完全相同。
4. 历史上存在 high-invalid 的旧 planner 结果；正式比较只接受固定 manifest 上的 `corrected_v1` 和 `unmasked` 语义。

因此，当前实验可回答“在修正并锁定的 DistanceHead 实验上，新 treatment 是否改善”，不能回答“是否逐比特复现历史临时 checkpoint”。

## 4. 本轮发现并修正的风险

| 风险 | 原行为 | 修正后行为 |
|---|---|---|
| seed release 过宽 | 继承大研究的 head0/1/2 与 backbone42/43/44 tier | Q1 只释放 42/0；Q2 只释放 42/0,1 |
| bootstrap 未分层 | 在所有 task 上整体重采样 | 每个 replicate 在各 maze-size stratum 内配对重采样 |
| result 仅做通用自检 | 未强制核对当前 method/split/seed/manifest cell | 每个 cell 与 quick lock、resolved method 和完整 row 数逐项核对 |
| diagnostic 校验偏浅 | 只看签名和少数字段 | 核对 method hash、sample count、cache、candidate bank、checkpoint |
| 证据链停在 cache index | shard 可变化而 index 不变 | decision 绑定每个 shard、rebound source index 和 source checkpoint |
| 上游 decision 只绑定顶层文件 | shortlist/winner 的早期 rows 或 diagnostics 变化可能不向后传播 | Q2/Q3 展开 Q1/Q2 的完整输入 hash 闭包，并重新推导 gate、排序和 winner |
| plan 签名只保证自洽 | 删减 cell 后重新签名仍可能结构合法 | worker 现场重建标准 phase plan 并要求完整相等 |
| 恢复路径偏宽松 | 无效已有产物可能被当作可重跑状态 | 已存在但不一致的正式产物立即失败，不静默覆盖或续跑 |
| 生成成功未统一写后复验 | 错误可能延迟到下游才暴露 | cache、candidate、checkpoint、diagnostic、result 在命令返回前重新加载验证 |
| quick 审计沿用上游 held-out 角色 | `legacy full-900` 未被上游新 split 审计显式覆盖 | 同时按 layout hash 和 task hash 检查 train 与全部 held-out、以及 held-out 两两隔离 |
| reachability 结论边界不够明确 | 容易被理解为 planner 已直接消费 reachability logits | 锁定同一 terminal-distance planner，明确本轮只测试辅助监督及共享 trunk 转化 |
| 动态方法来源不显式 | Q2/Q3 配置只列 anchor | profile 明确写入 `q1_shortlist` / `q2_winner` 来源 |

## 5. 统计一致性

所有 SR/SPL delta 先按 exact `task_id` 配对。bootstrap 使用锁定的 10,000 replicate seeds，并在每个 maze size 内有放回重采样；因此每次 replicate 都保持原设计中的各尺寸任务数和权重。

Q1/Q2 的门槛仍是预注册 effect-size gate，不把单 backbone CI 解释成确认性显著性。Q3 即使 full-900 通过也仍是单 backbone 探索性结果。

## 6. 服务器最终门

仓库本地审计不能代替服务器真实产物检查。Q0 必须验证：

- seed42 backbone payload、来源锁和 SHA256；
- source/quick cache index 及每个 shard；
- reference source checkpoint 到 quick rebound checkpoint 的 state tensor hash；
- source/quick candidate actions 逐元素相等；
- CUDA train/diagnose/evaluate 的正式 I/O；
- 四 worker 与 GPU 映射。

Q0 未通过时不得生成 Q1 性能结论。
