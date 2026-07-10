# Alignment and Inference Protocol v2

本文档是本轮实验的比较合同。代码、配置或报告只要违反一项 primary rule，
对应结果就只能标为 diagnostic，不能进入确认性主表。

## 1. 三个数据角色

| Role | Manifest | 数量 | Sizes | 用途 |
|---|---|---:|---|---|
| Train | `unisize_train_manifest.jsonl` | 2800 | 9-21，每 size 400 | 所有 final checkpoint |
| Development | `unisize_eval_manifest.jsonl` | 900 | 9-25，每 size 100 | 已反复使用，只能调试和选方案 |
| Confirmatory | `spatial_jepa_confirm_eval_manifest.jsonl` | 900 | 9-25，每 size 100 | 新 topology，一次性最终检验 |

完整 SHA256 位于 `configs/protocol_lock.json`。旧 Set-B eval900 已参与此前的
诊断和方法设计，因此不再被称为“未触碰测试集”。新确认集由
`generate_confirmatory_manifest.py` 确定性生成；正式运行前必须提交到 Git，
之后不得根据其 learned-model 结果修改模型、loss、K、seed 或判定规则。

`audit_protocol.py` 会重建全部 4600 个任务，并用与旧 manifest hash 无关的
canonical geometry hash 验证：

- 每个 split 内 topology/layout/task 唯一；
- train/development/confirmatory 两两 topology、layout、task overlap 为 0；
- 确认集与生成器逐条完全一致；
- manifest 文件 SHA256、size 计数、goal、wall、start 和 BFS distance 正确。

## 2. Primary 与辅助动作协议

Primary absolute-ability protocol：

| 字段 | 固定值 |
|---|---|
| Split | confirmatory 全部 900 个固定任务 |
| `max_steps` | 128 |
| Actions | `[UP, DOWN, LEFT, RIGHT] = [1,2,3,4]` |
| Action selection | `unmasked`，完全采用模型排序 |
| Learned field | 每个 task 起点只计算一次全图 field |
| Primary K | config 中预注册 |
| Eval sampling seed | 所有 checkpoint 固定为 42 |

必须对同一 checkpoint 额外运行：

- `model_valid`：使用模型自己的 valid head；
- `corrected`：使用真实墙体移除 no-move action，并尽量避免 immediate
  backtracking。

后两者用于量化外部帮助，不是绝对能力。报告必须同时展示三者，且不得用
`corrected` 替代 `unmasked`。`--recompute-every-step` 只用于 agent-marker
sensitivity，不进入确认性汇总。

decoded-map BFS 只根据预测 wall/agent/goal 选动作，不调用真实地图 correction。
它衡量 representation sufficiency，不等同于“JEPA 自己学会规划”。

## 3. Step-cap 与分层报告

确认集有 19 个任务的 shortest path 大于 128：

```text
SR@128 理论上限 = 881 / 900 = 0.9788888889
```

exact BFS 和 oracle VI K=256 必须达到该值，eligible SR 必须为 1.0。正式报告
同时给出：

- 全 900 `SR@128`、SPL；
- `optimal_length <= 128` 的 eligible SR；
- seen 9-21 与 size-OOD 23/25；
- per-size；
- shortest-path bins `1-16/17-32/33-64/65-128/129+`；
- invalid action 与 loop/cycle rate；其中 loop/cycle 固定定义为任一 state 在单局中
  被访问至少 4 次。

`129+` 是协议造成的结构性删失，不能解释为普通模型规划错误。

## 4. 训练一致性

所有 planner variants 固定：

- 30000 optimizer steps、map batch 8；
- hidden dim 64、recall 开启；
- AdamW、LR `1e-3`、weight decay 0、cosine schedule；
- gradient clip 1、distance scale 128；
- train/development/confirmatory manifests；
- training seeds 42-51，共 10 个。

四条 NumPy 随机流相互独立：maze entries、map states、JEPA trajectories、K
schedule。一个 ablation 多抽一次 K 或 trajectory 不会改变下一步训练样本。

正式 CLI 默认拒绝：

- 脏 worktree；
- 非确定性训练；
- 未锁定 training spec/analysis spec；
- 覆盖已有 checkpoint/result；
- NaN/Inf loss、gradient 或 evaluator output；
- 不同源码 fingerprint、checkpoint hash、runtime 或 source representation 的混跑。

每个 checkpoint 保存训练参数、manifest hashes、training/analysis spec、训练时的
Git commit/dirty 状态、代码 fingerprint、完整 runtime、源 representation SHA256、
训练样本记账和 inference Conv2d MACs。严格汇总不仅拒绝脏评估结果，也拒绝由脏
worktree 训练出的 checkpoint。

## 5. K 与计算量

Progressive recurrent 模型：

```text
K_train = {4, 8, 16, 32, 64, 128}
K_test  = {4, 8, 16, 32, 64, 128, 256}
```

Primary K：feedforward 4、fixed recurrent 64、progressive recurrent 128。不得从
confirmatory K curve 选择最大 SR。K=4 提供同一 recurrent checkpoint 的低计算
参考；报告必须分别给 planner、Spatial-JEPA encoder + planning projector 和总计的
size-25 参数量/GMACs。R4 K128 对 R2D 的差异是“完整 iterative system + 更多迭代
计算”的效果，不能单独归因于 weight sharing 或 recurrence。

## 6. 三条确认性假设

Family-wise alpha 为 0.05，三条假设使用 Bonferroni simultaneous CI，即每条
双侧 CI 的 alpha 为 `0.05 / 3`。seed 和 task 是交叉重复测量：bootstrap 每次
重采样 training seeds，并对所有 seed 共用同一批重采样 task IDs。

### H1：迭代系统的实际增益

`R4 progressive K128 - R2D dilated depth4`。

只有 simultaneous CI 的 lower bound `>= +0.03 SR` 才写“支持”。这验证完整
迭代系统具有至少 3 个百分点的可靠增益，不隔离额外计算量。

### H2：Spatial-JEPA 非劣于 raw input

`J1 frozen Spatial-JEPA - R4 raw`，相同 iterative planner family 与 primary K。

非劣界值为 `-0.03 SR`。只有 simultaneous CI lower bound `> -0.03` 才支持
“Spatial-JEPA 在该协议下保留了足够规划信息”。不能据此声称优于原始 LeWM，
因为本矩阵没有同 evaluator 的 compressed-LeWM 对照。

### H3：staged adaptation 的增益

`J2 last-block + map preservation - J1 frozen`。

只有 simultaneous CI lower bound `>= +0.03 SR` 才支持“受约束的 staged
adaptation 带来至少 3 个百分点增益”。该 estimand 是完整 adaptation package，
不单独归因于解冻或 map loss。

其余 R0-R3、J0、J3 比较均为 exploratory，不得改写成确认性成功。

## 7. Local metric 对齐

为延续旧 diagnostics，local top-1 固定：eval entry 使用 `seed+101` 顺序；先
消耗旧 pair-sampling RNG draws；每 maze 用 `seed+202` 不放回抽 24 个 free
states；只统计至少两个 valid actions 的非 goal states；并列 shortest actions
全部算正确；先按 maze 算，再做 macro average。

`all_cell_local_top1` 另存，但不能与旧约 0.588 的 sampled Local top-1 混用。

## 8. 可得与不可得的结论

本实验可以回答：

- topology hold-out 和 size 23/25 上的端到端绝对能力；
- 增加 recurrent test-time compute 是否带来稳定收益；
- Spatial-JEPA 是否在给定 planner 下达到预注册非劣标准；
- frozen、staged、joint integration 的差异和梯度冲突。

本实验不能单独回答：

- 是否超过旧 BC、latent-L2 或原始 LeWM；它们未在新确认集上同协议重跑；
- 是否具备纹理、颜色、背景、跨任务或跨环境泛化；
- recurrence 是否在等 FLOPs 下优于所有 feedforward architecture；
- corrected SR 是否代表模型无需外部帮助的能力；
- 单个 seed、最佳 K 或 development 最优值是否成立。

任何超出以上边界的文字必须标成后续假设，而不是本轮结论。
