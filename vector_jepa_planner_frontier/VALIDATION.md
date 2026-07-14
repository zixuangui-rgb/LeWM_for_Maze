# 验证记录与服务器验收标准

## 1. 本地验证范围

本地验证用于证明代码、协议和静态因果对照一致，不用于证明模型性能。最终提交执行：

```bash
uv sync --frozen --extra dev
uv run ruff format --check vector_jepa_planner_frontier tests/test_vector_jepa_planner_frontier.py
uv run ruff check vector_jepa_planner_frontier tests/test_vector_jepa_planner_frontier.py
uv run python -m compileall -q vector_jepa_planner_frontier
uv run pytest -q
uv run python -m vector_jepa_planner_frontier.lock_protocol --check
uv run python -m vector_jepa_planner_frontier.smoke_test
git diff --check
git diff --exit-code -- hdwm/planning.py
```

当前测试基线为：新增包定向测试 `65 passed`，全仓 `114 passed`。如代码再
修改，必须重新运行并更新该数字；不能只跑新增测试文件。全仓旧目录目前有
历史 Ruff debt，因此验收命令把 lint 范围锁定为本包和本包测试；这不是对旧
代码质量的声明，完整基线见 `IMPLEMENTATION_AUDIT.md`。

## 2. 协议审计覆盖

`audit_protocol.py` fail closed 检查：

- protocol/config/environment/amendment/code fingerprint 可重建；
- train `2800`、development `900`、validation `700`、confirmatory `900`；
- validation/confirmatory manifest 逐字节确定性；
- 四 split 的 topology/layout/task hash 两两隔离；
- 旧 source config/lock/train-manifest hash；
- P2 完整 5×4、P3 完整 `2^4`、P4 10 methods、P7 aligned control 和完整 `2x3x3x3` Track J grid、P8 3×4 frontier；
- 65 个 checked-in templates 展开为 118 个有效方法、66 个 confirmatory candidate-pool 方法；
- Corrected-v1 主终点、final-step checkpoint rule；
- 七层 oracle 与全 manifest inverse-action 关系；
- train topology 对 horizon-12 action chunk 的 eligibility。

## 3. 关键回归测试

### 旧 B0 一致性

1. Legacy schema 拒绝 horizon/candidate/elite/iteration/momentum/budget drift。
2. 新 adapter 与旧 `_latent_rollout_cost` 数值逐值相同。
3. 新 B0 与旧 `cem_plan` 的 sequence 和 cost 相同。
4. 新旧 controller 在真实 manifest maze 上首动作相同。
5. `legacy_warmup_v1` 与 `action_aligned_v2` 的差异被显式测试。

### Planner 与 compute

1. categorical CEM、iCEM、beam、best-first、MCTS、DTS、bidirectional 返回合法 shape。
2. 每个 planner 不超过 hard transition cap；assistance 不污染 planning ledger。
3. MCTS depth termination、leaf backup 和 simulation cap 有回归测试。
4. B0 只有 planner sentinel；learned methods 才有两个 planner seeds。
5. Block schedule 不会离开某 backbone 后再次回到该 block。

### 因果矩阵与继承

1. P3 恰有 16 factorial cells 和完整 factor codes。
2. P4 的 radical treatments 与 non-oracle controls 集合精确匹配。
3. P5 动态组件/radical 选择、逐 tensor assembly ownership、P5->P6->P7 parent chain、trainable component 集合和 round-3 owner 被验证。
4. P7 aligned control 与 P6 checkpoint path 完全相同，component jobs 为 0。
5. P7 恰有 54 个 joint cells；每个 cell 的四维超参数、T=8、30k budget、组件随机流隔离和 hard-negative provenance 被验证。
6. P8 每个 family 恰有 `0.5/1/4/16x`，alias 只复用 source checkpoint，component jobs 为 0。
7. P8 近优最小预算、Track F family、Track J fail-closed 和 hard-budget metric 聚合规则有单元测试。

### Oracle 与候选诊断

1. Oracle 名称精确覆盖 O0-O6；O6 生成的每个 future action 都真实移动。
2. P1 schedule 完整为 560 jobs 且 backbone-blocked。
3. Candidate sample 使用 size-stratified exact bottom-hash，而非独立 Bernoulli 近似。
4. Formal/replay action sequence 必须一致，候选 truth 事后计算且 compute excluded。
5. Coverage、regret、false optimism、diversity、ESS 和 rank correlation 的有限值/字段完整性被验证。

### 训练与统计

1. Reachability CDF 单调，非 bin budget 使用上界 bin。
2. Ranker batch 使用全部 matched roots，pairwise loss 保持梯度；P6 random 对照只替换 negative actions，P7 只消费三轮 hard negatives。
3. M1/M2/M3 fold deterministic、互斥且覆盖 train topology。
4. Hard memory 在 precision `<0.95` 时 fail closed。
5. 嵌套 task records 保留 decision count 与 compute，先平均 search；bootstrap 实际重采样 backbone、backbone 内 planner seed 和 size-stratified task。
6. Revisit/unique-state/two-cycle 指标的分母、边界和 Amendment 001 定义有确定性测试。
7. CSV header 唯一、样本数字段无歧义、factorial high-minus-low/DiD、backbone exact sign-flip 和功效计算有确定性测试。
8. Confirmatory primary family、opaque mapping、完整解盲前置条件由 schema/gate 检查。

## 4. Dry-run 结构预期

- P1：1 audit + 560 oracle jobs；
- P2：1 audit + 1600 paired evaluation jobs；
- P7：54 Track J train/calibrate families + 1 frozen aligned control；Track J 每个 family 为 20 backbones × 2 planner seeds，精确 job 数以冻结 schedule 为准；
- P8：Track J 通过 P7 时 1 audit + 1440 paired evaluation jobs；Track J 失败时只调度 960 个 Track F paired evaluation jobs，均为 0 train/calibrate/mining jobs；
- Confirmatory 在 P8 冻结前必须拒绝生成 schedule；P8 后只允许 K2 的 240 或 K4 的 400 jobs。不得调度全部 66 个候选方法形成伪“ceiling”。

实际 stage 中 train/retrieval/calibrate/mining job 数由 active heads 和 parent chain 决定，必须以冻结 CSV schedule 为准。

## 5. 本地无法替代的服务器验证

本仓库不包含正式 GPU checkpoints 和完整运行输出，因此工程师必须在服务器完成：

1. seed 42 的旧 `final_closure.evaluate` 与新 B0 task-level parity，包括 proposed/executed actions、SR、SPL、ledger；
2. seed 52 的 source extension 训练与旧 seed checkpoint schema/参数 shape 对齐；
3. 每类 learned head 的短诊断 backward、state-dict strict reload、calibration 和 evaluator build；
4. P6 round0->round1->round2->round3 的 parent hash 与只更新 ranker 参数审计；
5. P7 全部 54 cell 的 world-model gradients、full state reload、T=8 trajectory protocol、P6 三轮 hard-negative consumption 和 matched JEPA stability 重算；
6. P7/P8 reuse 方法的 checkpoint SHA256 相同、训练任务确实为 0；
7. CUDA deterministic algorithms 对所有实际算子的支持；失败时记录算子和新 protocol amendment，不得静默关闭；
8. full validation/confirmatory 后重新运行 protocol audit、summary completeness 和全仓 pytest。

## 6. 服务器验收失败条件

出现以下任一项，不能进入论文主结果：checkpoint 缺失或 hash 不符、dirty formal run、manifest overlap、非有限值、candidate replay 不一致、部分输出、P5/P8 gate 不可重现、Track J stability 不一致、required backbones 超过可用数、确认 mapping 泄露、提前生成具名 confirmatory 结果，或 summary 有 missing files。

“代码测试通过”只说明工程实现满足预注册不变量。只有完整服务器矩阵、一次性确认、全家族解盲和 backbone-level 统计通过，才能形成论文级结论。
