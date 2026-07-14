# 实现审计与验收说明

本文记录可执行代码如何覆盖研究协议，以及哪些结论仍依赖服务器正式运行。
它不是结果报告，不包含任何 validation 或 confirmatory 性能数字。

## 1. 审计结论

截至交接版本，本包完成以下静态与单元级审计：

- 旧 `hdwm.planning.cem_plan` 未修改；B0 adapter、cost、sequence 和真实 maze
  首动作有逐值 parity 测试。
- 65 个 JSON templates 确定性展开为 118 个有效方法；P2/P3/P4/P7/P8 矩阵
  完整性由 schema 和 `audit_protocol.py` 双重检查。
- P2、P5、P7、P8 决策均不可覆盖，并能从哈希锁定的 validation 工件重算。
- P5 组装验证 parent provenance、head ownership 和 tensor equality。
- P6 hard/random 控制初始和三轮训练预算相同，random 只替换 negative actions；
  三轮 dataset/checkpoint/fold/parent 链在确认冻结时逐项验证。
- `action_aligned_v2` 使当前 candidate action 影响对应 successor；legacy B0
  semantics 保持原样，二者有首动作影响回归测试。
- Track J 使用 T=8 trajectory、30k final-step、完整 world-model state、独立
  组件随机流和 P6 三轮 hard negatives；54-cell 选择验证全部 40 checkpoints。
- candidate diagnostics 通过 deterministic replay 事后计算，不向 planner 注入
  oracle candidate；false optimism 只有存在真正 progress candidate 时才成立。
- Amendment 001 在任何正式结果打开前修正 unique-state/two-cycle 的诊断分母；
  两项均由纯函数断言和回归测试保证位于 `[0,1]`，且不改变 planner action。
- 确认性 CI 是 backbone -> planner -> size-stratified task 的 paired nested
  bootstrap；p-value 是 backbone-level exact sign flip 并做 Holm；CI 做
  Bonferroni；factorial main effect 为 high-minus-low，interaction 为 DiD。
- confirmation freeze 对实际训练 steps、module limits、schedule、source hashes、
  retrieval bank、P5/P6/P7 provenance 做最终 fail-closed 验证。

## 2. 本地验证基线

交接前必须重新执行：

```bash
uv sync --frozen --extra dev
uv run ruff format --check vector_jepa_planner_frontier tests/test_vector_jepa_planner_frontier.py
uv run ruff check vector_jepa_planner_frontier tests/test_vector_jepa_planner_frontier.py
uv run python -m compileall -q vector_jepa_planner_frontier
uv run pytest -q tests/test_vector_jepa_planner_frontier.py
uv run pytest -q
uv run python -m vector_jepa_planner_frontier.lock_protocol --check
uv run python -m vector_jepa_planner_frontier.smoke_test
git diff --check
git diff --exit-code -- hdwm/planning.py
```

记录基线为本包 `65 passed`、全仓 `114 passed`。全仓 `ruff check .` 仍会命中
旧 `diagnostics/`、旧训练脚本等约 2326 个历史问题；本次没有为追求表面全绿
而重写不相关复现代码。新增包和新增测试的 Ruff/format 均必须为零问题。

## 3. 服务器必须补做的动态验证

本仓库不携带正式 GPU checkpoint，因此代码测试不能代替以下验收：

1. seed 42 旧 evaluator 与新 B0 的完整 task/action/ledger parity；
2. seed 52-61 真实补训及 source schema、parameter shape、loss 配置对齐；
3. 每类 head 的 CUDA backward、strict reload、calibration 与 retrieval fingerprint；
4. 全 20x2 P5 assembly、P6 三轮 chain、54x20x2 Track J checkpoint provenance；
5. CUDA deterministic algorithms 在实际算子上无静默降级；
6. 每阶段冻结 schedule 的完整性、partial-output 拒绝和 candidate replay equality；
7. confirmatory 只按 opaque schedule 一次运行，完整后统一解盲。

## 4. 已知研究边界

- Corrected-v1 使用真实当前状态合法动作和旧 one-step fallback，必须与 unmasked
  分账；它是允许的 assistance，不是纯自主 JEPA 能力。
- P5 gate 的科学解释由 reviewer 负责。代码能防字段遗漏和手改选择，不能证明
  reviewer 的机制证据本身充分，因此要求第二位 reviewer 逐表签核。
- P7 网格很大，但每个 cell 使用同一训练预算。它探索的是该预注册四维区域，
  不是所有可能 joint-training recipe 的全局最优。
- 20 backbone checkpoints 是最高层独立重复；planner seeds 是嵌套重复，tasks
  是配对测量。任何把 900 tasks 当作 900 个独立模型样本的分析都无效。
- 通过本地审计只证明实现满足协议，不预示 SR 会提高。负结果、gate failure 和
  Track J fail-closed 都是可报告的科学结果。

## 5. 变更控制

正式 validation 开始后，任何代码、config、manifest、lock、阈值、seed、steps、
candidate family 或 action protocol 变化都必须停止当前 run。确认集打开后只允许
完成已冻结 opaque schedule；不能以修复性能为由替换 checkpoint 或追加实验。
