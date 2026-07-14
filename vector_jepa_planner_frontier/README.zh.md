# Vector-JEPA Planner Frontier

本目录是 Procgen Maze 上“固定 pooled-vector JEPA 接口，探索 planner 性能上限与尺寸泛化边界”的完整实验包。工程师应把本目录、仓库原有 `hdwm/`、`final_closure/`、训练 manifest 和旧 checkpoint 视为一个不可拆分的复现单元。

## 研究问题

在不改变 `Unisize256` encoder、embedding projector 和 predictor **结构**的前提下：

1. 更强的搜索、距离/可达性表征、候选 proposal、记忆和反例训练能否提高 Maze SR？
2. 增益来自候选覆盖、局部排序、rollout drift、回溯/去环，还是 Corrected-v1 assistance？
3. 增益能否从训练尺寸 `9-21` 泛化到未见 topology，并外推到确认集中的 OOD 尺寸 `23/25`？
4. 固定表示的 Track F 达到什么上限；允许联合更新参数的 Track J 是否还有额外收益？

## 不可变边界

- B0 是 `b0_legacy_l2_cem`，直接调用旧 `hdwm.planning.cem_plan`；历史 `legacy_warmup_v1` 行为保留，不做静默修复。
- 主终点是 `corrected_v1` 的 confirmatory overall SR 和 size `23/25` OOD SR；`unmasked` 是严格配对的自主能力诊断。
- Corrected-v1 只使用当前真实状态的合法移动、上一真实位置和旧 one-step latent-L2 fallback。其收益必须单列为 assistance，不能写成 JEPA 自身能力。
- `B_plan`、`B_assist`、`B_total` 分账；候选诊断的事后 BFS 计算不计入 planner compute，也不能影响动作。
- Oracle ladder 只做定位，输出位于独立目录并带 `not_for_primary_table=true`。
- Confirmatory manifest 在 P8 选择、功效冻结和 checkpoint 哈希冻结前不可打开。

## 锁定矩阵

默认 JSON 登记了 65 个方法模板。配置 schema 会在读取时把唯一的 P7
Track J 模板展开为完整 `2x3x3x3=54` 网格，因此机器实际审计和执行的是
**118 个有效方法配置**。模板数和有效方法数用途不同，报告时不得混用：

| 阶段 | 数量 | 内容 |
|---|---:|---|
| P2 | 20 | 5 种搜索器 × `0.5x/1x/4x/16x` |
| P3 | 21 | 完整 `2^4` 因子设计 + 5 个机制负对照 |
| P4 | 10 | Vector-DTS、双向搜索、离散去噪 proposal 及匹配对照 |
| P5 | 1 | 由冻结的 P2/P5 决策导出的 Track F 组合 |
| P6 | 2 | 三轮 hard-negative ranker + random-negative 对照 |
| P7 | 55 | 冻结 P6 的 action-aligned 对照 + 54-cell Track J 网格 |
| P8 | 9 | P5/P6/P7 各增加 `0.5x/1x/16x` 预算 alias；`4x` 复用原结果 |

P3 的搜索器继承 P2 冻结的赢家；P5 只组合通过六项机制 gate 的 P3
组件，并最多采用一个通过 gate 的 P4 radical；P7 先从 54 个 Track J
cell 中冻结一个稳定且近优的赢家；P8 再冻结 Track F family、最小近优预算
和 Track J 是否进入确认集。每个派生方法都把决策工件 SHA256 写入自身
effective method spec，后续阶段会从原始结果重新计算并拒绝手改 winner。

随机性层级固定为 20 个 backbone seeds、每个 learned planner 2 个 planner-head seeds、每个 task 2 个 search seeds。B0 没有 planner-head 参数，只运行唯一 `planner_seed=0`，禁止复制制造伪重复。

## 数据角色

- Train：原 `2800` 个 size `9-21` topology，只用于梯度、retrieval 和反例采矿。
- Development：原 `900` 个任务，只用于 smoke/工程诊断。
- Validation：新生成的 `700` 个 size `9-21` topology，用于预注册选择、校准、功效估计。
- Confirmatory：新生成的 `900` 个 size `9-25` topology，含 size `23/25`，仅一次性盲化评估。

四组 manifest 的 topology、layout 和 task hash 必须两两不重叠。

## 关键文件

- `EXPERIMENT_PROTOCOL.md`：原始完整实验协议，内容哈希锁定。
- `configs/default.json`：唯一可执行方法矩阵与路径规范。
- `configs/protocol_lock.json`：协议、代码、环境、manifest 和旧 B0 的机器锁。
- `ENGINEER_RUNBOOK.md`：从环境检查到盲化解盲的逐步命令。
- `METHODS.md`：每个阶段的模型、训练链和因果对照。
- `COMPATIBILITY.md`：与旧 LeWM/B0 的逐字段兼容契约。
- `RESULT_SCHEMA.md`：checkpoint、task trace、候选诊断和确认工件 schema。
- `CLAIMS_AND_STOP_RULES.md`：允许与禁止的论文结论。
- `HANDOFF_CHECKLIST.md`：工程师交接和双人复核清单。
- `ADAPTIVE_DECISIONS.md`：P2/P5/P7/P8 派生方法与不可变决策链。
- `IMPLEMENTATION_AUDIT.md`：代码到协议条款的实现审计和已知边界。

## 最小自检

```bash
uv sync --frozen --extra dev
uv run ruff check vector_jepa_planner_frontier tests
uv run python -m compileall -q vector_jepa_planner_frontier
uv run pytest -q
uv run python -m vector_jepa_planner_frontier.lock_protocol --check
uv run python -m vector_jepa_planner_frontier.smoke_test
```

这些检查证明协议和代码不变量成立，不替代服务器上的真实 checkpoint/GPU 全矩阵运行。

## 固定执行顺序

```text
P0 audit/backbone extension
-> P1 oracle ladder
-> P2 search + freeze P2
-> P3/P4 + freeze P5 advancement
-> P5 -> P6 -> P7 + freeze P7
-> P8 + freeze P8
-> power analysis
-> freeze opaque confirmation
-> one-shot confirmatory execution
-> complete-family unblind
-> confirmatory summary and permanent closure
```

P8 会从 validation 自动冻结一个 Track F 方法及其最小近优预算；Track J
只有在 P7 赢家的全部 40 个 checkpoint 均满足 JEPA objective 相对恶化不超过
10%，且 validation SR 不比 Track F 低超过 `0.01` 时才晋级。最终确认集因此
只运行 B0、一个 Track F，以及可选的一个 Track J，不运行所有中间消融。

完整命令和失败处理见 `ENGINEER_RUNBOOK.md`。任何 stage gate 失败都是预定义研究结果，不是允许热改配置的理由。
