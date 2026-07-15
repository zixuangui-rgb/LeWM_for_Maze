# Vector-JEPA Planner Full-900 Screen

本目录是在 **完全复用历史 pooled-vector LeWM backbone** 的条件下，对12个
planner方法族进行完整900任务配对筛选的独立实验包。它解决两个同时存在的要求：

1. 不再用180任务子集，所有候选都在历史 `unisize_eval_manifest` 的900个任务上
   评测；
2. 不恢复此前118配置、20 backbone、2 search seed和多预算网格的论文级成本。

实验通过减少超参数、组合和晋级方法数量控制成本，不通过减少评测任务控制成本。

## 科学定位

- 研究对象：严格 pooled-vector Vector-JEPA planner。
- Backbone：原 `lewm_l2_cem_seqlen2`，结构、参数和训练全部不动。
- 主评测：`corrected_v1`；必须同时运行严格配对的 `unmasked`。
- 数据：历史 full-900 development split，每个尺寸100任务。
- 性质：探索性、严格配对的开发集方法筛选，不是新的盲测确认实验。
- 独立重复：backbone training seed；task和planner seed不是独立backbone重复。

## 方法范围

| 阶段 | 科学候选 | 匹配控制 |
|---|---:|---:|
| Q1 搜索 | iCEM、Beam、Best-first、MCTS | categorical-CEM bridge、历史B0 |
| Q2A 模块 | Reachability、Verifier、AR Proposal、Memory | Q1父方法或无记忆Best-first |
| Q2B 激进替代 | Vector-DTS、Bidirectional、Denoising-iCEM | Direct-DTS、forward-only、uniform-iCEM |
| Q2C 反例学习 | Hard-negative ranker | random-negative ranker |

一共12个可晋级方法族。所有Q2A方法独立运行，不累计装配。Q2B和Q2C也不与
Q2A组合。

## 执行结构

```text
protocol lock + audit
-> Q0 old/new B0 full-900 task parity
-> Q1 seed42 full-900 + freeze one scorer-compatible parent
-> Q2A/Q2B/Q2C seed42 full-900
-> freeze at most two candidates
-> Q3 add backbones43-44
-> freeze exactly zero or one winner
-> Q4 complete backbones42-51 and second planner seed when applicable
-> crossed backbone-by-task paired summary + permanent closure
```

详细命令见 `ENGINEER_RUNBOOK.md`。科学口径见 `EXPERIMENT_PROTOCOL.md`，方法
定义见 `METHODS.md`，与旧复现的逐字段契约见 `COMPATIBILITY.md`。

## 最小静态检查

从仓库根目录执行：

```bash
uv run ruff check vector_jepa_planner_full900_screen \
  tests/test_vector_jepa_planner_full900_screen.py
uv run python -m compileall -q vector_jepa_planner_full900_screen
uv run pytest -q tests/test_vector_jepa_planner_full900_screen.py
uv run python -m vector_jepa_planner_full900_screen.lock_protocol --check
uv run python -m vector_jepa_planner_full900_screen.audit_protocol
```

这些检查不代替服务器上的真实checkpoint、GPU训练和900任务执行。
