# Alignment Protocol

本文档是本轮实验的比较合同。违反其中任一 primary rule 的结果必须标成 diagnostic，不得放入主表。

## 1. 固定数据

| Split | 文件 | SHA256 | 数量 | Sizes |
|---|---|---|---:|---|
| Train | `data/splits/unisize_train_manifest.jsonl` | `1c477d...a05f6` | 2800 | 9-21，每 size 400 |
| Eval | `data/splits/unisize_eval_manifest.jsonl` | `210e2d...b8d9` | 900 | 9-25，每 size 100 |

完整 hash 位于 `configs/protocol_lock.json`。`audit_protocol.py` 会验证 topology、layout 和 task 三层 overlap 都为 0。

所有 final checkpoints 使用完整 2800 train manifest。若要调超参数，应另外建立 development train/validation 划分；确定配置后再从头用完整 2800 训练。不得根据 eval900 选择 loss、epoch 或 primary K。

## 2. 固定评估

Primary navigation protocol：

| 字段 | 固定值 |
|---|---|
| Eval tasks | manifest 全部 900 个固定 start/goal task |
| `max_steps` | 128 |
| Moving actions | `[UP, DOWN, LEFT, RIGHT] = [1,2,3,4]` |
| Action correction | mask no-move action；若可能则避免 immediate backtracking |
| Learned field | 每个 task 起点计算一次全图 field |
| Seen/OOD | seen 9-21；OOD 23/25 |
| Task output | 必须保存全部 900 个 task rows |

“每个 task 计算一次 field”适用于输出全图 policy/value 的模型。因为墙和 goal 在 episode 内不变，模型应学习与 agent marker 无关的全图解。`--recompute-every-step` 只用于检查 agent-channel sensitivity，不进入 primary summary。

## 3. Step-cap 上界

Eval900 中有 17 个 task 的 oracle shortest path 大于 128。因此：

```text
SR@128 理论上限 = 883 / 900 = 0.981111...
```

本目录的 exact BFS 和 `oracle_vi K=256` 都必须达到这个值。旧报告的 P4 VI 为 `0.957`，说明旧实现还存在约 22 个非 step-cap failure；它保留为历史锚点，但不再是新 evaluator 的预期 oracle。

正式报告应同时给出：

- `SR@128`，用于与旧实验比较；
- seen/OOD；
- per-size；
- 按 shortest-path bins 的补充结果。

不得把 17 个不可完成 task 解释成模型规划错误。

## 4. 训练预算对齐

`audit_protocol.py` 强制所有 planner variants 的以下字段一致：

- `steps=30000`；
- `map_batch_size=8`；
- AdamW、`lr=1e-3`、cosine schedule；
- `weight_decay=0`；
- `grad_clip=1`；
- `distance_scale=128`；
- train/eval manifests；
- training seeds 42/43/44。

Training seed 与 evaluation sampling seed 分离。三个 checkpoints 分别记录 42/43/44，但 local diagnostics 对所有模型固定使用 evaluation seed 42，确保抽取的是同一批 states。

允许变化的因素必须在 variant 表中显式列出：

- input 是 raw grid 还是 Spatial-JEPA；
- feedforward 还是 shared-weight recurrence；
- loss 开关；
- fixed/random/progressive K；
- encoder frozen/last-block/joint。

## 5. Local top-1 对齐

为了与旧 diagnostics 对齐，主 `local_top1` 使用：

- eval entries 按旧 `select_entries(seed+101)` 顺序；
- 先按旧 metric-alignment 代码消耗 `pairs_per_maze=128` 的 RNG draws；
- 每 maze 由 `seed+202` 不放回抽 24 个 free states；
- 只统计至少有两个 valid moving actions 的非 goal states；
- tie shortest paths 全部计为正确；
- 先算每 maze top-1，再对 maze 做 macro average。

结果同时保存 `all_cell_local_top1`，但不得把它与旧 `Local top-1 ~= 0.588` 直接互换。

## 6. K 的预注册

每个 variant 在 `configs/default.json` 中写死 `primary_iterations`：

- feedforward：depth 4；
- fixed recurrent：K=64；
- progressive recurrent：K=128。

测试时额外报告 `K={8,16,32,64,128,256}` 曲线，回答 test-time compute scaling 和 overthinking。不得从这条 test curve 取最大 SR 作为主结果。

## 7. 多 seed 与统计

正式结果至少 3 个独立 training seeds。`summarize.py`：

1. 验证每个结果的 eval hash、task count、max steps 和 action protocol；
2. 验证 candidate/baseline 的 task IDs 完全相同；
3. 先重采样 training seeds，再在 seed 内重采样固定 tasks；
4. 报告 delta SR/SPL 的 95% hierarchical paired-bootstrap CI。

若 95% CI 包含 0，结论应写成“未发现稳定提升”，不能根据单个 seed 宣称有效。

## 8. 可比性等级

### Level A：严格因果比较

由本目录同一代码、同一 evaluator、同一 seeds 和同一 manifest 生成的 variants。例如：

- R2 feedforward losses vs R4 recurrent same losses；
- raw recurrent vs Spatial-JEPA recurrent frozen；
- frozen vs last-block vs joint。

`decoded-map BFS` 单独使用 predicted-map action selection，不调用 oracle valid-action correction。它与 learned planner 不属于同一 action-selection protocol，因此只用于 representation sufficiency，不做 paired planner claim。

### Level B：协议一致参考

旧 latent-L2/BC/P4 报告使用同一 Set-B full900 和 max steps，但没有统一 task rows 或当前 evaluator。可以作绝对水平参考，不适合 paired significance claim。

### Level C：机制参考

- `r0_raw_value_only` vs 旧 FCVP；
- 新 oracle VI vs 旧 P4 VI。

它们回答机制是否复现，不是代码逐字节复现。

## 9. 禁止的比较

- full900 与 seen-only n=700；
- full900 与 max-per-size 40；
- `SR@128` 与不同 episode budget；
- corrected action selection 与 unmasked policy；
- test set 上调出的 K 与预注册 K；
- 单 seed 最优值与 3-seed mean；
- decoded-map BFS 与 fully learned planner 混称为“JEPA 学会规划”。
