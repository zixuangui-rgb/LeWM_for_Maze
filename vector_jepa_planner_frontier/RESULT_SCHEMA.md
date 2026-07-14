# 结果、checkpoint 与决策工件规范

## 1. Evaluation 文件身份

每个正式结果唯一对应：

```text
(method, backbone_seed, planner_seed, search_seed,
 split_role, action_selection)
```

learned planner 的 `planner_seed` 是两个锁定值之一；B0 使用路径中的 sentinel `planner0`，但 metadata 中写 `null`，不能把它当作训练重复。

顶层 JSON：

```text
metadata
stage = planner_evaluation
opaque_run_id
split_role
action_selection
manifest {path, sha256, count}
provenance
candidate_traces
rerun
summary
resources
tasks[]
```

`metadata` 保存 protocol ID、完整 MethodConfig、analysis-spec SHA256、代码 fingerprint、Git commit/dirty、runtime、device 和三个随机性层级。正式汇总拒绝 method、seed、split、manifest、code fingerprint 或 action protocol 不一致的文件。

## 2. Provenance

`provenance` 至少包含：

```text
source_checkpoint, source_checkpoint_sha256, source_training_seed
component_checkpoint, component_checkpoint_sha256
component_checkpoint_owner
retrieval_bank {path, sha256, fingerprint, task_count}
backbone_parameter_count, planner_parameter_count
total_system_parameter_count
```

P7 frozen control 和 P8 alias 的 `method.name` 与 `component_checkpoint_owner` 有意不同。Evaluator 用 owner 的 training spec 验证 checkpoint，再用 alias 的 planner config 执行；二者不能被汇总器合并成同一方法。

## 3. Task row

每个 task row 含：

```text
task_id, maze_size, topology_seed, start_cell, goal_cell
optimal_length, success, path_length, spl
invalid_actions, repeat_states, revisit_rate, unique_state_ratio
two_cycle_rate, short_cycle_event, short_cycle_periods
max_state_visits, loop_or_cycle, final_bfs_distance
shortest_path_bin
free_cell_count, dead_end_density, junction_count, mean_corridor_length
decision_count
assistance_rate, invalid_correction_rate, backtrack_correction_rate
dead_end_entries, dead_end_recoveries, dead_end_recovery_rate
episode_seconds
auxiliary
decision_traces[]
```

正式定义：

- `loop_or_cycle = (max_state_visits >= 4)`；
- `unique_state_ratio = (unique visited states excluding s_0) / executed_steps`；
- `two_cycle_rate = count(t >= 2: s_t = s_(t-2)) / max(executed_steps-1, 1)`；
- 上述两个 ratio 均必须位于 `[0,1]`，定义修正记录在 pre-run Amendment 001；
- 成功任务必须 `final_bfs_distance=0` 且 `path_length>=optimal_length`；
- 失败任务 SPL 必须为 0；
- `decision_count=path_length`；
- proposed invalid/backtrack 在 assistance 之前计算，不能被 corrected action 掩盖。

## 4. Decision trace 与 compute ledger

每一步保存：

```text
step, deterministic planner seed
proposed_action, executed_action, assisted, assistance_reason
best_cost, selected sequence
compute
planner_diagnostics
candidate_trace_recorded
candidate_trace_metrics
```

`compute` 字段：

```text
plan_transitions, assist_transitions, total_transitions
planner_forward_calls, assist_forward_calls, planner_max_batch
node_expansions, candidate_sequences, duplicate_candidates
verifier_forward_calls, reachability_forward_calls, ranker_forward_calls
proposal_forward_calls, join_forward_calls, dts_forward_calls
```

不变量：`total_transitions = plan_transitions + assist_transitions`，且 `plan_transitions` 不超过该方法每决策的 hard cap。P8 选择使用实际 `plan_transitions/decision`，不使用理论 multiplier 替代。

## 5. Candidate-trace 工件

每个 evaluation JSON 旁边必须有：

```text
<result_stem>.candidate_traces.jsonl
```

正式采样不是概率近似：先完成不记录候选真值的正式 pass，再在每个 maze size 内按 bottom-hash 精确选择最接近 10% 的 decisions，最后用同 checkpoint/seed 重放全部任务。重放必须逐 task 验证 proposed/executed action sequence 与正式 pass 完全相同。

每条 JSONL 记录包含候选序列及事后 BFS 标签，并标记：

```text
analysis_only_no_action_influence = true
diagnostic_rescore_excluded_from_planner_budget = true
```

汇总指标包括：first-action/prefix/goal-reaching coverage@64、selection accuracy/regret、false optimism、invalid/short-cycle/no-progress、unique route ratio、编辑距离、多种 effective sample size，以及 predicted score 与 true distance 的 Spearman 相关。

某些 legacy planner 在决策时不保存全部 candidate costs；缺失 cost 只可在正式动作完成后单独重算，并明确排除出 planner compute。

## 6. Component checkpoint

训练链中的 stage：

| Stage | 含义 |
|---|---|
| `component_training` | final optimizer-step 原始训练输出 |
| `component_calibration` | 无梯度 validation 校准后的不可变输出 |
| `counterexample_training_round` | P6 M1/M2/M3 轮次输出 |

checkpoint 必含 experiment family、format version、method、track、backbone/planner seed、analysis/training spec、train manifest、source checkpoint hash、parent path/hash、head config/state dict、validation metrics 和 protocol provenance。

Track F 不保存更新后的 world-model state；Track J 必须保存完整 `model_state_dict`。P6 final checkpoint 必须是 round 3。P8 和 P7 frozen control 不创建 checkpoint，只复用 owner。

P5 例外地是 0-step deterministic assembly。它必须保存
`initialization_parents[]` 和 `head_ownership`；每个 head tensor 必须逐值等于
其声明 parent。P6 每一轮 checkpoint 必须与对应 counterexample dataset、前一轮
checkpoint、fold 和恰好 20,000 steps 形成连续哈希链。Track J 除完整 model
state 外，还必须保存 `T=8` 的 train/validation schedule、30,000-step budget、
全部 module step limits，以及三轮 P6 hard-negative provenance。

## 7. 阶段选择工件

运行目录中的决策均为不可覆盖 JSON：

- `decisions/p2_selection.json`：P2 指标、规则、winner、P3 compatibility；
- `decisions/p5_advancement.json`：P3/P4 summary hashes、reviewer evidence、四组件和三 radical gates、所选 P3 cell 与最多一个 radical；
- `decisions/p7_selection.json`：54 个 Track J cell、40-checkpoint stability evidence、唯一赢家或 fail-closed 结果；
- `decisions/p8_selection.json`：Track J 失败时 8 个、成功时 12 个 frontier 点、输入结果 digest、Track F/Track J 预算、稳定性证据、K；
- `confirmatory_power.json`：8-backbone pilot 差异、方差代理、required/available backbones、claim status。

后续 gate 会从已哈希的 validation result/candidate trace 重新计算 P2、P5、P7、
P8 的选择，验证 effective method hash 和 Track J checkpoint stability；不能只改
JSON 中的 winner 字段。P5 evidence 中六个布尔判断属于具名 reviewer 的科学
复核边界：代码验证字段完整、证据引用非空、选择与规则一致，但不自动判断
一张机制表是否足以支持 reviewer 的结论。

## 8. 确认阶段工件

冻结后产生：

```text
confirmation_lock.json
private/confirmation_mapping.json   # 权限 0600
confirmation_schedule.json          # 只有 opaque run IDs
confirmation_opened.json
confirmatory_blinded/Rxxxxxx.json
confirmation_unblinded.json
```

`confirmation_lock` 保存 P8/power hash、primary family、source/component hashes 和完整 run count。执行器只能运行 mapping 中完全匹配的 opaque entry，不能直接写具名结果。

解盲采用两阶段提交：先验证每个 opaque result 和 candidate trace；全部成功后再复制为具名正式路径，最后写 unblinded marker。任何部分失败时不得发布半个主结果表。

## 9. 汇总工件

`summarize.py` 生成：

```text
summary.json
primary_results.csv
paired_effects.csv
assistance_effects.csv
paired_outcome_tables.csv
factorial_effects.csv
mechanism_results.csv
nested_seed_variance.csv
per_size.csv
structural_strata.csv
REPORT.md
```

描述性平均顺序固定为 search seed -> planner seed -> backbone。确认性 paired
CI 使用三层 nested bootstrap：最外层重采样 backbone；每个被抽中 backbone 内
独立重采样 candidate/baseline planner seeds；最后在每个 maze-size stratum 内
成对重采样 task，并在进入 bootstrap 前平均 search seeds。Confirmatory p-value
使用 20 个独立 backbone 配对差值的 exact two-sided sign-flip，family 使用
Holm；CI 使用 Bonferroni familywise alpha。

输出不得用模糊的 `n` 或 `seed_count` 掩盖重复层级。相应表明确写
`backbone_count`、`candidate_planner_seed_count`、
`baseline_planner_seed_count`、`unique_task_count`、
`maze_size_stratum_count`、`nested_row_count` 或 `run_count`。Factorial effect
定义为 high-minus-low；二阶 interaction 是 difference-in-differences，CI 同样
按 backbone/planner/task 层嵌套重采样。Oracle 文件不符合正式 result template，
汇总器不会收集。
