# 结果与证据结构

## 1. 目录约定

```text
a1_quick_validation_runs/
  cache/
  candidates/
  diagnostics/
  results/
  decisions/
  plans/
  logs/
  completions/
  train_state/
```

正式 checkpoint 位于：

```text
checkpoints/a1_quick_validation/heads/<method>/backbone42_head<head_seed>.pt
```

## 2. Result rows

评估继续使用 `distance-head-task-results-v1`。每个 cell 必须包含：

- `metadata.json`：方法 resolved spec、method hash、manifest、checkpoint、action protocol 和 run-spec hash；
- `rows.jsonl`：每个 task 一行，包含 success、SPL、路径、失败模式、循环等；
- `summary.json`：overall/seen/OOD/by-size/by-path-length 汇总。

选择器不直接相信 summary 均值，而是重新加载完整 rows，通过 `task_id` 配对计算 delta。

## 3. Diagnostics

诊断沿用 `distance-head-diagnostics-v1`，但输出到新 run root。必须包含：

- absolute BFS distance MAE/RMSE/bias/Spearman；
- true-latent local top-1/regret/margin；
- predicted-latent local top-1/regret/margin；
- candidate order Spearman/regret；
- closed-loop drift；
- reachability calibration（若适用）；
- cache、candidate bank 和 checkpoint provenance；
- `diagnostic_sha256`。

选择器还会展开并绑定 diagnostic 的 backbone/head checkpoint、candidate bank、cache index 和每个 cache shard；只验证顶层 JSON 签名不足以进入决策。

## 4. Q1 decision

路径：`decisions/q1_decision.json`。

核心字段：

- `ranked_passing_methods`；
- `selected_methods`；
- `metrics[method].paired_vs_a1`；
- `metrics[method].mechanism_gate`；
- `stopped_for_no_candidate`；
- `input_hashes`；
- `decision_sha256`。

若有候选，另写 `q1_shortlist.json`。其中 `selected_methods` 包含 A1 anchor 加最多两个新方法；`new_methods` 只包含新方法。

## 5. Q2 winner

路径：`decisions/q2_winner.json`。

核心字段：

- `eligible_methods`；
- `ranked_passing_methods`；
- `selected_method`，无 winner 时为 `null`；
- 每个 head seed/action protocol 的 paired metrics；
- 每个 head seed 的 `mechanism_rechecks`，明确标记 Q2 不重复使用 Q1 的 SR safety gate；
- 聚合 promotion gate；
- `stopped_for_no_winner`；
- `input_hashes`；
- `decision_sha256`。

winner 在 full-900 之前锁定。

## 6. Q3 assessment

路径：`decisions/q3_assessment.json`。

核心字段：

- `locked_winner`；
- 四个方法、两个动作协议的 paired metrics；
- overall/seen/OOD/SPL delta 和 bootstrap CI；
- `q3_gate_pass`；
- `confirmatory: false`；
- `evidence_status: exploratory_single_backbone_full900`；
- `decision_sha256`。

## 7. Job plan 和 completion

Plan 包含：

- phase；
- worker count；
- package/protocol lock hash；
- 每个 job 的固定 worker 和 argv command pipeline；
- `plan_sha256`。

Completion seal 包含：

- plan/job/worker/device；
- command list hash；
- 每个 log 的 path/hash；
- `all_commands_succeeded: true`；
- `completion_sha256`。

Plan、decision 和 completion 都是不可覆盖产物。

## 8. Paired metric 对象

每个 endpoint 至少包含：

```json
{
  "treatment_mean": 0.0,
  "reference_mean": 0.0,
  "delta": 0.0,
  "paired_bootstrap_95ci": [0.0, 0.0],
  "n": 0
}
```

paired metric 顶层还必须包含：

```json
{
  "pairing": "exact_task_id",
  "task_resampling": "paired_by_task_id_within_maze_size",
  "bootstrap_samples": 10000
}
```

SR 还包含 `seen` 和 `ood` 子对象（该 split 存在对应任务时）。所有 CI 使用锁定的 10,000 replicate seed schedule，并保持每个 maze-size stratum 的任务数。
