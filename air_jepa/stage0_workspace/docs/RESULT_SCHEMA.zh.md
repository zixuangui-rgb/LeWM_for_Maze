# AIR-JEPA Stage 0 结果与证据角色

## 1. 证据角色

| 标签 | 数据/条件 | 可以回答 | 不可以回答 |
|---|---|---|---|
| `HISTORICAL_BRIDGE` | 旧 confirmatory、J0/J1 static | 新旧 evaluator 是否等价 | AIR 新方法能力 |
| `EARLY_SIGNAL` | AIR_dev early210、seed42 | 尽早发现方向和技术问题 | 跨 seed 结论、最终 pass/fail |
| `PRIMARY_PROVISIONAL` | AIR_dev full900、3 seeds、K128 | 主 SR/OOD 初步结论 | 关闭阶段、选择最佳 K |
| `FINAL_CLOSURE` | 完整 locked DAG | Green/Yellow/Red | AIR_select/final 性能 |
| `MECHANISM_DIAGNOSTIC` | corrected/copy/permutation/zero/local | 定位瓶颈 | 绝对能力主分数 |
| `ORACLE_INTERVENTION` | true-future | predictor 上限/误差归因 | 可部署模型分数 |

## 2. Evaluation JSON

核心结构：

```text
schema
metadata
  experiment_id, method, seed, k, split_role
  evidence_role, action_protocol, intervention
  manifest/checkpoint/protocol/package/source hashes
  git/runtime/formal/elapsed_seconds
navigation
  overall, seen, ood, by_size, by_shortest_path
task_rows[]
  task_id, size, optimal_length, success, spl, path_length
  invalid_actions, repeats, loop_or_cycle, final_bfs_distance
  immediate_backtracks, distance_decrease/flat/increase_actions
  dead_end_recovery_opportunities/successes/failures
  failure_reason, elapsed_seconds, auxiliary
```

每个 full result 必须恰有 900 个唯一 task rows，early result 必须恰有 210 个，且
task ID 集合与锁定 manifest 完全一致。

## 3. Diagnostic JSON

`diagnose.py` 对每个 maze 确定性抽 24 个满足至少两个不同 successor 的 non-goal
states，保存：

- predicted/true/copy/permuted/zero energy ranking；
- tie-aware local top-1、regret、margin；
- predicted 与 true-future cost distribution 的截断距离 MAE/RMSE/Spearman、
  categorical accuracy 和 15-bin top-class ECE；
- normalized field/delta error 与 copy baseline；
- predicted/target variance 和四 candidate pairwise distance；
- `prediction_flip` 与 `energy_wrong_with_true_future` 错误分类。

Collapse 判据直接来自锁定 config，不由报告作者主观判断。

## 4. Release JSON/Markdown

`summarize.py` 生成签名 JSON 与人读 Markdown。JSON 保存所有精确统计；Markdown 是摘要，
不能替代 JSON。L3 包含：

- seed-level overall/seen/OOD SR/SPL；
- 七点 K curves；
- J0/J1 static bridges 和 corrected assistance gap；
- crossed seed × size-stratified task paired bootstrap；
- future intervention/diagnostic 汇总；
- exact-BFS step-cap ceiling（`EVALUATOR_ORACLE`，不得当作 learned score）；
- per-size、path-bin、episode failure taxonomy、动作级 backtrack/dead-end/distance
  progress 与完整 intervention rows；
- encoder/projector/AIR 子模块参数、训练显存、wall-clock、K-MAC 曲线与预定
  compute-matched K；
- K128-K16 的 OOD/长路径 delta、log2(K)-SR Spearman 与 SR/GMAC；
- 签名 L0 4×H800/paired-stream、historical bridge、direct/JEPA paired checkpoint 和
  sealed-role 审计；
- 每个 Green check 与唯一 `decision`。

Bootstrap 使用同一 seed/task resampling 规则，family size 固定为 4，并用 Bonferroni
simultaneous percentile interval。只有 3 个 training seeds，因此区间仍应写成严格的
探索性证据，不应宣称得到总体定理。

## 5. 主结果边界

论文式主表只允许 `AIR_dev full-900 + unmasked + K128 + normal`。corrected、历史集、
early210、非 primary K 和所有 future interventions 必须单独列出并保留标签。

`AIR_select` 与 `AIR_final` 在本阶段封存，评测 CLI 会主动拒绝这两个 role；L3 只说明
它们保持未打开，不报告任何分数。
