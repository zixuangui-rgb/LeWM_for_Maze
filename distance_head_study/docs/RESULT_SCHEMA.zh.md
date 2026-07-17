# Artifact 与结果 Schema

所有正式产物是 append-only 或 immutable。运行失败只能在相同 spec/hash 下 resume；
不得改 config 后覆盖旧目录。

Cache shard 额外绑定 `analysis_spec_sha256/protocol_lock_sha256`。Index 尚未写出时允许重跑
并复用逐 shard 原子文件，但必须重算 content hash 且 manifest/backbone/protocol 全匹配；
已经存在的完整 index 仍不可覆盖。

## Checkpoint

DistanceHead checkpoint 至少包含：

```text
experiment_family, format_version, stage, formal_run
method, method_sha256, decision_sha256s
training_spec_sha256, analysis_spec_sha256
backbone_seed, head_seed
backbone_path, backbone_sha256
head_spec, head_state_dict
calibrated_weights
candidate_bank(path/hash/metadata), cache_bindings
initialization(mode/parent hash/loaded keys)
checkpoint_selection = final_step
final_step
recent_metrics, elapsed_seconds
```

Joint checkpoint 额外保存 `model_config/model_state_dict`。Evaluator 必须加载 joint
checkpoint 内 backbone state；diagnostics 还要从 cache observation 重新编码所有相关
latent，不能误用原 frozen backbone latent。Operational resume state 额外保存
Python/NumPy/Torch/CUDA RNG state，它不进入最终 scientific checkpoint。

## Task rows

每个 `rows.jsonl` row 至少包含：

```text
task_id, maze_size, topology_seed, start_cell, goal_cell
optimal_length, success, path_length, spl
invalid_actions, repeat_states, max_state_visits, loop_or_cycle
final_bfs_distance, failure_mode
proposed_invalid, proposed_backtrack
assistance_count, assistance_rate
plan_transitions, fallback_transitions
mean_best_cost, episode_seconds
```

`spl=0` 对所有失败任务；成功 path 不得短于 oracle BFS。Summary 必须从 rows 重算。
Loader 还会读取 metadata 绑定的 manifest，核对每个 task 的 ID、size、topology、start、
goal 与 BFS optimal length；只有行数相同不算完整结果。

`failure_mode` 的优先级固定为：`success` -> `step_cap_ineligible` -> `invalid_action` ->
`loop_or_cycle` -> `insufficient_progress` -> `timeout_inefficient`。其中 D_confirm 有 12 个、
D_stress 有 33 个任务的 oracle shortest path 超过统一 128-step cap；这些任务仍保留以维持
paired manifest，对所有方法统一记为 `step_cap_ineligible`，不得按方法删行。

分析中的 `predictor_transitions_per_step` 固定为
`(plan_transitions + fallback_transitions) / max(path_length, 1)`。只报 search transitions
会漏掉 corrected fallback 的真实模型计算量。另报
`episode_seconds_per_step = episode_seconds / max(path_length, 1)`；前者是确定性的
world-model compute proxy，后者包含 head/search/encoder 与硬件噪声，两者都不能冒充总
FLOPs。Wall-clock 只在同一 runtime block 中作 secondary paired 比较，不参与 primary gate。

Loader 还会逐 row 重演 action/compute 语义：`corrected_v1` 的实际 invalid 必须为 0，非
test-BFS fallback 每次 assistance 必须恰好产生 5 次 predictor transition；`unmasked`
不得出现 assistance/fallback，且实际 invalid 必须等于 proposed invalid。Oracle 和
model-free greedy 的 plan transitions 必须为 0，predictor-greedy 必须恰好为每步 5 次，
search planner 必须每个实际决策步使用大于 0 且不超过 768 次 transition。违反任一条件的
结果即使 row 数、SR 和 summary 看似正常，也不能进入分析。

这里的 5 来自 LeWM 的 `0..4` model-action vocabulary；实际候选动作仍只有 `1..4`。因此
该 compute 记账不意味着 planner 有第五个可执行方向，也不改变 local top-1 的 `0.25`
随机基准。

## Diagnostics

```text
absolute_distance: MAE/RMSE/bias/Spearman/calibration
true_latent_local: top1/regret_steps/score_margin/score_margin_unit
predicted_latent_local: top1/regret_steps/score_margin/score_margin_unit
candidate_order: predicted/true dynamics Spearman and regret
closed_loop_drift_bfs_steps
reachability: per-budget Brier/ECE/AUROC and monotonic violation
candidate_bank(path/hash/metadata)
by_size
diagnostic_sha256
```

trajectory context 使用按 batch row 等距抽取的固定索引，覆盖按 topology 连续分组的
microbatch；drift 的 `n` 是所有已诊断 context 的全部 fixed-bank candidates 数，不能只
统计 candidate 0。Joint checkpoint 返回的混合 CPU/GPU batch 必须整体迁移到 evaluator
device 后再计算 mask、label 与 score。

legacy target 的 raw metric 使用 cache 内 `max_goal_distance` inverse-transform；该值只用于
诊断，绝不进入 planner action selection。

`score_margin` 的单位不是统一“步数”：DistanceHead/QRL 等 raw-distance scorer 报
`bfs_steps`，latent-L2 报 `latent_squared_l2`。不同单位的 margin 只能在各自方法内部比较，
不得直接横向解释为同样大小的 BFS 距离差。

`absolute_distance` 对 horizon-conditioned head 固定使用 `h=12`；
`true_latent_local` 与 action-aligned `predicted_latent_local` 固定使用 `h=1`。
历史 `b_dh_predictor_greedy` 是 corrected-v1 parity 接口，不得把它的 episode SR 当作
`predicted_latent_local` 的替代诊断。

## Signed decisions

Decision、seed release、原 shortlist、独立 negative fallback shortlist、confirmation n、
confirm-open、analysis 与 closure 都采用：

```text
unsigned_payload + canonical JSON SHA256 signature field
```

加载器会去掉 signature 字段重算。手改任何一个数都会失败。

会影响选择或结论的 artifact 不只绑定 aggregate rows。`input_hashes`/evidence bundle 必须
覆盖 metadata、task rows、summary、manifest、backbone/head checkpoint、candidate bank、
cache index，以及分片 merge 时的每个输入 rows 文件。任一上游文件变化或缺失，decision、
power、analysis 和 closure 都拒绝加载。

shortlist、negative closure、confirmation-n、seed release、confirm-open 与最终 closure
不仅绑定上一层 JSON 本身，还会把该 JSON 已验证的底层 `input_hashes` 扁平合并进自己的
hash map；同一路径出现不同 hash 会直接失败。最终 closure 因此可以从单个 artifact 直接
核验完整上游证据，而不是只验证一串仍可能引用已变化文件的中间指针。

Job plan 另有同名 `.metadata.json`，绑定 config/protocol/release/plan hash；executor state
绑定 plan hash，每个完成 job 再保存 output hashes。没有 completion seal 的中断 job 不会
因文件“看起来存在”而自动判为成功。正常退出也遵循同一规则：进程返回 0 后仍须调用该
artifact 类型的内容级 validator；缺文件或语义不一致会写入 failed state，不生成 seal。

formal 与 limited evaluator/diagnostics 使用同一 seed/split gate。Smoke artifact 虽然写入
独立目录且禁止进入决策，但不能借此提前读取未开放的 `D_select/D_confirm/D_stress`。

## Evidence status

| Status | 允许措辞 |
|---|---|
| `exploratory_single_backbone` | 单 backbone 有机制信号 |
| `replicated_development` | 三个 development backbone 初步复现 |
| `confirmatory` | 可按预注册 analysis 作正式结论 |

Seed-1/Seed-3 不得写“最终提升”“性能上限”或“已证明无效”。
