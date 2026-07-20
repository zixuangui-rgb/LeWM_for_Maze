# 工程运行手册

## 1. 原则

所有正式操作必须从仓库根目录运行，并且只能通过 `python -m a1_quick_validation.run` 或受锁 job runner 调用。不要直接编辑 JSON，不要使用 dirty override，不要复用另一个分支的 decision 文件。

以下示例假设仓库根目录为 `LeWM_for_Maze`，四张卡在同一进程命名空间中可见为 `cuda:0` 到 `cuda:3`。

## 2. 环境预检

```bash
git status --short
.venv/bin/python -m pytest a1_quick_validation/tests -q
.venv/bin/python -m a1_quick_validation.run audit
```

正式运行前 `git status --short` 必须为空。外层 package lock 会覆盖本目录全部代码、配置和文档；任何临时修改都会被拒绝。

## 3. 必需的原实验产物

至少需要：

```text
checkpoints/final_closure/lewm_l2_cem_seqlen2_seed42.pt
checkpoints/distance_head_study/heads/b_dh_cem/backbone42_head0.pt
checkpoints/distance_head_study/heads/b_dh_cem/backbone42_head1.pt
checkpoints/distance_head_study/heads/a1_log/backbone42_head0.pt
checkpoints/distance_head_study/heads/a1_log/backbone42_head1.pt
distance_head_study_runs/cache/{train,cal,screen,select}/backbone42/index.json
distance_head_study_runs/candidates/train/backbone42/bank.pt
```

参考 checkpoint 内记录的 cache 和 candidate bank 也必须仍可读取。导入器会校验这些依赖，不能只孤立复制 `.pt`。

## 4. Q0：输入准备

生成 plan：

```bash
.venv/bin/python -m a1_quick_validation.plan_jobs --phase q0
```

运行唯一的 CPU worker：

```bash
.venv/bin/python -m a1_quick_validation.run_jobs \
  --plan a1_quick_validation_runs/plans/q0.json \
  --worker-index 0 \
  --device cpu
```

Q0 会验证协议、重绑定四套 cache、生成 quick candidate bank、释放 seed1，并导入参考 head0。

### 原 cache 缺失时

可用 quick config 原生重建缺失角色：

```bash
.venv/bin/python -m distance_head_study.build_cache \
  --config a1_quick_validation/configs/default.json \
  --split-role screen \
  --backbone-seed 42
```

对缺失的 `train/cal/screen/select` 分别执行。原生 quick cache 会接受相同的完整 hash 校验。若参考 checkpoint 所绑定的原 cache 已丢失，则不能导入该 reference checkpoint，需要先恢复原依赖；不要手改 checkpoint metadata。

## 5. Q1：四卡快筛

```bash
.venv/bin/python -m a1_quick_validation.plan_jobs --phase q1
```

在四个终端分别运行：

```bash
.venv/bin/python -m a1_quick_validation.run_jobs \
  --plan a1_quick_validation_runs/plans/q1.json \
  --worker-index 0 --device cuda:0
```

其余终端将 `worker-index/device` 改为 `1/cuda:1`、`2/cuda:2`、`3/cuda:3`。

四个 worker 全部完成后才能做选择：

```bash
.venv/bin/python -m a1_quick_validation.plan_jobs --phase q1_select
.venv/bin/python -m a1_quick_validation.run_jobs \
  --plan a1_quick_validation_runs/plans/q1_select.json \
  --worker-index 0 --device cpu
```

检查：

```text
a1_quick_validation_runs/decisions/q1_decision.json
a1_quick_validation_runs/decisions/q1_shortlist.json
```

若只有 `q1_decision.json` 且 `stopped_for_no_candidate=true`，实验按预注册规则结束。不要生成 Q2 plan。

## 6. Q2：双 head seed 独立 split 复核

先串行释放 seed3 并导入两个 reference head1：

```bash
.venv/bin/python -m a1_quick_validation.plan_jobs --phase q2_gate
.venv/bin/python -m a1_quick_validation.run_jobs \
  --plan a1_quick_validation_runs/plans/q2_gate.json \
  --worker-index 0 --device cpu
```

训练最多两个候选的 head1：

```bash
.venv/bin/python -m a1_quick_validation.plan_jobs --phase q2_train
```

启动四个 worker；命令形式同 Q1，将 plan 改为 `q2_train.json`。空 worker 会立即退出。

完成后生成评估 plan：

```bash
.venv/bin/python -m a1_quick_validation.plan_jobs --phase q2_eval
```

再次启动四个 worker，plan 改为 `q2_eval.json`。所有 worker 完成后：

```bash
.venv/bin/python -m a1_quick_validation.plan_jobs --phase q2_select
.venv/bin/python -m a1_quick_validation.run_jobs \
  --plan a1_quick_validation_runs/plans/q2_select.json \
  --worker-index 0 --device cpu
```

检查 `a1_quick_validation_runs/decisions/q2_winner.json`。若 `selected_method` 为 `null`，实验结束，不跑 Q3。

## 7. Q3：full-900 探索性闭环

```bash
.venv/bin/python -m a1_quick_validation.plan_jobs --phase q3_eval
```

启动四个 worker，plan 使用 `q3_eval.json`。全部完成后：

```bash
.venv/bin/python -m a1_quick_validation.plan_jobs --phase q3_assess
.venv/bin/python -m a1_quick_validation.run_jobs \
  --plan a1_quick_validation_runs/plans/q3_assess.json \
  --worker-index 0 --device cpu
```

最终证据为：

```text
a1_quick_validation_runs/decisions/q3_assessment.json
```

Q3 后本实验无后续自动扩展，无论正负都应收尾。

## 8. 中断和恢复

- head training 总是以 `--resume` 调用；存在有效 train state 时恢复，不存在时从头开始。
- evaluation 总是以 `--resume` 调用；已有 rows 会按 task ID 跳过。
- 已完成 diagnostics、checkpoint、decision 和 completion seal 不会覆盖。
- 某个 job 失败时，查看 `a1_quick_validation_runs/logs/<phase>/<job>/`，修复外部环境后重跑同一 worker。
- 若 completion seal 已存在，runner 会验证它属于同一 plan 后跳过。

不要删除已经用于 decision 的输入产物；decision 的 `input_hashes` 会在后续门控时重新验证。

## 9. 四卡映射注意事项

若调度器为每个进程只暴露一张卡，则四个进程都传 `--device cuda:0`，但使用不同 `worker-index`。若一个节点内四卡全部可见，按示例传 `cuda:0` 到 `cuda:3`。

不要让两个 worker 使用同一 GPU，也不要同时运行两个相同 worker-index。

## 10. 故障判定

以下错误不能 bypass：

- protocol/package lock mismatch；
- dirty scientific worktree；
- cache shard/hash mismatch；
- source/quick candidate actions 不同；
- reference method hash 不同；
- split task 集不一致；
- 方法不在当前阶段 matrix；
- Q1 shortlist 或 Q2 winner dependency hash 改变。

这些错误意味着对照关系已不再成立，应恢复正确产物或重新开始该阶段。
