# Server Runbook v2

## 0. Formal-run 前提

```bash
cd /path/to/LeWM_for_Maze
git pull --ff-only
git status --short
python --version
python - <<'PY'
import torch, numpy
print(torch.__version__, numpy.__version__)
print(torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

`git status --short` 必须为空。不要修改 manifests，也不要在 formal run 使用
`--allow-dirty-worktree`、`--allow-protocol-mismatch`、`--allow-unlocked-spec` 或
`--overwrite`。这些参数只允许临时诊断；严格汇总会拒绝相应结果。
任何带 dirty/protocol 豁免的 evaluation 都会被强制标为
`comparable_to_primary=false`。

如果服务器已有旧版 `checkpoints/spatial_jepa_planning/` 或
`spatial_jepa_planning_runs/`，先整体移动到带日期的 archive。新代码会拒绝覆盖，
不要把 v1/v2 文件放在同一输出树。

## 1. 必做预检

```bash
python spatial_jepa_planning/generate_confirmatory_manifest.py --check
python spatial_jepa_planning/audit_protocol.py
python spatial_jepa_planning/smoke_test.py
python -m pytest -q tests/test_spatial_jepa_planning.py
python spatial_jepa_planning/run_plan.py --stages full --dry-run \
  > /tmp/spatial_jepa_plan_v2.txt
```

预期：

- train/development/confirmatory 数量为 2800/900/900；
- 三个 split 的 topology/layout/task overlap 全为 0；
- confirmatory manifest 与 generator 完全一致；
- seeds 为 42-51；
- dry-run 共 534 条命令、534 个唯一 output；
- 所有 learned confirmatory model 都有 unmasked/model_valid/corrected 三份结果；
- train command 同时有 training spec 与 analysis spec hash，eval command 复用同一
  analysis spec hash；
- Spatial-JEPA planner 引用同 seed representation checkpoint。

任一项失败都停止，不要开始 GPU 训练。

## 2. Development 阶段

旧 `unisize_eval_manifest.jsonl` 现在只作为 development。可以在这里检查实现、
loss 曲线和 representation gate，但不能把 development 数字放入最终确认性主表。

```bash
python spatial_jepa_planning/run_plan.py --stages train_representations
python spatial_jepa_planning/run_plan.py --stages eval_representations
```

representation development gate：

- 无 NaN/Inf；
- prediction、SIGReg、wall/agent/goal/valid losses 有稳定信号；
- decoded wall IoU 与 agent/goal accuracy 明显高于随机；
- decoded-map BFS 可以运行，且只使用 predicted map；
- 没有 token/channel collapse。

随后运行 raw 与 Spatial-JEPA planners 的 development 评估：

```bash
python spatial_jepa_planning/run_plan.py --stages train_planners
python spatial_jepa_planning/run_plan.py --stages eval_planners_development
```

若 development 暴露代码 bug或需要改超参数：修改、测试、提交，然后归档全部旧
checkpoint，从头训练。不能在保留旧 checkpoint 的同时换配置继续跑。

## 3. 正式冻结

在第一次 learned confirmatory evaluation 前：

1. 固定所有模型、loss、30000 steps、10 seeds 和 primary K；
2. 固定三条 hypothesis、0.03 阈值/非劣界和 Bonferroni 规则；
3. 运行全测试；
4. 提交代码，保证 worktree clean；
5. 从该 commit 重新训练全部 formal checkpoints。

训练 checkpoint 会记录 clean-worktree 状态、Git commit、code fingerprint、
training/analysis spec、完整 runtime、数据 hash 和源 representation hash。任何
dirty checkpoint 或 commit/config 混用都会在 evaluator 或 summary 阶段失败。

## 4. 推荐正式运行

最稳妥的是在同一 homogeneous runtime 上无人干预地执行完整矩阵：

```bash
python spatial_jepa_planning/run_plan.py --stages full \
  2>&1 | tee logs/spatial_jepa_confirm_v2.log
```

执行顺序为：audit、confirmatory oracle、representation training、development
representation check、planner training、development planner check、confirmatory
decoded-map、三种 action protocol 的 confirmatory planner evaluation、summary。

可以用 `--seeds` 分片到多卡，但所有分片必须使用相同 Git commit、PyTorch/CUDA/
cuDNN 版本和 GPU 型号。summary 会拒绝 runtime 混合。多卡示例：

```bash
python spatial_jepa_planning/run_plan.py \
  --stages train_representations,train_planners \
  --seeds 42,43 --device cuda:0
```

分片完成后，不带 `--seeds` 运行 confirmatory stages 和 summary。不要让两个进程
写同一 seed/output。

## 5. Oracle gate

confirmatory 预期：

```text
oracle_bfs SR@128 = 0.9788888889
oracle_vi K=256 SR@128 = 0.9788888889
eligible_sr = 1.0
invalid_rate = 0
```

如果不满足，停止全部 learned experiment，先修 evaluator。19 个 shortest path
超过 128 的任务不计作模型可避免的失败。

## 6. 训练期检查

每 500 steps 检查：

- total、action、Bellman、gap、map losses 均为 finite；
- Bellman 不是机器零；
- grad norm finite；
- progressive K 按预注册 schedule 变化；
- J2 `planner_map` 非零；
- J3 gradient cosine/norm 有记录。

代码会在 loss/gradient 非 finite 时立即退出。不要捕获异常后继续写 checkpoint。

CUDA OOM 时可以统一降低所有直接比较 variants 的 batch size，但这会改变
training spec，必须更新配置、提交并从头重跑整个比较 family。不能只改一个 seed。

## 7. Confirmatory 结果检查

每个 primary learned JSON 必须满足：

- `split_role=confirmatory`；
- `action_selection=unmasked`；
- `comparable_to_primary=true`；
- 900 个唯一 task rows；
- `max_steps=128`、evaluation seed 42；
- checkpoint/training/analysis/code hashes 完整；
- static field，没有 every-step recompute。

同一 checkpoint 还必须有 model_valid 和 corrected JSON，二者
`comparable_to_primary=false`。它们只用于量化 assistance gap。

## 8. 严格汇总

```bash
python spatial_jepa_planning/summarize.py
```

默认不允许缺 seed、缺 variant、缺 diagnostic protocol、缺 task row 或覆盖旧报告。
汇总会检查 source representation、checkpoint SHA、training spec、analysis spec、训练
与评估时的 clean-worktree 状态、代码 fingerprint，以及 Python/NumPy/PyTorch/CUDA/
cuDNN/GPU runtime 一致性。主表分别列出 planner、representation planning path 与
两者合计的参数量和 size-25 Conv2d GMACs，并另列 9 个 maze size 的 SR。

三条确认性判定：

- H1 R4-R2D：simultaneous CI lower bound `>= +0.03`；
- H2 J1-R4：simultaneous CI lower bound `> -0.03`；
- H3 J2-J1：simultaneous CI lower bound `>= +0.03`。

不满足就写 `not_supported`，不能用单 seed、nominal 95% CI、最佳 K 或 corrected
结果挽救结论。

## 9. 最终归档

归档以下内容：

- Git commit 和 clean-worktree 证明；
- config、protocol lock、confirmatory manifest SHA；
- 10 个 representation checkpoints；
- 100 个 planner checkpoints；
- development 与 confirmatory JSON；
- unmasked/model_valid/corrected task rows；
- oracle、summary Markdown/JSON 和完整日志；
- 失败运行、OOM、人工中断与任何 protocol deviation。

旧 BC、latent-L2、P4 数字只能标为 legacy-development context。没有在新确认集上
按同一 evaluator 重跑之前，禁止写“超过 BC/LeWM”。
