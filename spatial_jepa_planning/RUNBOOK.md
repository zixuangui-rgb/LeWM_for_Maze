# Server Runbook

## 0. 开始前

```bash
cd /path/to/LeWM_for_Maze
git rev-parse HEAD
git status --short
python --version
python - <<'PY'
import torch, numpy, gymnasium, pydantic
print(torch.__version__, torch.cuda.is_available())
PY
```

不要修改 manifests，不要把服务器绝对路径写进 config。checkpoint 和 result 目录已在 `.gitignore` 中。

## 1. Fail-fast 检查

```bash
python spatial_jepa_planning/audit_protocol.py
python spatial_jepa_planning/smoke_test.py
python -m pytest -q tests/test_spatial_jepa_planning.py
python spatial_jepa_planning/run_plan.py --stages full --dry-run \
  > /tmp/spatial_jepa_plan.txt
```

确认 dry-run 中：

- Python executable 是当前 conda/venv；
- seeds 为 42/43/44；
- outputs 不重名；
- Spatial-JEPA planner 指向同 seed representation checkpoint；
- eval 的 `--limit` 与 `--max-per-size` 均明确为 `0`；
- primary action selection 是 corrected。

## 2. Oracle 先行

```bash
python spatial_jepa_planning/run_plan.py --stages anchors
```

必须检查：

```text
oracle_bfs SR@128 = 0.981111...
oracle_vi K=256 SR@128 = 0.981111...
invalid_rate = 0
```

如果不满足，停止全部 learned experiments，先修 evaluator。

## 3. Representation

```bash
python spatial_jepa_planning/run_plan.py \
  --stages train_representations,eval_representations \
  2>&1 | tee logs/spatial_jepa_representation.log
```

每个 seed 检查：

- loss 无 NaN/Inf；
- `prediction`、`map_valid`、agent/goal losses 有下降；
- checkpoint 的 `protocol.eval_manifest_sha256` 正确；
- decoded-map JSON 有 900 个 unique task IDs；
- `comparable_to_full900=true`。

若训练内存不足，先降低 `map_batch_size`，但必须对所有 representation variants 同步修改，并更新 protocol/config；不要只改某个 seed。

## 4. Raw planner mechanism gate

先跑 R0-R4：

```bash
python spatial_jepa_planning/run_plan.py \
  --stages train_planners,eval_planners \
  --variants r0_raw_value_only,r1_raw_action_ce,r2_raw_bellman_gap,r2d_raw_dilated_bellman_gap,r3_raw_iterative_fixed,r4_raw_iterative_progressive \
  2>&1 | tee logs/spatial_planner_raw.log
```

检查：

- `planner_action` 接近并优于随机 CE `log(4)=1.386`；
- `planner_bellman` 不是接近机器零；
- K 曲线包含全部预注册点；
- R4 K=128 是 primary，不能改为 test SR 最大的 K；
- K=256 是否 overthink；
- OOD 与 long-path 是否从额外 iterations 中受益。

J2 还应看到非零 `planner_map`；若为零，说明 map-preservation loss 没有进入训练。

若 R4 不超过 full-receptive-field R2D，不应继续声称 recurrent planning 是主解决方案；先检查目标函数、更新稳定性和 K mask。

## 5. Spatial integration

```bash
python spatial_jepa_planning/run_plan.py \
  --stages train_planners,eval_planners \
  --variants j0_spatial_feedforward,j1_spatial_iterative_frozen,j2_spatial_iterative_lastblock,j3_spatial_iterative_joint \
  2>&1 | tee logs/spatial_planner_jepa.log
```

顺序上应先确认 J0/J1，再跑 J2/J3。J3 日志每 500 step 输出：

```text
gradient_audit cosine=... rep_norm=... plan_norm=...
```

若 cosine 长期显著为负且 J3 同时损伤 decoded-map 或 SR，应保留 J1/J2 作为主方法，不要为“端到端”强行使用 J3。

## 6. 汇总

```bash
python spatial_jepa_planning/summarize.py
```

脚本遇到以下情况会拒绝汇总：

- 少一个 seed；
- eval manifest hash 不同；
- 不是 full900；
- `max_steps != 128`；
- action selection 不是 corrected；
- learned primary 使用了 every-step recompute；
- paired candidate/baseline task IDs 不一致；
- primary K 缺失。

## 7. GPU/数值故障

默认 `device=auto`，CUDA 可用时选 CUDA，否则选 CPU。多卡服务器应使用
`run_plan.py --device cuda:N ...` 锁定设备；实际 resolved device 会写入训练
checkpoint 和 evaluation JSON。

### CUDA OOM

优先降低 `map_batch_size` 或 `trajectories_per_map`。不要降低 maze size、K 或 eval task count 来伪装成完整实验。变更 batch 后所有直接比较 variants 必须同步重跑。

### NaN/Inf

保存日志并停止该矩阵。依次检查：

1. SIGReg projection；
2. covariance loss；
3. gradient norm；
4. value field 极值；
5. mixed precision 是否被外部 wrapper 打开。

默认代码没有启用 AMP，便于第一轮定位数值问题。

### Local top-1 高但 SR 低

查看：

- margin 是否仍接近 0；
- 错误是否集中在长路径/少数 junction；
- static policy 是否形成 cycle；
- K 增加是否改变错误动作；
- corrected no-backtracking 是否掩盖 policy 本身的问题。

### Decoded BFS 高、learned JEPA 低

表征足够，planner 没学会算法。优先改 recurrence/Bellman/action gap，不要继续堆 map auxiliary。

### Raw recurrent 高、JEPA recurrent 低

规划器有效，JEPA planning branch 不够。检查 wall/goal decoder、token resolution 和 frozen/last-block 差异。

## 8. 报告时必须附带

- Git commit；
- config 和 protocol lock；
- 3 个 seed checkpoint metadata；
- full task rows；
- audit JSON；
- primary results 与完整 K curve；
- paired bootstrap CI；
- failed runs 和任何 protocol deviations。

任何 deviation 都应单独标注，不能覆盖同名正式结果。
