# Spatial-JEPA Iterative Planning

这是 Maze-JEPA 诊断和 `planning_repair` 之后的下一阶段实验包。它要回答的核心问题是：

> 保留逐格空间结构的 JEPA 表征，配合可重复执行的局部规划算法，能否突破当前约 `0.63-0.64` 的 latent/feedforward planning 水平，并在更大 maze 上通过增加 test-time iterations 获得泛化？

本目录不修改旧 `hdwm/`、`diagnostics/` 或 `planning_repair/` 的 checkpoint 格式。它复用同一个 Procgen Maze 环境和同一组 Set-B manifests，并通过 SHA256 protocol lock 固定比较协议。

## 目录

```text
spatial_jepa_planning/
  models.py                 # Spatial-JEPA、feedforward、gated recurrent planner、oracle VI
  losses.py                 # tie-aware action、Bellman、action-gap、map、collapse losses
  common.py                 # manifest、BFS labels、sampling、checkpoint、统计工具
  train.py                  # representation / planner / joint 三阶段训练
  evaluate.py               # learned field、decoded-map BFS、exact BFS、oracle VI
  audit_protocol.py         # 3700 个任务与配置的 fail-fast 审计
  run_plan.py               # JSON 驱动的完整实验编排
  summarize.py              # 多 seed 汇总和分层配对 bootstrap
  smoke_test.py             # CPU 前向、反向、EMA、checkpoint integration test
  configs/default.json      # 预注册实验矩阵
  configs/protocol_lock.json# manifest hash、旧结果锚点和主评估协议
  DESIGN.md                 # 科学问题、架构、损失、实验矩阵和判据
  ALIGNMENT_PROTOCOL.md     # 与旧实验严格对齐的规则
  RUNBOOK.md                # 服务器执行和故障排查手册
```

## 环境

使用服务器原项目环境，不要硬编码 Python 路径：

```bash
cd LeWM_for_Maze
pip install -e '.[dev]'
```

代码需要 Python 3.10+、PyTorch 2.0+、NumPy、Gymnasium、Pydantic 和 OmegaConf。总控脚本始终使用启动它的 `sys.executable`。
默认 `device=auto`：检测到 CUDA 时使用 CUDA，否则回退到 CPU。可用
`run_plan.py --device cuda:1 ...` 显式指定服务器 GPU。

## 必做预检

```bash
python spatial_jepa_planning/audit_protocol.py
python spatial_jepa_planning/smoke_test.py
python -m pytest -q tests/test_spatial_jepa_planning.py
python spatial_jepa_planning/run_plan.py --stages full --dry-run
```

`audit_protocol.py` 会逐个重建 train 2800 和 eval 900 个任务，并验证：

- manifest SHA256；
- topology/layout/task overlap 均为 0；
- goal、wall count 和 BFS path length 与 manifest 一致；
- 每个新 planner 的训练步数、batch、LR、scheduler 和 distance scale 对齐；
- 至少有 3 个独立 training seeds；
- reference JSON 没有被改写。

任何一项不一致都会直接退出，不会生成一个标为“可比较”的结果。

## 推荐运行顺序

### 1. 训练 Spatial-JEPA

```bash
python spatial_jepa_planning/run_plan.py \
  --stages audit,train_representations,eval_representations
```

默认运行 `seed=42,43,44`。representation 使用：

- stride-one full-resolution spatial tokens；
- online/EMA target encoder；
- action-conditioned next-spatial-latent prediction；
- 与旧实验同一类 SIGReg；
- wall、agent、goal 和 valid-action planning auxiliaries；
- 分开的 dynamics projector 与 planning projector。

`eval_representations` 使用 decoder 输出的 wall/agent/goal 执行 BFS。它不使用 oracle occupancy，是 representation sufficiency test。
decoded planner 的动作也只根据预测地图产生；`corrected` oracle action mask 不用于该主结果。

### 2. 训练 planner matrix

```bash
python spatial_jepa_planning/run_plan.py --stages train_planners
```

默认矩阵包括：

- raw value-only feedforward；
- raw all-state action CE；
- raw Bellman/action-gap feedforward；
- raw dilated full-receptive-field feedforward 强对照；
- fixed-K recurrent planner；
- progressive/random-K recurrent planner；
- Spatial-JEPA feedforward；
- Spatial-JEPA recurrent frozen/staged/joint variants。

staged last-block variant 对 representation 参数使用 `0.1x` learning rate，并继续施加 map-information loss，避免只为 planner 调整而破坏已经学到的墙/goal 信息。

所有 planner 都使用相同 train manifest、30000 steps、map batch size、optimizer、LR schedule 和 seed。只有表征来源、planner recurrence 和预注册 loss 开关不同。

### 3. 跑 oracle 和 full-900 evaluation

```bash
python spatial_jepa_planning/run_plan.py \
  --stages anchors,eval_planners,summary
```

主评估固定：

- `unisize_eval_manifest.jsonl` 的全部 900 个 task；
- `max_steps=128`；
- corrected valid moving action selection；
- learned value/action field 每个 task 计算一次；
- primary K 在训练前写入 config；
- K sweep 不用于在 test set 上选择最优模型。

### 4. 只跑一个 variant

```bash
python spatial_jepa_planning/run_plan.py \
  --stages train_planners,eval_planners \
  --variants r4_raw_iterative_progressive
```

仅做代码 smoke 时可以覆盖 seed：

```bash
python spatial_jepa_planning/run_plan.py \
  --stages train_planners \
  --variants r1_raw_action_ce \
  --seeds 42 \
  --dry-run
```

单 seed 不允许作为正式结论。

## 结果产物

```text
checkpoints/spatial_jepa_planning/<variant>_seed<seed>.pt
spatial_jepa_planning_runs/<variant>/seed<seed>/*.json
spatial_jepa_planning_runs/anchors/*.json
spatial_jepa_planning_runs/protocol_audit.json
spatial_jepa_planning_runs/summary.md
spatial_jepa_planning_runs/summary.json
```

每个 checkpoint 保存完整 architecture config、loss weights、manifest hashes、Git commit、seed、source representation、gradient-conflict history 和训练摘要。每个结果 JSON 保存全部 task rows，可进行严格 paired comparison。

## 重要边界

1. `r0_raw_value_only` 是对工程师 P4 FCVP **机制**的受控重实现。原 P4 源码/checkpoint 未进入当前仓库，因此不能声称 byte-identical reproduction。
2. 当前旧 latent/BC 数字作为 protocol-locked reference anchors；新 variant 之间的因果结论只使用同一 evaluator 重新跑出的逐任务结果。
3. `oracle_bfs` 与 `oracle_vi` 都读取真实墙体，属于算法上界，不是 learned JEPA 方法。
4. `decoded_bfs` 的地图来自 Spatial-JEPA decoder，但 BFS 是精确算法；它证明表征够不够，不证明模型学会了规划算法。
5. 默认任务 fully observable。访问记忆和跨任务泛化不在本轮主矩阵中，等当前 representation/planner 因果分解完成后再加入。

详细实验逻辑见 [DESIGN.md](DESIGN.md)，严格比较规则见 [ALIGNMENT_PROTOCOL.md](ALIGNMENT_PROTOCOL.md)。
