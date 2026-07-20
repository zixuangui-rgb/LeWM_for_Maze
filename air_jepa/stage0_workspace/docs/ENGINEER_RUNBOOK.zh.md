# AIR-JEPA Stage 0 工程执行手册

本手册假设执行者没有参与模型讨论。请按顺序执行，不要从中间开始，也不要把命令
改写成新的训练方案。完整科学理由见 `EXPERIMENT_PLAN.zh.md`。

## 1. 成功标准

工程任务不是“找到一个高分配置”，而是完整执行已经锁定的 143-job DAG，并生成：

1. L0 protocol/hardware/source/历史 bridge 审计；
2. 6 个 30,000-step final AIR checkpoints；
3. L1 `EARLY_SIGNAL`、L2 `PRIMARY_PROVISIONAL`；
4. L3 `FINAL_CLOSURE` JSON 与 Markdown；
5. full-900 per-task rows、全 K 曲线、corrected gap、future 干预、distance calibration
   和动作级 backtrack/dead-end diagnostics。

低分不是技术错误，不能触发重跑、追加 steps 或修改方法。

## 2. 服务器前置条件

- 4 张同型号 NVIDIA H800，编号 `cuda:0..3`；
- 同一台机器或同一个容器内可同时访问四张卡；
- Python 3.10+、项目 `uv.lock` 对应环境；
- 当前 Git checkout 必须 clean；
- `docs/SERVER_INPUTS.zh.md` 中 15 个 source files 已放在精确路径；
- 预计为 checkpoint、task rows、logs 预留足够磁盘；不要把正式输出放临时盘。

先执行：

```bash
cd /path/to/LeWM_for_Maze
uv sync --extra dev
uv run python -c "import torch; print(torch.__version__, torch.version.cuda); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
git status --short
```

必须看到至少四张 H800，且 `git status --short` 无输出。

## 3. L0-A：检查随仓库提交的锁

```bash
uv run python -m air_jepa.stage0_workspace.generate_manifests --check
uv run python -m air_jepa.stage0_workspace.protocol --target protocol --check
uv run python -m air_jepa.stage0_workspace.protocol --target package --check
uv run python -m pytest -q tests/test_air_jepa_stage0.py
uv run python -m air_jepa.stage0_workspace.smoke_test
```

任一命令失败都不要继续。尤其不能重新生成 manifest 或 package lock 来“适配”一个
意外变化；应先确认 checkout 的 commit 是否正确。

## 4. L0-B：锁定服务器上的 source artifacts

第一次运行：

```bash
uv run python -m air_jepa.stage0_workspace.lock_sources --write
uv run python -m air_jepa.stage0_workspace.lock_sources --check
```

`--write` 只允许执行一次，生成
`air_jepa_runs/stage0_workspace/locks/source_lock.json`。它会逐 seed 检查：

- representation/J0/J1 都是 `spatial_jepa_planning` format v2；
- seed 一致，且都是 final 30k checkpoint；
- J0/J1 内嵌的 representation tensors 与对应 source encoder 逐字节一致；
- 三个 seeds 共享同一 analysis-spec/source-code fingerprint，J1 参数量与 K-MAC 曲线可审计；
- 旧 J0/J1 confirmatory results 存在并被 hash。

若失败，按 `SERVER_INPUTS.zh.md` 修复文件放置。不要修改 config 来接受另一个模型。

## 5. 在任何分数出现前生成完整 DAG

```bash
uv run python -m air_jepa.stage0_workspace.plan_jobs
```

生成 `air_jepa_runs/stage0_workspace/job_plan.json`。该文件签名后不能修改。检查：

```bash
uv run python -c "import json; p=json.load(open('air_jepa_runs/stage0_workspace/job_plan.json')); print(p['job_count'], p['job_plan_sha256'])"
```

正式版本应报告 143 jobs。此时尚未读取任何 AIR 性能结果。
该总数必须同时分解为 135 个 protocol scientific cells 和 8 个 audit/smoke/release
门禁；runner 会从当前代码重建 DAG，并逐字段核对命令、依赖、GPU、优先级和输出路径。

## 6. 自动执行完整实验

建议在 `tmux` 中启动唯一调度器：

```bash
tmux new -s air0
uv run python -m air_jepa.stage0_workspace.run_jobs 2>&1 | tee air_jepa_runs/stage0_workspace/runner.log
```

调度器行为：

- 每张 GPU 同时最多一个 job；
- direct/JEPA 的物理 GPU 映射跨 seed 交叉；
- 依赖完成后自动继续，不等待 L1/L2 人工批准；
- L1/L2 分数不会改变剩余任务；
- 返回码非零或预期 artifact 缺失时标为 `technical_invalid` 并停止新任务。
- 通过 POSIX file lock 保证同一 run root 同时只有一个 runner；同伴 job 因技术失败
  被终止时也会写入 interruption status。
- 每个 formal 子进程固定并记录 `PYTHONHASHSEED=0`、
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；手工绕过
  runner 产生的 artifact 无法通过 release runtime signature。

`l0_benchmark` 会在 30k 正式训练前签名保存参数量、size-21/25 总 MAC 和
`K_compute_match`。该值只由结构与 `AIR <= 1.05 × J1@K128` 规则决定，不读取 SR。
L3 会复算并要求完全一致。
`l0_protocol_audit` 同时签名保存 4×H800、128-batch paired streams、matrix counts 和
code/source locks；正式 checkpoint 还必须证明其前 128 batches 与该签名流完全一致。
三个 release 都会主动读取并验证它，不能用手工跳过 L0 的方式生成报告。
`l0_benchmark` 固定使用 50 个 K128 forward tasks 和 5 次真实 batch8、deep
supervision 的 K128 forward/backward，以覆盖正式训练的最坏显存路径；改变 K 或次数
会被 release 拒绝。

手工 smoke 训练/评估或 `audit_protocol --skip-hardware` 必须显式指定隔离的
`--output`；CLI 会拒绝把非正式 artifact 写到正式默认路径。

不要同时启动第二个调度器。
若调度器本身意外退出，重新启动前先确认没有遗留子进程；runner 会检查旧 PID，
发现活进程时拒绝重复启动，发现失效 PID 时留下 interruption 状态再恢复。

## 7. 查看进度和快速结果

```bash
find air_jepa_runs/stage0_workspace/job_status -name '*.json' -maxdepth 1 | wc -l
tail -f air_jepa_runs/stage0_workspace/runner.log
find air_jepa_runs/stage0_workspace/logs -name '*.log' -maxdepth 1 -type f
```

释放文件：

```text
air_jepa_runs/stage0_workspace/releases/l1.json/.md
air_jepa_runs/stage0_workspace/releases/l2.json/.md
air_jepa_runs/stage0_workspace/releases/l3.json/.md
```

可以立即把 L1/L2 发给研究者讨论，但不能暂停或改写当前 DAG。只有 L3 的
`decision` 是本阶段 closure。

## 8. 技术失败处理

调度器标记 `technical_invalid` 后：

1. 保留 status JSON、log、已有 output 和 hash；
2. 记录 GPU、driver、OOM/NaN/文件系统错误及发生 step；
3. 不要删除 status，不要手动覆盖 output；
4. 判断是外部基础设施故障还是代码/协议故障，并把证据交给研究者；
5. 按 `REPLACEMENT_PROTOCOL.zh.md` 封存整个 attempt；AIR0-v1 不允许只重启受影响
   cell，也不允许复用该 attempt 已经产生的性能结果。

禁止用减小 batch、减少 K、开启 AMP、关闭 deterministic、换 seed 或从中间 step
恢复来修复正式 cell。这些会改变实验含义。

## 9. 工程师需要主动判断的边界

需要判断：磁盘是否足够、四卡是否同构、driver/runtime 是否一致、进程是否被系统
抢占、source 文件是否完整、失败是否客观技术故障。

不需要也不允许判断：哪个方法“值得继续跑”、是否删掉低分 seed、哪个 K 可称为
best、是否把 corrected/true-future 写进主表。后者由锁定分析代码处理。

## 10. 完成后验收

调度器必须打印 `complete=143/143`。随后执行：

```bash
uv run python -m air_jepa.stage0_workspace.generate_manifests --check
uv run python -m air_jepa.stage0_workspace.protocol --target protocol --check
uv run python -m air_jepa.stage0_workspace.protocol --target package --check
uv run python -m air_jepa.stage0_workspace.lock_sources --check
uv run python -m pytest tests distance_head_study/tests a1_quick_validation/tests -q
```

最终交付 `air_jepa_runs/stage0_workspace/`，但不要上传大 checkpoint 到 Git，除非仓库
另有 LFS 规则。必须保留 job plan、status、logs、locks、all per-task rows 和 L3 release。
