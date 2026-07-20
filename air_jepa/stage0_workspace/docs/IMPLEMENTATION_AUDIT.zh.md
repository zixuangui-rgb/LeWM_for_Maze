# AIR-JEPA Stage 0 实现审计记录

## 1. 审计目的

本文件记录代码与 `EXPERIMENT_PLAN.zh.md` 的对应关系及交付前验证边界。它不是实验
结果，也不能替代服务器 L0。任何正式结论仍必须来自 clean commit、锁定 source、
4×H800 和完整 143-job DAG。

## 2. 科学不变量与执行位置

| 不变量 | 执行位置 | 失败行为 |
|---|---|---|
| direct/JEPA 仅 loss 权重不同 | `schemas.py`、`train.py` | config 或 checkpoint 审计失败 |
| 同 seed 初始化、map/state/K 流一致 | `audit_protocol.py`、`train.py`、`summarize.py` | L0 或 release 拒绝 |
| encoder/projector 冻结且逐 seed 对齐 | `checkpoints.py`、`train.py` | source lock 或 checkpoint 审计失败 |
| 四动作包含真实 no-op successor | `data.py`、`evaluate.py` | batch property test 失败 |
| 主结果为 AIR_dev full-900/unmasked/K128 | `protocol.py`、`plan_jobs.py`、`summarize.py` | 非法 role/cell 无法进入 release |
| L1/L2 不改变后续矩阵 | 签名 job plan、`run_jobs.py` | runner 只按依赖与固定 priority 执行 |
| AIR_select/AIR_final 封存 | `protocol.py`、`audit_protocol.py`、`summarize.py` | 发现目录即拒绝 release |
| 结果与 task/checkpoint 一一对应 | `summarize.py::ArtifactLoader` | 缺行、重复、字段或 hash 不符即失败 |
| aggregate 可由 task rows 重算 | `summarize.py` | SR/SPL/failure summary 不可复算即失败 |
| exact BFS 只作为上限 | `evaluate_oracle.py` | 单独标记 `EVALUATOR_ORACLE` |
| cost classifier 无 token-only 旁路 | `models.py`、no-bypass property test | classifier 只消费 pooled future field |
| compute match 不按分数选择 | `benchmark.py`、`summarize.py::compute_accounting` | L0 先锁定，L3 复算一致 |
| 服务器与配对门禁不可绕过 | 签名 `audit_protocol.py`、`summarize.py` | 非 4×H800 或少于 128 pairing batches 时拒绝全部 release |
| job plan 不可漏项或换命令 | `plan_jobs.py`、`run_jobs.py` | runner 重建 DAG 并逐字段比较 |

## 3. 已完成的代码级检查

- config 对关键模型、loss、训练预算、seed、K、bootstrap 和 gate 二次硬锁；
- 五类 manifest 可确定性重建，train/旧 split/AIR 三个新 split 做 topology、layout、
  task 三层无泄漏检查；
- source lock 同时检查文件 hash、checkpoint metadata、manifest hash、analysis spec 和
  内嵌 representation tensor identity；
- package fingerprint 覆盖 AIR 全目录、测试、`pyproject.toml`、`uv.lock` 及正式入口的
  递归本地 import 闭包；runtime signature 同时锁定关键 Python package 版本；
- AIR checkpoint 以原子方式保存 final-step 身份、完整模型 hash、初始化 hash、样本流
  full/prefix hash；prefix 必须等于 L0 对同 seed 签名的前 128 batches；另保存
  可按 seed 重建的完整 30k progressive-K 序列 hash、window/cumulative K 计数、
  每 500-step window 吞吐/显存、模块参数量、训练耗时和
  size-21/25 MAC；
- MAC 静态口径覆盖全部 Conv/Linear/attention products，并要求 adapter/Reasoner/future
  decoder/energy head 分项之和严格等于总量；
- release 对每条任务重新检查 size、topology、起终点、BFS 最短路、成功、SPL、失败
  分类和 aggregate；
- L3 输出 oracle ceiling、七点 K 曲线、corrected gap、全部 future 干预、local
  diagnostics、per-size/path/failure、参数/MAC/runtime 和 compute-matched 描述表；
- runner 重启时若发现旧 `running` 状态，会检查 PID；活进程存在时拒绝重复启动，
  已消失时标记 interruption 后才允许恢复调度；
- evaluator/diagnostic 统一拒绝 smoke 或非 final-step checkpoint；
- formal artifact 均不可覆盖，技术失败按 `REPLACEMENT_PROTOCOL.zh.md` 保留整个 attempt，
  不允许按单元分数选择性重跑。

## 4. 本地可验证项

交付前必须重新运行 `VALIDATION.zh.md` 的 V1-V4。还应验证：

- 135 个科学 cells 加 8 个编排/门禁 jobs 恰为 143；不仅核对数量，还从每条实际命令
  反向提取 method/seed/K/role/protocol/intervention，逐 section 与 protocol matrix
  做 multiset 完全相等检查；
- early210 的 210 个任务和 AIR_dev 的 900 个任务均恰有 24 个 deterministic local
  states；
- distance MAE/RMSE/Spearman/15-bin ECE 可由 state rows 重算，movement counters 与
  path length/invalid count 强一致；
- exact BFS 在 AIR_dev 上逐步降低距离，成功集合严格等于
  `bfs_path_length <= 128`；
- 测试能主动拒绝篡改 aggregate、具体 DAG cell、job plan identity、L0
  hardware/compute lock、source lineage、越级 K schedule、非正式 checkpoint 和
  direct/JEPA 初始化/样本流不配对；checkpoint 序列化故障注入不会留下半文件。

本地 CPU 运行只能证明实现与数据合同自洽，不能证明训练会收敛、H800 显存足够或
真实 source checkpoint 合法。

## 5. 服务器必须完成的检查

`# CHECK-REQUIRED` 工程师必须在正式服务器完成：

1. 15 个 source artifacts 精确存在，`lock_sources --write/--check` 通过；
2. 四张 GPU 均为同构 H800，runtime audit 通过；
3. 历史 J0/J1 六个 bridge cells 逐 task exact parity；
4. 两个 1000-step smoke train 无 NaN/OOM，benchmark 给出真实 ETA/peak memory；
5. 6 个 final 30k checkpoint 全部完成且 direct/JEPA 配对审计通过；
6. 143/143 jobs 完成，L3 签名有效，sealed result paths 仍不存在。

任一项失败时只能标记 technical invalidity 并保留证据。不得用减少 batch/K/task、
开启 AMP、换 seed、挑 best checkpoint 或删掉低分 cell 来修复。

## 6. 解释边界

AIR0-jepa 与 AIR0-direct 的差异只能归因于“future latent + distributional cost”监督
包，不能在本阶段进一步拆成两个独立因果效应。corrected、true-future 和 BFS oracle
均为诊断或上限。只有 full-900、unmasked、normal、K128 的 learned method 分数可以
进入绝对能力主表；三个 seeds 的结果仍是严格的探索性架构证据。
