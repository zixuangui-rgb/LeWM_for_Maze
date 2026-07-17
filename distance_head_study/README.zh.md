# DistanceHead 分阶段严谨实验包

本目录实现 Procgen Maze 上 Vector-JEPA DistanceHead 的完整研究闭环。目标不是只把
BFS regression loss 降低，而是判断一个 goal-conditioned cost 是否能依次改善：

1. raw BFS distance 估计；
2. true-next 与 predicted-next 的局部动作排序；
3. h12 candidate trajectory 排序；
4. corrected/unmasked closed-loop SR、SPL 与 OOD；
5. 最终能否在独立 fresh backbone seeds 上超过 `B-DH-CEM` 与 `B-L2`。

代码状态：**实现完成，正式服务器实验尚未运行**。任何结果数字必须来自本包生成的
task rows 和 analysis artifact，不得从文档中的历史数字抄写。

## 科学合同

- `D_train`：2800 个 sizes `9-21` topology；
- `D_cal`：从 `D_train` 确定性抽取的 140 个 topology，只做数值/梯度标定；
- `D_screen`：140 个新 topology，只含 sizes `9-21`；
- `D_select`：210 个新 topology，只含 sizes `9-21`；
- `D_confirm`：sealed full-900，sizes `9-25`，其中 `23/25` 是 size OOD；
- `D_stress`：final closure 后的 sizes `27/29/31`；
- planner anchor：horizon 12、64 candidates、1 CEM iteration、每步 replan、128 step cap；
- 同时报告 `corrected_v1` 与 `unmasked`；
- checkpoint 一律使用 final step，禁止按 validation SR 挑 checkpoint；
- confirmation backbone 从 `1001` 开始，与历史 `42-61` 完全不重合；
- 独立统计单位是 backbone training seed，不是 task、candidate 或 head seed；
- BFS 只用于训练标签、诊断与 oracle，不得进入 learned method 的 test-time action selection。
- `D_confirm` 中 12/900、`D_stress` 中 33/150 个任务的最短路超过 128 步；这些任务按
  原任务 step cap 保留并标为 `step_cap_ineligible`，所有方法共享，不能从某个方法中删掉。

## 目录

```text
distance_head_study/
  configs/                  严格 config、方法目录与 protocol lock
  protocol/                 seed registry、baseline provenance、bootstrap schedule
  manifests/                D_cal/D_screen/D_select/D_confirm/D_stress
  docs/                     研究总纲、规范协议、工程手册和审计说明
  tests/                    单元、集成、确定性、泄漏与统计测试
  data.py                   goal-consistent cache 与 matched sampler
  models.py                 scalar/ordinal/distribution/reachability/quasimetric heads
  losses.py                 absolute/local/Bellman/multistep/predicted/TRM losses
  train_backbone.py         精确复用 final_closure LeWM recipe
  train_head.py             final-step、gradient-calibrated、joint-control trainer
  evaluate.py               task-level corrected/unmasked evaluator
  diagnose.py               raw distance/local/candidate/drift diagnostics
  analyze.py                crossed bootstrap、backbone sign-flip 与 Holm
  make_decision.py          签名的阶段决策
  plan_jobs.py/run_jobs.py  签名 job DAG、四卡 executor 与完成标记
```

## 从这里开始

工程师先阅读：

1. [实验协议](docs/EXPERIMENT_PROTOCOL.zh.md)
2. [运行手册](docs/ENGINEER_RUNBOOK.zh.md)
3. [方法目录](docs/METHOD_CATALOG.zh.md)
4. [服务器核查项](docs/CHECK_REQUIRED.zh.md)
5. [验证说明](docs/VALIDATION.zh.md)

首次检查：

```bash
cd /path/to/LeWM_for_Maze
uv sync --extra dev
uv run python -m distance_head_study.generate_manifests --role all --check
uv run python -m distance_head_study.audit_protocol --regenerate-manifests
uv run pytest distance_head_study/tests -q
```

所有命令必须从仓库根目录执行。新目录刻意不修改旧 `pyproject.toml`，因为该文件属于
既有 full-900 实验的锁定指纹；从仓库根目录运行时 Python 会直接加载本目录源码，同时
保持历史 protocol lock 可复核。

正式 job 必须在干净且已提交的 worktree 运行。`--allow-dirty-worktree`、
`--diagnostic-limit` 与 `--diagnostic-steps` 只用于 smoke test，生成的 artifact 不得进入
正式 decision、power 或 confirmation analysis。所有 limited/short-run 产物会写入独立的
`distance_head_study_runs/smoke/`，不会占用正式 cache/checkpoint/result 路径。

正式结果加载时会同时验证 metadata signature、checkpoint/candidate-bank/cache-index
hash、manifest、逐 task 身份与 summary 重算。阶段 decision、power 和 analysis 绑定完整
`metadata/rows/summary/manifest/checkpoint` 证据包，而不只绑定 aggregate。Joint 方法的诊断
会用 checkpoint 内更新后的模型重新编码 observation；不会拿旧 frozen cache latent 评价
新 projector。

limited evaluator/diagnostics 与正式运行走相同的 seed 和 sealed-split gate，不能作为提前
解盲入口。TRM 与 trajectory diagnostics 会在按 topology 分组的 batch 中确定性均匀抽取
context；rollout drift 统计 fixed bank 的全部 candidate，不再只取 candidate 0。Executor
只有在输出通过内容级语义验证后才写 completion seal。最终签名 artifact 扁平绑定所有
上游文件 hash，protocol code fingerprint 覆盖全部实际导入的科学计算依赖。

若正路线方法在 `D_select` 的 Seed-3 gate 失败，原 shortlist/finalist 保持不可变；完整
reserve closure 写入独立 `negative_shortlist_lock.json`，再由 fresh confirmation 验证两个
机制不同的 finalist。不得覆盖旧 shortlist 或把新候选补跑到已经打开的 `D_select`。

## 设计修复

本实现对旧 Simple DistanceHead 做了一个必须公开的协议修复：训练 observation、goal
latent 与 BFS target 现在始终指向同一个 manifest goal。旧脚本中 goal marker 与随机
pair label 不一致的采样不进入新排行榜。`baseline_provenance.json` 保存旧配置事实，
旧 checkpoint 只用于 parity，不用于新方法选择。

## 结论边界

成功时，本包最多支持“锁定方法在 corrected/assisted pooled Vector-JEPA 协议下，
相对锁定 baseline 有可复现提升”。失败时，只有完成 `negative_closure_lock`、确认两个
机制不同的 strongest finalists，并用 familywise upper bound 排除 MEI，才能形成
“本协议覆盖的方法族收益有限”的有界负结论。它永远不能证明所有未来 DistanceHead
或所有 JEPA 都不可能成功。
