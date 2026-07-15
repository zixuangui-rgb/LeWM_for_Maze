# 与历史复现的兼容契约

## 来源

- Source config：`final_closure/configs/default.json`
- Source lock：`final_closure/configs/protocol_lock.json`
- Baseline name：`lewm_l2_cem_seqlen2`
- Checkpoints：`checkpoints/final_closure/lewm_l2_cem_seqlen2_seed{42..51}.pt`
- Evaluation manifest：`data/splits/unisize_eval_manifest.jsonl`

本实验不修改上述文件，也不重签旧protocol lock。新代码通过独立协议锁管理。

## 逐字段一致性

| 字段 | 历史值 | 本实验 |
|---|---|---|
| encoder/projector/predictor结构 | `Unisize256` LeWM | 同一checkpoint对象 |
| backbone参数 | seed42-51 final-step | 原样加载、完全冻结 |
| history | 3 | 3 |
| context action初始化 | action ID 4重复 | 相同 |
| rollout | legacy warmup | 相同 |
| horizon | 12 | 12 |
| candidates/elites/iters | 64/8/1 | B0完全相同；其他方法受同1x预算 |
| action IDs | 1-4 | 1-4 |
| max steps | 128 | 128 |
| task RNG | eval seed42 + task index + step | 相同 |
| replan | 每步 | 每步 |
| terminal score | squared latent-L2 | Q1相同；Q2只改变声明的模块 |
| corrected fallback | 旧one-step latent-L2 | 复用同一实现 |

## Q0必要条件

旧 `LeWMCEMController` 与新 `LegacyCEMPlanner` 必须在seed42、900任务、两个action
protocol上逐任务一致。比较字段包括：task、成功、路径长度、invalid、repeat、
max visits、loop/cycle、final BFS distance以及完整executed-action序列。Q0不过，
Q1-Q4全部禁止运行。

categorical-CEM bridge还必须与新B0在相同任务上逐任务一致。它只解决旧B0实现不
消费可插拔scorer/proposal的问题，不可作为额外性能方法宣传。

## 历史数字的使用

历史development结果约为unmasked SR 0.211、corrected SR 0.639。它们只作为实现
锚点。seed42单checkpoint分数不得直接与历史10-seed均值作差。所有方法选择使用
同backbone、同task的本次配对B0；只有Q4补齐seeds42-51后才报告与历史10-seed
口径一致的聚合结果。

## 权重边界

Git仓库不包含大体积checkpoint。本地静态测试只能验证路径、schema和控制逻辑。
服务器正式运行前，`audit_protocol --require-checkpoints`必须确认10个历史权重均
存在；加载器还会验证seed、模型结构、source config/lock和checkpoint provenance。

## 依赖边界

新包从仓库根目录运行，复用 `hdwm`、`final_closure` 和
`vector_jepa_planner_frontier` 的底层算法实现。它不修改 `pyproject.toml`，避免改变
旧协议代码指纹。所有命令必须在仓库根目录通过 `python -m` 或 `uv run python -m`
执行。
