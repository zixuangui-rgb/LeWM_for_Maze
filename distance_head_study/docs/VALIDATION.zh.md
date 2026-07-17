# 验证与审计层级

本包采用四层检查。`VALIDATION` 只记录代码审计，不声称 server GPU full run 已完成。

## L1 静态与 schema

- `compileall`；
- Ruff E/F/I/UP/B；
- Pydantic `extra=forbid`；
- 45 个方法目录解析；
- static/dynamic parent cycle、alias、scientific diff 检查；
- target transform float64 round-trip；
- README 中英文同步检查。

## L2 单元与数值

- scalar/ordinal/distribution/multitask/quasimetric output shape/finite；
- quasimetric diagonal为零、非负与 triangle；
- tie-aware listwise 支持多个 optimal actions；
- Bellman/multistep/trajectory/reachability loss 梯度有限；
- sampler 同 seed/step byte-identical，不同 method 共用 schedule；
- candidate bank deterministic；
- crossed bootstrap 对每个 replicate 的所有 backbone 使用同一 task draw，并拒绝不一致
  task shape/maze-size strata；
- exact/Monte-Carlo backbone sign-flip、Holm/Bonferroni 与 MEI gate；
- shortest-path label 到 goal 后保持吸收态；
- compatible parent initialization 只加载 shape-matched shared keys。
- fresh backbone 在模型构造前复现 source entrypoint 的 Torch/NumPy/Python/CUDA seed 与
  deterministic 设置，相同 seed 的初始化 byte-identical；
- hierarchical head 缺 horizon、quasimetric 声明但忽略 horizon 等无效 schema 均拒绝；
- horizon-conditioned head 的 true-next/predicted-next 局部动作目标显式查询 `horizon=1`；
- TRM context 固定等距覆盖 topology blocks，max-distance label 与所选 row 对齐；
- drift 使用全部 candidate terminals，模型自由 true latent 与 predictor latent 的 domain
  adapter 标志明确分开。

## L3 合成集成

- 小型 synthetic cache shard -> sample -> head loss -> optimizer step；
- tiny LeWM checkpoint -> cache build -> predicted objective；
- task-row 必须逐项匹配 manifest，shard merge、错 task 与重复 task 均拒绝；
- signed artifact tampering 拒绝；
- seed release/select/confirm gate fail closed；
- formal/limited evaluator 与 diagnostics 共享同一 seed/split gate；
- interrupted evaluator/trainer resume spec mismatch 拒绝；trainer RNG round-trip；
- job DAG cycle/shared-output 拒绝，state/completion/output hash 三重绑定；
- 进程返回 0 后仍执行 artifact 内容级 validator，语义错误不得生成 completion seal；
- evidence hash flattening 检查冲突，protocol fingerprint 覆盖 transitive scientific code；
- dynamic method 产物重新解析 signed decision，并核对完整 resolved spec、method hash 与
  decision hash；historical/fresh backbone 分别核对原实验或本实验的训练 spec、manifest、
  final-step/model payload；cache shard 核对 topology/goal/max-distance 元数据；
- result validator 从 row 重演 corrected/unmasked assistance、fallback 和各 planner 的实际
  predictor transition 预算，防止只凭 aggregate 指标接受语义错误的结果；
- historical CEM 与 instrumented terminal CEM 的 candidate/sequence/cost parity。
- iCEM/Beam/Best-first 对相同 seed 确定性，实际 transitions 不超过 768 硬上限；iCEM
  精确用满，branch search 可记录未用的不可分支余量。

## L4 真实本地 smoke

- 使用真实 seed-42 checkpoint 的少量 topology cache；
- 运行 `server_preflight` 的真实 checkpoint I/O/cache/predictor/loss/rollout；
- 在 `distance_head_study_runs/smoke/` 做至少一个 short head train 与 limited evaluator；
- 需要测试中断时，确认 trainer RNG 恢复及 evaluator rows 与 uninterrupted run 一致；
- manifest 全量 regenerate 与 overlap audit。

limited evaluator/diagnostics 在服务器 smoke 时仍需先创建对应 seed release；sealed split
还必须已有 shortlist/confirm-open/closure gate。不要为了 preflight 使用门禁旁路。

L4 需要仓库外 checkpoint；本地缺失时明确记录 `not run: missing checkpoint`，不能用合成
测试冒充真实兼容性。工程师在服务器完成后保存 audit artifact 与日志 hash。

## 正式开始条件

```text
L1 pass
L2 pass
L3 pass
L4 real-checkpoint smoke pass on server
protocol lock matches committed code/config/manifests
seed1 release exists
```

任一项失败都停止下游 release。
