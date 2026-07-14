# 与旧复现实验的兼容契约

## 1. 唯一源基线

本实验的 pooled-vector B0 只允许来自：

- `final_closure/configs/default.json` 中的 `lewm_l2_cem_seqlen2`；
- `final_closure/configs/protocol_lock.json`；
- `data/splits/unisize_train_manifest.jsonl`；
- `checkpoints/final_closure/lewm_l2_cem_seqlen2_seed{42..61}.pt`。

seed `42-51` 是旧正式复现；seed `52-61` 由 `train_source_backbone.py` 使用相同 architecture、train manifest、optimizer、steps 和 loss 权重补齐。新包不接受换名 checkpoint、手工转换 state dict 或复制旧 seed。

`compat.py` 会重新读取旧 config/lock，逐字段验证 source baseline 和 checkpoint provenance。协议锁保存旧 config、lock、train manifest 的 SHA256。

## 2. B0 逐项映射

| 旧实现 | 新包 B0 | 要求 |
|---|---|---|
| `Unisize256` | 从 source checkpoint 原样构造 | state dict strict load |
| encoder/projector | `VectorWorldModel.encode` | 相同 dtype、shape、size conditioning |
| context | 起点 frame 重复三次，context action id `4` | 相同 |
| predictor rollout | `legacy_warmup_v1` | 保留历史 off-by-one 语义 |
| CEM | 直接调用 `hdwm.planning.cem_plan` | 相同函数，不重写 |
| RNG | `task_seed(eval_seed, task_index, step)` | 相同 |
| action | `[1,2,3,4]` | 相同 |
| score | terminal pooled latent squared L2 | 相同 |
| replanning | 每个真实环境步重新规划 | 相同 |
| horizon/candidates/elites/iters | `12/64/8/1` | 相同 |
| max steps | `128` | 相同 |

测试覆盖三层 parity：adapter 对旧 `_latent_rollout_cost`、planner sequence/cost 对旧 `cem_plan`、真实 manifest maze 的新旧 controller 首动作。

## 3. 历史 rollout 与动作对齐

旧 rollout 会先做一次 warmup prediction，再把候选 action 放入 history，因此
长度 H 的最后一个候选 action 不影响 terminal score。B0 和 P2-P6 均显式
标记为 `legacy_warmup_v1`。P3-P6 的 planner kind 不再静态假定为 best-first，
而是从 `p2_selection.json` 继承已登记的 4x 赢家；rollout semantics 仍保持
legacy，因此该派生不会顺带修复动作对齐。

`action_aligned_v2` 先放入当前候选 action，再预测 successor。它只在 P7 Track J 和 `p7_control_action_aligned_frozen` 使用。P7 control 复用完全相同的 P6 checkpoint，因此可以单独估计语义变化；没有该对照时，禁止把 P6->P7 总差写成纯联合训练收益。

## 4. Corrected-v1 一致性

唯一 assistance 实现复用 `final_closure.common.corrected_actions`：

1. 从当前真实 state 枚举四动作并去掉 wall/no-move；
2. 若存在非 immediate-backtrack action，则去掉 immediate backtrack；
3. planner 首动作不在允许集合时，才运行旧五动作 one-step latent-L2 fallback；
4. fallback 只在允许动作中选最小 cost。

未触发时 corrected 与 unmasked 执行同一 planner action。fallback 的 predictor transitions 只进入 `B_assist`，不能藏进 `B_plan`。Corrected-v1 是确认性主协议；unmasked 使用同 task/backbone/planner/search seed 严格配对运行。

## 5. 表示与参数边界

| 方法族 | encoder/projector/predictor 结构 | 参数 | 可新增内容 |
|---|---|---|---|
| B0/P2 | 不变 | frozen | search only |
| P3/P4/P5/P6 | 不变 | frozen | planner heads/proposal/memory/search |
| P7 frozen control | 不变 | frozen | rollout semantics only |
| P7 Track J | 不变 | joint update | 同一 planner heads + JEPA losses |
| P8 aliases | 不变 | 复用 source checkpoint | budget only |

因此“固定 Vector-JEPA 表示下的 planner 增益”只适用于 Track F。Track J 结果必须写成“结构固定、参数联合更新”。

## 6. 动态方法与 checkpoint 复用契约

JSON 文件中有 65 个模板；Pydantic schema 把 Track J 模板展开为 54 个联合
训练 cell，形成 118 个有效配置。任何正式 result/checkpoint 都保存运行时
effective method spec，而不只保存原模板。以下冻结决策会进入 effective spec
的 SHA256 链：

- P2：P3-P7 继承的 4x search planner；
- P5：通过 gate 的组件、所选 P3 cell 和可选 P4 radical；
- P7：54-cell Track J 网格的唯一赢家或明确失败；
- P8：Track F family、预算和可选 Track J admission。

P5 是确定性组装，不是重新训练。所选 P3 source 的同名 head 优先；可选
radical 只补齐缺失 head。`initialization_parents`、`head_ownership` 和逐 tensor
相等性在 confirmation freeze 时重验。

P8 不产生训练工件。`component_checkpoint_owner` 把每个 alias 解析回 P5/P6/P7 source；evaluator 使用 source method 的 `training_spec_sha256` 验证 checkpoint，同时在结果 metadata 中记录 alias MethodConfig 和真实 owner。

Schema 会拒绝 P8 alias 改变 scorer、proposal、memory、control、track、rollout semantics 或 planner 的其他字段。P7 frozen control同理，只允许从 P6 的 `legacy_warmup_v1` 切换到 `action_aligned_v2`。

## 7. 科学对齐表

所有主比较必须共享 topology hash、start/goal、max steps、动作空间、action protocol、backbone seed 和 search seed。learned planner 还共享 planner-head seed。

| 比较 | 必须固定 | 允许变化 |
|---|---|---|
| P2 search | latent-L2、uniform proposal、budget | search algorithm |
| P3 factorial | P2 winner、4x、training data | V/R/P/M 因子 |
| P4 matched control | family、4x、训练预算 | 指定 radical mechanism |
| P6 hard vs random | parent、root/goal/positive、ranker结构、初始和三轮 steps | negative action sequence |
| P6 vs P7 control | P6 checkpoint、4x | rollout semantics |
| P7 grid cells | parent、T=8、steps、数据、4x | 四个预注册 joint hyperparameters |
| P7 control vs Track J | action-aligned、结构、4x | parameter updates/loss |
| P8 family | checkpoint、所有 planner字段 | transition cap |
| confirmatory | frozen method、checkpoint、seeds | task topology/size |

实际 plan transitions、assistance transitions、wall time、forward calls、max batch 和 node expansions 均须报告。理论 cap 不能替代实际计算量。

## 8. 明确不兼容的更改

以下更改会创建新实验，不得混入当前 protocol ID：spatial latent、真实地图/坐标输入、在线 BFS、不同 encoder/projector 结构、不同 train topology、改变 action set、改变 Corrected-v1、改变 max steps、删除历史 warmup、根据 validation 分数追加 seed/steps/round，或在确认集打开后更改任何方法。
