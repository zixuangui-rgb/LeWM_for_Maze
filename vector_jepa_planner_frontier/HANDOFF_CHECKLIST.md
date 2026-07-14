# 工程师交接与双人复核清单

## A. 接收与环境

- [ ] 阅读 `README.zh.md`、`EXPERIMENT_PROTOCOL.md`、`ENGINEER_RUNBOOK.md`、`COMPATIBILITY.md`、`CLAIMS_AND_STOP_RULES.md`。
- [ ] 当前 Git branch/commit 已记录，正式运行 worktree clean。
- [ ] `uv sync --frozen --extra dev` 成功，Python/Torch/CUDA/driver 写入运行日志。
- [ ] Ruff、compileall、全仓 pytest、protocol lock check、smoke test 全部通过。
- [ ] seed `42-51` 的十个旧 B0 checkpoint 路径和 hash 可读。
- [ ] `backbones` 阶段只补训 seed `52-61`，最终共有 20 个不同 checkpoint。

## B. 协议与数据

- [ ] protocol audit 报告 65 checked-in templates、118 effective methods、16 factorial cells、54 Track J cells、9 P8 aliases、7 oracle rungs。
- [ ] train/development/validation/confirmatory 数量为 `2800/900/700/900`。
- [ ] 四组 topology/layout/task overlap 全为 0。
- [ ] validation/confirmatory manifest 与生成器逐字节一致。
- [ ] Confirmatory size `23/25` 未进入训练、retrieval、calibration、mining、P2/P5/P8 选择或功效方差估计。
- [ ] 主终点明确为 Corrected-v1；unmasked 是配对次要诊断。

## C. 调度与训练

- [ ] 每个阶段先审查 dry-run；实际 schedule 已冻结且以 backbone 为 block 随机化。
- [ ] P1 共有 560 个独立 oracle 输出，均位于 oracle 目录并标记不进主表。
- [ ] P2 完整运行 5×4 budget matrix；B0 没有伪 planner replicate。
- [ ] P2 selection 在 P3 前冻结；P3 全部 cell 的 effective planner 精确继承赢家的 4x spec。
- [ ] P3 16 factorial cells 和 5 controls 在每个 backbone block 内完整。
- [ ] Retrieval bank 的每个 task hash 均属于 train manifest，fingerprint 与 calibration 一致。
- [ ] Head 梯度只来自 train topology；validation 无 optimizer step。
- [ ] Hard memory 的 join precision 达到 `0.95`；未达时对应 hard variant 停止。
- [ ] P5 evidence 有 reviewer、四组件和三 radical 的六项逐项证据、表格/行号和 P3/P4 summary hashes。
- [ ] `selected_components` 精确等于通过全部 gate 的组件；radical 最多一个且能由预注册 tie-break 重算。
- [ ] P5 从所选 P3 cell 和可选 P4 radical 确定性组装，新增 0 steps；parent hashes、head ownership 和 tensor equality 全部通过。
- [ ] P6 hard/random 两支都有匹配的 30k initial ranker training 和三轮各 20k 更新；root/goal/positive 相同，只有 negative actions 不同。
- [ ] P6 M1/M2/M3 topology-disjoint；hard dataset/checkpoint/parent/fold/action 的三轮 provenance 连续且无 round 4。
- [ ] P7 frozen aligned control 与 P6 component SHA256 相同且生成 0 个训练任务。
- [ ] P7 54 个 Track J cell 完整，均从 P6 hard round 3 初始化，使用相同 T=8 数据协议和 30k final-step budget。
- [ ] Track J ranker 只消费已冻结的 P6 三轮 hard-negative datasets，没有退回 random negatives。
- [ ] P7 选择能从 54-cell validation/stability 工件重算；无稳定 cell 时 Track J 明确 fail closed。
- [ ] P7 Track J 保存 full model state、三轮 counterexample provenance 和 matched T=8 JEPA stability metrics。
- [ ] P8 九个 aliases 生成 0 个 train/calibrate/mining job，只改变 transition cap。

## D. 结果完整性

- [ ] 每个正式方法都有相同 task/backbone/search seed 的 corrected/unmasked 两套结果。
- [ ] 每个 learned planner 有两个 planner seeds；B0 只有 metadata `planner_seed=null`。
- [ ] 每个 evaluation 同时存在 result JSON 和 candidate-trace JSONL。
- [ ] Candidate sample 在每个 size 内精确约 10%，两遍 replay 的 proposed/executed actions 完全一致。
- [ ] Candidate truth/rescore 标记为 analysis-only，compute ledger 未计入事后诊断。
- [ ] `B_plan`、`B_assist`、`B_total`、forward calls、max batch、node expansions、wall time 完整且有限。
- [ ] Result/checkpoint 的 analysis spec、training spec、source/parent hash、code fingerprint、Git dirty 校验通过。
- [ ] Summary 缺失文件为 0；描述性平均顺序是 search -> planner -> backbone，CI bootstrap 为 backbone -> planner -> size-stratified task。

## E. 阶段冻结

- [ ] `p2_selection.json`、`p5_advancement.json`、`p7_selection.json`、`p8_selection.json` 均只写一次且 gate 可重复验证。
- [ ] P8 在 Track J 失败时覆盖 8 个、成功时覆盖 12 个四档 frontier 点，并保存所有输入 artifact digest。
- [ ] P7 赢家的 40 个 checkpoint stability gate 在 P8 和 confirmation freeze 时全部重验；失败时 K=2，不替换 checkpoint。
- [ ] Power record 使用前 8 个 validation backbones，candidate/K 与 P8 完全一致。
- [ ] 若 required backbones 超过 20，项目在确认集打开前标记 exploratory 并停止。

## F. 一次性确认

- [ ] Freeze confirmation 前具名/opaque confirmatory result 均不存在。
- [ ] Confirmation lock 保存 P8/power/source/component/mapping/schedule hashes。
- [ ] Private mapping 权限为 `0600`；公开 schedule 不含方法名。
- [ ] K=2 时 run count 为 240；K=4 时 run count 为 400。
- [ ] 执行期间只查看完成率、退出码、非有限值和资源信息，不汇总中途 SR。
- [ ] 所有 opaque outputs 完整后才运行 unblind；任何缺失时不产生具名结果。
- [ ] Unblinded marker、具名结果和 candidate hashes 全部通过后才运行 confirmatory summary。
- [ ] Nested Bonferroni simultaneous CI、backbone exact sign-flip p 和 Holm family correction 均在主表中。
- [ ] 主表样本数字段明确区分 backbone、planner seed、unique task、nested rows 和 run count，没有把 task 当独立 seed。

## G. 结题

- [ ] 报告明确区分 Track F、P7 aligned control、Track J、Corrected assistance 和 oracle。
- [ ] P7 joint-only 归因使用 aligned control，不把完整 H2 总差全归因于 joint training。
- [ ] 负结果和 gate failure 同样报告，不删除、不追加 seed/round/budget。
- [ ] 服务器日志、schedule、hash 清单、decision artifacts、所有 CSV/JSON/REPORT 一并归档。
- [ ] 无论结果正负，按 `CLAIMS_AND_STOP_RULES.md` 关闭当前 protocol ID。
