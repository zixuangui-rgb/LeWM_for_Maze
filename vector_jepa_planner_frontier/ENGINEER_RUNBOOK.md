# 工程执行手册

本文是服务器执行顺序的唯一操作说明。研究定义以 `EXPERIMENT_PROTOCOL.md`、`configs/default.json` 和 `configs/protocol_lock.json` 的共同约束为准。

## 1. 开始条件

在仓库根目录执行全部命令。正式运行必须使用 clean、已 commit 的 worktree；不得在训练过程中修改 Python、配置、manifest、`uv.lock` 或协议文件。

服务器必须先具有旧复现的十个正式 checkpoint：

```text
checkpoints/final_closure/lewm_l2_cem_seqlen2_seed42.pt
...
checkpoints/final_closure/lewm_l2_cem_seqlen2_seed51.pt
```

seed `52-61` 由本包使用与旧训练完全相同的配置补训。不要复制 seed `42-51` 结果充当新 backbone。

## 2. 环境与本地不变量

```bash
uv sync --frozen --extra dev
uv run ruff check vector_jepa_planner_frontier tests
uv run python -m compileall -q vector_jepa_planner_frontier
uv run pytest -q
uv run python -m vector_jepa_planner_frontier.lock_protocol --check
uv run python -m vector_jepa_planner_frontier.smoke_test
uv run python -m vector_jepa_planner_frontier.run_plan --stage audit --execute
```

审计必须至少报告：

- checked-in JSON 为 65 templates，加载后 `method_count=118`；另有 `factorial_method_count=16`、54 个 Track J cells、`frontier_alias_count=9`、`oracle_rung_count=7`；
- train/development/validation/confirmatory 数量分别为 `2800/900/700/900`；
- 任意 split pair 的 topology/layout/task overlap 均为 `0`；
- validation 和 confirmatory manifest 可以逐字节重新生成；
- 旧 B0 config/lock/train-manifest 哈希与兼容契约一致；
- 主动作协议为 `corrected_v1`，checkpoint 规则为 final optimizer step。

任何一项失败都先停止，不得通过删除校验或改 hash 继续。

## 3. 补齐 backbone seeds

```bash
uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage backbones --dry-run

uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage backbones --execute --resume-missing
```

调度器只为缺失的 seed `52-61` 生成训练命令。完成后应有 20 个独立 backbone checkpoint。每个 checkpoint 的实际 SHA256 会在确认冻结时记录。

## 4. P1 Oracle Ladder

```bash
uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage P1 --split-role validation --dry-run

uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage P1 --split-role validation --execute --resume-missing
```

P1 固定生成 `7 rungs × 20 backbones × 2 search seeds × 2 action protocols = 560` 个诊断文件。七个 rung 是：

```text
O0, O1_PROP, O2_SELECT, O3_DYN,
O4_VALUE, O5_JOIN, O6_VALID_FUTURE
```

`O4_VALUE` 的操作定义是：先使用真实 predictor rollout，再把 imagined terminal 以全自由格 latent 最近邻解码到真实 cell，最后读取精确 BFS remaining distance。它不能写成在线非 oracle 方法。`O5_JOIN` 应与 learned bidirectional 方法比较，不与 1x O0 混作等预算主比较。

## 5. P2 搜索与冻结

```bash
uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage P2 --split-role validation --dry-run

uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage P2 --split-role validation --execute --resume-missing

uv run python -m vector_jepa_planner_frontier.summarize \
  --split-role validation --stage P2 \
  --output-dir vector_jepa_planner_frontier_runs/summaries/P2

uv run python -m vector_jepa_planner_frontier.freeze_p2_selection
```

P2 是 5 种搜索器的完整四档预算矩阵，共 1600 个 paired evaluation 文件。
选择只看 validation 的 Corrected-v1；确认集和 size `23/25` 不参与。赢家会
确定性映射到同搜索器的 4x method，并写入 `p2_selection.json`。P3 全部 cell
在运行时继承这份 4x planner spec；不得继续使用模板中的占位 best-first，
也不得在 winner 之外人工挑另一搜索器。

## 6. P3、P4 与 P5 晋级证据

```bash
uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage P3 --split-role validation --execute --resume-missing

uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage P4 --split-role validation --execute --resume-missing

uv run python -m vector_jepa_planner_frontier.summarize \
  --split-role validation --stage P3 \
  --output-dir vector_jepa_planner_frontier_runs/summaries/P3

uv run python -m vector_jepa_planner_frontier.summarize \
  --split-role validation --stage P4 \
  --output-dir vector_jepa_planner_frontier_runs/summaries/P4
```

从 `protocol/P5_ADVANCEMENT_EVIDENCE_TEMPLATE.json` 建立一份运行目录中的证据文件，例如：

```text
vector_jepa_planner_frontier_runs/decisions/p5_evidence.json
```

指定 reviewer，并为 verifier、reachability、proposal、memory 以及
vector-DTS、bidirectional、denoising 三个 radical 的六项 gate 填入可追溯的
表格/行号。六项为：机制改善、overall 非劣、size-19/21 非劣、等计算、负对照
通过、方向一致。

`selected_components` 必须按固定顺序列出所有且仅列出六项全通过的组件。
`selected_radical` 最多一个：没有 radical 全通过时填 `null`；有一个时选它；
多个时必须按 corrected overall SR 距最大值 0.01、size-19/21 SR 距最大值
0.01、radical 名称字典序的规则重算。`radical_decision_reason` 始终必填。
建议由第一位 reviewer 填证据、第二位 reviewer 逐表复核；程序验证完整性和
可复算选择，但不会自动判断 reviewer 的科学解释是否成立。

```bash
uv run python -m vector_jepa_planner_frontier.freeze_p5_advancement \
  --p3-summary vector_jepa_planner_frontier_runs/summaries/P3/summary.json \
  --p4-summary vector_jepa_planner_frontier_runs/summaries/P4/summary.json \
  --evidence vector_jepa_planner_frontier_runs/decisions/p5_evidence.json
```

未通过的组件或 radical 不晋级，其他通过者仍可组成 P5。只有四组件与三
radical 均无通过者时，P5 门控才按设计停止。不得为了保留某模块把模板中的
`false` 手工改为 `true`。

## 7. P5、P6、P7

```bash
uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage P5 --split-role validation --execute --resume-missing

uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage P6 --split-role validation --execute --resume-missing

uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage P7 --split-role validation --execute --resume-missing
```

继承链必须是：

```text
P2 winner planner + P5 selected P3-cell final checkpoint
    + optional selected P4 radical final checkpoint
-> P5: 0 new optimizer steps
-> P6: ranker only, exactly M1/M2/M3
-> P7 Track J 54-cell grid: joint update from P6 hard round 3
```

`run_plan --stage P5` 调用 `assemble_p5.py`，逐 tensor 合并所需 head。P3 source
优先拥有同名 head，radical 只补齐缺失 head；输出必须保存 parent hashes 与
`head_ownership`。P6 hard 与 random 两支都先运行匹配的 30k ranker initial
training，再运行三轮各 20k；random 对照只替换 negative action sequence。

P7 同时评估 `p7_control_action_aligned_frozen` 和 54 个 Track J cell。每个
Track J cell 使用相同 P6 hard round-3 parent、T=8 trajectory protocol、30k
final-step budget 和三轮冻结 hard-negative datasets，只改变预登记的四维网格。
完成全部 P7 训练与评估后立即冻结赢家：

```bash
uv run python -m vector_jepa_planner_frontier.summarize \
  --split-role validation --stage P7 \
  --output-dir vector_jepa_planner_frontier_runs/summaries/P7

uv run python -m vector_jepa_planner_frontier.freeze_p7_selection
```

冻结器要求一个 cell 的 40 个 checkpoint 全部通过 10% JEPA stability gate，
再用 SR 近优和 JEPA 恶化 tie-break 选一个赢家。无稳定 cell 时写入
`selected_track_j=null`；这会关闭 P8 的 Track J aliases，但不会关闭 Track F。

Frozen control 直接复用 P6 round-3 checkpoint，训练 0 步，仅把 rollout 从
`legacy_warmup_v1` 改为 `action_aligned_v2`。因此：

- P6 vs aligned control：rollout 语义效应；
- aligned control vs Track J：同 rollout 下的参数联合更新效应；
- P6 vs Track J：完整 P7 package 的总效应。

P7 赢家存在时，可生成包含三者的专门汇总；先读取真实 winner，不能默认
canonical template 就是赢家：

```bash
P7=vector_jepa_planner_frontier_runs/decisions/p7_selection.json
JOINT=$(jq -r .selected_track_j "$P7")

uv run python -m vector_jepa_planner_frontier.summarize \
  --split-role validation \
  --methods p6_track_f_counterexample_ranked,p7_control_action_aligned_frozen,"$JOINT" \
  --output-dir vector_jepa_planner_frontier_runs/summaries/P7_attribution
```

## 8. P8 Compute Frontier

```bash
uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage P8 --split-role validation --dry-run

uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage P8 --split-role validation --execute --resume-missing

uv run python -m vector_jepa_planner_frontier.freeze_p8_selection
```

P8 的 9 个模板只改变 planner budget，分别指向 P5/P6/P7 checkpoint；调度器
为它们生成 0 个 train/calibrate/mining job。每个 finalist 的 `4x` 点复用原阶段
validation 结果，不重复运行。若 P7 没有赢家，三个 Track J aliases 被关闭，
只运行 6 个 Track F aliases；不要创建空白或伪 Track J 输出补齐矩阵。

冻结规则：先在共同 `4x` 下选择 P5 或 P6 Track F family；差值不超过
`0.01` 时选实际 plan transitions 更少者。随后在该 family 的四档预算中选择
距离最高 SR 不超过 `0.01` 的最小预算。P7 赢家使用同一预算规则，并只有在
全部 40 个 checkpoint 的 JEPA 稳定性 gate 再验证通过且 SR 对 Track F 非劣
`0.01` 时晋级。

## 9. 功效与确认冻结

读取 P8 决策：

```bash
P8=vector_jepa_planner_frontier_runs/decisions/p8_selection.json
CANDIDATE=$(jq -r .selected_track_f "$P8")
K=$(jq -r .comparison_count "$P8")

uv run python -m vector_jepa_planner_frontier.power_analysis \
  --candidate "$CANDIDATE" --comparison-count "$K" \
  --output vector_jepa_planner_frontier_runs/confirmatory_power.json
```

功效分析只使用前 8 个 validation backbone 的配对差异；先平均 search seed，
再平均 planner seed，最后在 backbone 层估计方差。若
`claim_status=exploratory_only`，必须在打开确认集前停止，不得用不足种子运行
确认性主张。

功效通过后冻结所有 source/component hashes、方法、seed 和 opaque schedule：

```bash
uv run python -m vector_jepa_planner_frontier.freeze_confirmation
```

K=2 时确认方法是 B0 + Track F，共 240 runs；K=4 时再加入 Track J，共
400 runs。这里的 K 是 overall/OOD 两个或四个主 contrasts，不是方法数量。
`freeze_confirmation` 会再次验证 P5 tensor ownership、各组件实际 optimizer
steps、P6 三轮 dataset/checkpoint chain、P7 赢家的训练 budget/完整 model state/
hard-negative provenance、retrieval bank 和所有 source SHA256；任何一项不符
都不能生成 opaque schedule。

## 10. 一次性确认、解盲与汇总

```bash
uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage confirmatory --dry-run

uv run python -m vector_jepa_planner_frontier.run_plan \
  --stage confirmatory --execute --resume-missing

uv run python -m vector_jepa_planner_frontier.unblind_confirmation

uv run python -m vector_jepa_planner_frontier.summarize \
  --split-role confirmatory \
  --output-dir vector_jepa_planner_frontier_runs/summaries/confirmatory
```

确认调度只显示 opaque run ID。`unblind_confirmation` 会先验证全家族结果和 candidate-trace hash；任一文件缺失时不发布任何具名结果。全部通过后才一次性解盲并写 marker。汇总器在 marker 之前拒绝运行。

## 11. 中断与重跑

- `--resume-missing` 只跳过完整输出；检测到一组多文件工件只存在一部分时会失败。
- 不得删除低分结果或改变 seed 后重跑。
- 允许的重跑原因只有：`interrupted_execution`、`missing_or_duplicate_task`、`manifest_checkpoint_or_code_hash_mismatch`、`non_finite_output`。
- 需要覆盖时必须调用具体子命令的 `--overwrite --rerun-reason <枚举值>`，保留旧工件哈希和基础设施日志；不要给整个阶段无差别覆盖。
- Confirmatory 打开后只允许完成原 schedule，不允许修改方法、预算、阈值、样本量或比较 family。

## 12. 交付物

最终交付至少包括 protocol/audit、所有 stage schedule、P2/P5/P8 决策、功效记录、confirmation lock/mapping/schedule、opened/unblinded marker、具名 task-level 结果、candidate traces、所有 summary CSV/JSON/REPORT、checkpoint 哈希清单和服务器环境日志。
