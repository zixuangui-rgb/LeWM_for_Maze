# 工程执行手册

本手册给拿到仓库代码与服务器 checkpoint 的工程师。不要从中挑选部分命令后直接跑
full-900；阶段门禁是实验设计的一部分。

## 0. 环境与人工核实

```bash
cd /path/to/LeWM_for_Maze
git status --short
uv sync --extra dev
uv run python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

后续全部命令均在仓库根目录执行。不要为了安装 `distance_head_study` 修改
`pyproject.toml`：该文件已被旧 full-900 protocol lock 指纹化；根目录执行会直接导入新目录，
并保持旧对照锁不变。

先逐项完成 [CHECK_REQUIRED.zh.md](CHECK_REQUIRED.zh.md)。尤其确认：

- 服务器有 `final_closure/lewm_l2_cem_seqlen2_seed42.pt` 至少到 seed 44；
- checkpoint metadata/model config 与 source protocol lock 一致；
- 四张 H800 在同一 runtime block，驱动/CUDA/PyTorch 一致；
- 正式运行使用干净、已提交的 commit；
- scratch/output 路径容量足够，不把 cache 写入临时会自动清理的位置。

## 1. P0 全审计

仓库已提交 manifests/bootstrap schedule/protocol lock。不得重新写，只检查：

```bash
uv run python -m distance_head_study.generate_manifests --role all --check
uv run python -m distance_head_study.audit_protocol --regenerate-manifests
uv run pytest distance_head_study/tests -q
uv run ruff check distance_head_study
uv run python -m distance_head_study.server_preflight \
  --backbone-seed 42 --device cuda
```

最后一条会读取真实 checkpoint，完成 model I/O、单 topology cache、all-action predictor、
head backward 和 legacy rollout。审计失败时停止，不得加 `--allow-dirty-worktree` 跑正式
任务。

## 2. Seed-1 释放与 P0/P1

```bash
uv run python -m distance_head_study.release_seed_tier --tier seed1
```

先跑 baselines 与 oracles。可用 DAG：

```bash
uv run python -m distance_head_study.plan_jobs \
  --phase seed1 \
  --methods b_dh_model_free,b_dh_predictor_greedy,o_dyn_true_rollout,o_score_true_bfs,o_bfs1 \
  --output distance_head_study_runs/jobs/seed1_p01.jsonl

uv run python -m distance_head_study.run_jobs \
  --jobs distance_head_study_runs/jobs/seed1_p01.jsonl \
  --gpus 0,1,2,3 \
  --state distance_head_study_runs/jobs/seed1_p01_state.json \
  --logs distance_head_study_runs/logs/seed1_p01
```

`plan_jobs` 同时生成 `<plan>.metadata.json`，其中绑定 protocol lock、config、seed
release、job IDs 与 JSONL hash。`run_jobs.py` 是单机四卡 executor；每个成功 job 还会
先按 artifact 类型做内容级验证，再写 output-hash completion seal。进程 exit code 0 但
输出缺失、schema/hash/provenance 不一致时仍标记 failed。若服务器使用 Slurm/Kubernetes，
按照 job JSONL 中的
`command/dependencies/outputs` 转写 launcher，保留 metadata 并自行记录同等 completion
hash 与内容级验收，不能改 scientific arguments。

## 3. Block A-D 顺序

每一 block 都先生成 DAG、跑完、diagnose，再创建签名 decision。动态 parent 必须先有
decision artifact，后续 method 才能 resolve。

### A target

运行 `b_dh_cem,a1_log` 后：

```bash
uv run python -m distance_head_study.make_decision \
  --name a_target_parent --criterion diagnostic \
  --eligible b_dh_cem,a1_log \
  --split-role screen --backbone-seeds 42 --head-seeds 0,1,2
```

### A sampling

运行 `a2_distance_balanced,a3_full_horizon` 后：

```bash
uv run python -m distance_head_study.make_decision \
  --name a_sampling_parent --criterion diagnostic \
  --eligible a2_distance_balanced,a3_full_horizon \
  --split-role screen --backbone-seeds 42 --head-seeds 0,1,2
```

### B structure/factorial

先跑 `b1_listwise,b2_bellman,b3_multistep`：

```bash
uv run python -m distance_head_study.make_decision \
  --name b_structural_winner --criterion diagnostic \
  --eligible b2_bellman,b3_multistep \
  --split-role screen --backbone-seeds 42 --head-seeds 0,1,2
```

再跑 `b5_local_structural`，对 parent/local/structural/combined 做 screen decision：

```bash
uv run python -m distance_head_study.make_decision \
  --name b_parent --criterion screen \
  --eligible b1_listwise,b2_bellman,b3_multistep,b5_local_structural \
  --split-role screen --backbone-seeds 42 --head-seeds 0,1,2
```

### C/D

运行 `c1_predicted_listwise,c2_dual_calibration`。二者必须都跑；否则固定 eligible set
不完整，不能创建 `c_parent`。然后：

```bash
uv run python -m distance_head_study.make_decision \
  --name c_parent --criterion screen \
  --eligible c1_predicted_listwise,c2_dual_calibration \
  --split-role screen --backbone-seeds 42 --head-seeds 0,1,2
```

运行 `d1_trm_short,d2_trm_full,d3_trm_shuffle,d4_reachability`。`d2` 只有同时超过
`d1/d3` 的 candidate ordering 才能归因 full-horizon supervision。

## 4. 锁定 shortlist

所有 11 个预注册主线候选都完成后创建 screen selection。`@main` 是代码内固定别名，
不能改成手工挑选的子集：

```bash
uv run python -m distance_head_study.make_decision \
  --name screen_selection --criterion screen \
  --eligible @main \
  --split-role screen --backbone-seeds 42 --head-seeds 0,1,2
```

有 ordinary-gate 候选：

```bash
uv run python -m distance_head_study.lock_shortlist \
  --screen-decision distance_head_study_runs/decisions/screen_selection.json
```

若主线没有 ordinary-gate 候选，不能直接锁空 shortlist。必须完成整个负向闭环：

```bash
uv run python -m distance_head_study.plan_jobs \
  --phase seed1 --methods @negative_closure \
  --output distance_head_study_runs/jobs/seed1_negative_closure.jsonl

uv run python -m distance_head_study.run_jobs \
  --jobs distance_head_study_runs/jobs/seed1_negative_closure.jsonl \
  --gpus 0,1,2,3 \
  --state distance_head_study_runs/jobs/seed1_negative_closure_state.json \
  --logs distance_head_study_runs/logs/seed1_negative_closure

uv run python -m distance_head_study.make_decision \
  --name closure_selection --criterion screen \
  --eligible @negative_closure \
  --split-role screen --backbone-seeds 42 --head-seeds 0,1,2

uv run python -m distance_head_study.lock_negative_closure \
  --screen-decision distance_head_study_runs/decisions/closure_selection.json

uv run python -m distance_head_study.lock_shortlist \
  --screen-decision distance_head_study_runs/decisions/closure_selection.json \
  --negative-closure-artifact \
    distance_head_study_runs/decisions/negative_closure_lock.json
```

`@negative_closure` 的 run set 和 candidate set 分别由代码集中定义；前者额外包含 matched
controls，后者只包含可成为 finalist 的方法。不得手工删减。负向闭环 artifact 会绑定每个
diagnostic、result、manifest、checkpoint、cache 与 candidate bank 的 hash。

## 5. Seed-3 与 D_select

```bash
uv run python -m distance_head_study.release_seed_tier --tier seed3
uv run python -m distance_head_study.plan_jobs \
  --phase seed3 --output distance_head_study_runs/jobs/seed3.jsonl
```

运行 DAG 后，用 shortlist 中最多两个方法创建：

```bash
uv run python -m distance_head_study.make_decision \
  --name finalist_lock --criterion select \
  --eligible <locked-shortlist> \
  --split-role select --backbone-seeds 42,43,44 --head-seeds 0,1
```

每 backbone 先平均两个 head seeds；不要把六个值当六个模型。

若初始 shortlist 有 ordinary-gate 候选，但 Seed-3 finalist 没通过扩大门槛，严格负结论
不能覆盖原 shortlist，也不能跳过 reserve。此时完成上一节的 negative-closure jobs、
`closure_selection` 和 `lock_negative_closure`，然后另写一个 fallback lock：

```bash
uv run python -m distance_head_study.lock_shortlist \
  --negative-fallback \
  --screen-decision distance_head_study_runs/decisions/closure_selection.json \
  --negative-closure-artifact \
    distance_head_study_runs/decisions/negative_closure_lock.json
```

该命令只允许在 `finalist_lock` 明确失败后执行，输出
`negative_shortlist_lock.json`，并同时绑定原 shortlist 与原 finalist。原文件保持不变。

## 6. Baseline-only power 与 confirmation n

Seed-3 DAG 会额外生成 backbone `42/43/44` 上三组 `legacy / b_dh_cem /
corrected_v1` baseline-only full-900 rows。这三组只估计 backbone 间方差，不进入候选排序，
也不读取 finalist effect。Power 输入只允许这些锁定的 baseline rows：

```bash
uv run python -m distance_head_study.power_analysis \
  --split-role legacy --baseline b_dh_cem \
  --backbone-seeds 42,43,44 \
  --output distance_head_study_runs/decisions/power.json

uv run python -m distance_head_study.lock_confirmation_n \
  --power-artifact distance_head_study_runs/decisions/power.json \
  --claim-route positive
```

严格负结论路线使用 `--claim-route negative`。代码会根据 Seed-3 gate 自动要求原 closure
shortlist 或独立的 `negative_shortlist_lock.json`，并拒绝不匹配的人工路线选择。

## 7. Seed-10

```bash
uv run python -m distance_head_study.release_seed_tier --tier seed10
uv run python -m distance_head_study.plan_jobs \
  --phase seed10 --output distance_head_study_runs/jobs/seed10.jsonl
```

Seed-10 DAG 的顺序是：fresh backbone -> train/cal caches -> required head owners ->
`open_confirmation` -> 直接从 manifest 在线执行 both-protocol full-900。Confirm evaluation
不读取预计算 confirm latent cache。Planner-only finalist 会训练它锁定的 owner head，
不会训练新的 planner-named head。

## 8. Analysis 与 closure

正路线示例：

```bash
uv run python -m distance_head_study.analyze \
  --candidate <finalist> --baselines b_dh_cem,b_l2_cem \
  --split-role confirm --backbone-seeds 1001,1002,1003,1004,1005,1006,1007,1008,1009,1010 \
  --head-seeds 0 --output distance_head_study_runs/analysis/final.json

uv run python -m distance_head_study.close_study \
  --analyses distance_head_study_runs/analysis/final.json
```

若 `confirmation_n_lock` 大于 10，seed 参数必须使用 lock 中的完整 ordered prefix。负路线
要分别分析两个 finalists，每个命令显式加 `--family-size-override 8`，再把两个文件同时
传给 `close_study`。

Primary closure 后才能跑 `D_stress`。Stress 只允许 closure 中的方法，并复用 confirmation
的完整 backbone seed prefix 与 head seed 0；不得新增 seed、换 checkpoint 或回流重选
模型。

## 9. 恢复与失败

- `plan_jobs` 生成的 head/evaluator 命令已经带 `--resume`：首次运行是 start，存在同 spec
  operational state/rows 时才恢复；
- Head trainer 每 1000 step 写 operational state；state 保存 Python/NumPy/Torch/CUDA RNG；
- Evaluator 每 task append 一行，恢复时逐 task 去重并核对完全相同的 metadata；
- Cache index 仍是 immutable；若构建在 index 写出前中断，重跑会复用通过 content hash、
  manifest、backbone 与 protocol-lock 校验的原子 shard，任何旧协议/损坏 shard 都会拒绝；
- DAG executor 只接受与 plan hash 一致的 signed state/completion seal。普通失败在核对
  日志后用同一命令加 `--retry-failed`；机器/进程中断先用 `ps -p <recorded-pid>` 确认原
  PID 已停止，再加 `--retry-interrupted`。后者只允许同一 host；若所有声明输出已存在，
  会按 artifact 类型重验完整 provenance 后补 completion seal，否则按原 spec 恢复；
- 正常完成和 interrupted recovery 调用同一 artifact validator；不得因为文件名存在或
  process return code 为 0 手工补写 success；
- 每次尝试写独立的 `attempt_NNN.log`，旧日志不覆盖；
- 正式 artifact 不允许 overwrite；
- `--diagnostic-steps/--diagnostic-limit/--diagnostic-batches` 的输出统一隔离到
  `distance_head_study_runs/smoke/`，正式 decision loader 还会核对固定 sample count；这些
  参数不绕过 seed/split gate，运行 limited evaluator/diagnostics 前仍须释放对应 seed tier，
  sealed split 还须已有 shortlist、confirm-open 或 closure；
- OOM/中断可以调整 microbatch 或并行 job 数吗？**不能直接改锁定 config**。先在
  diagnostic branch 验证等效 gradient accumulation，再以新 protocol version 审批；
- GPU worker 数、job 排队和 cache 存储位置可按环境调整，不改变 scientific config。

在 `close_study` 完成前，不能清理 cache index、candidate bank、backbone/head checkpoint、
task rows、summary 或 diagnostic。Decision/analysis 的完整证据链会在加载时重新计算这些
文件的 hash；shortlist、release、confirm-open 和 closure 还会扁平继承上游 hash map。
提前清理会使实验按设计 fail closed。
