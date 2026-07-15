# 工程师执行手册

以下命令全部从仓库根目录执行。正式运行前请先提交当前代码，保持受监控路径clean。

## 1. 环境与权重

```bash
uv sync --frozen --extra dev
for seed in $(seq 42 51); do
  test -f checkpoints/final_closure/lewm_l2_cem_seqlen2_seed${seed}.pt
done
```

不得转换state dict、复制seed或用别名checkpoint代替缺失权重。

## 2. 静态验证

```bash
uv run ruff check vector_jepa_planner_full900_screen \
  tests/test_vector_jepa_planner_full900_screen.py
uv run python -m compileall -q vector_jepa_planner_full900_screen
uv run pytest -q tests/test_vector_jepa_planner_full900_screen.py
uv run python -m vector_jepa_planner_full900_screen.lock_protocol --check
uv run python -m vector_jepa_planner_full900_screen.audit_protocol \
  --require-checkpoints
```

任何一步失败都不得进入正式矩阵。

`lock_protocol`默认拒绝覆盖已有锁。`--replace-before-run`只供交接前审查修订使用，
且只要run目录或本实验checkpoint目录已有任一正式文件就会拒绝；工程师正式执行时
只能使用`--check`。

## 3. Q0：旧新B0 full-900 parity

```bash
uv run python -m vector_jepa_planner_full900_screen.run_plan \
  --stage Q0 --dry-run
uv run python -m vector_jepa_planner_full900_screen.run_plan \
  --stage Q0 --execute --device cuda
```

Q0会分别运行旧reference和新B0的corrected/unmasked，共四次full-900，然后生成两
个parity工件。后续调度器会自动拒绝缺失或失败的parity。

## 4. Q1：搜索方法

```bash
uv run python -m vector_jepa_planner_full900_screen.run_plan \
  --stage Q1 --execute --device cuda
uv run python -m vector_jepa_planner_full900_screen.freeze_q1
```

`freeze_q1`同时验证categorical bridge与B0逐任务、逐执行动作一致，并永久冻结一个
Q1父方法。

## 5. Q2：全部主要方法族

三个阶段都依赖冻结的Q1父方法，但彼此不累计，可以在不同GPU作业中执行。每个stage
内部调度器按固定seed随机化方法block。

```bash
uv run python -m vector_jepa_planner_full900_screen.run_plan \
  --stage Q2A --execute --device cuda --resume-missing
uv run python -m vector_jepa_planner_full900_screen.run_plan \
  --stage Q2B --execute --device cuda --resume-missing
uv run python -m vector_jepa_planner_full900_screen.run_plan \
  --stage Q2C --execute --device cuda --resume-missing
uv run python -m vector_jepa_planner_full900_screen.freeze_shortlist
```

Q2B中的Vector-DTS以uniform-expansion MCTS作为正式晋级控制；Direct-DTS仍会运行，
但只是search-disabled描述性诊断。Q2C每个ranker依次执行train、calibrate、round1、
round2、round3，再评测。不得跳轮。

`freeze_shortlist`会对DTS三份checkpoint和Bidirectional/forward两份checkpoint执行
共享组件exact-parity gate。若报`shared learned components diverged`，不得手工修改
decision或继续Q3；应保留全部日志并把本次运行判为实现/确定性失败。

## 6. 运行量透明度

每个method必须分别运行corrected和unmasked，且每次均为完整900任务。因此seed42
阶段的固定评测量为：Q0 4次、Q1 10次、Q2A 8次、Q2B 14次、Q2C 4次，共40次
full-900，即36,000个episode；此外还有head训练、校准和Q2C反例挖掘。

Q3在shortlist=2且控制均不重合的最坏情况下增加20次full-900，即18,000个episode。
Q4的最坏情况是winner和匹配控制都含可训练head：增加82次full-900，即73,800个
episode。实际数量由冻结decision确定，并可先用`--dry-run`查看。取消candidate
replay避免把每次evaluation再近似执行一遍，但不会缩短900任务主评测。

这不是180-task快筛。完整对齐的代价必须在提交GPU作业前据实预算；不得运行中途
因为耗时而删掉某个action protocol、尺寸或低分方法。

## 7. Q3：三backbone复核

```bash
uv run python -m vector_jepa_planner_full900_screen.run_plan \
  --stage Q3 --dry-run
uv run python -m vector_jepa_planner_full900_screen.run_plan \
  --stage Q3 --execute --device cuda --resume-missing
uv run python -m vector_jepa_planner_full900_screen.freeze_final
```

Q3只运行shortlist、B0和每个shortlist的预注册匹配控制，新增backbones43/44。若shortlist
为空，stage没有方法作业，`freeze_final`会写入无胜者关闭决定。

## 8. Q4：十backbone最终对齐

```bash
uv run python -m vector_jepa_planner_full900_screen.run_plan \
  --stage Q4 --dry-run
uv run python -m vector_jepa_planner_full900_screen.run_plan \
  --stage Q4 --execute --device cuda --resume-missing
uv run python -m vector_jepa_planner_full900_screen.summarize
```

Q4为唯一胜者、B0和匹配控制补backbones45-51。有训练head的方法还在全部10个
backbone上补planner seed130363。`summarize`先在backbone内平均planner seeds，
生成 `summary.json`、`REPORT.md` 和永久closure工件。

## 9. 断点恢复

`--resume-missing`只跳过已存在的完整最终文件。训练、校准、round和evaluation各自
是独立原子文件；不存在最终文件时才重新运行。不得手工删除低分输出。

若进程中断且没有最终文件，可恢复。若最终文件已经产生但已知存在基础设施错误，
必须先保存原文件和SHA256、记录客观原因，再通过单独protocol amendment处理；
当前run-plan不提供按分数覆盖开关。

## 10. 产物位置

- Checkpoints：`checkpoints/vector_jepa_planner_full900_screen/`
- Results：`vector_jepa_planner_full900_screen_runs/`
- Schedules：`.../schedules/Q*.csv`
- Decisions：`.../decisions/`
- Counterexamples：`.../counterexamples/`
- Final report：`.../REPORT.md`

不要把run目录复制到其他protocol名下继续选择；任何后续组合或fresh holdout必须建
立新协议。
