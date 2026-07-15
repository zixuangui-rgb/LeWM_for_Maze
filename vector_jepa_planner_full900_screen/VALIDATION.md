# 验证范围

## 自动检查

- Pydantic拒绝缺失方法族、错误seed schedule、非1x预算、Track J和rollout drift；
- 方法角色与可执行method一一对应；
- Q1候选相对bridge只允许search planner变化；
- Q2B/Q2C匹配控制只允许预声明的叶子字段变化；
- DTS learned/uniform/direct和Bidirectional/forward的共享head、非计时训练摘要、
  校准指标及数据来源必须exact match；该门槛在seed42、3-backbone和最终汇总重验；
- train/development/validation/confirmatory在topology/layout/task三层零重叠；
- 四个manifest均检查尺寸计数、缺失hash和内部重复；
- B0 controller、rollout、CEM和corrected fallback已有数值单元对照；
- stage schedule、selection hash、checkpoint training spec和result metadata闭合；
- 预算、非有限值、task count和task hash在评测时检查；
- 最终统计先在backbone内平均planner seeds，再按backbone x task交叉结构重采样；
- seed42筛选的48项系统/机制比较使用固定Bonferroni区间，经验bootstrap分布精确
  枚举，不依赖20,000次模拟的极端尾部；
- Q0核对规范输入路径、当前source checkpoint、manifest任务顺序、完整结果字段和逐步
  实际执行动作；
- decision artifact绑定Q0 parity、所有输入result及上游decision哈希；
- final report同时输出mean/SD、逐尺寸SR、失败指标、assistance和每decision compute；
- `pyproject.toml`与`uv.lock`均进入锁和代码指纹。

## 必须运行的命令

```bash
uv run ruff check vector_jepa_planner_full900_screen \
  tests/test_vector_jepa_planner_full900_screen.py
uv run python -m compileall -q vector_jepa_planner_full900_screen
uv run pytest -q
uv run python -m vector_jepa_planner_full900_screen.lock_protocol --check
uv run python -m vector_jepa_planner_full900_screen.audit_protocol
```

## 自动检查不能证明的事情

- 服务器CUDA、驱动和显存足以完成全部作业；
- 未随Git分发的checkpoint真实存在；
- 新方法一定提升SR；
- full-900结果可外推为新的盲测性能；
- 训练head达到足够校准质量。

这些只能通过正式Q0-Q4运行和结果审查回答。测试通过表示实现满足锁定协议，不表示
科学假设已经得到支持。
