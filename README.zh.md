# hdwm — 假设驱动世界模型

[English](README.md) | [中文](README.zh.md)

基于 LE-WM 的 2D 网格导航世界模型，扩展了新的环境、规划算法和表征学习正则化方法。

本分支基于上游 [hdwm](https://github.com/qun-team/hdwm) 代码库（commit `4503b41`）。
代码差异对照：`hdwm-origin/` vs 当前代码树。

## 本分支新增内容

### 环境
- **Four Rooms** (`hdwm/envs/four_rooms.py`) — 经典 RL benchmark，四个连通房间，支持 virtual-border 训练/验证集划分。
- **Procgen Maze** (`hdwm/envs/procgen_maze.py`) — 程序化生成的迷宫，支持可配置的大小和障碍物密度。
- **Ice World 2D** (`hdwm/envs/ice_world_2d.py`) — 带滑移过渡的网格世界。

### 模型
- **LE-WM CNN** (`config/models/lewm_cnn.yaml`) — 用 CNN 观测编码器替换默认 MLP，为 2D 环境提供空间归纳偏置。
- **LE-WM v3** (`config/models/lewm_v3.yaml`) — concept-conditioned 旋转动力学，带 Cauchy loss 约束。
- **LIWM** (`config/models/liwm.yaml`) — 位置外推模型，使用可学习的 Lie 生成元。
- **ICWM** — in-context world model，支持轨迹打包。

### 正则化
- **VICReg variance loss** — 沿每条轨迹时间轴施加的 temporal variance hinge，防止分布偏移下的 embedding 坍缩。
- **Wasserstein SIGReg** (`hdwm/losses.py`) — sliced Wasserstein 高斯正则化，可作为 sketch-based SIGReg 的替代。

### 基础设施
- **规划模块** (`hdwm/planning.py`) — 基于 CEM 的模型预测控制，支持可配置的 horizon、population size 和 elite fraction。
- **旋转位置编码 (RoPE)** (`hdwm/models/shared.py`) — 可选的 rotary temporal position encoding，用于序列 transformer。
- **批次采样策略** — `same_within_batch` 和 `different_within_batch`，用于可控的 IID/OOD 数据生成。

### 实验
- `experiments/` 下的完整评估流程：跨模型、环境和 VICReg 设置的 IID vs OOD 对比（总计 48 次运行）。
- 产出：热力图 (`outputs/heatmaps/`)、规划动画 (`outputs/planning_gifs/`) 和[最终报告](experiments/outputs/final_report.md)。

## 核心发现

详见[最终报告](experiments/outputs/final_report.md)：

| 指标 | 无 VICReg | 有 VICReg |
|------|----------|----------|
| OOD embedding probe（均值） | 0.39 | **0.95** |
| OOD MPC 成功率（均值） | 18.5% | **45.7%** |
| IID–OOD embedding 差距 | +0.60 | **+0.04** |

VICReg 弥合了 IID–OOD embedding 差距，但 predictor probe 的差距依然存在——在分布偏移下改进动力学预测器仍有待研究。

## 安装

```bash
pip install -e '.[dev]'
```

## 快速开始

```bash
# 在 Grid World 上训练 LE-WM
python run_train.py --config-name train_lewm_sigreg

# 训练 LE-WM CNN
python run_train.py --config-name train_lewm_sigreg model=lewm_cnn env=grid_world_2d

# 运行 CEM planner 评估
python experiments/run_single.py
```

## 运行测试

```bash
python -m pytest -q
```

## Maze 实验包

- [`diagnostics/`](diagnostics/)：在锁定的 topology hold-out 协议下诊断表征、metric、rollout 和导航失败。
- [`planning_repair/`](planning_repair/)：P0-P2 修复矩阵。
- [`spatial_jepa_planning/`](spatial_jepa_planning/)：下一阶段 full-resolution Spatial-JEPA 与迭代规划实验，包含 protocol lock、多 seed 配对评估和 oracle 对照。
