# hdwm — Hypothesis-Driven World Model

[English](README.md) | [中文](README.zh.md)

LE-WM-based world model for 2D grid navigation, extended with new environments,
planning algorithms, and representation learning regularizers.

This branch builds on the upstream
[hdwm](https://github.com/qun-team/hdwm) codebase (commit `4503b41`).
Diff: `hdwm-origin/` vs current tree.

## New in This Branch

### Environments
- **Four Rooms** (`hdwm/envs/four_rooms.py`) — classic RL benchmark with four connected rooms, supporting virtual-border train/validation splits.
- **Procgen Maze** (`hdwm/envs/procgen_maze.py`) — procedurally generated mazes with configurable size and obstacle density.
- **Ice World 2D** (`hdwm/envs/ice_world_2d.py`) — grid world with slippery transitions.

### Models
- **LE-WM CNN** (`config/models/lewm_cnn.yaml`) — CNN observation encoder replacing the default MLP, providing spatial inductive bias for 2D environments.
- **LE-WM v3** (`config/models/lewm_v3.yaml`) — concept-conditioned rotation dynamics with Cauchy loss.
- **LIWM** (`config/models/liwm.yaml`) — position-extrapolation model with learnable Lie generators.
- **ICWM** — in-context world model supporting trajectory packing.

### Regularization
- **VICReg variance loss** — temporal variance hinge applied along each trajectory's time axis, preventing embedding collapse under distribution shift.
- **Wasserstein SIGReg** (`hdwm/losses.py`) — sliced Wasserstein Gaussian regularizer as a drop-in alternative to sketch-based SIGReg.

### Infrastructure
- **Planning module** (`hdwm/planning.py`) — CEM-based model predictive control with configurable horizon, population size, and elite fraction.
- **Rotary Position Embedding (RoPE)** (`hdwm/models/shared.py`) — optional rotary temporal position encoding for the sequence transformer.
- **Batch sampling strategies** — `same_within_batch` and `different_within_batch` for controlled IID/OOD data generation.

### Experiments
- The upstream project ran an IID vs OOD comparison across models,
  environments, and VICReg settings (48 runs total). Its original
  `experiments/` pipeline and generated report are not bundled in this checkout.

## Key Findings

The historical summary recorded for that upstream experiment is:

| Metric | without VICReg | with VICReg |
|--------|---------------|-------------|
| OOD embedding probe (mean) | 0.39 | **0.95** |
| OOD MPC success rate (mean) | 18.5% | **45.7%** |
| IID–OOD embedding gap | +0.60 | **+0.04** |

VICReg closes the IID–OOD embedding gap, but the predictor probe gap persists — fixing the dynamics predictor under distribution shift remains open work.

## Installation

```bash
pip install -e '.[dev]'
```

## Quick Start

```bash
# Train LE-WM on Grid World
python run_train.py --config-name train_lewm_sigreg

# Train LE-WM CNN
python run_train.py --config-name train_lewm_sigreg model=lewm_cnn env=grid_world_2d

# Run CEM planner evaluation
python experiments/run_single.py
```

## Run Tests

```bash
python -m pytest -q
```

## Maze Experiment Packages

- [`diagnostics/`](diagnostics/) diagnoses representation, metric, rollout, and
  navigation failures under the locked topology hold-out protocol.
- [`planning_repair/`](planning_repair/) contains the P0-P2 repair matrix.
- [`spatial_jepa_planning/`](spatial_jepa_planning/) contains the next-stage
  full-resolution Spatial-JEPA and iterative-planning experiments, including
  protocol locks, multi-seed paired evaluation, and oracle controls.
- [`final_closure/`](final_closure/) contains the fixed paper-closure baselines,
  provenance checks, statistical analysis, and immutable completion gate.
- [`air_jepa/`](air_jepa/) contains the staged AIR-JEPA architecture program.
  Stage 0 freezes the validated Spatial-JEPA representation and tests a shared
  recurrent goal/action/future workspace under paired training, sealed data
  roles, causal future interventions, and a score-independent four-GPU DAG.
- [`research_notes/pure_jepa_frontier_directions.zh.md`](research_notes/pure_jepa_frontier_directions.zh.md)
  is a Chinese research memo on open directions for the capability frontier and
  generalization boundaries of pure JEPA latent planning. It is not an executed
  experiment or a locked protocol.
