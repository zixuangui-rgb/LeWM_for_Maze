# AIR-JEPA Stage 0: Shared Workspace

[English](README.md) | [中文](README.zh.md)

Experiment ID: `procgen-maze-air0-workspace-v1`

Stage 0 freezes the validated Spatial-JEPA encoder/projector and feeds its full
`[64,H,W]` planning latent to one weight-shared reasoner repeated K times. The
workspace carries goal, action, and future tokens, predicts four
action-conditioned latent fields, and ranks them with one distributional cost
head. Navigation is receding: execute one action, observe again, and replan.

Formal execution is fixed at 135 scientific cells plus eight audit/orchestration
gates, for 143 jobs. The runner rebuilds and compares the DAG field by field,
and every release requires signed L0 evidence for four H800s, 128 paired-stream
batches, and the performance-blind compute-match lock.

`AIR0-direct` and `AIR0-jepa` have identical architecture, initialization,
sample/K streams, optimizer, training budget, and evaluation. Their only
scientific difference is the objective:

| Method | Action | Future latent | Distributional cost |
|---|---:|---:|---:|
| `air0_direct` | 1.0 | 0.0 | 0.0 |
| `air0_jepa` | 1.0 | 1.0 | 0.5 |

The primary path cannot read wall, valid-action, or BFS truth. BFS is used only
for training labels and offline diagnostics, and collisions remain exact no-op
successors. The package asks whether the new data path approaches
`j1-receding`, gains from explicit future/cost supervision, scales with K, and
uses future fields causally.

The complete scientific design is in
[`docs/EXPERIMENT_PLAN.zh.md`](docs/EXPERIMENT_PLAN.zh.md). Engineers should use
[`docs/ENGINEER_RUNBOOK.zh.md`](docs/ENGINEER_RUNBOOK.zh.md) as the only formal
execution entry point. `configs/default.json`, the protocol/package locks, and
the score-independent job DAG define the executable contract.

The L3 release separately reports the exact-BFS step-cap ceiling, recomputes
all aggregates from task rows, audits checkpoint pairing and sealed roles, and
emits parameter, MAC, runtime, size, path-length, and failure breakdowns. See
[`docs/IMPLEMENTATION_AUDIT.zh.md`](docs/IMPLEMENTATION_AUDIT.zh.md) for the
code-to-plan audit and the boundary between local checks and server-only L0.

Engineers may choose environment setup, process supervision, storage mounts,
and remedies for objective infrastructure failures. They may not change seeds,
K values, tasks, losses, batch size, steps, checkpoints, or result roles; skip
the historical bridge; rerun based on scores; or open `AIR_select/AIR_final`.
Objective failures follow the full-attempt archival procedure in
[`docs/REPLACEMENT_PROTOCOL.zh.md`](docs/REPLACEMENT_PROTOCOL.zh.md); AIR0-v1
does not permit score-selective single-cell retries.
