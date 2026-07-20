# AIR-JEPA Experiment Family

[English](README.md) | [中文](README.zh.md)

`air_jepa/` is the isolated experiment family for the more radical JEPA reasoning
architecture. Every stage has its own experiment ID, protocol lock, code,
manifests, tests, and interpretation boundary. A later stage must not overwrite
artifacts or retroactively change the rules of an earlier stage.

## Stages

| Directory | Experiment ID | Question | Status |
|---|---|---|---|
| `stage0_workspace/` | `procgen-maze-air0-workspace-v1` | Can one shared recurrent workspace connect action-conditioned futures, costs, and iterative reasoning through a causally testable path? | Code and protocol package |

“Code and protocol package” does not mean that results already exist. Formal
results are produced under `air_jepa_runs/` on the server, and only the `L3
FINAL_CLOSURE` release closes a stage.

## Family Rules

1. Reuse the validated Procgen Maze environment, topology hold-out, and source
   Spatial-JEPA checkpoints without changing transition semantics.
2. Lock questions, controls, data roles, seeds, budgets, statistics, and gates
   before materializing the complete job DAG.
3. Quicklooks accelerate feedback but never modify the remaining matrix.
4. Corrected and true-future assistance/oracle results never enter the absolute
   ability table.
5. A new stage requires a new experiment ID and an unopened data role.

See [`stage0_workspace/README.md`](stage0_workspace/README.md) for the Stage-0
engineering entry point.
