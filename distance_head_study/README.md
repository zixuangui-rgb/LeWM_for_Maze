# Rigorous Staged DistanceHead Study

This directory implements the complete Procgen Maze study for Vector-JEPA
DistanceHeads. The question is not merely whether BFS regression loss decreases,
but whether a goal-conditioned cost improves, in order:

1. raw BFS-distance estimation;
2. true-next and predicted-next local action ordering;
3. horizon-12 candidate trajectory ordering;
4. corrected and unmasked closed-loop SR, SPL, and OOD performance;
5. independent fresh-backbone confirmation against `B-DH-CEM` and `B-L2`.

Code status: **implemented; formal server experiments have not been run**. Result
numbers must be recomputed from task rows and signed analysis artifacts.

## Scientific contract

- `D_train`: 2,800 topologies at sizes `9-21`.
- `D_cal`: 140 deterministic training-topology calibration tasks.
- `D_screen`: 140 fresh topologies at sizes `9-21` only.
- `D_select`: 210 fresh topologies at sizes `9-21` only.
- `D_confirm`: sealed full-900 at sizes `9-25`; `23/25` are size OOD.
- `D_stress`: sizes `27/29/31`, after primary closure only.
- Anchor planner: horizon 12, 64 candidates, one CEM iteration, per-step replanning,
  and a 128-step cap.
- Both `corrected_v1` and `unmasked` are reported.
- Every formal checkpoint is the final training step; validation-SR checkpoint
  selection is forbidden.
- Confirmation backbones begin at seed `1001` and are disjoint from historical
  seeds `42-61`.
- The independent statistical unit is the backbone training seed, not a task,
  candidate, or nested head seed.
- BFS is permitted for labels, diagnostics, and oracles, never for learned-method
  test-time action selection.
- Twelve `D_confirm` tasks and 33 `D_stress` tasks have shortest paths above the
  128-step cap. They remain paired across every method and are labelled
  `step_cap_ineligible` rather than silently removed.

## Layout

```text
distance_head_study/
  configs/                  strict config, method catalog, protocol lock
  protocol/                 seed registry, baseline provenance, bootstrap schedule
  manifests/                D_cal/D_screen/D_select/D_confirm/D_stress
  docs/                     protocol, runbook, method and audit documentation
  tests/                    unit, integration, determinism, leakage, statistics
  data.py                   goal-consistent cache and matched sampler
  models.py                 scalar/ordinal/distribution/reachability/quasimetric heads
  losses.py                 absolute/local/Bellman/multistep/predicted/TRM losses
  train_backbone.py         exact final_closure LeWM recipe reuse
  train_head.py             final-step, calibrated, matched joint-control training
  evaluate.py               task-level corrected/unmasked evaluation
  diagnose.py               raw distance/local/candidate/drift diagnostics
  analyze.py                crossed bootstrap, backbone sign-flip, and Holm
  make_decision.py          signed stage decisions
  plan_jobs.py/run_jobs.py  signed job DAG, local executor, completion seals
```

## Start here

Engineers should read the Chinese normative documents in this order:

1. [Experiment protocol](docs/EXPERIMENT_PROTOCOL.zh.md)
2. [Engineer runbook](docs/ENGINEER_RUNBOOK.zh.md)
3. [Method catalog](docs/METHOD_CATALOG.zh.md)
4. [Server checks](docs/CHECK_REQUIRED.zh.md)
5. [Validation](docs/VALIDATION.zh.md)

Initial verification:

```bash
cd /path/to/LeWM_for_Maze
uv sync --extra dev
uv run python -m distance_head_study.generate_manifests --role all --check
uv run python -m distance_head_study.audit_protocol --regenerate-manifests
uv run pytest distance_head_study/tests -q
```

Run every command from the repository root. This package intentionally leaves the
historical `pyproject.toml` unchanged because that file is part of the prior
full-900 protocol fingerprint. Repository-root execution imports this source tree
directly while preserving reproducibility of the historical lock.

Formal jobs require a clean, committed worktree. `--allow-dirty-worktree`,
`--diagnostic-limit`, and `--diagnostic-steps` are smoke-test features; their
artifacts are ineligible for decisions, power analysis, or confirmation. Limited
and short-run outputs are isolated under `distance_head_study_runs/smoke/` and
cannot occupy formal cache, checkpoint, or result paths.

Formal result loading verifies metadata signatures, checkpoint/candidate-bank/cache
hashes, manifests, task identity, and recomputed summaries. Decisions, power, and
analyses bind complete metadata/rows/summary/manifest/checkpoint evidence bundles.
Joint-method diagnostics re-encode observations with the updated checkpoint model;
they never evaluate a new projector against stale frozen-cache latents.

Limited evaluation and diagnostics obey the same seed and sealed-split gates as
formal runs. Trajectory contexts are selected deterministically across topology
blocks, and rollout drift uses every fixed-bank candidate rather than candidate 0.
The executor semantically validates each artifact before writing its completion
seal. Final signed artifacts flatten transitive file hashes, and the protocol code
fingerprint covers every imported scientific dependency.

If a positive-path method fails the Seed-3 gate, its original shortlist remains
immutable. The complete reserve closure is written to a separate negative fallback
lock, and two mechanistically distinct finalists are tested only on fresh sealed
confirmation backbones.

## Protocol repair

Training observations, goal latents, and BFS labels now always refer to the same
manifest goal. The old arbitrary-pair pipeline could render one goal marker while
assigning distance to another cell. That legacy behavior is provenance/parity only
and is excluded from the new ranking.

## Claim boundary

A positive result supports a locked method under corrected/assisted pooled
Vector-JEPA. A negative result requires the preregistered closure set, two
mechanistically distinct finalists, and familywise upper bounds excluding the MEI.
It cannot establish that every future DistanceHead or JEPA method is impossible.
