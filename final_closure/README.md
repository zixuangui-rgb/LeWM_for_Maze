# Maze-JEPA Final Paper Closure

This folder closes the Procgen Maze experiment family with two missing, fixed baselines and one score-independent paper analysis. It does not begin another architecture-search cycle.

## What this stage answers

The prior `spatial_jepa_planning` experiment already tested whether preserving a spatial latent and adding iterative planning repaired the three diagnosed failures: compressed action information, projector information loss, and autoregressive latent drift. Its fresh 900-task confirmatory set has already been observed.

This addendum answers the two remaining benchmark questions on exactly those tasks:

1. How does the frozen Spatial-JEPA planner compare with a strong raw-observation imitation policy?
2. What is the system-level capability gap between the fixed vector LeWM latent-L2 CEM system and the fixed Spatial-JEPA iterative system on identical tasks?

The second comparison is deliberately descriptive. Because representation, supervision, training recipe, and inference compute all change together, it cannot attribute the gap to any one component.

The final table contains four methods:

| Method | Source | Primary execution |
|---|---|---|
| `r4_raw_iterative_progressive` | Existing confirmatory run | Unmasked, K=128 |
| `j1_spatial_iterative_frozen` | Existing confirmatory run | Unmasked, K=128 |
| `bc_deepcnn_fixed` | Trained here | Unmasked directional policy |
| `lewm_l2_cem_seqlen2` | Trained here | Unmasked receding CEM |

`corrected` is retained only as an oracle-assistance diagnostic. It uses true wall validity and immediate-backtracking information and is excluded from the primary table.

## Scientific status

The new cross-family comparisons are **post-confirmatory fixed secondary analyses**. This label is deliberate: the Spatial-JEPA confirmatory scores were known before these baselines were added. The code therefore does not claim that these are newly preregistered hypotheses.

Scientific safeguards:

- topology hold-out remains exact;
- the executable analysis has an independently stored SHA256 lock;
- closure requires a same-commit formal audit that regenerates all 4,600 entries;
- all four methods use the same 900 task hashes, starts, goals, and 128-step cap;
- ten training seeds, 42 through 51, are required; cross-method seeds are resampled independently while task hashes remain paired within maze-size strata;
- confirmatory data never selects a checkpoint or hyperparameter;
- SR is the sole multiplicity-controlled endpoint;
- four SR comparisons use Bonferroni simultaneous intervals at familywise alpha 0.05;
- SPL and subgroup results are fixed secondary descriptions;
- scores never determine whether a run is repeated;
- closure requires hashes, unique task rows, full seed coverage, tables, figures, and a report;
- every imported K curve must contain the same locked seven iteration budgets for every seed, with every row rechecked against the manifest;
- all aggregate navigation and compute fields are recomputed from task rows rather than trusted from stored summaries.

## Contents

| File | Purpose |
|---|---|
| `configs/default.json` | Entire executable matrix and statistical analysis |
| `configs/protocol_lock.json` | Immutable manifests, source commit, and oracle ceiling |
| `audit_protocol.py` | Full regeneration, leakage, and configuration audit |
| `models.py` | Historical DeepCNN and exact `Unisize256` compatibility constructors |
| `data.py` | Deterministic all-state BC targets and LeWM sequence sampling |
| `train.py` | Fixed final-epoch/final-step baseline training |
| `evaluate.py` | Shared task executor for unmasked and corrected evaluation |
| `summarize.py` | Crossed bootstrap, tables, report, and closure gate |
| `plot_results.py` | Publication PNG figures |
| `verify_closure.py` | Independent post-closure hash and schema verification |
| `run_plan.py` | Resumable orchestration and dry-run command generation |
| `smoke_test.py` | CPU integration test across both model paths |
| `VALIDATION.md` | Pre-commit static, parity, determinism, and full-shape E2E record |

Read [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), [COMPATIBILITY.md](COMPATIBILITY.md), and [ENGINEER_RUNBOOK.md](ENGINEER_RUNBOOK.md) before starting formal runs.

## Final artifacts

When every requirement passes, `summarize.py` creates:

- `final_closure_runs/summary.json`
- `final_closure_runs/PAPER_RESULTS.md`
- `final_closure_runs/tables/*.csv` (9 fixed audit-ready tables)
- `final_closure_runs/figures/*.png`
- `final_closure_runs/CLOSURE_COMPLETE.json`

The final gate is generated regardless of which method wins. A surprising result belongs in the paper; it is not permission to tune on the confirmatory set.

Once the gate exists, the training, evaluation, summary, and plotting entry points reject further formal mutation.
