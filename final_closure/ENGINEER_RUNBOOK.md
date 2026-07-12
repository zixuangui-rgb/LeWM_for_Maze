# Engineer Runbook

All commands run from the repository root. Formal commands must run from the committed clean revision containing this folder.

## 1. Install and verify

```bash
python -m pip install -e ".[dev,paper]"
python -m pytest -q
python -m final_closure.audit_protocol
python -m final_closure.smoke_test
```

The full audit regenerates all 4,600 manifest entries. Do not use `--skip-entry-regeneration` for the formal audit.

## 2. Inspect the exact command matrix

```bash
python -m final_closure.run_plan --stages full --dry-run > final_closure_commands.txt
```

Expected new training jobs: 2 baselines x 10 seeds = 20 checkpoints.

Expected new evaluations: 2 baselines x 10 seeds x 2 splits x 2 action protocols = 80 result files. Each result contains 900 task rows. The development runs are alignment diagnostics; the confirmatory unmasked runs form the new primary baseline rows.

## 3. Run on one machine

```bash
python -m final_closure.run_plan --stages audit,smoke,train,development,confirmatory --device cuda
```

The runner skips an existing output. It does not silently overwrite it.

## 4. Run distributed by seed

Each seed is independent. For example:

```bash
python -m final_closure.run_plan \
  --stages train,development,confirmatory \
  --seeds 42 \
  --device cuda:0
```

Use seeds 42 through 51 exactly once. Multiple hosts must use the same commit, environment package versions, and shared output tree. Do not edit code between training and evaluation.

To run one baseline only:

```bash
python -m final_closure.run_plan \
  --stages train,development,confirmatory \
  --baselines lewm_l2_cem_seqlen2 \
  --seeds 42,43 \
  --device cuda
```

## 5. Supply existing Spatial-JEPA results

The summary expects the prior files at:

```text
spatial_jepa_planning_runs/r4_raw_iterative_progressive/seed{42..51}/confirmatory_unmasked.json
spatial_jepa_planning_runs/j1_spatial_iterative_frozen/seed{42..51}/confirmatory_unmasked.json
```

Their referenced checkpoints must also exist at the paths embedded in result metadata. Do not regenerate R4 or J1 under the new commit merely to make paths look uniform; they are intentionally validated against source commit `0eca772...`.

## 6. Generate final paper artifacts

After all 20 checkpoints, 80 baseline results, and 20 imported spatial results are present:

```bash
python -m final_closure.summarize
```

Success creates `final_closure_runs/CLOSURE_COMPLETE.json`. That file is the hard stop. It records every input hash and explicitly denies score-triggered reruns.
The summary also requires `protocol_audit.json` from the same clean commit with full entry regeneration; it will not accept the shortened audit mode.

## 7. Objective reruns only

An interrupted or corrupt output may be rerun after the failure is recorded in the experiment log:

```bash
python -m final_closure.run_plan \
  --stages confirmatory \
  --baselines bc_deepcnn_fixed \
  --seeds 42 \
  --rerun-execution-failures \
  --device cuda
```

This flag overwrites every selected existing output, so select the narrowest baseline/seed/stage. Never use it because a score is low, unexpectedly high, or inconvenient.
It cannot be used after `CLOSURE_COMPLETE.json` exists. A completed study is immutable.

## 8. Failure triage

| Failure | Action |
|---|---|
| Dirty-worktree rejection | Commit intended files; do not bypass for formal runs |
| Manifest hash mismatch | Stop and restore the locked data file |
| Training/evaluation commit mismatch | Evaluate with the exact training commit |
| Missing spatial checkpoint | Restore the checkpoint referenced by the imported result |
| NaN/Inf | Preserve logs and checkpoint; classify as objective execution failure |
| CUDA OOM in BC | Reduce unrelated host load; batch size is locked |
| CUDA OOM in LeWM | Reduce unrelated host load; candidate/batch budgets are locked |
| Score differs from history | Report the difference; do not tune or rerun |

Formal outputs must never be produced with `--diagnostic`, `--allow-dirty-worktree`, task limits, shortened training, or reduced SIGReg projections.
