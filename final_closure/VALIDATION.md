# Validation Record

This record describes code validation performed before the package was committed. Synthetic scores mentioned here are test fixtures and carry no scientific meaning.

## Static and unit checks

- `ruff check final_closure tests/test_final_closure.py`: passed.
- `pytest -q`: 49 passed, including the pre-existing Spatial-JEPA suite.
- Python bytecode compilation and both JSON configuration parses: passed.
- `git diff --check`: passed.

The tests cover action order, exact observation rendering and historical padding, BFS target correctness, complete epoch coverage, uint8 cache equivalence, model shapes on size 21 and 25, LeWM configuration round-trip, unmasked/corrected action behavior, assistance/trajectory counterfactual consistency, wall-collision accounting, seed schedules, paired/independent/stratified bootstrap behavior, duplicate tasks, impossible paths, stale navigation and compute summaries, checkpoint hashes, imported checkpoint variant/training identity, exact cross-seed K curves, deterministic run order, source-code fingerprints, the independent analysis-spec lock, mandatory formal audit, constrained objective replacement, artifact-schema completeness, post-closure immutability, and independent gate tamper detection.

## Full protocol audit

`audit_protocol.py` regenerated all 4,600 manifest entries and passed:

- train: 2,800 tasks;
- development: 900 tasks;
- confirmatory: 900 tasks;
- topology/layout/task overlap: zero for all three split pairs;
- confirmatory paths longer than 128: exactly 19;
- oracle step-cap ceiling: exactly 881/900 = 0.978888...;
- source Spatial-JEPA commit and code fingerprint: exact match;
- fixed method, seed, action, inference, statistical, and run-order matrices: exact match.

## Training compatibility

The strict LeWM trainer and the historical `scripts/train/train_dim256.py` were each run for one CPU optimization step with seed 42, batch 256, sequence length 2, and 1,024 SIGReg projections.

Results:

- total loss: exactly `1.8702843189239502` in both;
- prediction/absolute/relative/goal losses: exact match;
- all 85 model-state tensors: bitwise identical after the optimizer step.

The fixed schedule therefore reproduces the original model, data, RNG, loss, and optimizer path before adding provenance controls.

Independent repeat tests also produced:

- BC diagnostic training: 62/62 state tensors bitwise identical;
- LeWM diagnostic training: 85/85 state tensors bitwise identical;
- repeated CEM evaluation: identical action-derived trajectory and failure metrics after excluding wall-clock fields.

## End-to-end closure test

A full synthetic result tree was generated with the production shape:

- 10 seeds;
- 900 confirmatory and 900 development task rows per run;
- 20 new checkpoints;
- 80 new baseline result files;
- 20 imported Spatial-JEPA result files plus their 20 checkpoints;
- 148 exact hashed source files in the closure gate: final config and protocol lock, source Spatial-JEPA config and protocol lock, three manifests, mandatory formal audit, 20 new checkpoints, 80 new baseline results, 20 imported Spatial-JEPA results, and their 20 checkpoints.

The production summarizer successfully validated the tree, executed 20,000-draw stratified crossed bootstraps, and generated:

- 9 CSV tables, including per-seed, per-size, assistance, and development-alignment records;
- 5 publication PNG figures;
- a populated paper report;
- `summary.json`;
- a score-independent `CLOSURE_COMPLETE.json`.

Negative injection was also exercised. The code rejected an impossible successful path beyond the 128-step cap, a stale compute total, an inconsistent loop flag, a non-finite auxiliary metric, a missing per-seed K value, a nonformal audit, a broad output-replacement request, and a post-closure artifact modification. After correcting the synthetic rows, the same complete tree closed and independently re-verified successfully.

## Scope

These checks validate code paths, determinism, compatibility, provenance enforcement, statistics, and artifact generation. They do not substitute for the 20 formal GPU training runs or their real scientific outcomes.
