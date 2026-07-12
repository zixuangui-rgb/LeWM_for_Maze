# Final Closure Experimental Protocol

## 1. Study role and timing

This is a fixed benchmark-completion addendum to the already executed Spatial-JEPA confirmatory study. The fresh confirmatory outcomes for R4/J1/J2/J3 were seen before this addendum was specified. Consequently:

- the original Spatial-JEPA hypotheses retain their original confirmatory status;
- BC-versus-JEPA and vector-LeWM-versus-spatial-JEPA are secondary comparisons;
- no wording in the generated report may call the new comparisons preregistered confirmatory tests;
- no baseline choice or parameter may depend on the fresh 900-task outcomes.

This separation prevents hindsight from being converted into false prospective evidence.

The four-way table is an absolute capability benchmark, not an equal-training-compute or equal-supervision ablation. BC and the learned planners receive oracle BFS-derived action/value supervision, while vector LeWM learns dynamics plus position auxiliaries. Therefore, cross-family differences estimate complete systems under their fixed established training recipes. They do not isolate representation learning, supervision efficiency, or sample efficiency.

## 2. Frozen datasets

| Role | Sizes | Tasks | Use |
|---|---|---:|---|
| Train | 9, 11, 13, 15, 17, 19, 21 | 2,800 topologies | Parameter fitting only |
| Development | 9 through 25 odd | 900 topologies | Historical implementation-alignment check only |
| Confirmatory | 9 through 25 odd | 900 fresh topologies | One fixed final evaluation |

Sizes 23 and 25 are OOD. Every manifest row fixes `topology_seed`, `env_seed`, `start_cell`, `goal_cell`, `layout_hash`, `task_hash`, and oracle BFS length. `audit_protocol.py` regenerates every maze and verifies topology, layout, and task disjointness between every pair of splits.

The 128-step cap yields an exact maximum SR of 881/900 = 0.978888..., because 19 confirmatory tasks have oracle paths longer than 128. This is a task-budget ceiling, not model failure.

## 3. Methods

### 3.1 Imported methods

`R4 raw iterative` and `J1 frozen Spatial-JEPA` are imported from commit `0eca77209429d86c71768195ba654d560cf35633`. Their primary K is 128 and their action selection is unmasked. Row-level outputs, checkpoint hashes, code fingerprints, training seeds, and task IDs must all validate before they enter a table.

J1 is selected because it was the original fixed primary Spatial-JEPA mechanism: frozen representation plus iterative planner. It avoids choosing J2 or J3 after looking at their confirmatory means.

### 3.2 BC DeepCNN

The BC baseline retains the historical architecture:

1. 5-channel raw observation;
2. 64-channel stem;
3. two 64-channel residual blocks;
4. stride-2 128-channel stage plus one residual block;
5. adaptive global pooling;
6. 512/256 MLP with dropout 0.3;
7. four directional logits in the shared action order: up, down, left, right.

Targets are exact BFS-optimal actions for every non-goal free state of every training maze. Procgen perfect mazes have a unique shortest action at each non-goal state; the loader asserts this property rather than silently resolving ties.

Historical bottom/right padding to a 21x21 training canvas is retained. OOD sizes 23 and 25 remain native size because the adaptive pool supports them. The policy is trained for 200 complete epochs with AdamW, batch 128, learning rate 1e-3, weight decay 1e-4, gradient clipping 1.0, and epoch-level cosine decay.

The historical script selected the best checkpoint on a random state-level split that shared topologies with training. This implementation deliberately removes that leakage: all training states are fitted and the final epoch is saved. Neither development nor confirmatory performance selects a checkpoint. The complete padded one-hot dataset is cached once as CPU `uint8`; each batch is converted to float on transfer, which preserves exact 0/1 inputs while avoiding 200 repeated image reconstructions.

### 3.3 Vector LeWM latent-L2 CEM

The vector baseline uses the repository's exact `Unisize256` implementation:

- CNN channels 64/128/256;
- 256-dimensional encoded and projected vectors;
- size embedding and fusion MLP;
- post-BN embedding and SIGReg stages;
- 16 predictor heads;
- prediction + SIGReg + absolute position + relative position + goal position losses;
- 30,000 steps, batch 256, sequence length 2, AdamW 1e-3.

The planner is fixed from the prior development experiments:

- receding replanning every environment step;
- context history 3;
- horizon 12;
- 64 candidates, 8 elites, one CEM iteration;
- squared L2 between predicted terminal embedding and goal embedding;
- candidate action alphabet 1 through 4;
- maximum 128 environment steps.

No distance head, QRL head, auxiliary BFS head, memory, new projector, or alternative horizon is tested here.

## 4. Action protocols

### Primary: `unmasked`

The model's proposed directional action is executed directly. A wall collision remains a wall collision. No true-wall validity mask, no anti-backtracking rule, and no oracle fallback is available.

### Diagnostic: `corrected`

The historical correction is preserved as a diagnostic. Invalid or immediate-backtracking proposals are replaced using the true maze's moving actions. BC picks its highest-logit allowed action; LeWM uses the historical one-step predicted-latent L2 fallback.

The corrected result answers only "how much did the old executor help?" It is never used to state absolute model ability.

## 5. Endpoints and inference

Primary endpoint: per-task success within 128 steps. The independent model-replication unit is a complete training run (`n=10` per method), not one of the 900 repeated task evaluations. Tasks provide a crossed generalization sample and are never misreported as 9,000 independent trained models.

Secondary endpoints: SPL, seen SR/SPL, OOD SR/SPL, per-size results, shortest-path bins, invalid-action rate, loop/cycle rate, correction-assistance rate, and compute diagnostics.

Each cross-method comparison is paired by exact task hash but treats the two methods' ten training runs as independent seed samples. A numeric seed label is not considered a meaningful pair across architectures that consume randomness differently. The crossed bootstrap independently resamples each method's training seeds and jointly resamples matched tasks within each maze-size stratum, preserving the designed 100-task weight of every size. In contrast, unmasked-versus-corrected diagnostics pair both seed and task because both executions use the same checkpoint. Four fixed SR comparisons form one family; percentile intervals use alpha 0.05/4. SPL receives ordinary 95% descriptive intervals and is not used for multiplicity-controlled claims.

The four SR contrasts are:

1. J1 Spatial-JEPA minus BC;
2. J1 Spatial-JEPA minus vector LeWM;
3. R4 raw iterative minus BC;
4. R4 raw iterative minus vector LeWM.

These remain secondary even when a simultaneous interval excludes zero.

## 6. Reproducibility and provenance

Formal training and evaluation require a clean worktree. Each checkpoint records the commit, scoped code fingerprint, runtime, device, seed, three manifest hashes, analysis-spec hash, training-spec hash, model configuration, and parameter count. Each result repeats training provenance and includes checkpoint SHA256 plus all 900 task rows.

The complete executable analysis specification has an independently stored SHA256 in `protocol_lock.json`. Loading a changed scientific setting under the same protocol ID fails before training. Final summarization additionally requires a clean, same-commit formal audit that regenerated all 4,600 manifest entries; the audit itself is hashed into the closure gate.

Training and evaluation of a new baseline must use the same commit and code fingerprint. Imported spatial results are validated against their frozen source commit instead of the new baseline commit.

`run_plan.py` applies a fixed seeded permutation (`run_order_seed=20260712`) to the 20 baseline-by-seed jobs instead of running every BC job before every LeWM job. This blocks avoidable chronological ordering effects while preserving an exactly reproducible schedule. Distributed execution must retain the same complete job set even when wall-clock order is controlled by the scheduler.

## 7. Completion rule

The study closes when all required files pass structural and provenance validation and paper artifacts are generated. Completion does not depend on effect direction, interval significance, or matching a historical score.

Only these objective failures permit rerun:

- interrupted execution;
- missing or duplicate task rows;
- manifest, checkpoint, commit, or code-fingerprint mismatch;
- NaN or infinite model output.

Low SR, high variance, or disagreement with the expected narrative must be reported as results and cannot reopen tuning on the confirmatory set.
After the closure gate is created, formal training, evaluation, re-summarization, and figure regeneration are rejected. A future study must receive a new protocol ID and untouched holdout.
