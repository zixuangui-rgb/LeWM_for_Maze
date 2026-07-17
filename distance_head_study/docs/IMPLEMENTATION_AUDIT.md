# Implementation Audit Record

This file records what must be checked before the package is handed to the server
engineer. It is not a result report.

## Design corrections incorporated

1. Confirmation seeds start at `1001`; historical `42-61` are excluded.
2. `D_screen/D_select` contain sizes `9-21` only; size OOD never selects methods.
3. Training images, goal latents, and BFS labels share the manifest goal.
4. Old validation-selected DistanceHead checkpoints are parity-only; new checkpoints
   are final-step only.
5. Dynamic parent decisions are signed and included in effective method hashes.
6. Planner-only reserves reuse `c_parent` or an explicit trained owner; they cannot be
   sent to `train_head`.
7. Joint treatments have matched continuation controls and retain original JEPA/SIGReg.
8. Confirmation uses fresh backbone seeds as independent units; nested head seeds are
   averaged within a backbone.
9. Corrected and unmasked task rows are separate artifacts.
10. Negative claims require a complete reserve-family artifact and two finalists.
11. Short/limited runs are isolated under `distance_head_study_runs/smoke/`; formal
    loaders require fixed sample counts and full manifests.
12. Multi-step path labels make the goal absorbing; legacy rollout labels execute
    exactly `horizon-1` true transitions.
13. Joint diagnostics re-encode observations with the updated checkpoint model.
14. Resume state preserves Python, NumPy, Torch, and CUDA RNG states.
15. Formal task rows are checked against the locked manifest task by task.
16. Superiority uses a backbone-level one-sided sign-flip test; effect intervals use
    crossed backbone/task bootstrap with one shared task draw across backbones.
17. Job plans, executor states, completion markers, and output hashes are bound into
    one immutable chain.
18. Normal job completion and interrupted recovery use the same artifact-level
    semantic validator before a completion seal is written.
19. TRM contexts are selected deterministically across grouped topology blocks;
    trajectory max-distance labels follow the selected rows.
20. Rollout drift aggregates every fixed-bank candidate, not a fixed candidate index.
21. Formal and limited evaluator/diagnostic paths share seed and sealed-split gates.
22. True-next model-free scoring and predicted rollout scoring use explicit, different
    domain-adapter flags.
23. Protocol fingerprints cover all transitive scientific Python dependencies.
24. Shortlist, release, confirmation, and closure artifacts flatten upstream evidence
    hashes and reject contradictory bindings.
25. The historical `pyproject.toml` is unchanged; the new package runs from the
    repository root so the prior full-900 protocol lock remains reproducible.
26. Horizon-conditioned one-step true-next and predictor-next objectives explicitly
    query horizon 1; trajectory horizons remain separately matched to rollout slots.
27. Result loaders replay corrected/unmasked action semantics and exact planner
    transition accounting instead of trusting aggregate compute fields.
28. Dynamic artifacts re-resolve their signed parent decisions; backbone checkpoints
    and cache shards are checked against their complete source protocol bindings.
29. Four executable movement actions and the five-class LeWM model vocabulary are
    separate locked constants; fallback compute counts all five predictor queries.
30. Fresh-backbone training restores the source entrypoint's deterministic seed call
    before model construction; the seed now governs initialization as well as data.
31. Predictor transitions are reported as a deterministic compute proxy, while paired
    seconds per decision are retained separately as hardware-sensitive secondary data.
32. Horizon-conditioned diagnostics use h=12 for current-state absolute distance and
    h=1 for action-aligned one-step local probes; the historical corrected-v1
    predictor-greedy fallback is explicitly separated as a parity interface.

## Remaining environment checks

See `CHECK_REQUIRED.zh.md`. In particular, real server checkpoint compatibility and
four-H800 runtime behavior cannot be proven from this local checkout alone.

## Local audit evidence

- Branch: `codex/distance-head-staged-study`.
- Full manifest byte regeneration: passed for `cal/screen/select/confirm/stress`.
- Pytest: 115 package tests passed at the final pre-lock audit.
- Historical repository compatibility: 149 tests passed, including reproduction of
  the checked-in full-900 protocol lock; the combined run passed all 264 tests.
- Ruff format/lint, compileall, all 23 CLI `--help` entrypoints, manifest byte
  regeneration, Markdown links, and protocol dry-build checks passed.
- Protocol lock is generated only after this docs/code freeze because it hashes this
  file; the post-lock audit must reproduce it exactly.
- Real checkpoint smoke: pending server checkpoint availability.
