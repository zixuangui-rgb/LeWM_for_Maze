# Result and Checkpoint Schema

## Baseline checkpoint

Every checkpoint contains:

- `experiment_family`, `format_version`, `stage`;
- `baseline_name`, `baseline_kind`, `training_seed`, `formal_run`;
- `analysis_spec_sha256`, `training_spec_sha256`;
- `protocol`: commit, dirty flag, code fingerprint, runtime, device, all manifest hashes, action IDs, and step cap;
- exact frozen training configuration;
- optional objective-rerun record containing the allowed reason and superseded-file SHA256;
- sample/step accounting and final training metrics;
- parameter count;
- serialized model configuration and state dictionary.

BC stores `policy_state_dict`; LeWM stores `model_state_dict`. LeWM configuration is serialized as plain Pydantic JSON rather than a pickled configuration object.

## Baseline result

Each JSON result has two top-level fields:

```text
metadata
results
  navigation
  task_rows
  compute
```

`metadata` binds the result to its checkpoint, training seed, split role, action protocol, evaluated manifest hash, current commit, code fingerprint, and runtime. `comparable_to_primary` is true only for a formal full-900 confirmatory unmasked result.

Each task row includes:

- immutable task identity and topology seed;
- start, goal, maze size, and oracle length;
- success, executed path length, and SPL;
- invalid action count;
- repeat-state count, maximum visits, and `loop_or_cycle`;
- final BFS distance;
- episode wall-clock time;
- controller-specific counters such as proposed invalid actions, oracle-assisted actions, CEM calls, and predictor transition counts.

The summarizer recomputes all navigation and compute aggregates from these rows. A stale task count, wall-clock total, controller-call count, assistance count, CEM transition count, or subgroup summary is rejected.

## Closure gate

`CLOSURE_COMPLETE.json` contains SHA256 values for the executable final config and lock, imported Spatial-JEPA config and lock, three manifests, full-regeneration formal audit, all source result files, all 20 new checkpoints, nine tables, five figures, report, and summary. It also records analysis/Git/code fingerprints, any objective replacement records, the permitted rerun classes, and explicitly sets `rerun_for_low_or_surprising_score` to false. Its existence makes all formal mutation entry points fail closed.

`python -m final_closure.verify_closure` independently reloads the gate, checks its schema and hard-stop fields, re-hashes every source and artifact, and cross-checks the summary. This command is read-only and is intended for archival verification after closure.
