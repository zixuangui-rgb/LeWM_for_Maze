# Result and Checkpoint Schema

## Baseline checkpoint

Every checkpoint contains:

- `experiment_family`, `format_version`, `stage`;
- `baseline_name`, `baseline_kind`, `training_seed`, `formal_run`;
- `analysis_spec_sha256`, `training_spec_sha256`;
- `protocol`: commit, dirty flag, code fingerprint, runtime, device, all manifest hashes, action IDs, and step cap;
- exact frozen training configuration;
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

## Closure gate

`CLOSURE_COMPLETE.json` contains SHA256 values for the full-regeneration formal audit, all source result files, all 20 new checkpoints, generated tables, figures, report, and summary. It also records the permitted objective rerun classes and explicitly sets `rerun_for_low_or_surprising_score` to false. Its existence makes all formal mutation entry points fail closed.
