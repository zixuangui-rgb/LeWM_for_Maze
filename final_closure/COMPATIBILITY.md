# Compatibility With Earlier Experiments

## Interface alignment

| Contract | Earlier experiment | Final closure |
|---|---|---|
| Environment | `ProcgenMazeEnv` | Same class and transition code |
| Observation | H x W x 5 one-hot | Same channel order and dtype |
| Direction order | action IDs 1/2/3/4 | Same IDs and slot mapping |
| Train manifest | `unisize_train_manifest.jsonl` | Same file and SHA256 |
| Old development manifest | `unisize_eval_manifest.jsonl` | Same file and SHA256 |
| Fresh evaluation | Spatial-JEPA confirm manifest | Same 900 rows and SHA256 |
| Start and goal | Manifest values | Manifest values, never resampled |
| Step cap | 128 | 128 |
| Seen/OOD boundary | <=21 / >21 | <=21 / >21 |
| SPL | oracle length / executed length on success | Same definition |
| Training seeds | 42-51 | Same ten labels |
| Evaluation RNG seed | 42 | Same fixed seed |

## Deliberate BC corrections

The old `scripts/eval/run_bc_cnn.py` cannot be used unchanged because it contains server-specific absolute paths, evaluates only 30 tasks per size, resamples a start state, and picks the best epoch from a state-level split that shares topologies with training.

`final_closure/train.py` reuses its DeepCNN topology and optimization family while making four protocol corrections:

1. four directional outputs match the shared action alphabet and remove unused STAY;
2. every non-goal free training state is included;
3. the final epoch is saved without validation selection;
4. evaluation uses all manifest starts and goals.

The 21x21 historical training canvas and native 23/25 OOD inputs are retained. Development corrected SR is reported beside the old 0.781 anchor, but no tolerance gate or parameter adjustment is allowed.

## Deliberate LeWM execution split

The old latent-L2 evaluator used `moving_actions` and a true-wall one-step fallback whenever CEM proposed an invalid or backtracking action. That behavior is now named `corrected`.

The strict trainer also preserves the historical `entries[step % len(entries)]` schedule and `numpy.default_rng(training_seed)` environment-seed stream. The new primary `unmasked` branch calls the same repository `cem_plan`, the same `Unisize256` encoder/projector/predictor, and the same latent-L2 score, then directly executes the first CEM action. The two branches share checkpoint, tasks, candidate RNG, horizon, and all other settings. Their paired difference isolates executor assistance.

## Imported Spatial-JEPA results

The new package does not retrain or rewrite R4/J1. It consumes row-level files produced by the prior strict package and validates:

- experiment family and format version;
- source Git commit and code fingerprint;
- clean training and evaluation flags;
- checkpoint path and SHA256;
- analysis-spec consistency;
- seed label;
- K=128 availability;
- unmasked confirmatory status;
- all 900 unique task hashes.

This permits a scientifically valid cross-stage table even though the baselines are trained on a later commit.
