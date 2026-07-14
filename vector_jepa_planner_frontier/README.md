# Vector-JEPA Planner Frontier

This package is the executable, fail-closed Procgen Maze study for measuring how far planner-side changes can push a pooled single-vector JEPA world model. The Chinese `README.zh.md` and `ENGINEER_RUNBOOK.md` are the operational sources of truth.

The checked-in JSON contains 65 method templates. Schema validation expands the
single Track-J template into a complete 54-cell grid, yielding 118 effective
method configurations: 20 backbone seeds, two planner-head seeds for learned
planners, and two paired search seeds. `corrected_v1` overall and size-23/25
success are the confirmatory primary endpoints; `unmasked` remains a paired
autonomy diagnostic.

The study separates:

- exact legacy B0 parity;
- oracle-only bottleneck localization;
- equal-budget search comparison;
- a complete verifier/reachability/proposal/memory factorial;
- radical planners and matched controls;
- result-derived frozen Track F, three-round counterexample training, and a
  preregistered 54-cell Track-J joint-training grid;
- a final `0.5x/1x/4x/16x` compute frontier;
- power-gated, opaque, one-shot confirmation.

Run the local integrity suite from the repository root:

```bash
uv sync --frozen --extra dev
uv run ruff check vector_jepa_planner_frontier tests
uv run pytest -q
uv run python -m vector_jepa_planner_frontier.lock_protocol --check
uv run python -m vector_jepa_planner_frontier.smoke_test
```

Do not begin formal execution before reading `ENGINEER_RUNBOOK.md`, `COMPATIBILITY.md`, and `CLAIMS_AND_STOP_RULES.md`. Formal runs require a clean committed worktree and immutable outputs.
