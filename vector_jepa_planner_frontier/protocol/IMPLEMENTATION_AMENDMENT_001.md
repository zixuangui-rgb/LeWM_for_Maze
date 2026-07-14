# Implementation Amendment 001: Pre-run Implementation Clarifications

## Status

This amendment is made before any formal validation or confirmatory result is
opened. It changes checkpoint selection and corrects two secondary trajectory
metric denominators. Architectures, datasets, maximum optimizer steps, planner
budgets, seeds, primary endpoints, and statistical contrasts remain unchanged.

## Original Rule

Section 10.7 of `EXPERIMENT_PROTOCOL.md` selected different intermediate
checkpoints using module-specific validation metrics. Section 10.6 selected a
Track J checkpoint using a validation SR/JEPA-loss Pareto rule.

## Effective Rule

Every method in a training family uses its checkpoint after the pre-registered
final optimizer step. Validation data performs no optimizer update and selects
no intermediate checkpoint. It remains permitted only for calibration,
pre-registered advancement decisions, power estimation, and stability checks.

Track J is evaluated from its final optimizer step. It may enter a four-contrast
confirmatory family only when its final validation JEPA objective is no more than
10% worse than the matched source backbone. If the gate fails, Track J is omitted
and the two-contrast Track F family proceeds. No earlier Track J checkpoint may
be substituted after observing validation results.

## Rationale

The executable implementation jointly trains several planner heads in factorial
and combined cells. Selecting a different intermediate step for each active head
would create a composite model that never existed at one optimizer step, while
selecting one shared step with different module metrics would introduce an
unregistered aggregation rule. Full planner evaluation at every interval would
also allocate method-dependent validation compute. A single final-step rule is
uniform, deterministic, and removes checkpoint-selection degrees of freedom.

This choice is conservative for performance: it can miss a transient better
checkpoint. The report must therefore describe the result as the performance of
the fixed-budget final-step training procedure, not as an oracle best-checkpoint
upper bound.

## Claim Boundary

This amendment preserves a valid comparison of fixed training procedures. It
does not permit claims about the best intermediate checkpoint and does not relax
the one-shot confirmatory protocol. Any later change to checkpoint selection
requires a new amendment before confirmation is opened.

## Secondary Trajectory Metric Correction

Section 12.5 intended `unique_state_ratio` and `two_cycle_rate` to be bounded
rates. Counting the initial state in the former can yield `1 + 1/T` on a
non-revisiting trajectory. The written denominator `executed_steps - 2` in the
latter provides only `T-2` slots for `T-1` valid lag-two comparisons and can
also yield a value above one.

The executable definitions are therefore:

```text
unique_state_ratio
  = number of unique states first discovered by executed actions
    / max(executed_steps, 1)
  = (number of unique visited states including s_0 - 1)
    / max(executed_steps, 1)

two_cycle_rate
  = count(t in [2, T]: s_t = s_(t-2)) / max(T - 1, 1)
```

Both values are required to lie in `[0, 1]`. This correction affects only
secondary loop/memory diagnostics. It does not change actions, success, SPL,
assistance, compute, model selection, power, or confirmatory hypotheses.
