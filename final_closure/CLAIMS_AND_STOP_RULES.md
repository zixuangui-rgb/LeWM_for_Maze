# Claims, Interpretation, and Stop Rules

## Claims this stage can support

After the closure gate passes, the paper may state:

1. the absolute unmasked performance of R4, frozen Spatial-JEPA, BC, and vector LeWM on the same fresh topology hold-out tasks;
2. the paired secondary SR/SPL differences between the fixed method pairs;
3. whether the direction of each SR difference remains stable under the simultaneous interval;
4. how performance changes from seen sizes to OOD sizes 23/25 and across path-length bins;
5. how much true-wall correction improves BC and LeWM;
6. whether LeWM's historical development score depended materially on the corrected executor;
7. the system-level capability difference between the fixed Spatial-JEPA iterative system and the fixed vector latent-rollout system, without attributing that difference to one component.

## Claims this stage cannot support

The paper must not claim:

- that BC-versus-JEPA was prospectively preregistered before any confirmatory result was seen;
- that corrected scores measure autonomous planning ability;
- that a nonsignificant interval proves equality;
- that size OOD establishes texture, background, multitask, or cross-environment generalization;
- that wall-clock time across BC and CEM is a hardware-independent compute comparison;
- that cross-family score differences isolate representation quality, because supervision and training compute are not equalized;
- that one Procgen Maze result proves JEPA superiority in general;
- that an untested head, memory module, or loss would not help.
- that the experiment was prospectively powered to detect a prespecified minimum cross-family effect; achieved uncertainty is reported through intervals instead.

## Decision language

Use effect sizes and intervals before adjectives. For example:

> On the fixed 900-task confirmatory manifest, J1 minus BC had a paired SR difference of X with a Bonferroni simultaneous interval [L, U]. This was a fixed post-confirmatory secondary comparison.

If the interval overlaps zero, say that the experiment did not resolve the direction under the chosen uncertainty model. Do not write "the methods are equivalent."

If corrected greatly exceeds unmasked, attribute the gap to oracle executor assistance, not model planning.

## Hard stop

Once `CLOSURE_COMPLETE.json` exists and validates, this Maze experiment family ends. The next work product is paper writing, figure interpretation, limitations, and release packaging.
The executable entry points enforce this boundary by refusing further formal training, evaluation, summarization, or figure regeneration while the gate exists.

The following are not reasons to run another architecture:

- BC wins;
- JEPA wins by less than expected;
- vector LeWM is worse than its old corrected score;
- one seed is an outlier;
- OOD size 25 is difficult;
- an interval overlaps zero.

Future texture, color, multitask, or cross-environment generalization belongs to a separately named study with a new untouched holdout and a new protocol. It must not be appended to this closure matrix.
