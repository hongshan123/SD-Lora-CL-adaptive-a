# Late current-branch operator averaging

This isolated hypothesis targets noise in the arriving task's learned LoRA
branch. After Task0, collect the raw current operators `s_j B_j A_j` after each
optimizer/retraction step in the final quarter of the task. The window contains
at least two updates when available. Average in input/output coordinates so
changes of factor coordinates do not corrupt the mean. Project that mean onto
the final A row space, write its coefficients as the current branch, and retain
the original historical LS and exact absorption. Historical G and final A are
not averaged. Record projection residuals and the actual write reconstruction;
operator residuals do not bound classification accuracy.

Task0 is unchanged. The parent training loop, SGD, 20 epochs, physical single-GPU
batch128, rank10, global HOEP5%, prototype classifier and transport are retained.
NormCap is off. Averaging uses training states only and adds no deployment
parameters. The active float64 accumulator has fixed shape24×768×768 (108MiB),
with additional temporary matrices during accumulation/projection. Per-task
archives contain the raw final A/B/scale and the mean operator for later review.

The disabled controls must match the existing three-task HOEP checks at every
model/RNG boundary. Enabled checks must preserve Task0, exercise multiple terms
in each average, retain A exactly and reconstruct the projected current branch.
Short-check accuracy is not a gate. Both datasets must pass before either formal
run starts. Full results and all invalid attempts belong in the canonical parent
RESEARCH_LOG.md. This is a proposed independently implemented averaging variant,
not a claim of a verified improvement or a novel general averaging algorithm.
