# SBGC Plasticity-Guard Decision (Pre-registered)

Date: 2026-09-23. This decision rule is recorded while the T=6 boundary
attribution queue is still running. Task 3 is available only for CUB-200;
Task 5 and the CIFAR-100/ImageNet-R later boundaries have not been inspected.

## Gate

At Task 3 or Task 5, count a dataset as showing an actionable
old/new tradeoff only if Fisher versus additive, under the same trained model:

- global FC old-class Top-1 improves by at least 0.30 percentage points;
- global FC new-class Top-1 decreases by at least 0.30 percentage points;
- with candidate-specific recalibration of *current-class* prototypes only,
  prototype old-class Top-1 does not fall by more than 0.10 points and
  prototype new-class Top-1 falls by at least 0.30 points.

Implement the plasticity guard only if at least two of the three datasets
meet all three conditions at one or both inspected later boundaries. Otherwise
stop this branch of SBGC: a smaller G-response risk has not demonstrated a
useful old-class benefit to trade against current-class plasticity.

The test-set attribution is diagnostic only. It must not select a deployed
candidate, budget, or dataset-specific threshold.

## Conditional Guard Protocol

If the gate passes, use a deterministic, class-stratified 10% holdout from
*current-task training data* for Task 1 onward. Exclude it from adapter/head
training, SBGC calibration, and prototype construction. Use the same split
for matched additive and SBGC controls. Task 0 remains unchanged.

For each later task, evaluate candidates on the existing SBGC solution path
at risk budgets `{0.05, 0.10, 0.20, 0.40, 0.80}` plus unconstrained additive.
For each candidate, keep historical prototypes fixed and recompute only
current-class prototypes from the non-holdout current-task training samples.
Use global prototype-classification cross-entropy on the holdout. Choose the
candidate with the smallest measured aggregate historical response risk among
those whose holdout CE is at most 1% above additive CE. Additive is always a
feasible fallback. Do not use old-class or test examples for selection.

Report the actual selected historical risk and budget. A selected budget above
5% is an explicit relaxation, not a claim that the original 5% constraint
still holds. Compare with same-split additive, uniform, and fixed-5% Fisher
controls before making any accuracy claim.
