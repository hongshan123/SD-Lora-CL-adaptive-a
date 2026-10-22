# Method overview

For each adapted Q/V projection, the training update is
`M = G_previous A / (||A||_F + eps) + s B A`.
The historical state uses shared A and aggregate G; the current task learns B
and scale s. Adapter factor shapes stay fixed as tasks arrive. Classifier
prototypes and optional training statistics are counted separately.

HOEP canonicalizes and rotates factor coordinates while preserving their
operator, then allocates plastic A directions using historical operator energy
and a global energy budget. See `backbone/operator_energy_partition.py` and
`backbone/sa_lora.py` for the low-energy policy and direction controls.

Historical least-squares alignment re-expresses the old operator in updated A
coordinates. Exact absorption incorporates current B into G. NormCap is a
separately configurable write modification. Prototype classification uses
training-set class means; optional transport fits a mapping using paired
current-task training features to update old prototypes.

The isolated `experiments/hoep-lora-adaptation-t50-cuda8-20261022/source/`
implementation adds two optional training losses after Task0:

1. Historical-input-weighted total update: accumulate layer input second
   moments C from each task's own training images, then penalize
   `sum trace((M-M0) C (M-M0)^T)` normalized by historical response energy.
   This includes A and current B and permits compensating factor changes.
2. All-seen-class prototype loss: initialize current prototypes from current
   training images, combine with stored old prototypes and train features
   against all seen classes. Detached new-class EMA targets are updated after
   each batch's loss is evaluated.

These are experimental mechanisms. Historical statistics can become stale as
upstream features change; a local response penalty is not a bound on classifier
forgetting. Diagnostic tools may read historical data for evaluation, which
must be distinguished from rehearsal-free training. Tests do not participate
in constructing prototypes or choosing an update rule.
