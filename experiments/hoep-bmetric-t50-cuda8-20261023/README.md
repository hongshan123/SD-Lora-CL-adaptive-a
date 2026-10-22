# Input-metric preconditioning of B gradients

This extension preserves current-input and historical-input metric variants,
their tests, paired configuration templates and queue code. No runtime records
or performance results are included.

After Task0, right-precondition each Q/V B gradient with a damped rank-space
input metric, then restore its instantaneous Frobenius norm. The current variant
uses each batch's `z = A x`; the historical variant also uses `A C_old A^T`,
with each metric normalized by its trace. Damping is 0.1. Historical moments
are collected when each task's own training data is available. The deployment
adapter shapes remain unchanged. SGD momentum means preserving gradient norm
does not preserve the actual parameter-step norm.

This queue reuses Frozen-A and HOEP controls from the adjacent paired suite.
Prepare that suite first and complete its three-task checks before this queue's
Task0 comparison gate. From the repository root:

```bash
python scripts/prepare_hoep_queue.py \
  --suite experiments/hoep-bmetric-t50-cuda8-20261023 \
  --primary-root experiments/hoep-t50-improvement-cuda8-20261023 \
  --data-root /path/to/data --gpus 4 5
cd experiments/hoep-bmetric-t50-cuda8-20261023
python run_queue.py
```

The preparation script creates ignored local paths and integrity manifests;
it never starts training or copies existing run state. Individual configuration
templates can also be used through `run_job.py` after setting a fresh absolute
output path and linking `source/data`.

This is an independent experimental implementation inspired by low-rank
optimization literature; it is not an official LoRA-RITE or LoRA-Pro release.
