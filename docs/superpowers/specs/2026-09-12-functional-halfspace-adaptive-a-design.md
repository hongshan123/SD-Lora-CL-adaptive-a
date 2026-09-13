# Functional-Halfspace Adaptive-A Design

## Objective

Replace the Pareto controller's operator-drift proxy with a task-start teacher
signal that directly measures historical-branch functional drift on current-task
images.  The controller changes only the shared LoRA down projections `A`; the
existing fixed `(A, G)` state, coordinate alignment, absorption, prototype
transport, classifier, and current-task `B` optimization remain unchanged.

## Stability Signal

For task `t > 0`, construct the same frozen task-start teacher already used by
HBD.  On each training minibatch, compare the student's historical branch
activations with the teacher's corresponding activations using the existing
`hbd_historical_branch_distance`.  This distance is used only to obtain a
stability gradient; it is not added to the scalar training loss.

Let `d_n` denote the optimizer's proposed shared-A update direction, including
SGD momentum, and let `g_s` denote the gradient of the teacher stability loss.
For every shared-A matrix, use its thin QR basis to decompose both directions:

```
d_n = d_n_parallel + d_n_perp
g_s = g_s_parallel + g_s_perp
```

All shared Q/V A matrices are treated as one product parameter space.  Define:

```
c = sum_i <g_s_i, d_n_i>
h2 = sum_i ||g_s_perp_i||^2
```

If `c >= -conflict_tol`, retain `d_n`.  If `c < -conflict_tol` and the normal
channel is numerically usable, apply the global minimum-change correction:

```
d_i* = d_n_i - (c / h2) g_s_perp_i
```

This preserves every parallel component and gives
`sum_i <g_s_i, d_i*> = 0` up to floating-point error.  It solves:

```
min_d 0.5 * sum_i ||d_i - d_n_i||_F^2
s.t.  d_i_parallel = d_n_i_parallel for every i
      sum_i <g_s_i, d_i> >= 0
```

The guarantee concerns only the first-order contribution of the shared-A
update.  It is not a guarantee on the whole optimizer step, current B, the
classifier, finite-step stability, or old-class accuracy.

Functional-halfspace projection tensors must use `torch.float32` or
`torch.float64`. Within every Q/V branch, the proposed direction, stability
gradient, and shared-A tensor must have exactly the same dtype. FP16 and BF16
are rejected because writing the correction back at those precisions can
quantize it across the hard stability halfspace boundary.

## Degenerate Normal Channel

The normal correction is considered unusable when either `h2` is below the
absolute threshold or `h2 / sum_i ||g_s_i||^2` is below the configured minimum
normal fraction.  In that case, use the full-space projection:

```
d_i* = d_n_i - (c / sum_j ||g_s_j||^2) g_s_i
```

This sacrifices parallel-component preservation but still enforces the
first-order stability halfspace.  If the complete stability gradient norm is
also below the absolute threshold, leave the update unchanged and record a
degenerate no-op.  Epsilon is a decision threshold only; it is not added to a
denominator because doing so would silently violate the hard constraint.

## Optimizer Semantics

For SGD momentum `mu` and existing momentum buffer `v`, project the direction
`d_n = grad + mu * v`.  Before `optimizer.step()`, write
`parameter.grad = d* - mu * v`, so PyTorch reconstructs `d*`.  The strategy is
restricted to SGD, matching the existing discrete Adaptive-A strategies.

## Lifecycle And DDP

- Task 0 has no teacher and keeps the unmodified live-A update.
- A frozen teacher is created before each later task without enabling the HBD
  scalar loss.
- The additional student/teacher pass preserves RNG state and disables input
  sketch capture.
- `autograd.grad` stability gradients are manually averaged across DDP ranks.
- The teacher and hooks are released by the existing end-of-task cleanup.
- No teacher, gradient, decision, or diagnostic is persisted in checkpoints.

## Configuration And Diagnostics

The new strategy name is `functional_halfspace`.  It uses:

- `sa_functional_conflict_tol`, default `1e-12`, strictly non-negative.
- `sa_functional_normal_tol`, default `1e-12`, strictly positive.
- `sa_functional_min_normal_fraction`, default `1e-4`, in `[0, 1]`.

Task diagnostics report observations, conflicts, normal-space projections,
full-space fallbacks, degenerate no-ops, mean pre/post inner products, mean
normal fraction, and mean correction-to-proposed-direction norm ratio.

## Non-Goals

- Do not add HBD to the scalar objective.
- Do not constrain B or classifier gradients.
- Do not change prototype transport, Dual-B/FC head behavior, coordinate
  alignment, NormCap/absorption, or persistent checkpoint state.
- Do not add a learned router, replay samples, old data, or per-task parameters.
