# Adaptive-A Coordinate Plasticity Design

## Objective

Add a layerwise adaptive shared-`A` update rule to the no-AOB
CoordinateStable SD-LoRA baseline at commit `4ca248a`. The rule must preserve
the fixed deployment state `(A, G)`, existing coordinate alignment, bounded
NormCap absorption, prototype transport, Dual-B, and classifier behavior.

The mechanism addresses the observed dataset-dependent trade-off: freezing
`A` after Task 0 improves CUB-200 and slightly improves CIFAR-100, but reduces
ImageNet-R accuracy. It therefore controls row-space plasticity per layer and
per optimization step instead of choosing one global freeze policy.

## Scope

- No AOB merge mode or AOB persistent state.
- No extra model copy, warm-up epoch, replay data, or old-task samples.
- No additional trainable parameters.
- No per-task persistent state. Adaptive statistics are task-local diagnostics.
- Task 0 is exactly the existing trainable-`A` baseline.
- Only `live_a_aggregate_b` with CoordinateStable alignment is supported.

## Adaptive Gradient Rule

For one Q or V branch in layer `l`, let:

- `A` have shape `[r, d]`;
- `D = dL/dA` have shape `[r, d]`;
- `B` and historical aggregate `G` have shape `[d, r]`;
- `s` be the current learned LoRA scale;
- `Q` have shape `[r, d]` and contain an orthonormal basis for the row space
  of `A`, obtained by the existing `canonical_down_projection` helper.

Split the gradient into row-space-preserving and row-space-changing parts:

```text
D_parallel = (D Q^T) Q
D_perp     = D - D_parallel
```

The first-order current-task and historical operator changes induced by
`D_perp` are:

```text
delta_cur  = (s B) D_perp
delta_hist = (G / (||A||_F + eps)) D_perp
```

Their Frobenius norms are evaluated without materializing a `[d, d]` matrix:

```text
||U V||_F^2 = trace((U^T U)(V V^T)).
```

Combine Q and V branch effects into one transformer-layer score:

```text
P_l = sqrt(||delta_cur_q||_F^2  + ||delta_cur_v||_F^2)
H_l = sqrt(||delta_hist_q||_F^2 + ||delta_hist_v||_F^2)
```

The instantaneous layer gate is:

```text
g_raw_l = clamp(P_l / (P_l + lambda H_l + eps), floor, 1)
```

If no historical aggregate exists, or both branch perpendicular gradients
are numerically zero, `g_raw_l = 1`. This makes Task 0 and gauge-only updates
identical to the baseline.

Use a task-local exponential moving average:

```text
g_l = momentum g_previous_l + (1 - momentum) g_raw_l
```

The first observation initializes the EMA directly from `g_raw_l`. Apply the
same layer gate to Q and V:

```text
D'_q = D_parallel_q + g_l D_perp_q
D'_v = D_parallel_v + g_l D_perp_v
```

This rule does not claim to freeze all entries of `A`; it adapts only the
coordinate-changing component. Pure within-row-space gauge motion remains
available because existing task-boundary alignment can represent it exactly.

## Configuration

The feature is opt-in and disabled by default:

```json
"sa_adaptive_a_enabled": true,
"sa_adaptive_a_stability_weight": 1.0,
"sa_adaptive_a_gate_floor": 0.05,
"sa_adaptive_a_gate_momentum": 0.9,
"sa_adaptive_a_eps": 1e-8
```

When enabled, configuration validation requires:

- `sa_train_a_all_tasks=true`;
- `sa_cumulative_state=true`;
- `sa_cumulative_merge="live_a_aggregate_b"`;
- `sa_live_a_coordinate_align=true`.

## Training And DDP Integration

The base SD-LoRA learner exposes a no-op post-backward hook. Both Task 0 and
incremental training loops call it after `loss.backward()` and before
`optimizer.step()`. `sa_sdlora.Learner` overrides the hook and delegates to
the backbone only when Adaptive-A is enabled.

DDP has already reduced parameter gradients when the post-backward hook is
called, so all ranks compute the same gate from the same `A`, `B`, `G`, scale,
and synchronized gradient. No new collective is required.

## Diagnostics

Accumulate task-local statistics and log after each task:

- mean/min/max layer gate;
- fraction of gate observations below `0.1` and above `0.9`;
- mean current impact `P_l` and historical impact `H_l`;
- mean perpendicular-gradient retention;
- per-layer mean gate vector.

No diagnostic is serialized into `sa_state.pt`; run configuration and logs
are sufficient for reproducibility.

## Tests

- Low-rank Frobenius norm equals explicit dense multiplication.
- Gradient decomposition reconstructs the raw gradient and makes
  `D_perp Q^T` approximately zero.
- No-history behavior preserves the original gradient exactly.
- Dominant historical impact lowers the gate; dominant current impact raises
  it; floor and EMA are respected.
- Q and V branches in one layer receive the same gate.
- Frozen-`A` configuration is rejected when Adaptive-A is enabled.
- Feature-disabled training executes the no-op hook and remains unchanged.
- Backbone state save contains only the pre-existing `(A, G)` state.

## Experiment Acceptance

First run one smoke task and one full seed per dataset using the same effective
batch size, optimizer, learning rate, epochs, seed, task order, and DDP world
size as the no-AOB baseline. Compare against fully trainable `A` and frozen
`A`. Promote to seeds 1-3 only if the gate is non-degenerate and at least two
datasets do not regress in both Final and AAA.

