# Risk-Budgeted Adaptive-A Design

## Goal

Replace the current magnitude-only Adaptive-A gate with an endpoint-capable
policy that can match Frozen-A, retain only coordinate-preserving motion, or
match Live-A independently at each layer. Keep the existing CoordinateStable
alignment, prototype transport, absorption, classifier, and fixed `(A, G)`
deployment state unchanged.

## Decision Rule

For each layer, decompose the synchronized training gradient into row-space
and normal-space components:

```
D = D_parallel + D_perp.
```

The three candidates are:

```
Frozen:  D' = 0
Tangent: D' = D_parallel
Live:    D' = D
```

A held-out control split from the same minibatch supplies `D_control`. The
signed utility of candidate `k` is the normalized gradient agreement:

```
U_k = <D_k, D_control> / (||D_k|| ||D_control|| + eps).
```

Historical risk is measured in effective-operator space and weighted by a
fixed-size diagonal activation sketch `S`:

```
R_k = ||G D_k S||_F^2 / (||G A_hat S||_F^2 + eps).
```

Q and V utility/risk are summed per layer. A deterministic Lagrangian search
selects one candidate per layer while keeping total selected risk below the
configured global budget. Negative-utility updates may therefore select the
true Frozen endpoint.

## State And Compatibility

- `sa_adaptive_a_strategy=impact_ratio` preserves the existing gate exactly.
- `sa_adaptive_a_strategy=risk_budgeted` enables the new policy.
- The historical activation sketch stores one vector per LoRA branch and a
  scalar sample count. Its size is independent of task count.
- Task-local control gradients, mode choices, and diagnostics are not saved.
- The deployment representation remains only shared `A` and aggregate `G`;
  the sketch is training metadata and is ignored by deployment forward.
- Task 0 uses Live mode because no historical operator exists.

## Training Integration

Only risk-budgeted Adaptive-A splits each minibatch deterministically into a
training subset and a disjoint control subset. The ordinary backward pass is
performed on the training subset. A second `autograd.grad` call obtains only
the shared-A control gradients; it does not mutate classifier, B, scale, or
prototype gradients. Under DDP, control gradients are explicitly averaged.

## Diagnostics

Log per-task mode fractions, selected/available risk, signed utility, and
per-layer dominant mode. Existing impact-ratio diagnostics remain unchanged.

