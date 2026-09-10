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

The already synchronized raw update from the preceding independently
shuffled minibatch supplies `D_control`. This preserves the original batch
and sample count while measuring cross-minibatch update agreement. The
signed utility of candidate `k` is the normalized update agreement:

```
U_k = <D_k, D_control> / (||D_k|| ||D_control|| + eps).
```

Historical risk is measured in effective-operator space and weighted by a
fixed-size diagonal activation sketch `S`:

```
delta_A_k = -learning_rate * D_k
A_hat_new = normalize(A + delta_A_k)
R_k = ||G (A_hat_new - A_hat) S||_F^2
      / (||G A_hat S||_F^2 + eps).
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

Risk-budgeted Adaptive-A caches only the preceding minibatch's synchronized
raw shared-A update. It performs no additional forward/backward, does not
change the effective batch or sample order, and cannot mutate classifier, B,
scale, or prototype gradients. Task 0 uses Live; the first minibatch of each
later task keeps A frozen while establishing its control update. Risk uses
the actual optimizer displacement `delta_A = -learning_rate * D`. With SGD
momentum,
candidates are formed from `momentum * buffer + gradient`, and the written
gradient compensates the old buffer so `optimizer.step()` applies the exact
selected Frozen, Tangent, or Live update.

## Diagnostics

Log per-task mode fractions, selected/available risk, signed utility, and
per-layer dominant mode. Existing impact-ratio diagnostics remain unchanged.
