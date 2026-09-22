# Functional-HOEP Phase A Results

Date: 2026-09-22

## Protocol

- T=10, rank 10, 20 epochs, two GPUs with batch size 64 per rank;
- CIFAR-100 seed 1993, ImageNet-R seed 1995, CUB-200 seed 1;
- operator-HOEP remained the active training mask;
- activation-aware functional masks were shadow diagnostics only;
- transport, Dual-B, HBD and bounded NormCap remained disabled.

All three runs ended with status 0. Calibration tensor/RNG invariance, finite-value checks, DDP synchronization and task-boundary operator equivalence passed throughout.

## Training Results

| Dataset | Final | AAA | Forgetting |
|---|---:|---:|---:|
| CIFAR-100 | 87.92 | 92.383 | 7.833 |
| ImageNet-R | 78.95 | 82.172 | 7.123 |
| CUB-200 | 84.10 | 89.445 | 8.739 |

These values reproduce the active operator-HOEP results. The shadow path did not alter its masks, gradients or deployed model.

## Diagnostic Results

| Dataset | Triggered transitions | Hit fraction | Mean Jaccard | Mean operator-mask functional exposure | Functional mixed branches |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 | 7/9 | 77.78% | 0.7850 | 5.23% | 65.28% |
| ImageNet-R | 1/9 | 11.11% | 0.8238 | 4.26% | 52.78% |
| CUB-200 | 0/9 | 0.00% | 0.8650 | 3.95% | 65.28% |

A transition is triggered when either:

```text
operator-mask functional exposure > 0.075
or Jaccard(operator mask, functional mask) < 0.8
```

No transition exceeded the 7.5% exposure threshold. CIFAR-100 triggers came entirely from mask Jaccard below 0.8. ImageNet-R crossed that threshold only at Task 3, while CUB-200 never crossed it.

## Preregistered Decision

The mixed-mask requirement passes globally, but only one dataset reaches a transition hit fraction of at least 30%. The registered requirement was at least two datasets.

Therefore:

```text
Phase A decision: NO-GO
Functional-HOEP Phase B: NOT STARTED
```

The diagonal activation weighting identifies a meaningful ranking mismatch on CIFAR-100, but not consistently across ImageNet-R and CUB-200. Under the preregistered interpretation, activation anisotropy is not a sufficiently general explanation for HOEP's lack of improvement over Frozen-A. This result should remain a failure-mechanism analysis rather than replace operator-HOEP as the method.
