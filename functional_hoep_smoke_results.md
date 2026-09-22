# Functional-HOEP Task0/1 Smoke Results

Date: 2026-09-22

## Protocol

- rank 10, Task 0/1, 2 epochs;
- two GPUs, batch size 64 per rank;
- `sa_hoep_energy_metric=operator`;
- `sa_hoep_functional_diagnostics=true`;
- transport, Dual-B and HBD disabled;
- deterministic task-boundary activation calibration with test preprocessing.

## Results

| Dataset | Seed | Final | AAA | Forgetting | Jaccard | Operator-mask functional exposure | Functional mixed branches |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 1993 | 95.60 | 96.00 | 0.80 | 0.8476 | 0.0544 | 79.17% |
| ImageNet-R | 1995 | 80.73 | 82.34 | 2.91 | 0.8255 | 0.0442 | 62.50% |
| CUB-200 | 1 | 92.49 | 94.68 | 6.09 | 0.8408 | 0.0359 | 66.67% |

All three runs completed both tasks without NaN/Inf. Every calibration reported unchanged deployment tensor and RNG hashes. Task-boundary absorption error was approximately `1.1e-7` to `1.3e-7`; HOEP retraction historical/current operator errors remained below `6.1e-8`.

## Shadow Invariance

A same-commit CIFAR-100 operator-HOEP control was run with diagnostics disabled. Control and shadow runs had identical:

- Top1 curve: `[96.4, 95.6]`;
- every common tensor in `sa_state.pt` (`max_abs_diff=0`);
- merged LoRA, FC head and prototype tensors (`max_abs_diff=0`).

The shadow state adds only `adaptive_a_input_rms` and `adaptive_a_input_counts`. The older 2026-09-21 P2 CIFAR run ended at 95.65 rather than 95.60, but its same-commit replacement establishes that this one-sample historical difference is unrelated to activation calibration.

## Decision

P2 smoke passes implementation and invariance checks. The single Task1 transition does not meet the registered transition-level trigger in any dataset, but this is not the formal Go/No-Go test. Mixed functional masks occur in more than 30% of branches on all three datasets. Proceed to the preregistered T=10 Phase A shadow diagnostic; do not start `functional_diag` training before analyzing Task1-9 artifacts.
