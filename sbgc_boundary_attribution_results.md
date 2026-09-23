# SBGC Task-Boundary Attribution Results

Date: 2026-09-23

## Protocol

All three T=1 joint-data references and three T=2 attribution runs exited with status 0. The attribution runs retain the P2 seed, rank 10, 20 epochs/task, and two GPUs with batch size 64 per GPU. At the Task 1 boundary, additive, uniform-budget, and Fisher-budget G candidates are evaluated using the **same trained model and fixed classifier heads**. The prototype head is computed once from the deployed Fisher state and is not recalibrated for counterfactual candidates. Test samples are used for diagnosis only, never to select a deployed candidate.

## T=1 Joint-Data Reference

| Dataset | Seed | T=1 Final |
|---|---:|---:|
| CIFAR-100 | 1993 | 90.61 |
| ImageNet-R | 1995 | 82.23 |
| CUB-200 | 1 | 87.99 |

These are same-rank, same-total-class references, **not** a theoretical or merge-only upper bound. They differ from T=10 in task order, initial A acquisition, and classifier optimization.

## Task 1 Counterfactuals

Numbers below are global Top-1 (%). Old/new refer to classes from Task 0/Task 1. `Proto` uses the fixed deployed prototype head.

| Dataset | Candidate | FC old | FC new | FC all | Proto old | Proto new | Proto all |
|---|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | Additive | 96.90 | **95.90** | **96.40** | 96.00 | **97.20** | **96.60** |
| CIFAR-100 | Uniform | **97.80** | 93.90 | 95.85 | **97.10** | 95.60 | 96.35 |
| CIFAR-100 | Fisher | 97.70 | 93.90 | 95.80 | **97.10** | 95.70 | 96.40 |
| ImageNet-R | Additive | 86.71 | **92.10** | **89.57** | 83.47 | **89.51** | **86.67** |
| ImageNet-R | Uniform | **87.20** | 88.51 | 87.89 | **85.90** | 86.35 | 86.14 |
| ImageNet-R | Fisher | 87.03 | 88.65 | 87.89 | **85.90** | 86.78 | 86.37 |
| CUB-200 | Additive | 87.83 | **88.34** | **88.08** | 90.61 | **95.71** | **93.18** |
| CUB-200 | Uniform | **89.04** | 86.45 | 87.74 | **91.13** | 95.03 | 93.09 |
| CUB-200 | Fisher | **89.04** | 86.45 | 87.74 | **91.13** | 95.03 | 93.09 |

Training-state versus additive FC logit relative L2 errors are `7.60e-7 / 1.12e-6 / 6.95e-7` for CIFAR-100 / ImageNet-R / CUB-200. Maximum absolute differences are `3.24e-5 / 2.34e-5 / 2.53e-5`: numerically equivalent at this precision, not bitwise equal.

The deployed Fisher T=2 prototype Final is `96.40 / 86.37 / 93.09`. These agree with the logged full evaluation. All Task 1 runs activate the 5% constraint in 24/24 Q/V branches; mean current-target distortion is `0.475 / 0.445 / 0.505` respectively.

## Interpretation

At the first merge, the constrained candidates improve old-class accuracy but reduce new-class accuracy more. Additive has the highest fixed-head overall Top-1 on all three datasets, and Fisher gives little improvement over uniform. This directly explains the P2 observation that frequent active constraints and output-gradient anisotropy do not translate into a consistent Final/AAA gain. The observed tradeoff is compatible with excessive restriction of the current write; this is a causal counterfactual for Task 1 only, not a proof that the same effect dominates all later tasks.

No tuning decision should use these test-set counterfactuals. Before changing the algorithm, inspect later task boundaries under a prespecified diagnostic protocol and separate classifier/prototype recalibration effects. Do not describe the T=1 reference minus T=10 performance as a merge-only gap.
