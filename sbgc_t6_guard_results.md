# SBGC T=6 Boundary Attribution and Guard Smoke

Date: 2026-09-23. Commit for T=6 diagnostic: `c12b227`; run configs:
`660d040`. Guard implementation: `3e82a45`. All three T=6 attribution
runs and all three Task 0/1 guard smokes exited with status 0.

## Boundary Attribution

Protocol: same trained Fisher state, same global FC, three candidate G values.
For `Proto`, historical prototypes stay fixed; each candidate recomputes
current-class prototypes from current-task training data with test transforms.
All numbers are Top-1 percent on seen test classes, used only for diagnosis.

| Dataset | Task | FC old A/F | FC new A/F | Proto old A/F | Proto new A/F | Proto all A/F |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 3 | 92.73 / 93.23 | 94.00 / 92.40 | 93.67 / 94.40 | 95.90 / 94.30 | 94.23 / 94.38 |
| CIFAR-100 | 5 | 89.36 / 90.66 | 91.90 / 89.60 | 91.06 / 92.28 | 93.60 / 91.20 | 91.48 / 92.10 |
| ImageNet-R | 3 | 83.27 / 84.08 | 84.56 / 83.06 | 80.97 / 82.96 | 86.06 / 82.91 | 82.26 / 82.95 |
| ImageNet-R | 5 | 79.98 / 81.73 | 85.31 / 85.00 | 79.91 / 80.74 | 86.72 / 83.91 | 81.10 / 81.29 |
| CUB-200 | 3 | 83.83 / 83.89 | 85.99 / 85.64 | 90.80 / 90.80 | 87.77 / 86.88 | 90.06 / 89.84 |
| CUB-200 | 5 | 75.95 / 75.95 | 83.19 / 82.12 | 87.97 / 88.15 | 92.74 / 92.21 | 88.75 / 88.81 |

`A/F` means counterfactual additive / Fisher. Uniform was also measured in
the JSON artifacts. CIFAR-100 and ImageNet-R meet the previously recorded
tradeoff gate in `sbgc_plasticity_guard_prereg.md`; CUB-200 does not. This is
why the guard was implemented. It is **not** evidence that relaxing Fisher
will improve overall accuracy: Fisher's prototype-all Top-1 exceeds additive
at both later boundaries in CIFAR-100 and ImageNet-R.

Deployed Fisher T=6 prototype Top-1 curves are:

- CIFAR-100: 98.50, 96.40, 95.30, 94.38, 93.10, 92.10.
- ImageNet-R: 90.60, 86.37, 85.26, 82.95, 81.70, 81.29.
- CUB-200: 96.87, 93.09, 91.60, 89.84, 89.59, 88.81.

## Guard Smoke

Three two-GPU Task 0/1 runs used 2 epochs/task, batch 64 per GPU,
class-stratified 10% holdout only from Task 1, and 1% relative CE tolerance.
All completed. DDP tensor/RNG checks passed and v7 states were saved.

| Dataset | Train / holdout | Selected | Additive CE | Selected CE | Limit |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 | 4500 / 500 | 0.05 | 2.504933 | 2.529818 | 2.529982 |
| ImageNet-R | 2396 / 268 | 0.05 | 3.302765 | 3.330446 | 3.335792 |
| CUB-200 | 540 / 60 | 0.05 | 3.199547 | 3.202441 | 3.231542 |

The smoke verifies execution, not method gain. In all three cases the 5%
candidate was already within the CE tolerance, so the guard made no different
choice. In particular, a relative CE allowance grows with the large
`log(number of classes)` baseline of a cosine-prototype classifier. The full
20-epoch T=10 run must report how often the guard actually relaxes the cap;
if it rarely does, this formulation has no distinct empirical contribution.
