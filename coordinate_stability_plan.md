# Coordinate-Stable Live-A Development Plan

## Objective

Test whether historical-operator alignment plus a constrained prototype
transport can reduce the representation/prototype mismatch of Live-A
Aggregate-B without replay, task IDs, a router, or task-growing LoRA state.

## Frozen Method

For every Q/V branch, preserve the task-start historical operator when the
shared down-projection changes:

```text
G_aligned = argmin_X ||X normalize(A_new) - G_old normalize(A_old)||_F
G_next = G_aligned + s_t B_t / ||B_t||_F
```

After rebuilding the saved deployment state, extract paired current-task
features before and after the update.  Fit a rank-10 orthogonal rotation only
inside the top feature-displacement subspace.  A deterministic 80/20 split
gates the map; without held-out improvement it becomes identity.  Apply it
once to historical prototypes and discard the basis and rotation.

## Protocol

- CIFAR-100: seed 1993, T=10, 20 epochs/task, batch 32.
- ImageNet-R: seed 1995, T=10, 20 epochs/task, batch 32.
- Four-card NCCL on GPUs 0-3 after the frozen P7 queue completes.
- All optimizer, task-order, classifier, and Dual-B settings match the
  corresponding completed development runs.
- No transport rank, regularization, damping, or schedule sweep is allowed.

## Development Gate

Reference full-method development results:

| Dataset | Final | AAA | Forgetting |
| --- | ---: | ---: | ---: |
| CIFAR-100 | 88.32 | 91.856 | 8.233 |
| ImageNet-R | 78.84 | 83.067 | 6.624 |

The direction is promising only if Final drops by no more than 0.20 and at
least one of AAA/Forgetting improves materially on CIFAR-100, without a new
greater-than-0.20 regression in the other metric.  ImageNet-R must then show
no greater-than-0.20 regression in Final or AAA.  These development seeds do
not support a paper claim; passing results require preregistered multi-seed
confirmation.

## State Boundary

Persistent learned state remains one shared A, one Aggregate-B G, classifier
state, and class prototypes.  Coordinate alignment and prototype transport
matrices are temporary and must not be written into the model artifact.
