# CUO Low-Rank Projection Design

## Goal

Add an isolated `sa_cumulative_merge="cuo_lowrank"` baseline that retains
the cumulative least-squares mechanism of Cumulative Unified Optimization
(CUO) while constraining the deployed Q/V adaptation state to the existing
rank-10, 0.369M Shared-A LoRA factor budget.

## Scope and Compatibility

`cuo_lowrank` is a new cumulative merge mode.  It must not modify the
behavior or artifact formats of `gauge`, `union_svd`, or
`live_a_aggregate_b`.

The mode is intentionally incompatible with options that require a live
historical shared-A coordinate system: Adaptive-A, CoordinateStable
alignment/transport, HBD, and function-safe Adaptive-A strategies.  The
classifier, prototype classifier, task protocol, optimizer, augmentation,
and evaluation code remain unchanged.  The initial experiments use no
Dual-B head, matching the current historical-signal protocol.

## CUO Restricted to a Fixed Low-Rank Coordinate System

For each selected ViT block and each Q/V branch, let `d=768`, `r=10`, and
let `P in R^(r x d)` be a persistent, row-orthonormal input projection.  Let
`H in R^(d x r)` be the persistent unified up projection.  At task `t`, the
temporary trainable LoRA factors are `A_t in R^(r x d)`,
`B_t in R^(d x r)`, and scalar `s_t`.

The training forward is

```
delta_t(x) = H_(t-1) P x + s_t B_t A_t x.
```

At Task 0 completion, `P` is initialized from the row-orthonormal QR
canonicalization of the trained `A_0`.  For later tasks, the trainable
`A_t` is initialized from `P`, but `P` itself remains immutable.  Thus every
CUO sufficient statistic uses one stable feature coordinate system even
though the temporary task LoRA is trained freely.

On one deterministic, no-gradient pass over the current task training set,
the final trained model supplies hidden states `x_j` and branch residual
targets `y_j=delta_t(x_j)`.  Define `z_j=P x_j`.  We accumulate

```
C_t = C_(t-1) + sum_j z_j z_j^T                  # [r, r]
D_t = D_(t-1) + sum_j y_j z_j^T                  # [d, r]
H_t = D_t (C_t + lambda I)^(-1).                 # [d, r]
```

Instead of storing `D_t`, the artifact stores `H_t` and `C_t`.  Before the
next task, recover `D_t = H_t (C_t + lambda I)`.  This is algebraically
equivalent in exact arithmetic and keeps the deployed forward directly in
the persistent `H_t P` form.

The update is the CUO normal equation restricted to the rank-10 operator
class `H P`; it is not a full CUO reproduction, which stores dense
`d x d` feature covariance and `d x d` adaptation matrices.

## State Budget

There are 12 blocks and two branches per block.  Persistent LoRA factors
contain

```
24 * (768 * 10 + 10 * 768) = 368,640
```

FP32 scalars, reported as `0.369M` LoRA state parameters, exactly matching
the rank-10 Shared-A aggregate baseline.  CUO also needs 24 small Gram
matrices `C_t`, contributing `24 * 10 * 10 = 2,400` FP32 bookkeeping
scalars.  Artifact reporting must show both values honestly: 0.369M
deployed low-rank factors and 2,400 cumulative-statistic scalars
(approximately 0.371M FP32-equivalent persistent scalars in total).

No per-task LoRA, sample feature, target, or full `768 x 768` statistic may
be saved.

## Runtime and DDP Semantics

`_CUOLowRankQKV` owns immutable buffers `projection_q/v`, persistent
`unified_up_q/v`, and task-local accumulation buffers.  During the
post-training calibration pass it accumulates the local `Z^T Z` and
`Y^T Z` for its branch.  The calibration pass uses the current task training
set with evaluation transforms, no shuffle, and no gradients.  Each branch
all-reduces its local statistics across DDP ranks before rank 0 solves the
normal equation and writes the artifact.  The solved `H_t` and `C_t` are
broadcast before any rebuild/evaluation path uses them.

The solver must use `torch.linalg.solve` on the `r x r` positive-definite
system rather than explicitly forming an inverse.  It computes in FP64 on
CPU or the statistics device, then stores FP32 factors.  Configurable
`sa_cuo_lambda > 0` provides Tikhonov damping.  Logs record calibration
token count, condition number of `C_t + lambda I`, residual fit error,
and the factor/statistic state count.

## Artifact and Configuration Contract

Add a new explicit artifact version.  It includes:

```
version, task_id, rank, merge_mode="cuo_lowrank",
projection_down, unified_up, projected_gram, cuo_lambda
```

Old artifacts must fail with a clear migration/incompatibility error; no
silent conversion is valid because prior states do not contain projected
normal-equation statistics.  The new mode requires
`sa_cumulative_state=true`, `sa_train_a_all_tasks=true`, rank equality
between `lora_rank` and `sa_cumulative_rank`, and positive
`sa_cuo_lambda`.  It rejects live-A-only settings described above.

## Tests and Acceptance

1. A synthetic projected regression test verifies that the cumulative
   `H` equals the direct least-squares solution over concatenated tasks.
2. A task-order/additivity test verifies that merged sufficient statistics
   produce the same solution as one combined calibration set.
3. A backbone test verifies Task 0 projection initialization, Task 1
   temporary-A reset from `P`, and that historical forward uses immutable
   `P` even after `A_t` changes.
4. Save/load and rebuild tests verify identical fixed-input outputs and
   exactly one persistent state entry per branch after multiple tasks.
5. DDP-style tests verify all-reduced `C`/`D` contributions are counted
   once and rank 0 is the only writer.
6. State accounting verifies 368,640 Q/V factor scalars at rank 10 and
   exactly 2,400 additional Gram scalars for a 12-block ViT.

## Initial Experiments

Create single-seed rank-10 configurations for CIFAR-100, ImageNet-R, and
CUB-200 with the current no-Dual-B protocol, the same task splits, epochs,
optimizer, learning rate, and effective batch size as the current
historical-signal runs.  Run the unit tests and a Task 0/1 DDP smoke test
before launching any full benchmark.
