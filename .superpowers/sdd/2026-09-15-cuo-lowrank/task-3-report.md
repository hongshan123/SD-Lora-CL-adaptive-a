# Task 3 Report: CUO State, Calibration, and DDP

## Scope

Implemented Task 3 only. Modified `backbone/sa_lora.py`,
`models/sa_sdlora.py`, `models/sdlora.py`, and
`tests/test_cuo_lowrank.py`; added `tests/ddp_smoke_cuo_lowrank.py`.
`models/sdlora.py` received the smallest required lifecycle extension: an
all-rank `_before_task_save()` no-op hook and one invocation immediately
before the unchanged rank-zero writer. Non-CUO learners retain the no-op
implementation and their save calls remain in the original branch.

## Code Paths

- `SharedALoRA_ViT_timm.prepare_cuo_calibration()` creates P from the trained
  Task-0 A only on rank 0, broadcasts it before collection, and never
  re-derives P for later tasks.
- `_CUOLowRankQKV` now retains branch-local FP64 C/D/count accumulators plus
  persistent-in-memory projected Gram buffers. `finalize_cuo_calibration()`
  performs one FP64 all-reduce each for C, D, and count per Q/V branch,
  recovers `D_old = H_old(C_old + lambda I)`, calls Task 1's
  `solve_projected_cuo()` on rank 0, broadcasts P/H/C, then removes the
  temporary B factors on every rank.
- `_save_cuo_lowrank_state()` writes only version-5 CUO fields:
  `version`, `task_id`, `rank`, `merge_mode`, `projection_down`,
  `unified_up`, `projected_gram`, and `cuo_lambda`. CUO rejects all other
  state formats. `save_merged_lora()` writes only H/P deployment factors and
  metadata.
- `sa_sdlora.Learner._before_task_save()` obtains current-task training data
  with `mode="test"`, shards it exactly across ranks without padding, runs a
  deterministic no-grad eval pass, and finalizes before the inherited
  rank-zero state writer. It logs calibration tokens, condition number,
  residual, LoRA-factor scalars, Gram scalars, and their persistent total.
- `cuo_state_scalar_counts(12, 10, 768)` reports 368640 deployed factor
  scalars, 2400 Gram bookkeeping scalars, and 371040 persistent scalars.

## TDD Evidence

Initial red run after adding the Task 3 tests:

```
ImportError: cannot import name 'combine_cuo_statistics' from 'backbone.sa_lora'
```

Learner-boundary red run:

```
AttributeError: type object 'Learner' has no attribute '_before_task_save'
```

All-rank deployment red run:

```
FAILED test_cuo_finalization_retires_temporary_b_on_every_rank
assert all(torch.count_nonzero(weight.weight) == 0 ...)
```

These became green after the corresponding reducer/state API, pre-save hook,
and finalization transition were implemented.

## Verification

CPU suite:

```
python -m pytest tests/test_cuo_lowrank.py tests/test_sa_cumulative.py tests/test_coordinate_stability.py tests/test_sdlora_dual_b.py -q
74 passed in 3.59s
```

Compilation and diff validation:

```
python -m py_compile backbone/sa_lora.py models/sa_sdlora.py models/sdlora.py tests/ddp_smoke_cuo_lowrank.py
git diff --check
```

Two-GPU DDP smoke, available on this host (`torch.cuda.device_count() == 8`):

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tests/ddp_smoke_cuo_lowrank.py
CUO_DDP_SMOKE_PASS ranks=2 final_hash=4789f533db40a2d4232abfb62e70795990a11f9ac032554484d7940a3fde0559 artifacts=['sa_merged_lora.pt', 'sa_state.pt']
```

The smoke calibrates/saves Task 0 and Task 1, asserts identical P/H/C hashes
on both ranks after each task, verifies the expected all-rank token count, and
asserts the fresh run directory contains only the two CUO artifacts with no
per-task B files. PyTorch emitted its standard `torchrun` OMP advisory and a
non-fatal NCCL `barrier()` device-context advisory.

## Commit

`feat: persist and synchronize low-rank CUO state` (Task 3 scoped commit,
including this report and the permitted `models/sdlora.py` lifecycle hook).

## Risks

- CUO is intentionally incompatible with every prior Shared-A state version;
  old artifacts now fail rather than being silently converted.
- The calibration pass adds one deterministic current-task eval traversal at
  every task boundary. Its ridge solve is intentionally a rank-constrained
  approximation, so residual diagnostics should be monitored in full runs.
- The DDP smoke covers two CUDA ranks and both task boundaries, but it is a
  tiny synthetic ViT rather than a full dataset benchmark.
