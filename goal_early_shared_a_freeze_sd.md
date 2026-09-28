# Early Shared-A Freeze: Paired Prefix Experiment

## Approved Scope

Evaluate delayed freezing before designing a gate. Train Task 0/1/2 with
Live-A, save one common completed boundary, then continue from that exact
boundary with either Frozen-A or Live-A. Do not select a policy using test
labels. The two continuations are research comparisons, not an online oracle.

- CUB-200 seed 1 and ImageNet-R seed 1995.
- T=10, rank 10, 20 epochs/task, two GPUs with batch 64/GPU.
- Constant SGD learning rate 0.01 and the existing complete Frozen protocol.
- Keep bounded NormCap, coordinate alignment, prototype transport, and the
  prototype classifier unchanged; no Dual-B, HBD, or Adaptive-A.
- Use GPU pairs 4,5 for CUB and 6,7 for ImageNet-R; leave 0,1 idle to reduce
  concurrency after the preceding three-pair NCCL timeouts. Never use 2,3.

## Implementation and Verification

1. Add an opt-in `sa_freeze_a_after_tasks=3`: A is trainable for tasks 0-2
   and frozen starting at task 3. An absent setting preserves all old modes.
2. Extend task-boundary recovery for deterministic, aligned Live-A state.
   Validate the source protocol, snapshot checksums, classifier, prototypes,
   and the entire shared-prefix A policy. No optimizer is recovered because
   the existing trainer creates a new optimizer at every task boundary.
3. Verify CPU lifecycle, gradient, endpoint, and snapshot recovery tests;
   run a two-rank NCCL smoke including the first frozen task on each pair.
4. Launch per-dataset queues: common prefix -> delayed Frozen -> Live.
   Use nohup inside user-systemd, unique outputs, and fail-fast queues.
5. Inspect services, GPUs, errors, latest task, and saved boundaries every
   30 minutes. Do not silently restart failures or report partial Final.

Ruling: train the prefix once and fork its post-task-2 snapshot rather than
retraining it twice. The experiment estimates a continuation-policy effect,
not the effect of choosing a different first-task initialization.

Ruling: reuse the existing full Frozen-A reference only as a secondary
comparison: CUB Final 84.40, INR Final 78.75. The primary causal comparison
is the two same-prefix continuations. Never mix the old three-seed table
with these new single-seed trajectories.

## Acceptance and Interpretation

- The two suffixes must reference the same audited snapshot and source log.
- Frozen suffix: every A tensor stays equal to the task-2 anchor; B/scale
  and the training head remain trainable. Persistent adaptation state stays
  at 368,640 scalars for 24 Q/V branches.
- Report complete Top-1 curves, Final, AAA, Forgetting, per-task old/new
  accuracy, wall time, and all snapshot audits.
- If the better same-prefix suffix cannot approach the original best
  endpoint, an early gate cannot repair that prefix's accumulated damage.
- If the suffixes are promising, test whether legal early-task signals
  predict the better suffix across held-out seeds/orders before building
  a gate. Do not optimize the switch point on the reported test sets.

## Execution Ledger

### 2026-09-28: Setup

- Workspace: existing linked worktree `codex/functional-halfspace-adaptive-a`.
- Base commit: `b26a45f`; no pre-existing tracked changes; unrelated
  untracked experiment files are untouched.
- No torchrun/main.py processes were running at preflight; GPUs were idle.
- Prior anchored joint A/B formal runs ended with NCCL timeouts, not valid
  full results. This experiment does not resume those partial trajectories.
- Implementation, tests, smoke results, and launch manifests are pending.

### 2026-09-28 15:34 CST: CPU Implementation Verified

- Added the opt-in task-count schedule and aligned Live-A boundary restore.
- Initial TDD run: 10 failures for missing schedule/restore, 3 existing tests
  passed. Added an extra protocol-key rejection test and observed its failure
  before making the Live restore compare both source and target keys.
- Focused verification: 32 passed. Complete CPU suite: 536 passed, 1 skipped.
- `git diff --check` and queue `bash -n` passed.
- Cached pre-trained ViT loaded successfully with `HF_HUB_OFFLINE=1`:
  12 blocks, 768 features. Dataset symlinks match the full Frozen references.
- Twelve configs prepare smoke/formal prefix and two suffixes for each
  dataset. Smoke is explicitly truncated to four tasks, one epoch/task;
  none of its metrics will be counted as a formal result.
- Queues preserve failures, reject existing outputs, and stop before the
  next item on any nonzero exit. Both suffixes audit their common prefix,
  full metric history, finite state, and the fixed 368,640-scalar budget.
- Scoped Terra read-only review is pending; GPU smoke is not launched yet.

### 2026-09-28 15:42 CST: Fork Audit Verified

- The scoped read-only review found that an arm audit must prove its requested
  policy and exact source identity, not infer both from the child's config.
- Added import-time snapshot SHA256 provenance, explicit expected-source and
  expected-policy checks, and a paired audit of source fingerprints and all
  shared-prefix Top-1/Top-5 metrics. No model numerical path changed.
- Regression RED: five expected failures before the provenance/guard fix.
- Focused GREEN: 20 passed. Complete CPU suite: 542 passed, 1 skipped.
- Queue syntax and diff whitespace checks passed; reviewer confirmed no
  remaining blocking issue. Real-data smoke starts next on 4,5 and 6,7.

### 2026-09-28 15:50 CST: DDP Restore Race Fixed

- CUB completed its three-task smoke prefix, but rank 0 imported files while
  rank 1 was still constructing its fresh task-0 model. The latter correctly
  rejected the now-existing state. This was a restore ordering error, not a
  training loss or NCCL timeout.
- Added a barrier after all per-rank freshness checks, before rank-0 copy;
  retain the post-copy barrier. Two regression tests observed RED `[True]`
  before the fix and GREEN `[False, True]` after it.
- Focused tests: 22 passed. Complete CPU suite: 544 passed, 1 skipped.
- Archived the failed CUB output and both log copies under
  `EARLY_A_FAILED_CUB_RESTORE_RACE_20260928`; reuse the audited successful
  prefix. Queue supports an explicit `freeze`/`live` starting item and still
  refuses to overwrite any child output.
- INR smoke prefix is still running; no formal experiment has started.

### 2026-09-28 15:57 CST: Completed Frozen Smoke and Metric Parser Fix

- CUB frozen continuation completed Task 3. Its 24 A tensors are exactly
  equal to the task-2 source; 368,640 persistent adaptation scalars remain.
- The audit exposed mixed scalar formatting in resumed curves: old points
  are Python floats, new points are `np.float64`. The old regex extracted
  only the latter. Normalize the NumPy scalar wrapper and parse the entire
  list with `ast.literal_eval`; do not change the logged training values.
- Two regression tests observed RED for plain/mixed curves before the fix.
  Focused GREEN: 24 passed. Complete CPU suite: 546 passed, 1 skipped.
- Re-auditing the existing CUB frozen output succeeded without retraining it.
- INR's original queue shell failed after its successful prefix because the
  script was edited while that shell was reading it. Restarted only the
  continuations with the new shell; the successful prefix is unchanged.
  Do not edit a launch script while its queue is active in future stages.
- INR frozen Task 3 is finishing evaluation. CUB Live suffix starts next.
  One-epoch smoke metrics are not formal performance evidence.

### 2026-09-28 16:02 CST: CUB Smoke Accepted

- CUB prefix and both one-epoch Task-3 suffixes finished successfully.
  `early_a_cub_seed1_smoke_20260928_paired_audit.json` confirms identical
  imported source fingerprints and identical three-task Top-1/Top-5 history.
- All 24 Q/V branches are finite; frozen A is bit-identical to the anchor;
  persistent adaptation state remains 368,640 scalars in both suffixes.
- CUB formal queue starts on GPUs 4,5: 20-epoch common Live prefix, then
  seven-task delayed-Frozen continuation, then seven-task Live continuation.
- INR Live smoke is in its final evaluation; formal INR launch remains
  contingent on its paired audit. No new algorithm or budget changes.

### 2026-09-28 16:04 CST: Both Formal Queues Launched

- INR one-epoch prefix and both Task-3 continuations passed the paired audit:
  `early_a_inr_seed1995_smoke_20260928_paired_audit.json`. Frozen A is
  bit-identical and both suffixes preserve the 368,640-scalar state budget.
- Formal launch code: `0734f7c`; CUB launch documentation commit `1c3d4b3`.
  All CPU tests pass (546 passed, 1 skipped); both two-rank real-data smoke
  experiments pass after the recorded restore/parser fixes.
- Running user-systemd services:
  - `codex-early-a-cub-formal-20260928.service`, PID 1358500, GPUs 4,5,
    seed 1, started 16:01:52 CST.
  - `codex-early-a-inr-formal-20260928.service`, PID 1363010, GPUs 6,7,
    seed 1995, started 16:03:53 CST.
- Both queues use nohup, per-GPU batch 64, rank 10, 20 epochs/task and
  preserve the approved complete Frozen-protocol modules. Prefix is trained
  once per dataset; both seven-task suffixes import its finalized Task2
  snapshot. No learned gate or test-based policy selection is introduced.
- Queue logs: `early_a_cub_formal_queue_20260928.log` and
  `early_a_inr_formal_queue_20260928.log`. Per-item logs use config basename
  plus `.log`; artifacts use the corresponding uppercase output directory.
- `codex-early-a-monitor-20260928.timer` is active every 30 minutes; first
  scheduled check is 16:33:53 CST. Monitor output:
  `monitor_early_a_freeze_20260928.log`. It checks only these two queues and
  stops its own timer when both are terminal; failures do not auto-restart.
- Startup verification at 16:04:32 CST: CUB completed Task0 epoch 10/20 and
  INR completed Task0 epoch 1/20. Both formal services are active/running,
  each selected GPU uses about 8.5 GiB, and no startup traceback or NCCL
  timeout is present. Formal metrics are pending; smoke accuracy is not
  reported as a formal result.
