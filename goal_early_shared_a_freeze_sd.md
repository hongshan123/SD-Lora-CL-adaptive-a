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
