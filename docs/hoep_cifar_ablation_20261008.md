# CIFAR-100 HOEP direction ablations

User authorized the 15 additional runs on the current server (cuda7).

## Protocol

Base revision: 6380f77, the original HOEP P3 training revision. T=10,
rank=10, 20 epochs/task, SGD, two GPUs with batch64 per rank. LS alignment
and exact absorption on; NormCap, transport, Dual-B and HBD off.
Seeds: 1993, 1994, 1995. Epsilon=0.05, no dataset/seed tuning.

Six policies: Frozen, raw Live, canonical all-plastic Live, low-energy
HOEP, matched-count random, matched-count high-energy. Reuse the completed
P3 Frozen/Live/HOEP seed1993 runs; train the other 15 in sequence.

Random/high controls use the low-energy HOEP reference's per-task, per-Q/V
branch counts for the SAME seed. They retain complete spectral tie groups.
If an exact count is impossible on their own evolving spectrum, stop with
an explicit error rather than split a tie or silently change dimensions.
Random selection uses an independent CPU generator, never training RNG.
Their actual selected energy is reported; the 5% risk cap does not apply.
Canonical Live retains all HOEP coordinate/optimizer operations and opens
every direction. Full training trajectories differ after intervention.

## Implementation and verification

1. Test and implement alternate masks, config validation and provenance.
2. Verify seed1993 reference counts from the prior shadow diagnostics;
   verify original and shadow shared A/G/prototypes and ten-task curves.
3. Generate an 18-cell manifest with three explicitly reused cells and
   15 queued jobs. Train missing HOEP references before dependent controls.
4. Run the test suite, two-rank NCCL integration smoke, and a short real
   two-GPU Task0/Task1 smoke for the new controls before formal training.
5. Freeze production sources, configs, reference provenance and manifest,
   including the actual experiment revision and an archived source snapshot.
   Verify their SHA256 before every job. Launch a systemd user service on
   GPU0,1, with event/status files, no overwrites and fail-fast behavior.
   Check actual training logs. Resume completed runs only with explicit
   successful subprocess events and a revalidated complete log.

## Rulings

- The old P3 improvement gate failed. These user-authorized runs are
  exploratory mechanism ablations; they do not reverse that NO-GO result.
- Use an isolated linked worktree at the original protocol revision, since
  the current diagnostic branch has many subsequent unrelated changes.
- Preserve existing results and jobs. The new queue targets cuda7 only.

## Progress

- Baseline current diagnostic branch: 616 tests passed.
- Original-protocol baseline: 380 tests passed.
- Alternate-mask tests: failed first, then 21 focused tests passed.
- Queue tests: failed first; fixed NumPy scalar log parsing with a failing
  regression, then passed. Reuse audit passed (including exact prototypes).
- First full suite: 390 passed, one legacy defaults test required updating
  for the three new backward-compatible fields; rerun: 392 passed.
- Random, high-energy and all-plastic two-rank NCCL smokes: passed.
- Fresh review: no mathematical/configuration blocker. Important source
  provenance gap fixed with immutable snapshot/hashes and a failing source/
  config mutation test. Minor recovery limitation fixed with success-event
  recovery and a failing incomplete-log regression. 14 focused tests pass.
- Final full suite: 394 passed in 35.39s. Review fix-pass found no blocker;
  source/config hashes are also rechecked after any GPU-idle wait.
- Formal queue: pending; four real T2/E1 smoke runs precede the 15 full runs.
- First real preflight stopped before training because the queue generator
  omitted the output parent directories required by the existing run guard.
  Added a real run-guard regression (RED), then created both parent roots.
  Failed logs are preserved in the first runtime; restart uses a fresh runtime.
