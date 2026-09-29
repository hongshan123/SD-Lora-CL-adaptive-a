# ImageNet-A Paired Live-A / Frozen-A Study

## Binding Protocol

- Dataset: the existing `/data/dataset/Imagenet-A/{train,test}` split, 200 classes, 5,981/1,519 images. Do not regenerate the split.
- Seeds: 1995, 1996, 1997; both A policies for every seed. Do not select seeds after observing results.
- T=10, 20 classes/task, rank10, 20 epochs/task, SGD, constant LR 0.01, two GPUs with batch64 each (effective128).
- Inherit `exps/frozen_margin_inr_t10_seed1995.json`; change only dataset, identity/output path, seed and `sa_train_a_all_tasks`.
- Keep LS coordinate alignment, prototype transport and bounded NormCap ON in both arms. Prototype classifier ON; Dual-B, Adaptive-A, replay OFF.
- Smoke: seed1995, Task0/1, 2 epochs/task, two GPUs with batch64. These are not formal results.

## Execution Ledger

- [x] Existing isolated worktree verified; baseline full suite: 551 passed.
- [x] Content SHA256 / image decode / class mapping audit; link `data/imagenet-a` without overwriting existing data.
- [x] Tests and queue implementation; independent configuration commit. Final focused: 27 passed; full suite: 578 passed.
- [x] Frozen/Live smoke: Task0 equality, Task1 A policy, snapshots, rebuild logits, DDP consistency. Report: `.runtime_imageneta_live_frozen_20260929_paired/smoke_verification.json`.
- [x] Six formal results: completed 2026-09-29 13:09:56 CST, all exit status 0; all 60 checksummed task snapshots verified. GPU slots 0,1 / 4,5 / 6,7; GPU 2 and 3 excluded.
- [x] Final / AAA / global old-new curves and paired sample mean/std. Forgetting recomputed from numeric class-range identities; the raw trainer metric has a class-range string-sorting defect.

## Rulings And Interfaces

- Ruling: use an immutable local Git clone for training, inside an ignored runtime directory. This prevents another agent's edits from changing the code halfway through the six runs. The implementation remains in the current isolated worktree.
- Review fix: enforce an independent, clean, detached clone at `runtime/source` and recheck before each run. Check the sealed pretrained tensor hash whenever training constructs a backbone; use offline HF cache and the same hash for smoke reconstruction. This adds diagnostics only, not a new model or RNG operation.
- Review clarification: the initial prepare succeeded before configurations were committed. Subsequent preparation may validate identical registered configurations, but must never replace changed configurations or reuse a different output path under the same tag.
- Smoke fix: the trainer's run guard needs the result-parent directory before opening its lock. The scheduler creates only that parent, never the guarded run directory. Initial failed output contained no training and is archived; resealing is forbidden once any formal run has started.
- Numerical-audit ruling: retain the planned FP32 logits `max_abs_diff < 1e-5`. The extra feature-absolute gate was not part of the plan and rejected CPU/GPU normalization rounding; replace it with relative L2 `<1e-5`, explicitly report absolute error, and additionally require each effective operator's relative error `<1e-6`. Use the training backend settings. Training remains pinned to `38fe295`; the read-only verification tool has a separately recorded Git/script SHA.
- Ruling: cross-split file-content hashes must be disjoint. Relative names alone are not a leakage test; record their overlap separately.
- Ruling: archive failed logs and partial artifacts together; rerun only unfinished items from Task0, never silently resume a partial task sequence or change batch/LR.
- Interface: committed configurations point at independent runtime result directories; the run manifest records code/config/data SHA256 and both GPU ranks.
- Interface: a successful subprocess is not enough: each completed result needs the full metric curve and all checksummed task snapshots.
- Use `nohup` inside user-systemd services; independent monitor service checks every 30 minutes. A freed slot immediately receives the next job.

## Claim Boundary

This compares A policies with all other components held fixed. It does not independently establish the causal benefit of alignment, transport or NormCap, and does not claim an official ImageNet-A train/test protocol beyond the locally audited split. Report every registered seed and either direction of the paired difference.

## Runtime Handoff

- Training Git: `38fe295c98b19b6c2a0ab1bb1ae1be944b28a4bc`; verification Git: `2e974bfefe87f8b311252850ff5b0842cc981cae`.
- Queue/monitor started 2026-09-29 12:12 CST via user-systemd and nohup. Monitor interval: 1,800 seconds.
- Runtime: `.runtime_imageneta_live_frozen_20260929_paired/`; individual logs: `logs/`; results/snapshots: `results/`; automatic JSON/CSV: `formal_results.{json,csv}`.
- Start/restart unfinished work: `bash run_imageneta_live_frozen_queue.sh .runtime_imageneta_live_frozen_20260929_paired formal '0,1;4,5;6,7'`. Never invoke this while the queue is active; systemd rejects duplicate units. Successful complete runs are reused, failed/partial attempts are archived.
- Monitor: `bash run_imageneta_live_frozen_queue.sh .runtime_imageneta_live_frozen_20260929_paired monitor '0,1;4,5;6,7'`.
- Final analysis completed 2026-09-29 14:50 CST; queue and monitor exited successfully, all GPUs idle. Smoke is not a formal accuracy result.

## Formal Results

All numbers below use all three registered seeds. Standard deviations are sample standard deviations. F is the corrected task-identity-aligned forgetting, not the raw logged value.

| Seed | Policy | Final | AAA | Corrected F |
|---|---|---:|---:|---:|
| 1995 | Frozen | 63.33 | 69.273 | 10.2333 |
| 1995 | Live | 62.48 | 69.385 | 13.4922 |
| 1996 | Frozen | 62.54 | 68.820 | 11.1400 |
| 1996 | Live | 62.80 | 68.995 | 11.2222 |
| 1997 | Frozen | 63.33 | 70.014 | 9.6933 |
| 1997 | Live | 62.34 | 69.772 | 13.7389 |

| Policy | Final mean +/- std | AAA mean +/- std | Corrected F mean +/- std |
|---|---:|---:|---:|
| Frozen | 63.0667 +/- 0.4561 | 69.3690 +/- 0.6028 | 10.3556 +/- 0.7310 |
| Live | 62.5400 +/- 0.2358 | 69.3840 +/- 0.3885 | 12.8178 +/- 1.3873 |

- Paired Live minus Frozen Final: `[-0.85, +0.26, -0.99]`, mean `-0.5267` pp. AAA: `[+0.112, +0.175, -0.242]`, mean `+0.0150` pp. Corrected F mean difference: `+2.4622` pp.
- Unweighted mean of the nine Task1--9 new-class accuracy differences per seed: `[+2.2267, +0.3678, +2.7033]` pp, across-seed mean `+1.7659` pp. Corresponding old-class mean difference: `-0.3559` pp. These are descriptive paired accuracies, not causal attribution to an individual module.
- At the final boundary, new-class accuracy difference averages `+3.5200` pp and old-class accuracy difference averages `-1.0733` pp. Final is sample-weighted over all seen classes and must not be replaced by an equal-weight average of old and new accuracies.
- Task0 training tensor/RNG hashes match within every seed pair. This protocol does not support a stable Live advantage on ImageNet-A: AAA is essentially tied, Live Final is lower in two of three pairs, and corrected F is higher in all three. It does show new-class plasticity gains; it does not establish that A updates have no value in all settings or statistical significance with three seeds.

## Forgetting Audit

The executed `trainer.py` sorts keys using `sorted(cnn_keys)`. Labels from `utils/toolkit.py` are padded to two digits only, so `100-119` sorts before `20-39` once Task5 is introduced. Accuracy-matrix rows therefore change task identity across columns; the printed forgetting is not task-aligned. This does not affect `grouped["total"]`, old/new accuracies, Final or AAA.

Recompute from each logged `CNN` dictionary, retaining the same class-range key across all task boundaries:

```text
F = mean_{i=0..8}(max_{t=i..9} accuracy[class_range_i, t]
                 - accuracy[class_range_i, 9])
```

Raw logged Frozen F for seeds1995/1996/1997: `11.5022/13.5667/9.9778`; raw logged Live F: `14.2922/14.2344/12.2378`. The existing generated `formal_results.json/csv` preserves these raw logged values. Use the corrected values in this document for reporting; do not silently overwrite raw evidence. Training, immutable source, configurations and artifacts are unchanged. A future trainer/reporting fix is separate from this read-only result audit.
