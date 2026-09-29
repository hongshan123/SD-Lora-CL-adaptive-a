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
- [x] Tests and queue implementation; independent configuration commit. Final focused: 24 passed; full suite: 575 passed.
- [ ] Frozen/Live smoke: Task0 equality, Task1 A policy, snapshots, rebuild logits, DDP consistency.
- [ ] Six formal runs, three dynamic double-GPU slots: 0,1 / 4,5 / 6,7. Never use GPU 2 or 3.
- [ ] Final / AAA / Forgetting / global old-new curves and paired sample mean/std.

## Rulings And Interfaces

- Ruling: use an immutable local Git clone for training, inside an ignored runtime directory. This prevents another agent's edits from changing the code halfway through the six runs. The implementation remains in the current isolated worktree.
- Review fix: enforce an independent, clean, detached clone at `runtime/source` and recheck before each run. Check the sealed pretrained tensor hash whenever training constructs a backbone; use offline HF cache and the same hash for smoke reconstruction. This adds diagnostics only, not a new model or RNG operation.
- Review clarification: the initial prepare succeeded before configurations were committed. Subsequent preparation may validate identical registered configurations, but must never replace changed configurations or reuse a different output path under the same tag.
- Ruling: cross-split file-content hashes must be disjoint. Relative names alone are not a leakage test; record their overlap separately.
- Ruling: archive failed logs and partial artifacts together; rerun only unfinished items from Task0, never silently resume a partial task sequence or change batch/LR.
- Interface: committed configurations point at independent runtime result directories; the run manifest records code/config/data SHA256 and both GPU ranks.
- Interface: a successful subprocess is not enough: each completed result needs the full metric curve and all checksummed task snapshots.
- Use `nohup` inside user-systemd services; independent monitor service checks every 30 minutes. A freed slot immediately receives the next job.

## Claim Boundary

This compares A policies with all other components held fixed. It does not independently establish the causal benefit of alignment, transport or NormCap, and does not claim an official ImageNet-A train/test protocol beyond the locally audited split. Report every registered seed and either direction of the paired difference.
