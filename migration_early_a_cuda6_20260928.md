# Early Shared-A Experiment: Conditional Migration to cuda6

## Scope

- Requested on 2026-09-28: inspect cuda6 and launch if GPUs are available.
- Preserve the approved common Live-A Task0/1/2 prefix, then compare a
  Task3-onward frozen continuation against a Live continuation.
- CUB seed1 and ImageNet-R seed1995; T10, rank10, 20 epochs/task,
  two GPUs with batch64/GPU (effective128). Keep bounded NormCap,
  coordinate alignment, prototype transport and the prototype classifier.
- No Dual-B, HBD, controller or algorithm change. Training code is identical
  to `de107fd`; only operational waiting/monitoring helpers are added.

## Resource Audit

- Host: `zhaoyang@192.168.10.206`, hostname `cuda6`, four RTX3090 GPUs.
- GPU1 is idle (3 MiB). GPU0 has another user's CIFAR generation/evaluation
  process (about 1.7 GiB). GPUs2/3 contain another user's large-model training
  (about 24 GiB each). None of these processes may be stopped or preempted.
- Therefore no idle pair exists at deployment time. Without explicit sharing
  approval, the conditional queue must wait rather than consume GPU0.
- Poll GPUs0/1 every60 seconds. Require two consecutive observations with
  no compute PID, at most64 MiB allocated, and at most5% utilization on each
  card before launching a dataset. Query/parse failures are fail-closed.
- With only one potential pair, run complete CUB queue first, then wait for
  the same pair again and run complete ImageNet-R queue. Never run both
  datasets concurrently on GPUs0/1; never use GPUs2/3.

## Isolated Deployment

- Repository: `/home/zhaoyang/SD-Lora-CL-early-freeze-20260928`.
- Branch: `cuda6/early-freeze-20260928`, imported from a Git bundle rooted
  at `de107fd`. Existing cuda6 repositories are unchanged.
- `data` links to `/home/zhaoyang/SD-Lora-CL/data`; both CUB and ImageNet-R
  train/test directories exist. No dataset is copied or modified.
- Python/torchrun: `/home/zhaoyang/miniconda3/envs/sdlora/bin`.
- PyTorch2.4.1+cu121 and timm1.0.9 match cuda7. NumPy/scipy differ:
  cuda6 uses2.1.1/1.14.1; cuda7 uses2.2.6/1.15.3. Do not claim bitwise
  equivalence across hosts. Each completed result keeps its own run manifest.
- Cached pre-trained ViT loads offline: 12 blocks, 768 features. No package
  installation or model download is needed for this launch.
- Restart from Task0 on cuda6 in the fresh directory; do not combine partial
  cuda7 training with cuda6 Final/AAA. Retain all cuda7 logs and snapshots.
- cuda7 jobs were observed progressing very slowly rather than definitively
  crashed. They are not stopped as part of the conditional waiting setup.
  No kernel-level hardware diagnosis has been established.

## Verification Ledger

- CPU waiting guard covers occupied GPUs, memory/utilization thresholds,
  malformed queries and jobs on unrelated cards. Four direct checks pass.
- Focused cuda6 tests: 28 passed, covering idle guard, freezing, restoration
  and exact prefix provenance.
- Initial full cuda6 CPU suite: 548 passed, two failed, one skipped. Failures:
  `test_adaptive_a_queue_runs_three_four_gpu_experiments_with_complete_logs`
  and `test_three_way_queue_runs_nine_strict_two_gpu_controls`.
- Both failures reproduced on untouched `de107fd`. SSH did not provide a
  `python` command on PATH, while the old mocked-conda queue tests expect
  one. Run with the actual sdlora bin on PATH; no test or old queue edits.
- Corrected full CPU suite: 550 passed, one skipped, 36.91 seconds.
- New parent uses the actual interpreter's bin on PATH, so its child queue
  resolves both `python` and `torchrun` from cuda6's sdlora environment.
- Real two-rank smoke was completed on cuda7 before this migration. No
  CUDA6 GPU smoke or formal epoch is claimed while waiting for the pair.

## Runtime

- Waiting service: `codex-early-a-cuda6-20260928.service`, under user-systemd
  with nohup. Entrypoint: `scripts/wait_early_a_pair.py --gpu-ids 0,1`.
- Parent log: `early_a_cuda6_migration_queue_20260928.log`.
- Dataset logs: `early_a_cub_formal_queue_20260928.log` and
  `early_a_inr_formal_queue_20260928.log`, created when each dataset starts.
- Per-training stdout logs use the unchanged config basename plus `.log`;
  artifact directories use the uppercase corresponding run name.
- The 30-minute monitor observes this controller via `--controller-unit`.
  It must not mistake a waiting controller for a completed experiment.
- A failed dataset stops the parent; no failed item is silently overwritten
  or automatically restarted. Waiting is not reported as training.
- Actual commit, service status and log verification are recorded below
  after deployment. This file is the authoritative migration ledger.

## Deployment Verification: 2026-09-28 16:51 CST

- Code commit: `efcba37` on both hosts. Only this ledger is updated after
  that implementation commit; the training code and configuration remain
  unchanged from `de107fd`.
- Waiting service started at 16:44 CST; MainPID `660272`, active/running,
  `ExecMainStatus=0`. Repeated separate SSH connections confirm that the
  service persists and continues polling without initializing CUDA.
- Enabled user linger for `zhaoyang`; `loginctl show-user` confirms
  `Linger=yes`. The service and monitor can survive SSH logout.
- Monitor timer is active, with first scheduled check at 17:14:15 CST and
  subsequent checks every30 minutes.
- GPU0 still has another user's job (1681 MiB,8% utilization); GPU1 is
  idle; GPUs2/3 are busy. Latest controller messages report
  `WAIT_FOR_GPUS ... idle=False streak=0/2`.
- Status is WAITING, not TRAINING. No cuda6 training accuracy, Task0/1
  smoke or dataset completion is reported. No other user's process is
  stopped or shared. No local cuda7 experiment is stopped.
- Once the pair becomes idle, the existing audited queue runs CUB's
  common Live prefix, frozen continuation and live continuation, then
  repeats the same sequence for ImageNet-R. Each complete result uses
  only its own host's prefix; no cross-host checkpoint mixing occurs.
- Subsequent cuda7 check: CUB service is failed. Its log records a rank1
  NCCL ALLREDUCE timeout at 16:44:20 after600 seconds (sequence469),
  followed by SIGABRT. ImageNet-R service remains active. This establishes
  a communication failure, not a specific hardware root cause. Preserve
  the failed run unchanged and never use its partial curve as a result.
