# Historical-only Seeds 4--13

## Registered Protocol

- User request on 2026-09-30: retain completed seeds 1--3; run ten NEW seeds 4--13, not 1--10.
- Three datasets: CIFAR-100, ImageNet-R and CUB-200; 30 new full-sequence runs.
- T=10, rank10, 20 epochs, SGD; dataset-specific schedules/LR and splits unchanged.
- Dual GPU, batch64 per process, effective batch128.
- Slots: physical GPU 0/1, 2/5 and 6/7; GPU 3/4 excluded.
- Run seed-major, dataset-interleaved; dispatch immediately when a slot completes.
- All training uses nohup under user-systemd; independent checks every 1800 seconds.

## Immutable Method

Train from detached independent clone at `c68e12a557bef0b197841a0a8ecb9a308c4b83cf`, the completed seeds 1--3 training commit. No algorithm changes.

Historical-only means `function_safe_pareto`, historical student branch for teacher consistency, cross-fit interval4. Preserve live-A coordinate alignment, bounded NormCap absorption, prototype transport and the original prototype classifier. Dual-B and HBD remain off.

Generate from the original registered per-dataset seed1 configuration. Formal configurations differ ONLY in seed, prefix and output filepath. Record configuration checksums, cached pretrained-weight checksum and environment in the manifest. Existing seeds1--3 must pass log, provenance and 24-branch artifact checks before reuse.

## Execution Gates

1. Focused scheduler/config tests, including exact preservation of source settings, seed coverage, no overwrite, exclusions, dynamic refill and no cherry-picking.
2. Three short seed4 Task0/1 smoke checks (one epoch per task), strictly excluded from performance tables.
3. Formal queue starts only after all three smoke artifacts/logs pass. All formal runs start afresh from Task0; interrupted outputs are archived, never treated as complete or resumed at an arbitrary task.
4. Completion requires status0, ten task evaluations and finite 24 Q/V branch artifacts with correct code/config provenance.
5. Summarize all seeds1--13 only when each dataset has all thirteen complete unique results. Report mean and sample standard deviation, all individual seeds and full curves; no selecting winning seeds.

## Metrics

Report Final, AAA and Forgetting. Keep raw logged Forgetting separately; recompute the reported Forgetting from numeric class-range identities in the original CNN dictionaries because the legacy trainer string-sorts two-digit/three-digit class ranges. This is read-only reporting, not a training-code change.

## Status

- Preparation complete: all nine existing seeds1--3 records verified; none scheduled or overwritten.
- Implementation/config commits: `d32c8fa` / `67311d7`; 21 focused tests passed.
- Real-data Task0/1 smoke: all three seed4 runs status0, finite 24-branch artifacts and training/config provenance verified. Finished at 14:40:55 on 2026-09-30; short-test scores excluded from formal results.
- Formal queue launched 2026-09-30 14:41:55 with 30 jobs. At 14:42, C100/INR seed4 completed Task0 epoch1 and CUB seed4 reached epoch7; no startup exceptions.
- Formal service: `historical-only-formal-.runtime-historical-only-seed4-13-20260930.service`.
- Monitor service: `historical-only-monitor-.runtime-historical-only-seed4-13-20260930.service`; 1800-second interval. User linger enabled.
- Runtime: `.runtime_historical_only_seed4_13_20260930/`.
- Logs: `logs/<dataset>_historical_only_formal_seed<seed>_t10_i4_bs128.log`.
- Reports: `formal_results.json`, `formal_results.csv`; full summary withheld until complete.
- Queue log: `formal_queue.log`; periodic checks: `monitor_queue.log`.
- Launch via `bash run_historical_only_seed4_13_queue.sh formal` and independent `monitor` service after smoke passes.

## Failure Audit: 2026-09-30

- Formal queue stopped with status1 after all three first-wave seed4 processes hit rank1 NCCL ALLREDUCE watchdog timeout (`600000 ms`). CUB exited at 15:08:11, C100 at 15:13:02, INR at 15:26:26; systemd recorded queue exit1 at 15:27:40.
- Last progress: C100 Task0 trained all20 epochs but final evaluation/hash work did not finish; INR Task1; CUB Task2 epoch10. No new run completed the ten-task sequence; remaining27 jobs never started. Existing nine seeds1--3 records remain complete.
- Dispatcher intentionally uses stop_on_failure; monitor later exits on failed queue state. Missing collected transient systemd units are a consequence of completed/failed services, not evidence of a reboot or SSH termination.
- C100 rank1 waited while rank0 performed prototype/eval/hash work: prototype sync 15:01:53, RNG check 15:05:54, tensor invariance 15:10:41, rank1 timeout 15:11:53. This demonstrates a long rank0-only interval; logs alone do not prove a collective-order bug or a faulty GPU.
- Host observations at 15:57--16:04: memory availability about242GiB, swap0, user cgroup oom/oom_kill0. Sysfs reported all96 CPU frequencies around116--172MHz, below configured minimum1000MHz; CPU temperatures about50--58C. A 45-second PyTorch-import probe timed out before reaching CUDA allocation. These support severe host-performance degradation as a leading hypothesis, not a confirmed PSU/thermal diagnosis. Kernel/BMC logs require administrator access.
- No algorithm/config changes or automatic restart during this read-only diagnosis. Preserve failed logs/artifacts; recover the host and validate sustained CPU/CUDA/DDP progress before restarting failed seed4 jobs from Task0. Do not count partial accuracy curves as Final/AAA.

## Restart: 2026-09-30 20:28

- User explicitly requested restart. CPU sysfs frequencies now report about 1000MHz; PyTorch import took 1.321s and a small CUDA matmul/allocation check 0.306s. This is recovery evidence for the earlier short-probe slowdown, not proof of long-run hardware stability.
- All three slots 0/1, 2/5, 6/7 passed fresh NCCL all_reduce and barrier checks with correct summed value 3. Source remains clean at `c68e12a`; all 30 config checksums match, original nine full runs and three earlier data smokes remain verified. No method, optimizer, batch, LR or timeout changes.
- Formal queue restarted at 20:28:47 with 30 pending jobs. The three failed seed4 log/result pairs were archived under `archive/<job>_20260930_202847_*/`; each restarts from Task0, without partial resume.
- Same nohup/user-systemd formal and monitor services are active. Monitoring resumed only after queue state changed from `failed` to `running`, avoiding an immediate exit against the stale failure status.
- At 20:29, C100/INR seed4 had completed Task0 epoch1 and CUB seed4 reached epoch7, with no startup exceptions. Continue immediate slot refill for the remaining seeds5--13; long-duration stability and task-boundary evaluation remain to be verified.

## Results Audit: 2026-10-01

- Restart outcome: 2 complete, 2 failed, 26 not started out of 30 registered new runs. Queue ended with status1 at 08:24:23 after its last active job completed; monitor detected failed state at 08:32. No automatic restart in this audit.
- CUB seed4 finished at 2026-09-30 22:07:04; C100 seed5 was immediately dispatched to its freed GPU6/7 slot and finished at 2026-10-01 08:24:23. Both have ten evaluations, verified provenance and finite 24-branch artifacts.

| Dataset | Seed | Final | AAA | Corrected Forgetting |
|---|---:|---:|---:|---:|
| CUB-200 | 4 | 83.26 | 90.077 | 8.095556 |
| CIFAR-100 | 5 | 87.31 | 90.354 | 6.922222 |

- CUB seed4 Top1: `[98.98, 96.09, 93.40, 91.21, 89.83, 88.69, 87.81, 86.07, 85.43, 83.26]`.
- C100 seed5 Top1: `[94.20, 93.65, 93.57, 91.90, 89.92, 88.92, 88.19, 87.99, 87.89, 87.31]`.
- C100 seed4 failed at Task4 epoch15, exit23:02:22; INR seed4 failed at Task5 epoch5, exit23:10:00. Both again report rank1 scalar ALLREDUCE timeout600000ms, with severe epoch slowdown preceding the timeout. Root cause is not established by these logs; partial curves are excluded.
- Corrected CUB seed4 F is8.095556, not raw logged7.311111. The original task-range string-sort issue persists in the sealed training version; reporting remains read-only.

Accumulated complete results only (mean +/- sample standard deviation; NOT a completed 13-seed table):

| Dataset | Complete Seeds | Final | AAA | Corrected Forgetting |
|---|---|---:|---:|---:|
| CIFAR-100 | 1,2,3,5 | 87.715 +/- 0.463 | 91.499 +/- 0.784 | 6.703 +/- 0.836 |
| ImageNet-R | 1,2,3 | 78.370 +/- 0.260 | 82.399 +/- 0.749 | 6.023 +/- 0.420 |
| CUB-200 | 1,2,3,4 | 83.743 +/- 0.461 | 89.151 +/- 0.812 | 7.260 +/- 0.742 |

- New CUB seed4 has higher AAA but lower Final than the previous three-seed mean; C100 seed5 lowers both pooled Final and AAA. These are cross-seed observations, not paired evidence of algorithm improvement or deterioration.
- Direct host GPU and user-systemd checks are unavailable under this session's restricted permissions. Do not interpret NVML/systemd access failures as a confirmed driver fault. Queue status above is established from persisted logs and validated artifacts.
- No algorithm, configuration, training source or runtime artifact was changed during this results audit.

## Restart Request Blocked: 2026-10-01

- User requested the unfinished queue. Fresh read-only validation confirms 28 pending/incomplete jobs, with CUB seed4 and C100 seed5 complete; prior nine runs and three smokes are valid, and sealed source/config/weight hashes remain unchanged.
- Attempted the existing `bash run_historical_only_seed4_13_queue.sh formal` launcher. It exited1 before creating a service: `Failed to connect to bus: Operation not permitted`. Current sandbox also cannot directly access GPUs or inspect host processes. No training was started and queue state remains `failed`.
- Run the same launcher from an ordinary host terminal. It preserves the two completed runs and archives/restarts incomplete ones from Task0, using the registered dual-GPU slots and batch64. Start the monitor only after a NEW `QUEUE START phase=formal jobs=28` entry and running status appear, not against the stale failed status.
- No sandbox bypass, algorithm/config changes or artifact modifications were attempted.

## Per-Seed Results: 2026-10-01 14:21 CST

Freshly inspected all 39 registered/reused logs and verified the eleven complete artifacts. Cells below are `Final / AAA / corrected Forgetting`; incomplete runs have no final metrics.

| Seed | CIFAR-100 | ImageNet-R | CUB-200 |
|---:|---|---|---|
| 1 | 87.55 / 92.122 / 7.177778 | 78.52 / 83.090 / 5.977778 | 84.31 / 89.582 / 6.320000 |
| 2 | 88.38 / 91.830 / 5.466667 | 78.52 / 81.604 / 5.626667 | 83.50 / 88.556 / 7.492222 |
| 3 | 87.62 / 91.688 / 7.244444 | 78.07 / 82.504 / 6.463333 | 83.90 / 88.388 / 7.133333 |
| 4 | Running: Task1 epoch2/20 | Running: Task1 epoch10/20 | 83.26 / 90.077 / 8.095556 |
| 5 | 87.31 / 90.354 / 6.922222 | Running: Task1 epoch10/20 | Pending |
| 6 | Pending | Pending | Pending |
| 7 | Pending | Pending | Pending |
| 8 | Pending | Pending | Pending |
| 9 | Pending | Pending | Pending |
| 10 | Pending | Pending | Pending |
| 11 | Pending | Pending | Pending |
| 12 | Pending | Pending | Pending |
| 13 | Pending | Pending | Pending |

- Persisted queue log shows restart at14:10:02, `jobs=28`, state `running`. The agent's earlier sandbox-blocked attempt did not start it; the new host-side start is established from logs, not attributed to that blocked command.
- Current new-run coverage: 2 complete, 3 active with recent progress, 25 pending. The two previous failed seed4 attempts have been archived and replaced by fresh Task0 runs. No additional full sequence has completed since the preceding results audit.
- Aggregate means/sample standard deviations remain the 4/3/4-seed figures above. This is not a completed 13-seed comparison, and no partial Final/AAA values enter the table.

## Final 13-Seed Audit: 2026-10-08

- The host-side queue of 28 remaining jobs finished at 2026-10-02 08:20:15 with `QUEUE END phase=formal status=0`; `formal_queue_status.json` is `complete`. All 30 new seeds4--13 and nine reused seeds1--3 pass the log, provenance, ten-task, and final 24-branch artifact checks: 39/39 complete.
- Entries are `Final / AAA / corrected Forgetting`, in percentage points. The earlier running/pending table is retained above as a dated snapshot; this table supersedes it for completed results.

| Seed | CIFAR-100 | ImageNet-R | CUB-200 |
|---:|---|---|---|
| 1 | 87.55 / 92.122 / 7.178 | 78.52 / 83.090 / 5.978 | 84.31 / 89.582 / 6.320 |
| 2 | 88.38 / 91.830 / 5.467 | 78.52 / 81.604 / 5.627 | 83.50 / 88.556 / 7.492 |
| 3 | 87.62 / 91.688 / 7.244 | 78.07 / 82.504 / 6.463 | 83.90 / 88.388 / 7.133 |
| 4 | 87.26 / 90.851 / 6.811 | 78.35 / 82.066 / 5.629 | 83.26 / 90.077 / 8.096 |
| 5 | 87.31 / 90.354 / 6.922 | 78.38 / 82.634 / 5.371 | 84.45 / 88.747 / 6.592 |
| 6 | 88.64 / 92.358 / 5.400 | 78.57 / 82.033 / 5.352 | 84.50 / 89.244 / 6.354 |
| 7 | 88.12 / 91.635 / 5.933 | 78.17 / 83.194 / 6.146 | 84.00 / 88.841 / 6.782 |
| 8 | 88.17 / 91.085 / 6.078 | 78.53 / 82.150 / 5.777 | 82.93 / 88.684 / 8.412 |
| 9 | 88.31 / 93.164 / 6.089 | 77.88 / 81.927 / 4.884 | 83.62 / 89.187 / 7.083 |
| 10 | 87.65 / 91.836 / 7.000 | 78.73 / 81.564 / 5.278 | 82.29 / 88.815 / 9.088 |
| 11 | 87.91 / 92.564 / 5.744 | 78.35 / 82.987 / 5.254 | 83.14 / 89.001 / 8.052 |
| 12 | 88.25 / 92.986 / 6.367 | 78.62 / 83.309 / 5.061 | 84.00 / 88.887 / 7.560 |
| 13 | 87.77 / 91.555 / 6.767 | 78.17 / 79.820 / 5.281 | 83.86 / 88.027 / 7.714 |

Mean +/- sample standard deviation over all 13 registered seeds:

| Dataset | Final | AAA | Corrected Forgetting |
|---|---:|---:|---:|
| CIFAR-100 | 87.918 +/- 0.430 | 91.848 +/- 0.808 | 6.385 +/- 0.644 |
| ImageNet-R | 78.374 +/- 0.244 | 82.222 +/- 0.933 | 5.546 +/- 0.449 |
| CUB-200 | 83.674 +/- 0.642 | 88.926 +/- 0.523 | 7.437 +/- 0.835 |

- C100 Final ranges87.26--88.64, INR77.88--78.73, CUB82.29--84.50. INR seed13 AAA79.820 and CUB seed10 Final82.29 are retained, as required by the sealed seed protocol.
- The original trainer string-sort bug affects reported Forgetting for INR/CUB. The table uses `corrected_forgetting` from numeric task class ranges; raw logged F is preserved in the JSON/CSV and must not replace these corrected values. Final and AAA are unaffected.
- The older `monitor_queue.log` ends with the previous failed attempt; formal queue status and validated per-run logs establish final completion. This does not verify that the independent 30-minute monitor was relaunched for the successful attempt.
- The full 13-seed Historical-only result supports its own stability estimate. A claim of superiority over Frozen-A/Live-A still requires matching same-seed, same-protocol baselines; single-seed development comparisons do not establish that claim.
