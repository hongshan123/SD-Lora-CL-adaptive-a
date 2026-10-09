# CIFAR HOEP-A 15-run queue

Launch: 2026-10-08 22:42 CST, current host **cuda7**, GPU0 and GPU1.

## Protocol and cells

T10, rank10, 20 epochs/task, SGD, batch64 per GPU (global128).
LS alignment and exact absorption on. NormCap, prototype transport,
Dual-B and HBD off. Initially one full experiment at a time; changed to four
parallel two-GPU experiments on 2026-10-09 as recorded below.

| Method | Existing seed | New seeds |
|---|---|---|
| Frozen-A | 1993 | 1994, 1995 |
| Raw Live-A | 1993 | 1994, 1995 |
| Low-energy HOEP-A | 1993 | 1994, 1995 |
| Canonical all-plastic Live-A | — | 1993, 1994, 1995 |
| Matched-count random directions | — | 1993, 1994, 1995 |
| Matched-count high-energy directions | — | 1993, 1994, 1995 |

15 new full runs, three reused runs, 18 cells in the final comparison.
Four short T2/E1 runs check the real training flow before full experiments;
they are verification runs and are excluded from the results table.

## Implementation and provenance

Isolated project:
`/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-cifar-ablation-20261008`

Base protocol revision: `6380f77`. Experiment revision: `6e295e4`.
Random/high controls match the SAME seed low-HOEP per-task, per-Q/V counts.
Complete eigenvalue ties remain indivisible; impossible matching stops the
queue. Random masks use an independent CPU generator. Their selected energy
is recorded, and they do not inherit the 5% energy cap.

The seed1993 reference comes from the previous shadow diagnostic reproduction.
Its final A/G/prototypes exactly equal the original HOEP run and its ten-task
accuracy curve matches. Runtime stores the reference and original-log hashes.

Production source snapshot and source/config hashes are archived and checked
before each run (again after any GPU wait). All 395 tests passed; random,
high-energy and all-plastic two-rank NCCL integration smokes passed.

The original P3 performance NO-GO remains in the record. These are exploratory
mechanism ablations explicitly requested by the user.

## Runtime and status

Runtime:
`/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-cifar-ablation-20261008/.runtime_hoep_cifar15_20261008_r2`

Service: `codex-hoep-cifar15-cuda7-20261008-r2.service`

Read `status.json` and `events.jsonl` for current status, `manifest.json` for
the exact sequence, `logs/` for each run, and `results_summary.json` for
completed full runs. Successful completed runs can be recovered from explicit
completion events; partial/failed outputs are never overwritten.

At launch verification the service was active and the first real preflight
was training Task0 on both GPUs. The scheduler automatically starts the
15 full runs after all four real preflights pass, including Task0 weight/RNG
equivalence and count-matching checks.

The first startup attempt failed before training because output parent
directories were missing. That attempt is preserved in
`.runtime_hoep_cifar15_20261008`; a failing real run-guard regression was added,
the generator was fixed, and r2 uses a fresh runtime.

## Four-pair handover — 2026-10-09 00:57 CST

User requested relaunch across multiple GPUs. The current cuda7 server now
runs four experiments concurrently, each still world_size2, batch64 per GPU
and global128. Training source/config SHA256s remain identical to the pinned
original experiment. No completed result was overwritten.

| Physical GPUs | Active cell at handover |
|---|---|
| 0,1 | Raw Live-A seed1994, existing training continues |
| 2,3 | Low-energy HOEP-A seed1994 |
| 4,5 | Low-energy HOEP-A seed1995 |
| 6,7 | Frozen-A seed1995 |

Frozen-A seed1994 completed: Final87.71, AAA92.057. Three seed1993 results
remain reused. Of the 15 new runs: one complete, four active, ten pending
at handover. Random/high direction runs wait for the same-seed HOEP full
completion and validation of all nine per-task, per-Q/V count vectors.

New service: `codex-hoep-cifar15-cuda7-parallel-20261009.service`.
Runtime subdirectory: `parallel/` (queue.log, events.jsonl, status.json,
job_results/ and the archived dispatcher source/hash). The original
status.json and results_summary.json also show the aggregate parallel state.

Only the old serial scheduler parent was paused to prevent duplicate
dispatch; existing Live torchrun695244 and rank PIDs695408/695409 kept
running. It continued Task3, and new jobs entered Task0. Verification showed
all eight GPUs computing, 8480–8734MiB each, with no failed jobs. The old
scheduler will be retired after the adopted child has a verified exit status;
the old service may show failed after this planned retirement.

Parallel scheduler regressions passed (eight tests); full suite passed403.
Independent review's worker-recovery blocker was fixed and re-reviewed.
See the isolated worktree's `docs/hoep_cifar_parallel_20261009.md` for
process-handover details and dispatcher provenance. Full experiments continue;
this records a launch, not final experiment results.

## Restart excluding unstable GPU3/4 — 2026-10-09 09:39 CST

The host rebooted around01:35, interrupting all four active experiments.
User requested restart while excluding physical GPU3 and GPU4. The new
queue uses0,1 /2,5 /6,7 and retains the completed Frozen1994 and original
seed1993 cells. Fourteen unfinished cells restart fromTask0 in
`.runtime_hoep_cifar15_20261009_stable`; all interrupted files are retained.
Current service: `codex-hoep-cifar15-cuda7-stable.service` (persistent/enabled).
Use the new runtime status/results instead of the interrupted r2 state.
See [restart record](hoep_cifar15_stable_restart_20261009.md).
