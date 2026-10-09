# CIFAR HOEP-A restart excluding GPU3 and GPU4

Launched on cuda7 at 2026-10-09 09:39 CST following the user's restart request.
The previous server boot ended around01:33; the server restarted around01:35.
Four incomplete experiments lost their running processes. Their original
runtime, partial logs and checkpoint files are retained unchanged.

## Protocol and reuse

T10, rank10, 20 epochs/task, two ranks, batch64 per rank/global128. Training
sources retain the original SHA256s. Relocated configurations differ only
in prefix, output path and paired reference path; scientific settings,
seeds and datasets remain identical. LS and exact absorption remain on;
NormCap, transport, Dual-B and HBD remain off.

Frozen-A seed1994 is reused after verification of the original completion
event and ten-task log: Final87.71, AAA92.057. Three original seed1993 cells
are also reused. The other14 new cells are restarted in a fresh runtime.
Incomplete cells start from Task0, because the saved files do not provide
the complete optimizer/scheduler/RNG state needed for an equivalent resume.

## GPU allocation and queue

| Physical GPUs | Initial experiment |
|---|---|
| 0,1 | Low-energy HOEP-A seed1994 |
| 2,5 | Low-energy HOEP-A seed1995 |
| 6,7 | Frozen-A seed1995 |

GPU3 and GPU4 are excluded in the scheduler and worker CLI. Workers use
CUDA_DEVICE_ORDER=PCI_BUS_ID; cuda7's PCI ordering matches nvidia-smi indices.
Each free pair automatically takes another experiment. Random/high controls
wait for complete same-seed HOEP counts and validate all nine24-branch vectors.

At startup verification, three jobs entered Task0, allowed cards allocated
about8GiB and computed; GPU3/4 stayed at1MiB and0% utilization. No failures.
Of the15 new cells: one completed, three active, eleven pending.

## Service and recovery

Persistent user unit: `codex-hoep-cifar15-cuda7-stable.service`, enabled for
default.target; user linger is enabled. KillMode=process preserves healthy
workers if only the coordinator fails. A pipe gate prevents training from
starting before the worker PID/start ticks are recorded atomically.

On a subsequent host reboot, the coordinator detects a changed boot ID,
archives incomplete attempts under parallel/interrupted_attempts/ and
restarts those cells from Task0. Completed cells remain reused. Same-boot
recovery adopts living workers by PID and start ticks. A worker disappearing
without a terminal proof fails safely; the narrow case where the coordinator
dies after recording ownership but before releasing the gate requires manual
recovery or next-boot archival. Training failures stop new dispatch.

Runtime:
`/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-cifar-ablation-20261008/.runtime_hoep_cifar15_20261009_stable`

Read status.json, results_summary.json, parallel/queue.log and logs/ here.
This runtime supersedes the interrupted .runtime_hoep_cifar15_20261008_r2.
restart_audit.json records every original/restarted cell and config change.
Source snapshots and source/config/dispatcher SHA256s are archived and
verified before every dispatch. Full suite:409 passed; launch gate and
stable-pair regressions included. Independent review's startup ownership
blocker was fixed and re-reviewed before launch.
