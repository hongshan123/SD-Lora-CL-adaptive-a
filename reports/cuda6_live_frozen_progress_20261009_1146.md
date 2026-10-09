# Live-A / Frozen-A progress — 2026-10-09 11:46 CST

Read directly from cuda6 runtime
`/home/zhaoyang/SD-Lora-CL-live-frozen-20261008/.runtime_live_frozen_diagnosis_20261008`.
The service remains active. Single GPU per cell, batch128, T10/rank10/20epochs,
LS, NormCap and transport retained. Every complete training result below has
a ten-task curve, forgetting metric and explicit successful completion event.
Snapshot-off audit replicates are excluded from seed counts.

## ImageNet-R: all three paired training runs complete

| Seed | Live Final | Frozen Final | Difference (pp) | Live AAA | Frozen AAA | Difference (pp) |
|---|---:|---:|---:|---:|---:|---:|
| 1995 | 79.67 | 78.53 | +1.14 | 82.565 | 81.991 | +0.574 |
| 1996 | 79.42 | 78.53 | +0.89 | 82.506 | 81.932 | +0.574 |
| 1997 | 79.58 | 78.45 | +1.13 | 82.602 | 81.781 | +0.821 |
| Mean | 79.557 | 78.503 | +1.053 | 82.558 | 81.901 | +0.656 |

Forgetting Live/Frozen by seed:1995 5.906/5.907;1996 4.737/4.893;
1997 3.996/4.312. Live is higher in Final and AAA for each of the three
prespecified seeds under this protocol. This does not establish superiority
on other datasets or prove HOEP superior to Frozen.

## Mechanism diagnostics remain incomplete

ImageNet-R1995/1996 have all nine paired boundaries diagnosed. Frozen1997
also has all nine; Live1997 has Tasks1–3 complete and Task4 in progress.
Current checks pass for those three new boundary files: S3/S4 max absolute
logit differences8.643e-7/8.047e-7/8.941e-7; S5/deployment difference0;
formal prototype differences below3.651e-7. Do not extrapolate partial
seed1997 results as the full three-seed mechanism summary.

The previous two-seed diagnosis remains: pre-merge global advantage small,
NormCap expands it, and transport slightly reduces it. Direct fixed-head
S4→S5 global changes average+0.866/+0.620pp for Live1995/1996 and
+0.202/+0.108pp for Frozen. Those are boundary counterfactuals, not an
end-to-end no-NormCap training ablation. Detailed stage/old/new/residual
results are in cuda6_live_frozen_progress_20261009.md.

## CIFAR and CUB

| CIFAR seed | Frozen Final | Frozen AAA | Frozen Forgetting |
|---|---:|---:|---:|
| 1993 | 88.34 | 92.372 | 5.600 |
| 1994 | 87.07 | 91.401 | 8.000 |

CIFAR Frozen1993 has all nine diagnostics. Frozen1994 has Tasks1–5 and
Task6 is in progress. CIFAR Live has not produced completed runs in this
queue, so no paired CIFAR conclusion is available. CUB has no results yet.

## cuda7 HOEP ablations: separate protocol

The six allowed cards0,1,2,5,6,7 are computing;3/4 remain at1MiB/0%.
The persistent stable service is active, with one new cell previously
completed (Frozen1994), three active and eleven pending. Three original
seed1993 cells remain reused. No additional full result is available.

Since the09:39 launch, two changed-boot recoveries archived all three
interrupted attempts and restarted them (events09:56 and11:10 CST).
Current jobs are HOEP1994/HOEP1995/Frozen1995 on0,1/2,5/6,7, still Task0.
Thus automatic recovery has operated on actual reboots, but repeated host
reboots have delayed completion despite exclusion of GPU3/4. The cause of
those host reboots has not been established by this status inspection.

This cuda7 two-GPU HOEP direction ablation disables NormCap/transport;
its results must be kept separate from cuda6's boundary-diagnostic protocol.
