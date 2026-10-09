# cuda6 Live-A / Frozen-A progress, 2026-10-09

Direct SSH observation around09:43–09:46 CST, host cuda6
(zhaoyang@192.168.10.206). Experiment repository:
`/home/zhaoyang/SD-Lora-CL-live-frozen-20261008`.
Runtime: `.runtime_live_frozen_diagnosis_20261008`.
Service `codex-live-frozen-cuda6-20261008-r2.service` remains active.
This is the single-GPU batch128 boundary-diagnostic protocol, with LS,
NormCap and prototype transport retained; it is distinct from the cuda7
two-GPU HOEP direction-ablation protocol, where NormCap/transport are off.

## Completed full runs

Only snapshot-on runs count as formal cells. Snapshot-off runs are audit
replicates, not extra seeds. Each row below has a ten-task log, forgetting
metric, recorded completion event and nine completed boundary diagnostics.

| Dataset | Seed | Policy | Final | AAA | Forgetting |
|---|---:|---|---:|---:|---:|
| ImageNet-R | 1995 | Live | 79.67 | 82.565 | 5.906 |
| ImageNet-R | 1995 | Frozen | 78.53 | 81.991 | 5.907 |
| ImageNet-R | 1996 | Live | 79.42 | 82.506 | 4.737 |
| ImageNet-R | 1996 | Frozen | 78.53 | 81.932 | 4.893 |
| ImageNet-R | 1997 | Frozen | 78.45 | 81.781 | 4.312 |
| CIFAR-100 | 1993 | Frozen | 88.34 | 92.372 | 5.600 |

Paired ImageNet-R Final differences: +1.14pp for1995 and +0.89pp for1996.
AAA difference is +0.574pp for both. Seed1997 Live remains incomplete;
do not present these two finished pairs as the completed three-seed result.

## In progress at observation

- GPU0: ImageNet-R Live seed1997, Task2 Epoch9/20; Task0/1 accuracies88.39/87.08.
- GPU1: CIFAR-100 Frozen seed1994, Task5 Epoch4/20; last completed
  five-task curve94.70/94.65/94.57/93.12/92.30.
- CIFAR Live and all CUB cells have not yet produced results in this queue.
- GPU2/3 are occupied by other work; this diagnostic service uses GPU0/1.

## Completed ImageNet-R paired boundary diagnostics

The values below are equal-weight means of the nine Task1–9 global Top1
differences (Live minus Frozen), not Task9 final accuracy. Within each run,
S2–S5 use a fixed prototype head. S6 changes only old prototypes; the final
row separately recomputes formal new prototypes. Heads across policies are
their own traced feature/prototype states; this is a conditional operator
diagnosis and does not establish an independent causal effect of A or B.

| Stage | Seed1995 delta (pp) | Seed1996 delta (pp) |
|---|---:|---:|
| S2 trained, unmerged | +0.0264 | +0.0544 |
| S3 history LS alignment | -0.0031 | +0.0574 |
| S4 exact absorption, no NormCap | -0.0031 | +0.0574 |
| S5 NormCap | +0.6612 | +0.5697 |
| S6 prototype transport | +0.5964 | +0.5544 |
| Formal new prototype recomputation | +0.6386 | +0.6377 |

Live already improves new-class S2 accuracy relative to Frozen by2.1944pp
(1995) and1.6599pp (1996), while old-class S2 accuracy is lower by0.4219pp
and0.3266pp. Thus its pre-merge global advantage is small. NormCap then
enlarges the global paired advantage. Transport slightly reduces that
advantage in both finished pairs. These observations prioritize examining
the interaction between task learning and bounded branch absorption.

A-only S0→S1 mean old-class changes for Live are +0.1286pp/+0.3018pp;
Frozen S0 and S1 are identical in these completed diagnostics. This
counterfactual does not separate jointly trained A/B into independent causes.

### Direct NormCap effect: S4→S5

Rechecked directly on cuda6. Mean Top1 changes over Task1–9, in percentage
points; backbone boundary state, sample order and prototype head are fixed
for this operator counterfactual.

| Policy | Seed | Global delta | Old-class delta | New-class delta | Global improved boundaries |
|---|---:|---:|---:|---:|---:|
| Live | 1995 | +0.8660 | +2.3782 | -3.7777 | 8/9 |
| Live | 1996 | +0.6203 | +1.7797 | -3.6180 | 8/9 |
| Frozen | 1995 | +0.2018 | +1.5476 | -2.9807 | 8/9 |
| Frozen | 1996 | +0.1081 | +1.0352 | -2.8804 | 7/9 |

Across these two seeds, mean global changes are +0.7432pp for Live and
+0.1549pp for Frozen. Global accuracy decreases at Task1 in both policies
and both seeds; Frozen1996 also decreases at Task5. Old-class accuracy
improves and new-class accuracy decreases at every one of these boundaries.
This is a same-state boundary comparison. A full training run with NormCap
disabled would change later training states and requires its own paired
ablation; these numbers are not that end-to-end ablation.

## Reconstruction and theory checks

Seed1995 snapshot-off/on audit passes for both policies: Task0 model/RNG
equal, no mismatches, and all ten task accuracies identical. Across the54
completed boundary JSON files, maximum S3/S4 logit absolute error is
8.918345e-6 (relativeL2 max6.393765e-7), S5/deployed error is0, and formal
new-prototype max absolute error is8.121133e-7. These meet the stated tolerances.

Live mean normalized R_rec is0.006929/0.006452 for1995/1996; weighted
residual means0.027659/0.025546. Per-seed correlations use only nine
boundaries and are descriptive:

| Seed | Residual | S0→H_LS old accuracy loss Spearman | S0→formal loss Spearman |
|---|---|---:|---:|
| 1995 | R_rec | -0.6167 | -0.3333 |
| 1995 | R_weighted | -0.6833 | 0.0000 |
| 1996 | R_rec | -0.0333 | -0.5667 |
| 1996 | R_weighted | -0.5167 | -0.4000 |

These finished pairs do not show the expected positive residual/forgetting
relationship. They do not establish a recoverability theory or a general
Live-A advantage. All prespecified remaining seeds/datasets stay in the queue.

Sources read directly on cuda6: manifest.json, logs/lfdiag_*.log,
smoke_audit.json, pipeline_events.jsonl and diagnostics/*/task_*.json.
The full automatic summary remains pending the18 snapshot-on runs.
