# Frozen-A G / head counterfactual, 2026-09-24

## Protocol

- Read-only evaluation on saved post-merge task snapshots from the CUB-200 seed 1 and CIFAR-100 seed 1993 T=10 runs.
- For each anchor -> target pair, evaluate the *same anchor-class test samples* in the target task's full class space.
- Cross the anchor/target merged G with anchor/target **old-class prototype rows**. Every class introduced after the anchor keeps its target-task prototype. Frozen A is bit-identical across the paired snapshots.
- Repeat the same 2x2 grid for the saved linear FC weight **and bias**, but this is a shadow analysis: the deployed classifier is the prototype cosine head, not FC. The prototype head itself is the deployed classifier, so prototype and FC cannot be treated as two independently active modules.
- Report global Top-1, old-class-restricted Top-1, and later-class prediction rate. The target-G/target-prototype arm reproduces the task accuracy matrix in the training log.
- The interventions are diagnostic mismatches between independently trained states. They are not deployable algorithms, and the four cells are not an additive causal decomposition.

## Deployed prototype-head Top-1 (%)

| Dataset / pair | Old G + old rows | Old G + target rows | Target G + old rows | Target G + target rows |
|---|---:|---:|---:|---:|
| CUB Task 0 -> 9, 575 old samples | 83.65 | 80.52 | 66.43 | **70.43** |
| CIFAR-100 Task 0 -> 9, 1000 old samples | 92.40 | 89.10 | 78.60 | **83.60** |
| CUB Task 8 -> 9, 5197 old samples | 85.15 | 85.05 | 84.82 | **84.74** |
| CIFAR-100 Task 8 -> 9, 9000 old samples | 88.08 | 88.12 | 88.04 | **87.98** |
| CIFAR-100 Task 6 -> 7, 7000 old samples | 90.20 | 90.04 | 89.14 | **89.56** |

All nine adjacent transitions (Task 0 -> 1 through Task 8 -> 9) were evaluated for each dataset. The table above shows the long-horizon and selected transition examples; all transition JSON/PT files are present in this directory.

| Dataset | Old-G rollback with target prototypes: positive transitions | Mean Top-1 effect | Old-prototype rollback with target G: positive transitions | Mean Top-1 effect |
|---|---:|---:|---:|---:|
| CUB-200 | 5 / 9 | +0.046 pp | 2 / 9 | -0.082 pp |
| CIFAR-100 | 8 / 9 | +0.143 pp | 1 / 9 | -0.159 pp |

These are unweighted means of paired *old-class-cohort* accuracy differences, not changes in the full-task Final metric. For CIFAR-100, the Task 0 -> 1 G rollback is -1.00 pp: the new G helps old classes at that boundary. Thus a blanket freeze/attenuation of every G update is not supported.

## Interpretation

1. **Long-horizon G effect is large under a fixed target prototype head.** Replacing the final G by Task-0 G recovers +10.09 pp on CUB and +5.50 pp on CIFAR-100 for Task-0 samples. The current Task-0 restricted accuracy remains high (96.87% / 98.20%); global loss is primarily competition against later classes, not confusion among the original classes.
2. **Prototype transport/calibration is not simply harmful.** With final G fixed, reverting only the Task-0 prototype rows *lowers* Task-0 Top-1 by 4.00 pp on CUB and 5.00 pp on CIFAR-100. Old prototypes and new G are mismatched. With old G fixed, target prototype rows are worse by 3.13 / 3.30 pp. This sign reversal demonstrates a strong G/prototype interaction.
3. **New-class competition exists in the mixed-state old-operator counterfactual.** Task-0-only Top-1 at training time was 96.87% / 96.90%; with old G and old prototype rows but later-class target rows present, it is 83.65% / 92.40%. The final model predicts a later class for 28.87% / 15.90% of Task-0 samples. However, later-class prototype rows were trained under the target G, so this mixed-state comparison does **not** isolate a pure class-count or head-only effect.
4. **The final transition is small.** Task 8 -> 9 old-class Top-1 falls by 0.40 pp on CUB and about 0.10 pp on CIFAR-100 in the old-G/old-rows versus target/target comparison. CIFAR-100 Task 6 -> 7 is more consequential: +0.49 pp is recovered by reverting G with target prototypes, while reverting old prototype rows with target G costs 0.41 pp.
5. **The saved FC is a different, unused head.** With final G on Task-0 samples, the shadow linear FC gives 64.00% CUB / 80.40% CIFAR-100, versus deployed prototype head 70.43% / 83.60%. FC is not a third independent intervention within the deployed model.

## Decision

Do not regularize historical prototype updates toward their old coordinates, and do not claim current-branch intrusion alone explains the long-term decline. A fixed-strength G freeze is also unsupported by the mixed signs of adjacent effects. The next mechanism test should target **when cumulative G changes erode old-vs-later class margins**, while measuring the coupled response of prototype recalibration. Compare selective G consolidation with a separate old-vs-new logit calibration diagnostic on permitted current-task data; report both old and new accuracy, and never use old test labels to set a training constraint or hyperparameter.

Machine-readable results and per-sample predictions are in the neighboring `.json` and `.pt` files. Reproduce with `scripts/counterfactual_g_head_swap.py` and the saved run directories.
