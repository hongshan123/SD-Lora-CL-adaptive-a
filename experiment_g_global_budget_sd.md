# Global-Budget G Consolidation Experiment Ledger

## Objective and frozen protocol

Seek a rehearsal-free, nearly task-constant-state continual LoRA variant that
beats complete Frozen-A Final by **more than 0.20 percentage points** on at
least two of CIFAR-100 seed 1993, ImageNet-R seed 1995, and CUB-200 seed 1.
Use T=10, rank 10, 20 epochs/task, two GPUs with batch 64 each, the same
SGD/constant schedule, prototype transport, and no Dual-B/HBD. Frozen-P/G
attempts keep the Task-0 input basis; explicitly labelled Live-A attempts
may move it but must keep the same fixed-rank persistent state. Do not
select a method using test labels or change a dataset's budget separately.
Record every modification and its results below.

Complete Frozen-A references (Final/AAA/F): C100 88.17/92.385/5.811,
INR 78.75/81.941/5.946, CUB 84.40/89.596/8.286.
Directly verified 10-task baseline logs:
`frozen_branch_intrusion_c100_t10_seed1993_20260923.log`,
`frozen_margin_inr_t10_seed1995_20260924_r3.log`, and
`frozen_branch_intrusion_cub_t10_seed1_20260923.log`. Their configs use
`live_a_aggregate_b`, `bounded_norm_calibrated_absorb` (NormCap), and
`sa_coordinate_stable_transport=true`. These are the complete Frozen-A
references for the strict two-dataset Final criterion, not QR-additive.

## Attempt 0: Per-branch Fisher SBGC with prototype transport (completed)

Code/data: `sbgc_replacement_20260924.md`, commit `fb2dee4`.
Each Q/V branch separately met a 5% diagonal-Fisher response-risk budget.
Final/AAA: C100 88.50/92.887, INR 78.93/82.525, CUB 84.04/89.409.
This clears the >0.20 Final threshold only on C100. Versus same-QR additive,
the Fisher merge helps C100 (+0.54 Final) but hurts INR (-0.10) and CUB
(-0.41). Old holdout-CE guard was also tried previously and selected the
unchanged 5% candidate on every task; do not repeat it.

## Attempt 1: One global function-risk budget (implementation verified; GPU pending)

Keep the frozen input basis P, trained task-local B and scale, classifier,
prototype transport and Fisher/activation calibration unchanged. Only change
how the 24 Q/V matrices are absorbed into G. For branch b let O_b be old G,
T_b=O_b+sB_b be additive target, H_b historical projected activation
covariance, C_b current covariance, and f_b^h/f_b^c their diagonal output
sensitivities. Choose all G_b together:

min sum_b,j f^c_bj (g_bj-t_bj) C_b (g_bj-t_bj)^T

subject to

sum_b,j f^h_bj (g_bj-o_bj) H_b (g_bj-o_bj)^T
  <= 0.05 * sum_b,j f^h_bj o_bj H_b o_bj^T.

A single nonnegative dual multiplier eta produces the same row-wise closed
form as current SBGC; bracket and bisect eta using the **sum** of historical
energies. This reallocates a fixed global risk allowance toward branches
where the current-target benefit is largest, rather than enforcing 5% in
each branch. It needs no additional tensor state beyond existing C/f.

Commit a distinct artifact version containing `budget_scope=global`; reject
incompatible old checkpoints. Preserve `budget_scope=branch` as the exact
existing behavior. Report aggregate risk, maximum individual branch risk,
current-target distortion, solver time, Final/AAA/Forgetting, and all task
curves. The 5% quantity remains a branch-response surrogate, not a bound on
network forgetting. Do not claim global budget improves accuracy until the
three matched runs finish.

### Verification gates

1. CPU: global KKT stationarity, monotone risk, zero update, singular
   covariance, FP32 deployment and rank-1 equivalence to branch solver.
2. DDP Task 0/1 smoke on all datasets: same frozen P, exact Task-0
   canonicalization, finite statistics, full-state save/rebuild, prototype
   transport, and aggregate risk <=0.050001.
3. Formal T=10 paired against the three complete Frozen-A references and
   against Attempt 0; check 10 artifacts and 10 task snapshots per dataset.

### Results

2026-09-25 implementation: added `sa_g_budget_scope=global` and a single
FP64 dual solve over all Q/V branches. The actual FP32 deployed matrices are
rechecked against the aggregate 5% budget. The `uniform` setting uses unit
sensitivities in the global solve; `fisher_diag` uses calibrated sensitivities.
The v8 state records `budget_scope=global` and rejects loading as branch mode.
The original branch scope remains the default with v6/v7 artifacts unchanged.
Six isolated configs were added for Task 0/1 smoke and T=10 formal runs.

CPU evidence: 45 focused tests pass, including global KKT stationarity,
risk reallocation, monotonicity, singular covariance, low-precision return,
Task-0/1 state roundtrip, and strict scope mismatch. The full suite passed
487 tests after the final implementation/config patch. Code commit: `6fa7888`.

Three two-GPU Task 0/1 real-data smokes completed without traceback on
2026-09-25, using GPU pairs 0,1 / 4,5 / 6,7 and batch 64 per GPU. Each has
two verified task snapshots, passing calibration tensor/RNG hashes, rebuilt
deployment state, and prototype transport. Task-0 canonicalization operator
errors were 1.25e-7 (C100), 1.23e-7 (INR), and 1.25e-7 (CUB). Task-1 risk:

| Dataset | Aggregate risk | Max branch risk | Current distortion | Eta |
| --- | ---: | ---: | ---: | ---: |
| C100 | 0.050000 | 0.186029 | 0.458036 | 2.579047 |
| INR | 0.050000 | 0.094485 | 0.584564 | 3.550698 |
| CUB | 0.050000 | 0.111883 | 0.593997 | 3.707582 |

The branch maxima above 5% verify that this is one global budget, not the
old per-branch rule. This smoke used 2 epochs/task and is not an accuracy
comparison. T=10 Final/AAA/Forgetting remain pending. No accuracy benefit is
claimed yet.

Formal T=10 runs were launched after all three smokes passed, at code commit
`6fa7888` plus documentation commit `1a5af18`. Each uses two GPUs, batch 64
per GPU, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `nohup`/`setsid`:

| Dataset | GPU pair | Config | Log |
| --- | --- | --- | --- |
| C100 | 0,1 | `exps/sbgc_global_c100_full_20260925.json` | `sbgc_global_c100_full_20260925.log` |
| INR | 4,5 | `exps/sbgc_global_inr_full_20260925.json` | `sbgc_global_inr_full_20260925.log` |
| CUB | 6,7 | `exps/sbgc_global_cub_full_20260925.json` | `sbgc_global_cub_full_20260925.log` |

The runs are in progress; do not treat partial task curves as Final.

### CUB formal result (2026-09-25)

CUB seed 1 completed Task 0-9 with ten verified snapshots, finite statistics,
no traceback, and nine logged aggregate transition risks at 0.050000. The
Final/AAA/Forgetting are **84.17 / 89.389 / 7.911**. Final Top-1 curve:
`[96.87, 92.92, 91.54, 89.76, 89.46, 88.75, 87.53, 87.13, 85.76, 84.17]`.
Against complete Frozen-P: Final **-0.23**, AAA **-0.207**, Forgetting
**-0.375** (better). Against per-branch Fisher: Final **+0.13**, AAA
**-0.020**. Against QR additive: Final **-0.28**, AAA **-0.213**.
This is a negative CUB result for the required >0.20 Final gain. C100 and INR
are still running; no cross-dataset success claim is justified.
All ten CUB task logs report the same 389,520-scalar persistent state. At Task
9 the global model has old/new accuracy 84.61/80.40, versus complete Frozen-P
84.74/81.41. The larger new-class deficit (1.01 point) is consistent with
current-target distortion limiting plasticity, while the old-class result is
near the baseline. This is an observation, not a causal proof; the 5% risk
surrogate did not translate into a CUB Final gain.

### Interrupted C100/INR runs and exact-boundary recovery

The host rebooted at **2026-09-25 03:16** (`who -b`). The C100 process ended
mid Task 8 epoch 9/20 and INR ended during Task 5 prototype transport. Neither
log contains a model traceback. C100 Task 7 is the last fully evaluated
boundary; INR Task 4 is the last fully evaluated boundary. Both completed
snapshots pass checksum audit. The incomplete INR Task 5 snapshot fails audit
and is explicitly excluded from recovery. These partial runs are not Final.

Added a fresh-directory snapshot resume path, restricted to global SBGC v8.
It verifies the snapshot checksum, protocol keys, state version, classifier
shape, prototype coverage and prior log metrics; restores P/G/C/f, FC and
prototypes; and continues from the next task with the previous accuracy curve
and matrix. No old run directory or incomplete artifact is modified. Two new
CPU tests cover successful restore and protocol-mismatch rejection; the full
suite passes **489 tests**. First validate on CUB by restoring Task 8 and
replaying Task 9 under the original 20-epoch protocol. Only if its Final, AAA,
Forgetting and full curve match the completed CUB run will C100/INR resume.

CUB replay validation **passed**: restoring `task_008` and training Task 9
again yielded exactly the original Final **84.17**, AAA **89.389**, Forgetting
**7.911**, all ten Top-1 values, Task-9 aggregate risk 0.050000, eta
1.065696, and identical post-evaluation tensor/RNG hashes. Its Task-9
snapshot checksum audit also passed. This is evidence that the boundary resume
is trajectory-equivalent for the tested CUB case, not a proof for every task.
Recovery code/config/tests commit: `f101a97`.

C100 and INR recovery jobs were launched after the CUB replay check, with
`nohup`/`setsid`, the original two-GPU batch-64 protocol, and new artifact
directories. C100 resumes from Task 7 on GPUs 0,1 using
`exps/sbgc_global_c100_resume_20260925.json` and writes
`sbgc_global_c100_resume_20260925.log`. INR resumes from Task 4 on GPUs 4,5
using `exps/sbgc_global_inr_resume_20260925.json` and writes
`sbgc_global_inr_resume_20260925.log`. Startup confirmed `[SnapshotResume]`
next tasks 8 and 5 respectively. INR remains in progress.

### C100 formal result (completed via verified resume)

C100 seed 1993 completed Tasks 0-7 in the original directory and Tasks 8-9
from the audited Task-7 snapshot in the fresh resume directory. All ten task
boundaries are represented by verified snapshots; the resumed Task-9 eval
tensor/RNG hashes pass. Final/AAA/Forgetting are **88.55 / 92.876 / 5.467**.
Final Top-1 curve:
`[98.50, 96.40, 95.43, 94.45, 93.12, 92.15, 91.57, 89.71, 88.88, 88.55]`.
Against complete Frozen-P: Final **+0.38**, AAA **+0.491**, Forgetting
**-0.344** (better). Against per-branch Fisher: Final **+0.05**, AAA
**-0.011**. Against QR additive: Final **+0.59**, AAA **+0.370**.
C100 passes the >0.20 Final gate; INR must also pass because CUB did not.

### INR formal result and Attempt-1 decision

INR seed 1995 completed Tasks 0-4 in the original directory and Tasks 5-9
from the audited Task-4 snapshot in the fresh resume directory. All five new
snapshots pass checksum audit; every recorded transition has aggregate risk
0.050000 and Task-9 eval tensor/RNG hashes pass. Final/AAA/Forgetting are
**78.70 / 82.538 / 6.440**. Final Top-1 curve:
`[90.60, 86.52, 85.20, 82.95, 81.76, 81.46, 79.91, 79.50, 78.78, 78.70]`.
Against complete Frozen-P: Final **-0.05**, AAA **+0.597**, Forgetting
**+0.494** (worse). Against per-branch Fisher: Final **-0.23**, AAA
**+0.013**. Against QR additive: Final **-0.33**, AAA **-0.081**.

**Attempt 1 fails the requested gate**: only C100 exceeds Frozen-P Final by
more than 0.20; INR and CUB do not. A uniform 5% aggregate branch-response
risk surrogate is not sufficient to predict final classifier accuracy. Do not
promote global SBGC as the main method on this evidence. Next inspect the
transition-wise additive/Fisher/global tradeoff and design a training-only,
dataset-independent consolidation decision; do not select a policy using
test-set Final values.

## Attempt 2: Uniform 10% global budget (registered, pending)

Mechanistic observation from Attempt 1: at the final task the constrained
model's new-class accuracy is lower than complete Frozen-P by 0.30 on INR and
1.01 on CUB, while its old-class accuracy differs by only -0.02 and -0.13.
C100 gains +0.78 old-class points over its QR additive control but loses 1.10
new-class points. The final-task global current-target distortions are 0.151
(C100), 0.058 (INR), and 0.241 (CUB), all nonzero. These are associations,
not proof of causality, but motivate a less restrictive budget.

Change exactly one global hyperparameter, `sa_g_risk_budget: 0.05 -> 0.10`,
for all three datasets. Keep the code, frozen P, Fisher calibration, rank 10,
20 epochs, batch 64/GPU, seed, transport and classifier unchanged. Use new
output directories and complete T=10 runs from Task 0. This is a uniform
budget-sensitivity experiment, not dataset-specific model selection; because
the same seeds informed it, any apparent win remains exploratory and needs
independent-seed confirmation. Pre-registered checks: aggregate risk <=
0.100001 every transition; 10 snapshots/dataset; compare Final against
complete Frozen-P thresholds C100 >88.37, INR >78.95, CUB >84.60. Two of
three must exceed their threshold. If not, stop budget-only tuning and move
to a different training-only consolidation signal.

Results pending.

The three Attempt-2 runs were launched from fresh Task 0 after config commit
`cb8cb5c`, with `nohup`/`setsid`, batch 64 per GPU, and deterministic CuBLAS.
C100 uses GPUs 0,1 and `sbgc_global_c100_b10_20260925.log`; INR uses 4,5
and `sbgc_global_inr_b10_20260925.log`; CUB uses 6,7 and
`sbgc_global_cub_b10_20260925.log`. All three reached Task 0 without startup
error.

### CUB 10% formal result

The CUB seed-1 T=10 run completed all tasks. `audit_sa_task_snapshots.py`
verified all ten finalized snapshots; all nine aggregate-risk records are
0.100000, and the Task-9 RNG/evaluation tensor hashes pass. Final/AAA/
Forgetting are **84.33 / 89.524 / 7.927**. Final Top-1 curve:
`[96.87, 93.01, 91.60, 89.97, 89.63, 88.95, 87.70, 87.23, 85.95, 84.33]`.
Versus complete Frozen-P this is Final **-0.07**, AAA **-0.072**, and
Forgetting **-0.359** (better). Versus global 5% it is Final **+0.16**
and AAA **+0.135**. It does **not** meet the strict CUB Final threshold
`>84.60`. At Task 9 the global current-target distortion is 0.00137,
so loosening the budget has nearly recovered the additive target at that
boundary, but has not produced the required Final gain. C100/INR remain
necessary for the two-dataset goal.

### INR GPU-4 failure and recovery boundary

The initial INR 10% job completed and verified Task 0, then trained Task 1
for all 20 epochs and recorded a 0.100000 aggregate merge risk. Before its
Task-1 prototype synchronization, GPU 4 became unreadable to `nvidia-smi`
(`Unknown Error`); rank 1 timed out on a four-element NCCL broadcast. The
subsequent absurd tensor size was observed *after* the NCCL failure and is
not evidence of an SBGC numerical failure. The Task-1 snapshot contains
only `config.json` and `pre_merge.pt`, so it is invalid as a resume point.
The Task-0 snapshot has all eight files and passes the SHA256 audit.
Only the failed INR process group was terminated; C100/CUB were not touched.
Resume from the completed Task-0 snapshot on healthy GPUs 6,7 using
`exps/sbgc_global_inr_b10_resume_t0_20260925.json`, a fresh output
directory, and the same per-GPU batch 64 and all algorithm settings. The
resume config differs only in prefix, filepath, and `sa_resume_snapshot`.
Do not count the interrupted Task-1 run as a result.

The first local resume attempt on GPUs 6,7 failed before model loading:
even a separate minimal `CUDA_VISIBLE_DEVICES=6,7` PyTorch process could
not initialize the CUDA driver after GPU 4's failure. The already-running
C100 process was left intact. A fresh checkout of committed HEAD `a3ee5c0`
was deployed on cuda6 (`192.168.10.206`) at
`/home/zhaoyang/SD-Lora-CL-sbgc-b10-20260925`. The remote sdlora
environment reports PyTorch 2.4.1+cu121/timm 1.0.9 and initializes two
GPUs; the ImageNet-R dataset has 200 train-class directories. The exact
Task-0 snapshot and source accuracy log were copied to that checkout, and
the remote SHA256 snapshot audit passed (8/8 files). On cuda6 GPUs 0,1,
the two-rank, batch-64/GPU resume launched with deterministic cuBLAS;
the log confirms `[SnapshotResume] restored through task 0; next task 1;
previous top1=[90.6]`. Remote log:
`/home/zhaoyang/SD-Lora-CL-sbgc-b10-20260925/sbgc_global_inr_b10_resume_t0_20260925.log`.
At launch the resumed result was held pending until ten completed snapshots,
risk audit, and final metric verification.

### Complete 10% results and decision

All three T=10 runs completed. C100 and CUB each have ten locally audited
finalized snapshots. INR has the audited original Task-0 snapshot plus nine
audited Task-1–9 snapshots from cuda6; the latter were synced back to
`SBGC_GLOBAL_INR_B10_RESUME_T0_20260925/` and re-audited locally. Each of
the nine aggregate-risk records per dataset is <=0.100000. Task-9 RNG and
evaluation tensor hashes pass in all three logs. INR's Task-9 additive
target already satisfies the budget (risk 0.0690544, eta 0); this is not a
missing constraint application.

| Dataset | Frozen-P Final / AAA / F | Global 5% Final / AAA / F | Global 10% Final / AAA / F | 10% Final delta vs Frozen |
| --- | --- | --- | --- | ---: |
| C100 | 88.17 / 92.385 / 5.811 | 88.55 / 92.876 / 5.467 | 88.07 / 92.676 / 6.900 | -0.10 |
| INR | 78.75 / 81.941 / 5.946 | 78.70 / 82.538 / 6.440 | 78.63 / 83.257 / 4.844 | -0.12 |
| CUB | 84.40 / 89.596 / 8.286 | 84.17 / 89.389 / 7.911 | 84.33 / 89.524 / 7.927 | -0.07 |

C100 10% Top-1 curve:
`[98.50, 96.40, 95.50, 94.58, 93.08, 92.03, 91.20, 89.08, 88.32, 88.07]`.
INR 10% Top-1 curve (audited Task-0 source + resumed Tasks 1–9):
`[90.60, 88.58, 86.52, 83.67, 82.29, 82.56, 80.83, 80.05, 78.84, 78.63]`.
CUB curve is recorded above. INR remote training log is mirrored locally at
`sbgc_global_inr_b10_resume_t0_20260925_remote.log`.

**Attempt 2 fails the objective**: zero of three 10% Final results exceed
the matched complete Frozen-P reference, let alone the strict >0.20 point
criterion on two datasets. The increased INR AAA and lower INR forgetting
do not satisfy a Final-based goal. Uniformly relaxing the risk cap also
removes the C100 Final benefit seen at 5%; the C100 10% Final is 0.48
points below C100 5%. These are single-seed system-level comparisons; the
Frozen-P recipe retains NormCap and a different Task-0 parameterization.

Per the pre-registered decision, **stop budget-only tuning** of this G
response-risk surrogate. Do not select 5% for C100 and 10% for CUB/INR or
claim an improvement from the partial AAA gains. The next proposal must
change the training/selection signal, establish it with a separate diagnostic
that does not use test labels to choose deployment, and then run a new
uniform three-dataset protocol.

## Attempt 3: Train-time projected current branch (pre-registered)

The previous SBGC variants train the temporary `sB` branch unconstrained,
then replace its operator at the task boundary. This creates a measurable
train/deployment mismatch: in the 5% runs, current-target distortion can
be large at early transitions. Attempt 3 changes **when** the constraint
enters optimization, not its fixed per-branch 5% budget. Task 0 and its
QR-canonicalization remain unchanged. For each Task `t>0` Q/V branch, let
`P` be the frozen row-orthonormal basis, `G` the historical up matrix,
`C_h` the saved projected-activation covariance, and `f_h` the saved
output-gradient diagonal sensitivity. Set `D=sB` and

`E_h(X)=sum_j f_h[j] X[j,:] C_h X[j,:]^T`,
`R_h(D)=E_h(D)/(E_h(G)+1e-12)`,
`alpha(D)=min(1, sqrt(0.05/(R_h(D)+1e-12)))`.

For FP32 accumulation, the executable projection uses `0.05*(1-1e-4)`
inside the square root. This fixed numerical margin leaves the declared
5% upper bound unchanged and prevents a roundoff-only second projection
at deployment.

During every Task `t>0` forward, use `G P x + alpha(D) D P x`. The scalar
`alpha` remains in the autograd graph; it is not a detached post-hoc cap.
At task end, absorb exactly `alpha(D_end) D_end` into `G`. The existing
per-branch SBGC solver is retained as a numerical feasibility check; its
unconstrained target must already be within the 5% historical risk ball,
so it should return the target unchanged (eta 0, near-zero distortion).
Calibration and prototype transport remain otherwise unchanged. This
eliminates train/deployment mismatch in the LoRA branch while allowing
SGD to optimize the direction of its feasible current update. It does
**not** claim the response surrogate bounds classification forgetting.

Implementation is opt-in via `sa_g_train_projected=true`, valid only for
per-branch Fisher SBGC with shadow/holdout guard disabled. The default
remains the audited legacy behavior. No new persistent tensor or per-task
LoRA archive is allowed; the 389,520-scalar adaptation state is unchanged.
Record per-task min/mean `alpha`, fraction of active branch projections,
pre-save/post-absorb operator error, risk, Final/AAA/Forgetting and curves.

First pass CPU tests: zero `B`, exact risk-ball projection, finite gradients,
disabled-path identity, and pre-save/post-absorb operator equivalence.
Then run real-data Task 0/1, two-GPU, batch64/GPU smokes on C100/INR/CUB.
If these pass, run the exact matched T=10, rank10, 20-epoch single-seed
protocol (C100 1993, INR 1995, CUB 1) with one uniform 5% budget and no
dataset-specific thresholds. Compare with complete Frozen-P, QR additive,
and the already completed post-hoc per-branch Fisher 5% runs. Acceptance
for the user objective remains **strictly more than +0.20 Final points
over complete Frozen-P on at least two datasets**, with all ten task
snapshots and budget checks. A positive single-seed result is exploratory
and would require independent-seed confirmation for publication.

### Implementation and CPU verification

Added a differentiable historical-response-ball projection for the
effective current update `D=sB`. The Q/V wrapper applies its scalar in
every Task 1+ current-branch forward and records alpha/active statistics;
the task boundary uses the same projected `D` as the absorption target.
Pre-merge snapshots record the effective current up matrix and alpha.
Projected-mode artifacts are v9 and reject loading under the legacy mode
or vice versa. No new persistent tensors were added. Three Task0/1 smoke
configs use the same seeds, rank10, batch64/GPU, 2 epochs and independent
output directories. The focused SBGC/config tests passed 50/50; the full
CPU suite passed 493 tests with one pre-existing skip. Local GPU smoke is
unavailable while cuda7's CUDA driver fails initialization, so the
real-data smoke will use cuda6's healthy GPUs. Formal accuracy remains
unmeasured; no improvement is claimed here.

### Three-dataset real-data Task 0/1 smoke (cuda6)

Commit `cc405a9` was deployed in a separate checkout at
`/home/zhaoyang/SD-Lora-CL-sbgc-trainproj-20260925`. The remote focused
suite passed 50/50. All three runs used two RTX 3090s (GPU 0,1), batch
64 per GPU, rank 10, two epochs per task, and their pre-registered
dataset/seed protocol. Each run completed Task 0 and Task 1, passed
calibration parameter/RNG checks, rank prototype synchronization, eval
tensor/RNG hash checks, and both immutable task-snapshot audits.

| Dataset (seed) | Task 1 Top-1 (smoke only) | Max Task 1 risk | Mean alpha | Projection active fraction | Max absorption error |
| --- | ---: | ---: | ---: | ---: | ---: |
| C100 (1993) | 95.50 | 0.049995 | 0.692284 | 0.760937 | 2.853555e-8 |
| INR (1995) | 77.32 | 0.049995 | 0.654206 | 0.762897 | 2.994958e-8 |
| CUB (1) | 92.66 | 0.049995 | 0.715517 | 0.620833 | 2.959602e-8 |

The Task-0 QR operator relative errors were 1.247822e-7, 1.212892e-7,
and 1.252387e-7 respectively. Each task artifact reported exactly 389,520
persistent adaptation scalars. At Task 1 the boundary solver was inactive
(`active=0`, `mean_distortion=0`) because the training-time projected update
was already feasible. These are **engineering smoke results**, not a matched
20-epoch accuracy comparison or evidence of a Final improvement.

Formal T=10 configs are `exps/sbgc_trainproj_{c100,inr,cub}_t10_20260925.json`.
They retain the corresponding prior Fisher-SBGC protocol and set only the
new training-time projection plus a new output name/path. The next required
evidence is ten complete snapshots and matched Final/AAA/Forgetting for
all three datasets, compared to complete Frozen-P under the objective's
strict >0.20 Final-point criterion on at least two datasets.

### Formal T=10 queue launched

The three formal configs and audited dual-GPU runner were committed as
`523e0b8` and `c3fa1f8`. The cuda6 checkout is at `c3fa1f8`.
`run_sbgc_trainproj_t10_queue.sh` launched at 2026-09-25 14:20:10 UTC
on GPU 0,1, using 64 samples/GPU, in the fixed order C100 -> INR -> CUB.
Queue log: `sbgc_trainproj_t10_queue_20260925.log` in the remote checkout.
Each experiment writes its own same-named `.log`, a separate result directory,
and a ten-task snapshot audit JSON before the next experiment starts. The
first C100 run entered Task 0 and GPU utilization was nonzero; no full
result exists at this point. Do not infer success from smoke or early epochs.

After cuda7 recovered and local PyTorch confirmed all eight CUDA devices,
the same committed INR and CUB formal configs were additionally launched
there on GPU 0,1 and 4,5 respectively (two processes, 64 samples/GPU).
Their local logs are `sbgc_trainproj_inr_t10_20260925.log` and
`sbgc_trainproj_cub_t10_20260925.log`; they write independent local
result directories, not the cuda6 checkout. To prevent redundant remote
INR/CUB runs, the cuda6 queue controller PID 1513245 was SIGSTOPed while
its C100 torchrun child PID 1513248 continued training. The C100 child was
observed advancing from Task-0 epoch 13 to 14 after this intervention.
After C100 completes, audit its ten snapshots manually and terminate the
stopped controller before it can dispatch another dataset. This change
only affects scheduling; it does not alter model code, configuration, or
the already-running C100 training process.

### Attempt 3 formal results: INR and CUB complete

The local INR and CUB two-GPU runs both completed all 10 tasks on the
pre-registered configs. All 10 task snapshots per run passed checksum
audit, all nine noninitial boundary risks were <= 0.04999501, the boundary
solver reported `active=0` and `mean_distortion=0` on every task, and every
task reported 389,520 persistent adaptation scalars. No traceback or
nonfinite failure was found in either log.

| Dataset | Full Frozen-A Final / AAA / F | Post-hoc Fisher Final / AAA / F | Train-projected Final / AAA / F | Final delta vs full Frozen-A |
| --- | --- | --- | --- | ---: |
| INR | 78.75 / 81.941 / 5.946 | 78.93 / 82.525 / 6.196 | 78.78 / 82.653 / 6.357 | +0.03 |
| CUB | 84.40 / 89.596 / 8.286 | 84.04 / 89.409 / 8.122 | 84.02 / 89.413 / 8.130 | -0.38 |

INR Top-1 curve:
`[90.60, 87.05, 85.56, 83.14, 81.80, 81.35, 79.89, 79.46, 78.90, 78.78]`.
CUB Top-1 curve:
`[96.87, 92.83, 91.60, 89.84, 89.66, 88.72, 87.68, 87.21, 85.70, 84.02]`.
These are **negative or inconclusive Final results** under the stated
criterion: neither exceeds its full Frozen-A reference by >0.20 points.
Train-time projection improved INR AAA relative to post-hoc Fisher but
reduced its Final by 0.15, while CUB was essentially unchanged. Its
operator-preserving absorption and zero boundary distortion were verified,
so train/deploy mismatch alone does not explain the CUB deficit. This is
an inference from paired single-seed runs, not a causal proof or a
general claim about all risk budgets. C100 remains in progress and no
Attempt-3 success decision is made from partial results.

### Attempt 3 complete three-dataset decision (2026-09-26)

C100 completed Task 0-9 on cuda6. Ten task snapshots passed checksum audit;
all ten `[SBGC]` records have 24 Q/V branches, fixed 389,520-scalar
persistent state, boundary solver `active=0`, and zero target distortion.
The maximum reported risk was 0.04999501; the task-9 absorption error was
2.667357e-8. All ten evaluation tensor-hash checks passed, with no
traceback. After C100 finished, the stopped cuda6 queue controller was
terminated before it could dispatch duplicate INR/CUB experiments. The
complete remote C100 log was copied locally as
`sbgc_trainproj_c100_t10_20260925_remote.log`.

| Dataset (seed) | Complete Frozen-A Final / AAA / F | Train-projected Final / AAA / F | Final delta | >+0.20? |
| --- | --- | --- | ---: | --- |
| C100 (1993) | 88.17 / 92.385 / 5.811 | 88.47 / 92.846 / 5.633 | +0.30 | yes |
| INR (1995) | 78.75 / 81.941 / 5.946 | 78.78 / 82.653 / 6.357 | +0.03 | no |
| CUB (1) | 84.40 / 89.596 / 8.286 | 84.02 / 89.413 / 8.130 | -0.38 | no |

C100 Top-1 curve:
`[98.50, 96.35, 95.53, 94.35, 93.10, 92.12, 91.47, 89.69, 88.88, 88.47]`.
The user objective is **not met**: only one of the three datasets exceeds
the complete NormCap-plus-prototype-transport Frozen-A Final by strictly
more than 0.20 points. C100 is 0.03 below the earlier post-hoc Fisher
SBGC Final (88.50); INR is 0.15 below it; CUB is 0.02 below it. Thus the
train-time projection is technically valid, but does not establish a
cross-dataset Final improvement. Do not tune risk budgets per dataset or
select a method using test labels. A separate fixed-rank joint A/B
compression question was raised; it requires its own pre-registered
diagnostic because changing P creates historical recoverability loss that
the present projected-covariance state cannot measure.

## Pre-A/B offline feasibility audit (2026-09-26; no training change)

Question: can a task-local joint `A_t/B_t` update introduce new input
directions while returning to one persistent rank-10 `(P,G)` state? This is
an offline capacity diagnostic, not a CIL accuracy result or an approved
new training mode.

The complete Frozen-A Task-0 artifacts save 24 Q/V pairs as `shared_a=A`
and `aggregate_up=G`; their deployed historical operator is
`M=G A/(||A||_F+1e-8)`. For `A^T=QR`, setting `P=Q^T` and
`G_can=G R^T/(||A||_F+1e-8)` preserves `M=G_can P` exactly. Across the
three Task-0 artifacts, the maximum FP64 relative operator error was
`2.97e-16`; maximum `cond(A)` was C100 `1.049`, INR `1.109`, CUB `1.006`.
Thus canonicalization is a compatible starting representation, but
conditioning of Task-0 A is not the apparent bottleneck. The existing
snapshot-resume utility accepts only global-SBGC v8, not these Frozen-A
v4 artifacts, so a matched new-mode probe cannot use it unchanged.

For each saved Frozen-A operator, the singular values of the historical
rank-10 operator were computed through the equivalent `768x10` factor
`G R^T/(||A||_F+1e-8)`. At both Task 0 and Task 9, all 24 branches in
each dataset had weakest-direction energy <=5% of their own operator
Frobenius energy. The number `k` of weakest singular directions whose
*combined* energy is <=5% had median C100/INR/CUB values of `2/1/2`
at Task 0 and `2/2/2` at Task 9. This suggests approximately 1-2
rank slots per branch may be exchangeable under an **operator-energy**
criterion. It does not bound branch responses, old-class predictions,
or forgetting.

Existing `CAUSAL_CUB_INTERVENTION_20260923/counterfactual/t2_raw.json`
is a warning against equating low operator reprojection risk with old-class
function safety: CUB Task 2 old-class global prototype Top-1 was `39.55`
with the trained current branch enabled versus `93.09` with exact history
alone, although the reported historical row-space risk was `0.02435`.
That diagnostic used a temporary prototype setup and is not the formal
Final metric. Earlier HOEP-A already opened low-historical-energy A
directions and failed its three-dataset improvement gate; merely freeing
weak directions would repeat that idea. A genuinely distinct joint A/B
probe must use the *learned current operator* as current-task demand and
separately test pre-merge current-branch intrusion and rank-10 compression.
No per-task Live-A pre-merge `A_t/B_t` pair was found in the existing
artifacts, so this cannot be measured from completed runs alone.

## Matched-protocol route audit (2026-09-26; no new training)

The >+0.20 Final objective requires two datasets, not necessarily CUB.
Existing `final_a_policy_live_a_inr_t10_seed1995_bs64_2gpu_20260919_104122_final_a_policy.log`
reports INR Final `79.52` versus the complete Frozen-A `78.75` (+0.77).
Both logs report the same seed 1995, T=10, rank 10, constant schedule,
learning rate 0.01, 20 epochs/task, two GPUs with batch 64/GPU, NormCap,
prototype transport, and no Dual-B. This makes INR a plausible positive
Live-A endpoint, subject to a final code/protocol reproduction audit.

The analogous C100 Live-A log reports Final `88.19` versus `87.99` for
its *own* matched Frozen-A run, both with cosine scheduling and learning
rate 0.008. The complete Frozen-A reference used by this objective is
`88.17`, but its C100 config uses constant scheduling and learning rate
0.01. Therefore `88.19 - 88.17 = +0.02` is only a cross-config numerical
contrast, **not** a controlled A-policy effect, and the +0.20 contrast
against the cosine Frozen-A control does not satisfy the strict >+0.20
criterion against complete Frozen-A. A new C100 Live-A run matched to the
complete Frozen-A protocol would be required before combining C100 and
INR as evidence. The current C100 SBGC +0.30 result also begins at
Task-0 Top-1 `98.50` versus `96.90` for complete Frozen-A, so it must
not be attributed solely to its later G constraint.

This audit changes experiment priority: matched C100 Live-A is potentially
more informative than repeating a weak-direction A-opening experiment
on CUB. No existing result yet proves the two-dataset objective, and no
test-labelled candidate selection is licensed by this retrospective
inspection.

## Attempt 4: Full-protocol C100 Live-A paired control (pre-registered)

This is a **baseline and route-selection experiment**, not a claim that
Live-A is a new algorithm. The existing INR final-policy Live/Frozen pair
uses the same commit `db039a1`, differs only in
`sa_train_a_all_tasks`, and reports `79.52/78.75` Final. Its favorable
INR contrast motivates checking whether the same A/B training path can
also clear the complete Frozen-A C100 reference under the *actual*
constant-LR protocol. The older C100 final-policy pair used cosine/
0.008, whereas complete Frozen-A C100 used constant/0.01, so it cannot
answer that question.

Two new C100 seed1993 configs copy
`exps/frozen_branch_intrusion_c100_t10_seed1993.json` exactly except
for unique prefix/output paths and `sa_train_a_all_tasks` (`true` for
Live-A, `false` for its same-commit Frozen-A control). Both use T=10,
rank10, 20 epochs/task, constant learning rate 0.01, SGD, effective
batch128 (two GPUs, 64/GPU), NormCap, prototype transport, no Dual-B,
and task snapshots. They write separate logs and artifacts. No code
or dataset-specific hyperparameter is changed.

Pre-registered decision: Live-A C100 must exceed the complete
Frozen-A Final `88.17` by **strictly** >0.20 points (`>88.37`) *and*
exceed its same-commit paired Frozen-A Final by >0.20. If both hold,
reproduce the INR Live/Frozen pair and evaluate CUB under the same
code/protocol before claiming a three-dataset, two-win result. If
C100 fails, do not combine the earlier INR result with mismatched C100
or select a dataset-specific A policy; return to a pre-registered
algorithmic intervention. Record all ten task curves, Final, AAA,
Forgetting, artifact integrity, and any runtime failure. No partial
task result counts as a completed experiment.

### Launch and deterministic-environment recovery

Configs and pre-registration were committed as `3330029`. Both first
`nohup` commands were issued from a short-lived command shell; this
environment cleaned their ordinary background processes on shell exit,
leaving empty logs. A 60-second `nohup sleep` reproduced the launcher
behavior. A user-systemd test unit stayed active, so both training jobs
were placed under separate user-systemd services while retaining `nohup`
and separate file logs.

The first systemd attempt reached Task 0 but both ranks then failed
before any completed task with PyTorch deterministic CuBLAS requiring
`CUBLAS_WORKSPACE_CONFIG`. Their logs and manifest-only output directories
were preserved with `_failed_cublas` / `_FAILED_CUBLAS` suffixes; they are
not experiment results. The retried units add only
`CUBLAS_WORKSPACE_CONFIG=:4096:8` to the launch environment. Frozen-A
uses GPUs 0,1 under `codex-c100-frozen-pair-20260926-r2.service`, Live-A
uses GPUs 4,5 under `codex-c100-live-pair-20260926-r2.service`. Both have
been observed entering Task 0. Current logs are
`c100_frozen_pair_full_frozen_protocol_20260926.log` and
`c100_live_pair_full_frozen_protocol_20260926.log`. Final metrics and
snapshot audits are pending; do not count launch status as success.

### Attempt 4 outcome: Live-A fails the C100 Final gate

Both retried runs used commit `3330029` and the two pre-registered
configs. Live-A completed Task 0-9 with no traceback. All ten of its
task snapshots passed `audit_snapshot` checksum verification. Its
Final/AAA/Forgetting were `88.23 / 92.411 / 6.289`, with Top-1 curve
`[96.90, 96.30, 95.17, 94.08, 92.74, 91.65, 91.14, 89.22, 88.68, 88.23]`.
Against the complete Frozen-A C100 reference `88.17`, its Final delta
is only `+0.06`, short of the strict `>+0.20` goal. Thus Live-A alone
does not supply a second winning dataset alongside the matched INR
Live-A result, and no three-dataset success can be claimed.

The same-commit Frozen-A control completed Tasks 0-8; all nine
corresponding snapshots passed checksum verification and its Top-1
curve through Task 8 exactly matched the earlier complete Frozen-A
reference:
`[96.90, 95.80, 95.10, 94.02, 92.80, 91.93, 91.29, 89.28, 88.56]`.
Task 9 training and the `sa_state.pt` save began, but rank 1 then timed
out on a four-element NCCL broadcast after 600 seconds. Rank 0's
underlying stall/exit was not visible in the log; user-level journal
recorded torchrun exit status 1, and kernel logs were inaccessible.
Task-9 snapshot lacks the post-merge state/prototypes/classifier and
**is invalid**. Do not infer a same-commit Frozen-A Final from its
partial artifacts or count this as a completed paired comparison.
The earlier independently completed Frozen-A `88.17` remains the
reference for the user's threshold. No automatic restart or partial
result reuse was performed.

Interpretation limited to this single seed: Live-A improved early
C100 stages relative to the complete Frozen-A curve (Task 1 `+0.50`),
but the margin faded by the final task. This suggests that unrestricted
A motion alone is insufficient under the full protocol. It does not
establish that A drift caused the loss; classifier/prototype changes
remain possible confounders. Return to algorithmic intervention rather
than repeat an identical full Live-A run.

### Attempt 5: Joint A/B mechanism diagnosis before intervention

This is a read-only diagnosis of the completed C100 Live-A run, not a
new training result. The 24 Q/V branch tensors were read from each
`task_snapshots/task_XXX/pre_merge.pt`; the preceding task's persistent
`A,G` came from `sa_state.pt`. All ten Live-A snapshots passed the
existing checksum audit. No parameter, checkpoint, or training config
was changed.

For each branch, use the actual bounded-NormCap merge gain
`c=min(1,1/(||A_t||_F ||B_t||_F))`, historical operator
`M=G_{t-1} A_{t-1}/(||A_{t-1}||_F+1e-8)`, and absorbed current operator
`D=c s_t B_t A_t`. In FP64 compute current novelty
`||D(I-P_{A_{t-1}})||_F^2`, historical loss
`||M(I-P_{A_t})||_F^2`, and the exact best-rank-10 tail of `M+D`
using a 20-by-20 QR/SVD core. Values below are sums across 24 branches,
normalized by `sum ||M||_F^2` except for the last column, which is
normalized by `sum ||M+D||_F^2`.

| Task | Mean cap gain | New novelty / old energy | Old loss / old energy | New novelty / old loss | Best-rank-10 tail |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.773 | 0.00759 | 0.01342 | 0.566 | 0.001234 |
| 2 | 0.846 | 0.00078 | 0.00533 | 0.146 | 0.000413 |
| 3 | 0.817 | 0.00067 | 0.00666 | 0.101 | 0.000416 |
| 4 | 0.821 | 0.00032 | 0.00512 | 0.063 | 0.000239 |
| 5 | 0.790 | 0.00046 | 0.00587 | 0.079 | 0.000312 |
| 6 | 0.833 | 0.00027 | 0.00394 | 0.067 | 0.000201 |
| 7 | 0.799 | 0.00041 | 0.00539 | 0.076 | 0.000301 |
| 8 | 0.802 | 0.00019 | 0.00500 | 0.038 | 0.000159 |
| 9 | 0.787 | 0.00039 | 0.00584 | 0.067 | 0.000293 |

Accounting for the *deployed* cap reverses the earlier raw-operator
comparison: in Tasks 2-9 the new direction energy is only 3.8%-14.6%
of the historical energy lost by accepting the trained new A. The
rank-10 tail of their sum is small, so joint A/B refactorization has
room to reduce alignment error without increasing persistent rank.
These are operator-space measurements, not a bound on accuracy or
proof that such refactorization improves it.

The existing `scripts/diagnose_main_branch_intrusion.py` was then run
on Live-A pre-merge Tasks 1 and 9, using fixed saved old prototypes and
new prototypes constructed from new-task *training* data. Test examples
were used only to evaluate the two counterfactual forwards (historical
branch alone vs historical plus current branch), never to tune a
candidate. Generated reports are in the corresponding snapshot folders
as `branch_intrusion.json` and `.pt`.

| C100 transition | Old Top-1 historical -> full | New Top-1 historical -> full | Old correct -> new-class error |
|---|---:|---:|---:|
| Task 1 | 96.50 -> 94.80 | 89.70 -> 97.30 | 2.30% of old test examples |
| Task 9 | 88.43 -> 86.66 | 85.00 -> 93.00 | 1.42% of old test examples |

The current `sBA` branch is useful for the new classes but also causes
old-class intrusion before task-boundary absorption. Thus a scalar
shrink of B is not a sufficient explanation or obvious remedy. The
next intervention should separate (i) boundary rank-r refactorization
of `M+D`, which addresses A-coordinate loss, from (ii) training-time
functional protection against the current branch, which addresses B-side
intrusion. Test the first intervention without changing the classifier,
prototype transport, NormCap, or task protocol; compare against the
same-commit Frozen-A control before attributing any improvement.

## Attempt 6: Fixed-rank joint A/B boundary consolidation (pre-registered)

Code commit `faae111` adds opt-in
`sa_live_a_boundary_merge="joint_svd"` to the existing Live-A aggregate
mode. Default `aligned` artifacts remain byte-schema compatible; a
`joint_svd` artifact carries its mode and cannot be silently loaded by
an `aligned` run. Task 0 is unchanged. For each Task `t>0` Q/V branch,
the historical effective operator is
`M=G_old A_old/(||A_old||_F+eps)` and the *deployed* current operator is
`D=c s_t B_t A_t`, where bounded NormCap chooses
`c=min(1,1/((||A_t||_F+eps)(||B_t||_F+eps)))`. The new boundary state
solves `min_{rank(X)<=r} ||X-(M+D)||_F^2` with a factorized 2r-by-2r
QR/SVD core. Its right singular basis becomes the next shared A; the
left factors become G. A is rescaled to retain `||A_t||_F`, while G is
rescaled so `G A/||A||` remains the rank-r approximation. No per-task
factor or calibration state is stored. Training, optimizer, NormCap,
prototype transport, classifier, and evaluation remain unchanged.

This intervention isolates boundary coordinate loss, **not** the
training-time old-class intrusion observed in Attempt 5. The exact
Frobenius optimum may still hurt accuracy, because network outputs and
class margins are not Frobenius objectives. In particular, a good Final
cannot be assumed from the small offline truncation tail.

The focused math/rebuild tests and the full local suite passed
(`496 passed`). Three T=10 single-seed configs were created:
`exps/joint_ab_boundary_c100_seed1993_t10.json`,
`exps/joint_ab_boundary_inr_seed1995_t10.json`, and
`exps/joint_ab_boundary_cub_seed1_t10.json`. Each was structurally
compared against its reference config after excluding only prefix,
filepath, `sa_train_a_all_tasks`, and the new mode. Effective batch is
128 (two GPUs, 64 per GPU). They write distinct logs/artifacts and
retain task snapshots.

Pre-registered evaluation: first require Task 0/1 DDP completion,
24 finite branch diagnostics, valid snapshots, and saved/rebuilt state
consistency. Then report all ten Top-1 points, Final, AAA, Forgetting,
and the mean/max joint truncation error for each dataset. Success under
the user's objective requires **strictly** `Final >88.37` on C100,
`>78.95` on INR, or `>84.60` on CUB, in at least two datasets, versus
the complete Frozen-A references. If only one or none passes, do not
relabel the method successful or tune the SVD using test labels; analyze
the remaining B-side functional intrusion as the next separate cause.

Status: implementation and configs verified; GPU runs pending.

### P0 launch failure and initial-factory fix (2026-09-26)

The first three user-systemd, `nohup`-wrapped launches used physical GPU
pairs 0/1 (C100), 4/5 (INR), and 6/7 (CUB), with local `device=[0,1]`,
64 examples per rank, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and commit
`50502e1`. All entered real Task 0 training. CUB completed its 20
epochs, but failed during the Task-0 deployment rebuild: `sa_state.pt`
had no `boundary_merge`, so loading a `joint_svd` model correctly
rejected the artifact as saved `aligned`. The cause was not the SVD:
initial Task-0 construction uses `utils/inc_net.py:get_backbone`, while
the new argument had only been threaded through
`models/sa_sdlora.py:update_network` (the later rebuild path). Thus
Task 0 trained and saved with the default `aligned` mode. The CUB
Task-0 snapshot was never finalized and is invalid. C100 was stopped
mid Task-0 around epoch 9 and INR around epoch 1; neither is a result.
Their three original logs and generated directories remain in place
for audit, and none will be resumed or used as a partial result.

Commit `6cfd20a` adds a failing-then-passing test for the initial
factory's mode propagation and forwards `sa_live_a_boundary_merge`
there. The focused factory/save/rebuild tests passed. Full local suite:
`496 passed, 1 skipped`; the skip/warnings accompanied failed NVML/CUDA
initialization, not an algorithm test failure.

Before rerun, the server's GPU driver became unhealthy. `nvidia-smi -L`
reports `Unable to determine the device handle for gpu 0000:21:00.0`
(GPU 4), GPU 5 shows 8480 MiB without a listed process, and a fresh
PyTorch process with `CUDA_VISIBLE_DEVICES=0,1,6,7` reports
`torch.cuda.is_available() == False` and zero devices. All three joint
experiment services are terminal, with no `torchrun`/`main.py` process
remaining. No GPU restart/reset was attempted on this shared server.
Rerun all three experiments from new output paths only after a fresh
PyTorch CUDA probe succeeds; do not infer any Final/AAA from these
interrupted artifacts.

### Isolated cuda6 continuation (2026-09-26; running)

The local CUDA probe remained false, so the committed `f005e47` branch
was transferred as a Git bundle and cloned into the new, isolated
`/home/zhaoyang/SD-Lora-CL-joint-ab-20260926` on cuda6. No existing
remote research tree was changed. Its `data` link points to the existing
cuda6 CIFAR-100, ImageNet-R and CUB-200 data. cuda6 has four idle
3090 GPUs and `/home/zhaoyang/miniconda3/envs/sdlora` reports CUDA
available with four devices. Remote focused tests passed (`26 passed`).

Three full-config runs use commit `f005e47`, not the invalid first
launch's `50502e1`:

| Dataset | Physical GPUs | user-systemd service | Current state |
|---|---|---|---|
| CUB seed1 | 0,1 | `codex-joint-ab-cub-20260926-r2.service` | Running |
| C100 seed1993 | 2,3 | `codex-joint-ab-c100-20260926-r2.service` | Running |
| INR seed1995 | 0,1 after CUB terminal | `codex-joint-ab-inr-queue-20260926.service` | Waiting; no INR process yet |

Each uses a separate `nohup`-wrapped two-rank torchrun inside its user
service, batch64 per rank, deterministic CuBLAS, and distinct artifacts.
The queue polls the **live** CUB service and waits another 15 seconds
after it terminates before using its pair of GPUs. Logs are under the
isolated cuda6 directory as `joint_ab_boundary_{cub,c100,inr}_...log`.

The first genuine CUB Task 0/1 smoke succeeded: task-0 state saves
`boundary_merge=joint_svd` with 24 A/G branches, both post-merge
snapshots pass `audit_sa_task_snapshots.py`, and Task 1 logs
`[JointAB] branches=24 mean_relative_truncation=5.877022e-03`
(`max=1.873270e-02`). Post-task Top-1 is `96.87, 93.44`, compared with
Frozen-A `96.87, 93.18` at the same two stages. This is only a partial
curve, not a Final result or the two-dataset goal. Continue auditing
all tasks and record Final/AAA/F only for complete, valid runs.

### Attempt 6 first full-run audit: one data copy is not comparable

All three cuda6 services terminated with status 0. Each produced Task
0-9 snapshots and all **30/30** snapshots passed checksum audit. Each
Task-9 state has `boundary_merge=joint_svd`, 24 A/G branches, exactly
368,640 LoRA scalars and the same shared A in `sa_state.pt` and the
merged artifact. The run manifests identify commit `f005e47` and the
respective committed configs. No task output was inferred from an
unfinished snapshot.

| Dataset | Joint-A/B Final | AAA | Forgetting | Frozen-A Final | Apparent Final delta | Validity |
|---|---:|---:|---:|---:|---:|---|
| C100 seed1993 | 88.24 | 92.390 | 5.767 | 88.17 | +0.07 | Comparable |
| CUB seed1 | 84.21 | 89.524 | 8.993 | 84.40 | -0.19 | Comparable |
| INR seed1995 | 79.01 | 82.319 | 3.894 | 78.75 | +0.26 | **Not comparable: different image split** |

C100 Top-1 curve: `96.90,95.80,95.03,94.02,92.74,92.10,91.24,89.24,88.59,88.24`.
CUB Top-1 curve: `96.87,93.44,91.89,90.19,89.21,88.61,87.66,87.15,86.01,84.21`.
INR on the original cuda6 copy: `87.52,85.08,84.38,82.83,82.03,82.37,81.02,79.79,79.16,79.01`.
Only the first two curves can be compared to the recorded local
Frozen-A references. Neither exceeds the strict `+0.20` Final gate.

The data audit found exact C100 `train`/`test` archive SHA-256 matches
and identical CUB file-path/size manifest hashes across the two
servers. However the existing cuda6 `data/imagenet-r/train` and
`test` use different image filenames and their manifest hashes differ
from the local Frozen-A data. The Task-0 accuracy difference (`87.52`
versus `87.68`) was an early warning; it cannot be explained as a
controlled method effect. The apparent INR `+0.26` is therefore **not
evidence for the user's objective**. The preceding provisional
comparison is superseded by this data audit.

For a valid matched pair, the local `data/imagenet-r/train` and `test`
were copied into the isolated cuda6 directory
`SD-Lora-CL-joint-ab-20260926/data_matched/imagenet-r` (30,000 images,
2.168 GB transferred). Both train/test path-and-size SHA-256 manifests
now match the local data exactly. A second independent clone
`SD-Lora-CL-joint-ab-matched-inr-20260926` points its `data` link to
this copy. It will run both the committed Joint-A/B config and the
committed Frozen-A INR config on separate GPU pairs and compare those
two complete runs. No existing remote data directory was changed.

The matched pair was launched from Git commit `f005e47` with two-rank
batch64 jobs, 20 epochs/task, T=10, deterministic CuBLAS:

| Method | GPUs | Service | Log |
|---|---|---|---|
| Joint-A/B | 0,1 | `codex-joint-ab-inr-matched-20260926.service` | `joint_ab_boundary_inr_seed1995_t10_matched_20260926.log` |
| Frozen-A | 2,3 | `codex-frozen-inr-matched-20260926.service` | `frozen_margin_inr_t10_seed1995_matched_20260926.log` |

The logs/artifacts live in the second isolated cuda6 clone. Their
Task-0 first four epoch loss/accuracy sequences match exactly. This
launch is **not** yet a result; both complete curves and snapshots are
required before any Final comparison.

### Post-hoc operator diagnosis of Attempt 6 (read-only)

Using only saved pre-merge A/B/scale and adjacent post-merge A/G states,
the same capped current contribution `D` was compared under two
task-boundary representations on each Task 1-9 branch:

- Existing aligned merge: `M P_{A_t} + D`.
- Joint-SVD merge: saved `G_t A_t^{joint}/||A_t^{joint}||`.

For each transition, squared errors were summed over all 24 Q/V
branches and divided by `sum ||M+D||_F^2`; the table reports the mean
of those nine transition ratios. Historical recoverability separately
uses `sum ||M(I-P)||_F^2 / sum ||M||_F^2` per transition.

| Dataset | Aligned target error | Joint target error | Trained-A old loss | Joint-A old loss | Final vs Frozen |
|---|---:|---:|---:|---:|---:|
| C100 | 0.007728 | 0.000471 | 0.009337 | 0.000294 | +0.07 |
| CUB | 0.000941 | 0.000094 | 0.001214 | 0.000073 | -0.19 |

For C100 Task 9 specifically, aligned target error was `0.008924`
versus joint `0.000368`; old loss was `0.009762` versus `0.000080`.
For CUB Task 9, aligned target error was `0.002315` versus joint
`0.000164`; old loss was `0.002690` versus `0.000069`.
The FP64 diagnostic used QR row-space projectors and directly reloaded
the committed boundary tensors; it did not update any model or choose
hyperparameters from test labels.

This confirms that the proposed SVD successfully improves its stated
*operator-space* objective while failing to deliver a corresponding
Final improvement on C100/CUB. It does not prove which downstream
mechanism is responsible: current-branch old-class intrusion, feature
nonlinearity, prototype transport and classifier margins remain
possible causes. The next algorithmic hypothesis must target one of
these functions directly instead of further reducing an already small
Frobenius residual. The matched-data INR pair remains running and is
not included in this conclusion.

### Read-only A/B output-space diagnosis (Attempt 6)

Using the saved Task 0-9 states and each subsequent `pre_merge.pt`, the
current task's `current_up` was projected onto the column space of the
previous aggregate `G`. For every transition, the reported ratio is
`sum ||Q_G^T B||_F^2 / sum ||B||_F^2` across all 24 Q/V branches;
the second number is the cosine between the historical effective
operator `G A_old / ||A_old||_F` and the unabsorbed current operator
`s B A_current`. These diagnostics do not use labels and do not alter
the completed runs.

| Dataset | Mean B-in-G energy | Mean old/current operator cosine | Task 1 B-in-G | Task 9 B-in-G |
|---|---:|---:|---:|---:|
| C100 seed1993 | 0.0672 | +0.1122 | 0.1171 | 0.0463 |
| CUB seed1 | 0.0796 | +0.1117 | 0.1542 | 0.0776 |

There is no evidence here that most of the current `B` simply overwrites
the existing output column space. Orthogonality to `G` is **not**
orthogonality to old-class classifier margins, however: a small or
orthogonal branch response can still change decisions after later ViT
blocks. Consequently, a plain `B -> (I-P_G)B` rule is not justified by
these data. Before changing both A and B, test their separate and joint
effects on old/new class margins using identical saved checkpoints and
evaluation tensors. Keep the matched-data INR control pending until
both services complete.

### A/B boundary causal diagnostic on complete CUB run

Added `scripts/diagnose_joint_ab_boundary.py` to compare four states on
the **same** evaluation images and fixed, actually deployed prototype
head: training-state pre-merge; pre-merge with the task's NormCap gain;
the original LS-aligned merge counterfactual; and saved joint-SVD merge.
The script audits both task snapshots, uses no training updates and
stores paired logits. The joint-SVD weighted old/new accuracy reproduces
the logged Task 2 and Task 6 Top-1 to the logging precision. This is
an attribution analysis with a fixed head, not a newly trained method.

| CUB task | State | Old Top-1 | New Top-1 | Old mean true margin | New mean true margin |
|---|---|---:|---:|---:|---:|
| 2 | pre-merge/capped | 91.105 | 93.412 | 0.179887 | 0.152193 |
| 2 | aligned | 91.105 | 93.412 | 0.179841 | 0.152088 |
| 2 | joint-SVD | 91.537 | 92.568 | 0.179135 | 0.149056 |
| 6 | pre-merge/capped | 87.626 | 87.478 | 0.117972 | 0.113103 |
| 6 | aligned | 87.655 | 87.478 | 0.117952 | 0.112875 |
| 6 | joint-SVD | 87.742 | 87.120 | 0.117413 | 0.109017 |

Both tasks have NormCap gain exactly one on all 24 branches, so the
differences above arise from boundary representation, not attenuation.
Aligned-to-joint prediction changes are small: at Task 2, five old
examples become correct while five new examples become wrong (7 and 5
predictions changed in total); at Task 6, old net +3 correct and new
net -2 (23 and 5 predictions changed). A direct FP64 decomposition of
the saved joint operator into projected old operator plus residual
current operator gives current-relative L2 residuals 2.09% (Task 2)
and 2.76% (Task 6), despite the much smaller full-operator relative
errors. These are not proof that every future-task loss is caused by
truncation, but show why whole-operator Frobenius error can hide
current-class decision changes.

On *current-task training data only*, with deterministic test
preprocessing, the deployed prototype head's aligned/joint new-class
Top-1 and mean true margin are Task 2: `94.482/0.162431` versus
`94.147/0.159222`, and Task 6: `93.823/0.129435` versus
`93.656/0.125155`. The pre-merge training FC head also sees lower
joint mean margin at both tasks (Task 2 `4.7467 -> 4.5860`, Task 6
`3.9277 -> 3.7961`), though its Task 6 Top-1 rises slightly. Thus
current-data margin is a plausible *candidate* for rejecting harmful
joint merges, but it has not yet been used for training or validated
at Final. The fixed deployed prototype head was calibrated using the
joint representation, so these comparisons isolate the backbone but
do not prove what an aligned-trained prototype head would do.

Code-only tests: four new focused tests and the full suite `501 passed`.
The next cheap falsification is the Task 9 Final boundary: compare
aligned and joint with the same head and current-train margin before
implementing an online gate. The matched-data INR paired T=10 runs
remain in progress; their Task 0-2 curves are partial, not Final.

### Task 9 falsifies a current-margin-only A/B gate

The Task 8/9 CUB snapshots were evaluated locally with the same saved
Task-9 prototype head and all 5,794 Task-9 test samples. The
`joint_svd` recomputation is 84.2078% and reproduces its logged Final
84.21%; the counterfactual `aligned` deployment is 83.5865%. Thus a
gate that chose the candidate with better current-task margin would
make the final boundary worse on this run.

| CUB Task 9 candidate | Current-train Top-1 | Current-train prototype margin | Current-train FC margin | Old test Top-1 (n=5197) | New test Top-1 (n=597) | Full test Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Aligned | 87.458 | 0.103751 | 2.329680 | 83.510 | 84.255 | 83.587 |
| Joint-SVD | 86.288 | 0.096851 | 2.186024 | 84.299 | 83.417 | 84.208 |

Both current-train margin measures correctly signal the new-class
test drop, but neither captures the larger old-class aggregate gain.
The task's old/new test split is approximately 90/10, so old and new
class effects cannot be balanced by a current-only margin heuristic.
This is a one-boundary causal comparison with a **fixed joint-trained
prototype head**, not a complete aligned-training counterfactual.
It does not justify selecting candidates using test labels, and it
does not make the joint method meet the Frozen-A goal (84.21 < 84.40).
Any deployable A/B boundary selector needs a validated old-class
function surrogate available without retaining old samples; current
task margin alone is ruled out by this Task-9 check.

### Fixed current-write attenuation at the same boundary

One read-only Task-9 CUB counterfactual scaled the **already
NormCap-absorbed** current contribution to `0.75 * sBA` before the
same rank-10 joint SVD. The earlier tasks and Task-9 classifier were
held fixed. This factor was fixed before evaluating any labels and is
only a mechanism probe, not a new trained method.

| Current-write factor | Old test Top-1 | New test Top-1 | Full test Top-1 | Current-train Top-1 |
|---:|---:|---:|---:|---:|
| 1.00 (saved joint) | 84.299 | 83.417 | 84.208 | 86.288 |
| 0.75 (counterfactual) | 84.376 | 82.747 | 84.208 | 85.953 |

The gain in old classes is offset by a larger loss in new classes;
uniformly shrinking the current write does not improve this complete
Task-9 Final. This does not exclude a direction-aware or training-time
constraint, but provides no reason to sweep scalar weights using test
labels. The matched-data INR pair was still live through Task 5 when
this was recorded; no Final comparison is inferred from its partial
curve.

### Rejected P0: a joint 5% operator-space risk cap

Before adding code, the complete joint A/B pre-merge snapshots were
used to estimate the scalar current-write attenuation required if
Live-A's final row space were kept and the **total historical operator
drift** were bounded at 5% across all 24 branches. For each transition,
`L = sum ||M_old(I-P_Anew)||_F^2` is the irrecoverable history loss,
`U = sum ||D_current||_F^2` is the actual NormCap-absorbed current
operator energy, and `E = sum ||M_old||_F^2`. Since the two terms are
orthogonal in input row space, the largest feasible scalar is
`alpha = min(1, sqrt(max(0, 0.05 E - L) / U))`. These numbers were
computed from immutable snapshots in FP64; no training or selection
used test labels. The old cuda6 INR split is marked unmatched and used
only for this operator diagnostic, never a Frozen-A accuracy comparison.

| Dataset | Mean L/E | Mean U/E | Alpha by Task 1-9 |
|---|---:|---:|---|
| C100 | 0.0093 | 0.1202 | 0.277, 0.569, 0.603, 0.809, 0.731, 0.807, 0.799, 1.000, 0.816 |
| CUB | 0.0012 | 0.2439 | 0.229, 0.358, 0.493, 0.576, 0.583, 0.652, 0.726, 0.754, 0.703 |
| INR (unmatched split) | 0.0104 | 0.1705 | 0.253, 0.401, 0.479, 0.733, 0.574, 0.652, 0.786, 0.651, 0.833 |

The cap would often remove more than half of the current effective
operator, including on the dataset where live A may be useful. At the
fully measured CUB Task 9, even the milder fixed `alpha=0.75` did not
improve Final. Thus a plain 5% Frobenius cap on A/B jointly is rejected
as a main experiment. Operator energy is not a proven classifier-risk
surrogate; any next budget should first validate a *function-space*
old-class signal rather than simply retune the scalar threshold.

## Attempt 7 P0: function-sensitive Live-A/G consolidation kernel

The existing C100 benefit from SBGC and the historical INR benefit
from Live-A motivate a **single** prospective method that updates both
the shared input basis A and the cumulative B/G coefficient, rather
than selecting unrelated methods per dataset. This is a new hypothesis,
not a claim that either endpoint will transfer or that the desired
two-dataset Final criterion has been met.

For old effective operator `M = G_old A_old / ||A_old||`, trained
`A_new`, and an already absorbed current `current_up`, let
`a=A_new/||A_new||`. Historical input moments use a fixed-size diagonal
approximation `V_h=diag(v_h)` and historical output sensitivity uses
`F_h=diag(f_h)`. The historical risk of a candidate G is

`R(G)=tr[F_h (G a - M) V_h (G a - M)^T]`.

Its weighted LS center is

`G0 = M V_h a^T (a V_h a^T)^(-1)`.

The irrecoverable residual `R_min = R(G0)` is paid **before** G can
write anything. If `R_min/E_h > epsilon`, with
`E_h=tr(F_h M V_h M^T)`, the trained row space is infeasible and the
kernel explicitly rejects it. Otherwise a current-response weighted
convex solve finds G nearest `G0 + current_up` subject to
`R(G) <= epsilon E_h`. The existing row-wise SBGC solver is reused on
the remaining budget; the FP32 deployed operator is rechecked in
FP64. This is a per-transition diagonal-response surrogate, not a
full-network or forgetting bound. No old images or feature samples
are stored; a later training integration would maintain only the
streaming per-block input second moment and per-branch output
sensitivity, approximately `12*768 + 24*768 = 27,648` FP32 values
besides the fixed 368,640-scalar A/G state (about +7.5%).

Code added in `backbone/live_a_functional.py`; six focused tests cover
weighted risk, irrecoverable loss, feasible additive behavior,
infeasibility, A-scale gauge invariance, and zero/rank-deficient A.
The full repository suite is `507 passed`. This is **P0 only**: the
historical activation/sensitivity calibration, DDP synchronization,
checkpoint version, real-data smoke, and T=10 comparisons are not yet
implemented. The next gate is to validate a task-boundary calibration
that preserves RNG/model tensors and to measure actual current-target
distortion under this function-space risk before running full jobs.

### P0 global-budget correction and real-snapshot feasibility

Per-branch feasibility was checked from all nine transitions of the
complete C100, CUB and unmatched-split INR joint runs. With unit input
moments/sensitivities, just **1/216** C100 branches (Task 1, maximum
irrecoverable ratio 0.0575) and **1/216** INR branches (Task 2, maximum
0.0508) exceed a local 5% budget; CUB has **0/216** (maximum 0.0104).
All three have substantially lower *global* irrecoverable ratios.
Consequently the P0 kernel now has a global variant sharing a single
5% risk allowance across branches, including branches that would be
locally infeasible. A focused test covers exactly this case and another
covers globally infeasible row-space rotation. The full suite is now
`509 passed`.

As a real-shape numerical check, the saved CUB Task-8/9 factors were
passed through the global solver using **unit** input moments and
sensitivities. Across 24 branches it returned irrecoverable risk
`0.00268969`, deployed total risk `0.0499999998`, current-response
distortion `0.08803`, and dual multiplier `0.42187`; the constraint
was active. These unit-weight values are only a solver smoke test.
The actual activation and output-sensitivity statistics are not yet
calibrated, so neither function sensitivity nor CIL accuracy has been
validated. The unmatched INR split remains excluded from any accuracy
claim. No new T=10 method experiment has been launched from this P0.

### Attempt 6 matched-INR completion audit (2026-09-28)

Both matched-data cuda6 services are terminal (`MainPID=0`, inactive,
`ExecMainStatus=0`). Both runs have all ten Task 0-9 snapshots;
**20/20** checksum audits pass. Their run manifests use the same
code commit `f005e4776757dfbcde603650e8aad1d1c1fa03a7`, the registered
seed1995/T10/rank10/20-epoch/two-GPU batch64 protocol and the isolated
ImageNet-R copy whose manifests match the local reference data.
Each final state has 24 Q/V A/G pairs, exactly 368,640 LoRA scalars,
and identical A tensors in state and merged artifact. The Frozen-A
control exactly reproduces the earlier local Top-1 curve.

| Method | INR Final | AAA | Forgetting | Final delta vs complete Frozen |
|---|---:|---:|---:|---:|
| Frozen-A aligned | 78.75 | 81.941 | 5.946 | -- |
| Joint-A/B SVD | 79.13 | 81.949 | 5.644 | +0.38 |

Joint curve: `87.68,84.92,83.98,82.07,81.83,81.10,79.96,79.76,79.06,79.13`.
Frozen curve: `87.68,85.30,84.23,82.26,81.93,81.13,79.72,79.61,78.80,78.75`.
The remote logs are copied locally as
`joint_ab_boundary_inr_seed1995_t10_matched_20260926_remote.log` and
`frozen_margin_inr_t10_seed1995_matched_20260926_remote.log`.
The original unmatched cuda6 INR result 79.01 remains excluded.

Attempt 6 now has one **valid** strict >+0.20 win (INR +0.38), while
C100 is +0.07 and CUB is -0.19. The user goal is still **not achieved**.
The P0 function-sensitive Live-A/G code is not yet a deployed method
and cannot be counted as a second win. Subsequent experiments must
retain these complete Frozen-A controls, not substitute stripped or
dataset-specific baselines.

### Attempt 7 P1: offline functional calibration and CUB boundary check

Added an opt-in, offline QKV statistics collector and a sequential
calibration command. Each saved task is calibrated on **its own training
classes only**, with deterministic test preprocessing and the saved
deployed backbone/prototype cosine head. Frozen parameters receive no
`.grad`; Q/V output derivatives are collected with `autograd.grad` and
the batch CE mean is undone before squaring. The collector preserves
the graph through earlier QKV outputs and removes all temporary hooks.
Calibration wraps model creation and loading in RNG preservation and
checks the model tensor hash. It is not wired into training or DDP.

CUB Task 0-8 sequential calibration used batch16 on local GPU4. Sample
counts were `600,600,598,600,599,600,599,600,600`; all nine model/RNG
hash checks passed. All statistics are finite. The final mean sensitivity
CV is 1.357554. Running state has 12 input vectors and 24 sensitivity
vectors, **27,648 FP32 values**, plus 36 counts. Task metadata and paired
diagnostic logits are research artifacts, not deployment state.

For a cheap falsification check, only the Task-9 boundary was replaced.
Historical statistics stop at Task 8. Current statistics use Task 9's
own 598 training samples; they are not added to the history before
solving. Importantly, the current calibration uses the **already saved
joint-SVD backbone and prototype head**. This is diagnostic lookahead,
not an online-valid pre-merge calibration or a newly trained method.
The candidates retain the trained pre-merge A and solve G jointly across
24 branches with the preregistered 5% functional surrogate budget.
Uniform uses the same input moments but unit output sensitivities.

| CUB Task-9 candidate | Old test Top-1 | New test Top-1 | Full test Top-1 | Total historical surrogate risk | Irrecoverable risk | Current distortion | Dual eta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Saved joint-SVD | 84.299 | 83.417 | 84.208 | -- | -- | -- | -- |
| Functional uniform | 84.145 | 83.752 | 84.104 | 0.050000000 | 0.008865687 | 0.140206594 | 0.599110010 |
| Functional Fisher | 84.068 | 83.752 | 84.035 | 0.050000000 | 0.008806661 | 0.092014209 | 0.432624652 |

All candidates use the exact same saved prototype head and 5,794 test
samples (5,197 old; 597 new). Fisher improves new-class accuracy versus
joint but loses more old-class accuracy, giving a net -0.173 points;
uniform is -0.104. Fisher's smaller reported target distortion is
**not a common-metric improvement over uniform**, because the two
objective weightings differ. Meeting the branch surrogate risk budget
does not demonstrate improved class margins or reduced forgetting.

The six raw statistics/candidate/paired-logit files were retained under
`LIVE_A_FUNCTIONAL_DIAGNOSTIC_CUB_T9_20260928/` rather than left only in
`/tmp`. Focused collector/calibration tests and the full CPU suite pass:
**514 passed, 1 skipped**. No new formal T=10 run has been launched.
Offline calibration implementation/tests are committed as `50346e9`.
A separate read-only review was requested but returned no findings
before it was stopped; it is not counted as completed verification.
This single-boundary negative result neither proves that all A/B
function-sensitive training must fail nor supports integrating this
5% diagonal-Fisher boundary solver as the next expensive main experiment.
Before any such integration, calibration must be made pre-merge and
the historical-function proxy validated against paired old-class
prediction changes. The two-dataset Final improvement goal remains
unmet; Attempt 6 still contributes only the matched-INR +0.38 result.

## Attempt 8: Task-anchored history with joint A/B consolidation

Pre-registration (2026-09-28), implementation commit `60cad65`.
Use `sa_live_a_history_forward="anchored"` together with the existing
`sa_live_a_boundary_merge="joint_svd"`. Default `shared` retains the
existing Live-A behavior and artifact format. No baseline or old
experiment configuration has been modified.

At each task start, existing nonpersistent anchor buffers capture
`A_old` and `G_old/(||A_old||+eps)`. Historical branch forward is now
`M_old x`, where `M_old=G_old A_old/(||A_old||+eps)` stays fixed within
the task. Current A and B are both trainable, initialized exactly as
before, and the current branch remains `s B A x`. Historical output
must remain differentiable with respect to x so earlier current LoRA
branches can learn; only the historical factors are frozen.

This isolates an actual coupling in Attempt 6: shared historical
forward used `G_old A_current/||A_current||`, while joint consolidation
targeted the task-start historical operator plus the current update.
The new trial removes that **within-task history-factor mismatch**.
It does not freeze the full old-model function: earlier current LoRA
branches can still change inputs to later historical branches.

Boundary processing remains unchanged. Existing NormCap produces the
current committed operator `D_cap`; fixed-rank joint SVD consolidates
`M_old + D_cap` into one A/G pair per branch. Prototype transport and
the prototype classifier remain enabled; Dual-B and HBD stay disabled.
The new mode requires joint SVD and rejects mismatched history modes
when loading a saved state. Its artifact adds only a string marker,
not any persistent tensor: 24 pairs still total **368,640 scalars**.
Old-task anchor tensors overwrite existing buffers, never form a bank.

Hypothesis: current A/B will learn the new task without exploiting
artificial drift of the historical factorization, improving the
quality of the subsequent joint merge. This is not a guarantee.
In particular, B=0 again implies zero current A gradient at the first
batch; shared history previously provided an extra A gradient, so the
trial may also lose useful plasticity. Fixed-rank truncation and
NormCap still cause train/deploy changes and can still harm accuracy.

Validation sequence:

1. TDD tests first failed for the missing history-forward API (three
   failures, one Task-0 equivalence control already passed). New tests
   then verified immutable historical output, no historical A gradient,
   retained input/current-factor gradients, Task-0 shared equivalence,
   same-size save/rebuild, mode guards and initial factory propagation.
   Full repository CPU suite: **519 passed, 1 skipped**.
2. Three real-data Task0/1 smoke runs: two epochs, rank10, two GPUs with
   batch64 each. GPU pairs C100 `0,1`, INR `4,5`, CUB `6,7`; exclude 2/3.
   Audit both task snapshots, 24 branch pairs, history marker, finite
   factors, and exactly 368,640 persisted LoRA scalars before full runs.
3. Only after smoke passes, run fresh T10/20-epoch configurations with
   seeds C1001993/INR1995/CUB1, effective batch128 and unchanged SGD,
   learning rates, class order, NormCap and prototype transport. Use
   `nohup` under user-systemd; keep output and snapshots separate from
   Attempt 6. No test-label selection or dataset-specific gate/budget.
4. Compare complete Final against **88.17/78.75/84.40**, and against
   Attempt 6 **88.24/79.13/84.21**. The active goal still requires the
   same complete method to exceed Frozen-A by strictly >0.20 on at
   least two datasets; smoke or one-boundary evidence is not success.

Configs are `exps/anchored_joint_ab_{c100,inr,cub}_seed*_t10_20260928.json`
and corresponding `t2_e2_smoke` configs. All six preserve the reference
per-task class counts; smoke changes only epochs/max_tasks and output.

### Attempt 8 smoke launch and deterministic-runtime correction

The initial three user-systemd/nohup smoke services exited with status 1
at the first forward because their environment omitted
`CUBLAS_WORKSPACE_CONFIG`. Both ranks report the explicit PyTorch
deterministic-CuBLAS error, not an algorithm/optimizer or communication
failure. All three terminal services have MainPID=0. Their logs and
incomplete output directories were archived under
`ANCHORED_JOINT_AB_INVALID_ENV_20260928/`; no partial accuracy is used.

A minimal GPU forward/backward with deterministic algorithms and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` returns finite activations and input
gradients. The existing task dispatcher uses the same setting. The
three smoke jobs were restarted with fresh output directories and new
service IDs `codex-anchored-joint-{c100,inr,cub}-smoke-r1-20260928`.
Launch command: user-systemd with append log output, `/usr/bin/nohup
/usr/bin/env`, explicit `CUDA_VISIBLE_DEVICES`, `HF_ENDPOINT`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=4`, then the sdlora
Python `-u -m torch.distributed.run --standalone --nproc_per_node=2
main.py --config=./exps/anchored_joint_ab_*_t2_e2_smoke_20260928.json`.
Training code/config commit is `38604c8`. Formal T10 runs are still
gated on completed real-data smoke audits, not launcher success.

### Attempt 8 real-data diagnostic compatibility fix

The corrected CUB smoke completed Task0 training, absorption, deployment
rebuild, rank-consistent prototypes and snapshot checks, then failed at
the first Task1 batch. The existing `live_a_gradient_diagnostics` called
`autograd.grad(hist_loss, current_A)` unconditionally. Anchored history
has no direct dependence on current A, so this loss has no grad_fn when
its input is frozen, or unused A inputs when x requires grad. This is a
logging-path incompatibility, not a non-finite training gradient.

Two regression cases first reproduced both errors, then the diagnostic
was changed to handle absent/unused historical gradients as zero while
leaving current gradients and previously accumulated `.grad` unchanged.
The full suite is now **521 passed, 1 skipped**. The same-mode shared
diagnostic tests still pass. No optimizer, classifier, absorption,
transport or training-forward formula was changed by this fix.

The C100/INR r1 smoke services were explicitly stopped rather than left
running code with the same known diagnostic incompatibility. All three
MainPIDs are now zero. Their logs/output directories were archived to
`ANCHORED_JOINT_AB_INVALID_DIAGNOSTIC_20260928/`. Restart smoke from fresh
Task0; do not combine the partial r1 results with later runs or count
their Task0 accuracy as a completed experiment.

### Attempt 8 validated smoke and independent-basis snapshots

Audit time: 2026-09-28 10:20 CST. All three r2 user-systemd/nohup
services are terminal with `MainPID=0`, `ExecMainStatus=0`. Each has
exactly two completed real-data tasks, two epochs per task, two ranks
and batch64 per rank. These runs use training commit `ce9d5fb` and are
path checks, not T10 performance evidence.

| Smoke | Task0 Top-1 | Task1 Top-1 | Two-task AAA | Forgetting | Task1 mean/max relative SVD truncation |
|---|---:|---:|---:|---:|---:|
| C100 seed1993 | 96.90 | 95.90 | 96.400 | 0.900 | 0.003778 / 0.017029 |
| INR seed1995 | 83.95 | 82.10 | 83.025 | 1.130 | 0.002390 / 0.014059 |
| CUB seed1 | 96.87 | 92.57 | 94.720 | 6.090 | 0.000092 / 0.000341 |

All six Task0/1 checksum audits pass, with eight hashed files each.
Every state/merged artifact records `history_forward="anchored"` and
`boundary_merge="joint_svd"`; factors are finite, all 24 Q/V A/G pairs
are present, and A tensors match between state and deployment artifact.
Persisted LoRA factors remain exactly **368,640 scalars** at both tasks.
Prototype rank synchronization, evaluation RNG hashes and model tensor
hashes pass for all six evaluations. No communication hang or traceback
occurs in the validated r2 logs.

The existing v1 research pre-merge snapshot assumed a shared historical
and current down basis. That assumption is false for anchored history.
Commit `26b1bd5` introduces research snapshot v3 storing the immutable
historical down factor and its already-normalized up factor separately
from current A/B. Replay uses these explicit factors, without a second
normalization, and preserves current A independently. Tests first
reproduced incorrect v1 capture/replay, then verify exact branch replay.
The full suite passes **524 tests** in this GPU-visible environment.
This is a research capture/replay change only: training, persistent
deployment state, NormCap, joint SVD, transport and classifier are
unchanged. Default shared snapshots stay v1; SBGC capture stays v2.
This particular shared/anchored intrusion replay command now explicitly
rejects unsupported v2 instead of silently applying shared normalization.

Completed r2 smoke snapshots still contain the older v1 pre-merge
format, because their processes started before this capture change.
Do not rewrite their files or completion hashes. The diagnostic CLI
rejects legacy anchored v1 rather than silently producing wrong outputs;
manual reconstruction would require the preceding saved task state.
Fresh formal runs will capture explicit v3 factors. Rerunning real-data
smoke solely for this analysis-format change is not required: the
training code is identical and the new capture/replay tests pass.

Attempt 8 is ready for its registered three full T10 runs, not declared
successful. Complete Frozen-A Final references remain C100 88.17,
INR 78.75, CUB 84.40. Strict >+0.20 on two datasets is still unverified.

### Attempt 8 independent review and unsupported-HBD guard

A bounded read-only `gpt-5.6-terra` review completed before formal
launch. It found one unsupported combination: anchored history with
`sa_hbd_enabled=true` makes the old HBD diagnostic assume nonzero A
gradient graphs that anchored history no longer supplies. The reviewer
found no other executable-code defect in the reviewed forward,
input-gradient, save/load marker, factory or fixed-size state paths.
This is a scoped review, not proof of accuracy or full-network stability.

The Learner now rejects this combination before constructing the
network. A regression test first failed because invalid flags reached
parent/model construction, then passes with the early guard. Existing
shared-mode HBD remains supported and unchanged. Formal Attempt 8
already disables HBD, so the guard does not change its numerical path.
Fresh full CPU suite: **524 passed, 1 skipped**; `git diff --check`
passes. No smoke replay is needed for a guard that is inactive in all
six preregistered configurations.
