# Global-Budget G Consolidation Experiment Ledger

## Objective and frozen protocol

Seek a rehearsal-free, nearly task-constant-state continual LoRA variant that
beats complete Frozen-A Final by **more than 0.20 percentage points** on at
least two of CIFAR-100 seed 1993, ImageNet-R seed 1995, and CUB-200 seed 1.
Use T=10, rank 10, 20 epochs/task, two GPUs with batch 64 each, the same
SGD/constant schedule, frozen Task-0 input basis, prototype transport, and
no Dual-B/HBD. Do not select a method using test labels or change a dataset's
budget separately. Record every modification and its results below.

Complete Frozen-A references (Final/AAA/F): C100 88.17/92.385/5.811,
INR 78.75/81.941/5.946, CUB 84.40/89.596/8.286.

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
