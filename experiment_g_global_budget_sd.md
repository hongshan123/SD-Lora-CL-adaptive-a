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
Task-0/1 state roundtrip, and strict scope mismatch. The pre-config full suite
passed 483 tests; rerun after the final patch remains pending.

Real-data DDP smoke and T=10 Final/AAA/Forgetting are pending. No accuracy
benefit is claimed yet.
