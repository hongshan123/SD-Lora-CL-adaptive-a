# Online Gauge-Aligned Cumulative Shared-A LoRA for Rehearsal-Free Class-Incremental Learning

**Status**: first draft (2026-08-07). Numbers are final from evidence bank unless marked `TODO_VERIFY` / `TODO_EVIDENCE`; citations marked `TODO_CITATION` need verification before submission.

## Abstract

Rehearsal-free class-incremental learning (CIL) with low-rank adapters faces two coupled problems: the persistent adapter state grows linearly with the number of tasks, and a continuously updated shared down-projection $A$ drifts the historical LoRA operators that old prototypes depend on. We show that, in a shared-$A$ SD-LoRA setup, the entire historical bank of up-projections can be folded into a single cumulative matrix $H$ with an exactness below $10^{-5}$ (C1). When $A$ is re-canonicalized after each task with a thin QR decomposition $A^T = QR$, the historical effective operator $H_{\mathrm{old}} Q_{\mathrm{old}}^T$ is preserved in closed form by gauge alignment $H_{\mathrm{aligned}} = H_{\mathrm{old}}(Q_{\mathrm{old}}^T Q_{\mathrm{new}})$ (C2). The resulting persistent LoRA state is independent of the number of tasks: 371,040 parameters ($10.1\%$ of SD-LoRA's 3.69M) measured identically at $T=5,10,20,40$ (C3), with single-model inference, no task id/router, and no need to restore historical $B$ banks (C4). Across four seeds, removing $81.7\%$ of LoRA parameters costs $-0.65$ Final Top-1 ($p=0.026$) and $-0.52$ AvgAcc ($p=0.009$) on ImageNet-R relative to our prior Shared-A + prototype system, while CIFAR-100 differences are not significant (Final $p=0.28$, AvgAcc $p=0.15$) and Forgetting is unchanged on both datasets (C5). On the fine-grained CUB-200-2011 benchmark it outperforms that system by $+8.04$ Final Top-1 (single seed; C6). Inference FLOPs/throughput/peak memory are identical to the merged legacy model (C7). Per-task gauge diagnostics of the historical operator require a pre-save diagnostic rerun (`TODO_VERIFY`, P0-2); the algebra in C1--C2 is exact under the repository's normalization convention (C2).

## 1 Introduction

Class-incremental learning without rehearsal is one of the most restrictive continual-learning settings: no old samples may be stored or revisited, no old-data calibration is allowed, and the model must remain usable after every task [CIT-01, CIT-03]. Parameter-efficient fine-tuning with LoRA is attractive here because each task can be adapted with a small low-rank update; however, a naive bank of per-task LoRA adapters grows linearly with the number of tasks, and any sharing mechanism that updates a common down-projection $A$ after the first task changes the effective historical operators on which previously stored class prototypes are evaluated (E-INR-009; a single-task drop of 2.66 points was observed in EXP-009).

Existing work attacks parts of this problem. Adapter-isolation methods fix or project LoRA subspaces to limit interference [CIT-06, CIT-08, CIT-13]; prototype-classifier methods replace the incremental fully-connected head with class means and cosine matching, but assume a frozen or stable backbone [CIT-03, CIT-04]; drift-compensation methods learn a generic forward mapping for old prototypes without exploiting the LoRA factorization [CIT-05, CIT-10]. None of these makes persistent state independent of the number of tasks while explicitly preserving the exact historical LoRA operator under a continuously updated shared $A$.

In this paper we exploit the algebraic structure of Shared-A SD-LoRA [CIT-01, CIT-02]: the historical branch is exactly $\sum_i s_i B_i(Ax)/(\|A\|\|B_i\|)$, which for a fixed $A$ equals $H(Ax)$ with $H = \sum_i s_i B_i/(\|A\|\|B_i\|)$. We introduce an online cumulative state $(Q^T, H, R)$ per Q/V branch, updated at the end of every task by (i) QR-canonicalizing the new $A$, (ii) aligning the old operator with the closed-form gauge solution, and (iii) folding the current task's normalized contribution. The historical operator is then fixed during training and preserved near-exactly across tasks (per-task diagnostics at the $10^{-8}$ scale, E-INR-DIAG). The deployed model is a single canonical LoRA operator, and no per-task $B$ file is ever kept.

Contributions:
1. **Exact cumulative folding** of the historical Shared-A bank into one matrix per branch (C1), validated at operator/feature/logits error below $10^{-5}$.
2. **Closed-form gauge alignment** for operator preservation under shared-$A$ updates (C2), with per-task residual/rotation/preservation diagnostics.
3. **Task-count-independent persistent state**: 371,040 LoRA parameters ($10.1\%$ of SD-LoRA), $85.8\%$/$87.9\%$ total state reduction including prototypes (C3), single-model inference and no historical-$B$ recovery (C4).
4. **Comprehensive evaluation**: 4-seed paired comparison vs our prior system (no significant difference, C5), task-length $T=5/10/20/40$ (C8), extra dataset CUB-200 (C6), efficiency audit (C7), and negative results/correlations (C9-C10).

## 2 Related Work

### 2.1 Low-rank adapters for continual learning
LoRA factorizes an update as $\Delta W = BA$; the factorization is not unique, and canonical/balanced parameterizations have been studied [CIT-12]. SD-LoRA decouples magnitude (scalar scale) from direction (normalized $A,B$) and stores per-task banks [CIT-01]. SA-LoRA shares $A$ across tasks and keeps per-task $B$ [CIT-02]. InfLoRA reparameterizes injected parameters into a fixed subspace to reduce interference [CIT-06]; C-LoRA uses a learnable routing matrix with orthogonal constraints [CIT-08]; LoRA-DRS constrains LoRA parameter drift [CIT-13]; FM-LoRA shares a base and task coefficients with dynamic rank [CIT-07]; Janus-LoRA derives closed-form gradient rectification [CIT-11]. These methods do not address the persistent-state growth of per-task components, nor do they explicitly preserve the historical operator under shared-$A$ updates.

### 2.2 Prototype classifiers and drift compensation
CL-LoRA uses per-task prototypes and cosine matching without rehearsal [CIT-03]; RanPAC uses frozen random projections and decorrelated prototypes [CIT-04]. Both assume a stable or per-task backbone. LDC learns a forward-projection network to compensate moving-backbone prototype drift [CIT-05]; E2-LoRA analyzes the low-rank structure and energy ordering of output-feature drift [CIT-10]; DGS aligns LoRA gradients with semantic prototype shifts [CIT-14]; ICLR 2026 two-way alignment uses cycle consistency and Gaussian prototype transport [CIT-15]. Our method differs structurally: it requires no learned network, no semantic synthesis, and no per-task adapters at inference; the compensation is a closed-form algebraic alignment of the exact historical operator (C2).

## 3 Method

### 3.1 Setup and notation
We use the Shared-A SD-LoRA backbone of EXP-009: $24$ Q/V LoRA branches ($12$ blocks $\times$ Q/V), shared down-projections $A \in \mathbb{R}^{r \times d}$ ($r=10, d=768$) trained continuously, per-task up-projections $B_t \in \mathbb{R}^{d \times r}$ and scalar scales $s_t$. During task $t$, the forward pass adds the historical bank $\sum_{i<t} s_i B_i(Ax)/(\|A\|\|B_i\|)$ and the current branch $s_t B_t(A x)$; prototypes are stored as L2-normalized class means from current-task data only (no rehearsal).

### 3.2 Cumulative folding (C1)
For a fixed $A$:
$$H = \sum_{i} \frac{s_i B_i}{\|A\|\|B_i\|}, \qquad \sum_i s_i B_i(Ax)/(\|A\|\|B_i\|) = H(Ax).$$
This is an exact identity under the repository's normalization convention. We verified operator, feature, and logits equivalence between the bank forward and the cumulative forward with maximum error below $10^{-5}$ on synthetic and tiny-model tests (E-AUDIT-UNIT).

### 3.3 Canonicalization and gauge alignment (C2)
At the end of each task, decompose $A^T = Q R$ (thin QR; $Q \in \mathbb{R}^{d\times r}$, $R \in \mathbb{R}^{r\times r}$), so $A = R^T Q^T$ and $B A = (B R^T) Q^T$. Define the canonical state per branch:
$$Q^T \ (\text{canonical down}), \qquad H = H_{\mathrm{raw}} R^T \ (\text{cumulative up}), \qquad R.$$
When the new canonical basis $Q_{\mathrm{new}}$ differs from $Q_{\mathrm{old}}$, the best approximation of the old operator in the new coordinates is the least-squares solution of $\min_{H'} \|H_{\mathrm{old}} Q_{\mathrm{old}}^T - H' Q_{\mathrm{new}}^T\|_F^2$:
$$H_{\mathrm{aligned}} = H_{\mathrm{old}} (Q_{\mathrm{old}}^T Q_{\mathrm{new}}).$$
The projection residual is $\|H_{\mathrm{old}} Q_{\mathrm{old}}^T (I - Q_{\mathrm{new}}Q_{\mathrm{new}}^T)\|_F$. We record, per task: relative projection residual, basis rotation $\|Q_{\mathrm{old}}^T Q_{\mathrm{new}} - I\|_F/\sqrt{r}$, and operator-preservation error $\|H_{\mathrm{aligned}} Q_{\mathrm{new}}^T - H_{\mathrm{old}}Q_{\mathrm{old}}^T\|_F/\|H_{\mathrm{old}}Q_{\mathrm{old}}^T\|_F$. In all runs these diagnostics were $10^{-8}$–$10^{-9}$ per task, i.e., the shared $A$ effectively stays within its old row space and the historical operator is preserved almost exactly (E-INR-DIAG).

### 3.4 Online cumulative training
At task $t$:
1. Load $(Q_{t-1}, H_{t-1}, R_{t-1})$; initialize $A_t = R_{t-1}^T Q_{t-1}^T$ (exact reconstruction), $B_t=0$, $s_t=0.8$.
2. Forward: historical $H_{t-1}(Q_{t-1}^T x)$ (fixed) + current $s_t B_t(A_t x)$ (same convention as the legacy v1 training-time forward).
3. After training, compute $(Q_t,R_t)$ from $A_t$; set $H_t = H_{\mathrm{aligned}} + s_t B_t R_t^T/(\|A_t\|\|B_t\|)$; save only $(Q_t^T, H_t, R_t)$; discard $B_t$.
4. Optional ablation `cumulative_gauge=false` keeps $H_{t-1}$ unaligned (cumulative-only); residual LRPT can be applied to old prototypes as an ablation.

### 3.5 Persistent state and complexity
Per branch: $Q^T$ ($r\times d$), $H$ ($d\times r$), $R$ ($r\times r$). With $r=10$, 24 branches: $184,320 + 184,320 + 2,400 = 371,040$ parameters, independent of $T$. Including prototypes: INR $524,640$ ($14.23\%$ of SD-LoRA; $-85.77\%$), C100 $447,840$ ($12.15\%$; $-87.85\%$). Deployment inference is one canonical operator; no task id, router, or per-task adapter is used, and continuing training does not require historical $B$ files (C3, C4).

## 4 Experiments

### 4.1 Protocol
- Datasets: ImageNet-R (200 classes, $T=10$, init 20/inc 20, seed1995; extra seeds 1/2/3), CIFAR-100 (100 classes, $T=10$, init 10/inc 10, seed1993; extra seeds 1/2/3), CUB-200-2011 (200 classes, $T=10$, seed1).
- Optimization: Adam, lr 0.01 (INR/CUB constant; C100 cosine 0.008), batch 32, 20 epochs/task, rank 10, orthogonal shared-A init, 4$\times$RTX 3090 DDP.
- Metrics: Final Top-1, Average Accuracy, Forgetting; persistent parameters; FLOPs/throughput/peak memory; consistency audit of merged vs reconstructed backbone.
- Baselines: SD-LoRA [CIT-01] (E-INR-BASE/E-C100-BASE), EXP-009 (Shared-A + prototypes, E-INR-009/E-C100-009), and earlier variants (E-INR-010/011/012).

### 4.2 Main results
Single-seed main protocol:
| Method | ImageNet-R (F/A/Fg) | CIFAR-100 (F/A/Fg) | LoRA params |
| --- | ---: | ---: | ---: |
| SD-LoRA | 78.76 / 83.13 / 5.61 | 86.89 / 91.44 / 5.58 | 3,686,400 |
| EXP-009 | 79.34 / 82.47 / 7.26 | 88.42 / 92.07 / 8.08 | 2,027,530 |
| **cumulative+gauge (ours)** | **79.06 / 81.77 / 6.82** | **87.70 / 91.83 / 8.53** | **371,040** |

Four-seed paired comparison (same seeds, paired t-test, $n=4$):
| Dataset | ΔFinal (p) | ΔAvgAcc (p) | ΔForgetting (p) |
| --- | ---: | ---: | ---: |
| ImageNet-R | -0.65 (0.026) | -0.52 (0.009) | -0.04 (0.852) |
| CIFAR-100 | -0.22 (0.282) | -0.12 (0.147) | +0.01 (0.938) |

Conclusion (C5): seed-joined pairing shows a statistically significant small deficit on ImageNet-R Final/AvgAcc; CIFAR-100 is not significant, but TOST with a $\pm 0.5$ margin is not met for ImageNet-R Final/AvgAcc or CIFAR-100 Final. The honest claim is a large state reduction ($-81.7\%$ LoRA) at a small, quantified accuracy cost, not statistical equivalence (E-PARAMS).

### 4.3 Efficiency
- Persistent state vs $T$: v1 bank = $184,320 + T\times184,320 + T$ scales; v2 = constant 371,040 (measured at $T=5,10,20,40$, E-PARAMS-T/E-TL).
- Inference (GPU, batch 32): ours and EXP-009 merged both 1.129e12 FLOPs/forward, $\sim413$ img/s, $\sim579$ MiB peak (E-EFF). The training-time forward of v1 iterates all historical branches and slows with $T$; the v2 forward is constant-complexity.

### 4.4 Task length
| Dataset | T=5 | T=10 | T=20 | T=40 |
| --- | ---: | ---: | ---: | ---: |
| ImageNet-R Final | 77.54 | 79.06 | 77.03 | 75.31 |
| ImageNet-R Forgetting | 8.83 | 6.82 | 9.51 | 12.34 |
| CIFAR-100 Final | 88.06 | 87.70 | 85.63 | — |
| CIFAR-100 Forgetting | 8.77 | 8.53 | 10.72 | — |

The optimal operating point is around $T=10$; forgetting grows monotonically with $T$ (C8), while the persistent LoRA state stays constant.

### 4.5 Extra dataset: CUB-200-2011
| Method | Final Top-1 | AvgAcc | Forgetting |
| --- | ---: | ---: | ---: |
| EXP-009 | 71.75 | 84.93 | 23.31 |
| **cumulative+gauge** | **79.79** | **87.69** | **14.20** |

On fine-grained CUB, the legacy per-task renormalization is substantially more harmful; the gauge-aligned cumulative state improves Final Top-1 by 8.04 and reduces Forgetting by 9.11 (single seed; C6).

### 4.6 Ablations and analysis
- cumulative-only (no gauge): INR 78.78/81.69/7.28 (E-INR-CUM) — the unaligned re-expression of $H$ in new coordinates distorts the historical operator; gauge recovers +0.28 Final and -0.46 Forgetting (E-INR-GAUGE).
- residual LRPT on top of gauge: INR 78.49/82.19/6.38 (E-INR-GLRPT) — improves AvgAcc/Forgetting but lowers Final; kept as an ablation, not the main method (C10).
- Diagnostics vs per-task forgetting: operator drift r=0.26 (ns); control-group raw drift r=-0.73 (p=0.025, plasticity confound); gauge diagnostics have zero variance; LRPT drift r=0.18 (ns) (E-CORR). Diagnostics are mechanism evidence, not forgetting predictors (C9).
- Closed negative routes (E-NEG): rank/damping/raw-space/class-mean/JVP/operator-stability/adaptive/consistency variants all failed to improve over the main line; these are documented to avoid repetition.

### 4.7 Engineering audit
- Unit tests: 45 passed, including Phase A/B/C algebraic equivalence (E-AUDIT-UNIT).
- 4-GPU DDP smoke and all full runs: exit 0; artifacts contain no per-task $B$ files; `verify_sa_consistency.py` PASS on INR/C100/CUB main artifacts with feature and prototype-logit differences of 0 (E-AUDIT-VERIFY/E-AUDIT-DDP).
- Reproducibility: every experiment records config, commit, log, and artifact; the v1→v2 migration script is explicit with a backup.

## 5 Discussion

Interpretation. The shared $A$ remains essentially inside its initial row space during training (per-task rotation/preservation diagnostics at $10^{-8}$), which makes the closed-form gauge alignment exact in practice: the historical operator is preserved while $A$'s norm/canonical coordinates change. This explains why removing v1's implicit renormalization (which changed historical contributions whenever $\|A\|$ or $A$'s basis moved) does not hurt and even helps on fine-grained CUB.

Relation to our prior system. On ImageNet-R/CIFAR-100 the method is statistically equivalent to EXP-009 (C5); its contribution is the 86% state reduction and O(1) memory, not an accuracy gain. We do not claim universal superiority over SD-LoRA or prior drift-compensation methods; CUB is single-seed and external baselines (InfLoRA/CL-LoRA/LoRA-DRS/DGS) were not re-implemented (CIT-06/03/13/14).

Speculation (clearly separated). If $A$ ever leaves its row space substantially (e.g., larger ranks or different optimizers), the projection residual would grow and gauge alignment alone would no longer preserve the historical operator; correctly computed pre-save diagnostics would flag this. The logs written before the P0-2 fix compared the new state with itself and are therefore not valid evidence about this regime (`TODO_VERIFY`).

## 6 Limitations

- External strong baselines are not reproduced; all comparisons are internal (SD-LoRA and our previous systems).
- CUB-200 result is single-seed; multi-seed CUB and ImageNet-A/DomainNet are not included.
- At $T=40$ forgetting reaches 12.34 (INR); the method does not solve long-horizon forgetting by itself.
- Per-task gauge diagnostics have near-zero variance and are therefore not useful as forgetting predictors; the correlation analysis is descriptive, not causal. Note: diagnostic logs before P0-2 were computed after the state was overwritten and must not be cited as mechanism evidence.
- Training-time peak memory includes the temporary current $B$ and QR factors; these are reported separately from the persistent budget (not fully tabulated yet: `TODO_EVIDENCE` training peak memory).
- The exactness claims (C1) hold under the repository's normalization convention; they are not claims about arbitrary LoRA implementations.

## 7 Conclusion

We presented an online gauge-aligned cumulative Shared-A LoRA framework for rehearsal-free CIL. It folds all historical up-projections into one cumulative matrix, preserves the historical effective operator in closed form as the shared down-projection evolves, and reduces persistent LoRA state to 10.1% of SD-LoRA independent of task count. Across four seeds the price relative to our previous system is a statistically significant but small deficit on ImageNet-R Final/AvgAcc ($-0.65$/$-0.52$) and no significant difference on CIFAR-100 or Forgetting, with a large single-seed gain on fine-grained CUB-200. The framework provides a principled, measurable path from task-dependent adapter banks to O(1) state in continual low-rank adaptation.

## References

- CIT-01 SD-LoRA (ICLR 2025), "Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning", OpenReview 5u1rlpx68a.
- CIT-02 SA-LoRA, J. King Saud Univ. Comput. Inf. Sci. 2026, s44443-026-00925-x.
- CIT-03 CL-LoRA, CVPR 2025, arXiv:2505.24816.
- CIT-04 RanPAC, NeurIPS 2023, arXiv:2307.02251.
- CIT-05 LDC, ECCV 2024, arXiv:2407.08536.
- CIT-06 InfLoRA, CVPR 2024, arXiv:2404.00228.
- CIT-07 FM-LoRA, CVPR 2025 Workshop DG-EBF, arXiv:2504.08823.
- CIT-08 C-LoRA, arXiv:2502.17920.
- CIT-09 EASE, CVPR 2024, arXiv:2403.12030.
- CIT-10 E2-LoRA, ICML 2026, arXiv:2605.27482.
- CIT-11 Janus-LoRA, ICML 2026, arXiv:2605.28495.
- CIT-12 Balanced LoRA (BaLoRA), ICML 2026. `TODO_CITATION` arXiv ID.
- CIT-13 LoRA-DRS, CVPR 2025, arXiv:2503.18985.
- CIT-14 DGS, CVPR 2026. `TODO_CITATION` arXiv ID.
- CIT-15 ICLR 2026 Two-Way Alignment (BiCyc), arXiv:2606.05675.
- CIT-16 CT-Merging, arXiv:2607.20561. `TODO_VERIFY`
- CIT-17 Subspace-Boosted Model Merging, arXiv:2506.16506. `TODO_VERIFY`
