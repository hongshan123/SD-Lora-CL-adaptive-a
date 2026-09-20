# Recoverability-Constrained Accessibility Adaptive-A Design

## Goal

Add an opt-in Adaptive-A strategy that rotates each shared LoRA down-projection toward current-task effective-weight gradients while bounding the fraction of the task-start historical operator that cannot be recovered by least-squares coordinate alignment.

## State And Complexity

For every Q/V branch, the persistent adapter state remains one shared row basis and one aggregate up-projection:

\[
A_l\in\mathbb R^{r\times d},\qquad G_l\in\mathbb R^{d\times r}.
\]

During task \(t\), a detached task-start anchor \((A_l^-,G_l^-)\) is retained and overwritten at the next task. It is transient and is not serialized. The operator-controller state is constant in the number of tasks. Classifier and prototype storage are reported separately because they scale with the number of classes.

## Canonical Row Basis

The recoverability strategy keeps every shared down-projection on the row-Stiefel manifold:

\[
A_lA_l^\top=I_r.
\]

The strategy uses polar retraction after each accepted subspace update. Existing checkpoints are interpreted through their row-orthonormal basis and their aggregate up-projection is converted to the corresponding coordinates before an online anchor is created.

## Historical Recoverability

The task-start historical operator is fixed throughout the task:

\[
M_l^-=G_l^-A_l^-.
\]

For a candidate basis \(A_l'\), define

\[
R_l(A_l')=
\frac{\min_X\|XA_l'-M_l^-\|_F^2}
{\|M_l^-\|_F^2+\epsilon}
=
\frac{\|M_l^-(I-P_{A_l'})\|_F^2}
{\|M_l^-\|_F^2+\epsilon}.
\]

This is evaluated without dense operators. With

\[
S=(G_l^-)^\top G_l^-,\quad
H_o=A_l^-(A_l^-)^\top,\quad
C=A_l^-(A_l')^\top,\quad
H'=A_l'(A_l')^\top,
\]

the total and captured energies are

\[
E_l=\operatorname{tr}(SH_o),\qquad
E_l^{\rm cap}=\operatorname{tr}(SC(H')^\dagger C^\top).
\]

## Current-Task Accessibility

For the effective-weight gradient \(H_l=\nabla_{W_l}L_t\), define

\[
J_l(H_l,A_l)=\|H_lP_{A_l}\|_F^2.
\]

For row-orthonormal \(A_l\), its horizontal Grassmann ascent direction is

\[
Z_l=A_lH_l^\top H_l(I-P_{A_l}).
\]

The candidate family is

\[
A_l(\gamma)=\operatorname{Retr}_{A_l}
\left(\alpha\gamma\frac{Z_l}{\|H_l\|_F^2+\epsilon}\right),
\qquad \gamma\in\Gamma.
\]

The implementation uses a configurable finite candidate grid \(\Gamma\), including 0 and 1, so exact candidate utility and risk can be verified with low-rank statistics:

\[
U_l(\gamma)=J_l(H_l,A_l(\gamma))-J_l(H_l,A_l).
\]

Q/V effective-weight gradients are represented by the current minibatch input \(X\) and output gradient \(Z\), where \(H=Z^\top X\). No dense \(d\times d\) gradient is stored.

## Global Budget

The full strategy solves the finite multiple-choice problem

\[
\max_{\gamma_1,\ldots,\gamma_L\in\Gamma}
\sum_l U_l(\gamma_l)
\]

subject to

\[
\frac{\sum_l\|M_l^-(I-P_{A_l(\gamma_l)})\|_F^2}
{\sum_l\|M_l^-\|_F^2+\epsilon}\le\varepsilon.
\]

The selector uses deterministic Pareto-frontier pruning. Before the global-budget stage, each branch independently uses the same normalized recoverability tolerance.

## Online Alignment

At stages enabling online alignment, after accepting \(A_l^{(k)}\), the in-memory historical coordinates are replaced by

\[
G_l^{(k)}=M_l^-(A_l^{(k)})^\top
\left(A_l^{(k)}(A_l^{(k)})^\top\right)^\dagger.
\]

The anchor remains fixed until task consolidation. The deployed artifact remains the existing \((A,G)\) state; no anchor history is serialized.

## Validation Stages

1. `exact_risk`: original synchronized shared-A gradient direction, exact fixed-anchor recoverability risk, no online realignment.
2. `accessibility`: replace the direction and utility with effective-weight gradient accessibility.
3. `anchor_realign`: accessibility plus fixed-anchor online least-squares realignment.
4. `global_budget`: accessibility, fixed-anchor online realignment, and one global recoverability budget. This is the complete method.

Task 0 has no historical risk and selects the full accessibility candidate.

## Non-Goals

- Do not change prototype transport, classifier, Dual-B, or task-boundary absorption.
- Do not change existing Adaptive-A strategies or their default behavior.
- Do not claim total class-incremental state is constant when class prototypes are enabled.
