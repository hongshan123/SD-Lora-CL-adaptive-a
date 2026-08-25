# CoordinateStable Effective-Operator Coordinate Alignment Audit

**Audit scope.** This document audits the executable implementation in
`backbone/coordinate_stability.py`, `backbone/sa_lora.py`, and
`models/sa_sdlora.py` on the CoordinateStable branch. It does not use code
comments or prior experiment summaries as evidence.

## Notation and tensor shapes

For one Q or V branch, let `d = qkv.in_features` and `r = lora_rank`. The
current CoordinateStable configurations use `d=768` and `r=10`.

| Symbol | Runtime tensor | Shape |
| --- | --- | --- |
| `A_old`, `A_new` | `w_a.weight` | `[r, d]` = `[10, 768]` |
| `B_t` | `w_b.weight` | `[d, r]` = `[768, 10]` |
| `G_old`, `X` | `aggregate_up[idx]`, `aligned` | `[d, r]` |
| `M_old` | `old_operator` | `[d, d]` = `[768, 768]` |
| least-squares LHS | `new_down.t()` | `[d, r]` |
| least-squares RHS | `old_operator.t()` | `[d, d]` |
| least-squares result | `solution` | `[r, d]` |

## 1. Actual alignment objective

`align_live_a_aggregate` computes

\[
\hat A_{old}=\frac{A_{old}}{\lVert A_{old}\rVert_F+\epsilon},\qquad
\hat A_{new}=\frac{A_{new}}{\lVert A_{new}\rVert_F+\epsilon},
\]

\[
M_{old}=G_{old}\hat A_{old},
\]

then solves

\[
X^*=\arg\min_{X\in\mathbb{R}^{d\times r}}
\left\lVert X\hat A_{new}-M_{old}\right\rVert_F^2.
\]

The direct implementation is:

```python
solution = torch.linalg.lstsq(new_down.t(), old_operator.t()).solution
aligned = solution.t()
```

Source: `backbone/coordinate_stability.py`, lines 28-37.

## 2. A normalization

Both `A_old` and `A_new` use one global matrix norm:

```python
A / (torch.linalg.vector_norm(A) + 1e-8)
```

`torch.linalg.vector_norm` is called without `dim`, hence it is the Frobenius
norm over all entries. The implementation does not normalize rows, columns, or
per-layer vectors independently. Inputs are detached and converted to
`float64` before the alignment solve.

Source: `backbone/coordinate_stability.py`, lines 7-11 and 28-30.

## 3. Historical target operator and retained magnitude

The target is exactly

\[
M_{old}=G_{old}\frac{A_{old}}{\lVert A_{old}\rVert_F+\epsilon}.
\]

The saved aggregate is updated in this order:

\[
G_t=\operatorname{Align}(G_{t-1},A_{t-1},A_t)
 +s_t\frac{B_t}{\lVert B_t\rVert_F+\epsilon}.
\]

Therefore historical learned scale values `s_i` are contained in `G`, while
the raw Frobenius magnitude of each `B_i` is divided out. The raw magnitude of
`A` is likewise divided out in the historical forward path. The `epsilon`
terms mean this is not perfectly scale invariant near zero.

Sources: `backbone/sa_lora.py`, lines 1641-1659 and 659-665.

## 4. Solver

The code uses `torch.linalg.lstsq`; it does not explicitly invoke `pinv`,
`solve`, `inverse`, or SVD for the alignment. No `driver` or `rcond` argument
is supplied.

Source: `backbone/coordinate_stability.py`, line 35.

## 5. Meaning of `sa_coordinate_transport_reg`

`sa_coordinate_transport_reg=1e-4` is **not used by operator alignment**.
`align_live_a_aggregate` has no regularization argument.

The value is passed only to `fit_residual_orthogonal_transport` for prototype
transport. There the code computes

\[
C=\frac{(Z_{old}U)^T(Z_{new}U)}{n}+\lambda I,
\qquad R=UV^T,\quad [U,\_,V^T]=\operatorname{SVD}(C).
\]

Before the validation gate, this is equivalent to the identity-biased
orthogonal Procrustes problem

\[
\min_{R^TR=I}
\lVert Z_{old}UR-Z_{new}U\rVert_F^2
+n\lambda\lVert R-I\rVert_F^2.
\]

Sources: `models/sa_sdlora.py`, lines 347-349 and 1340-1345;
`backbone/coordinate_stability.py`, lines 98-103. A later validation gate may
replace `R` by the identity.

## 6. Closed form and comparison with candidate formulas

The actual formula is

\[
X^*=M_{old}\hat A_{new}^{\dagger}.
\]

If `A_new_hat` has full row rank, the same solution is

\[
X^*=M_{old}\hat A_{new}^T
(\hat A_{new}\hat A_{new}^T)^{-1}.
\]

Thus it is not exactly either of the following expressions when they use raw
`A_new`:

\[
M_{old}A_{new}^{\dagger},\qquad
M_{old}A_{new}^T(A_{new}A_{new}^T+\lambda I)^{-1}.
\]

It becomes the second form only after replacing `A_new` by normalized
`A_new_hat` and setting `lambda=0`. There is no ridge term in alignment.

## 7. Operator-alignment log values

For each aligned branch, the diagnostics are

\[
e_{before}=\frac{\lVert G_{old}\hat A_{new}-M_{old}\rVert_F}
{\lVert M_{old}\rVert_F+\epsilon},
\]

\[
e_{after}=\frac{\lVert X^*\hat A_{new}-M_{old}\rVert_F}
{\lVert M_{old}\rVert_F+\epsilon}.
\]

The emitted `before` and `after` values are arithmetic means over the aligned
Q/V branches, not one error computed after concatenating all branches.

Sources: `backbone/coordinate_stability.py`, lines 31-51;
`backbone/sa_lora.py`, lines 1671-1693; `models/sa_sdlora.py`, lines 537-550.

## 8. `max_condition`

For every branch the code computes

\[
\kappa(\hat A_{new})=
\frac{\sigma_{max}(\hat A_{new})}
{\sigma_{min}(\hat A_{new})+\epsilon}.
\]

`max_condition` is the maximum of these values across branches. It is a
diagnostic only and does not alter the solution.

Sources: `backbone/coordinate_stability.py`, lines 46-51;
`backbone/sa_lora.py`, lines 1684-1686.

## 9. Execution order relative to current B aggregation

At the end of a task, per branch the code performs:

1. Read `G_old`, `A_old`, current `A_t`, current `B_t`, and `s_t`.
2. Align only historical `G_old` to the current `A_t` coordinates.
3. Add the normalized current contribution `s_t * B_t / (||B_t||_F + eps)`.
4. Save `aggregate_up=G_t` and `shared_a=A_t`.

This occurs in `save_lora_parameters` after training. During training, the
current branch is raw `s_t B_t A_t x`; it is not normalized until save.

Sources: `backbone/sa_lora.py`, lines 1450-1459, 1641-1713, and 663-667.

## 10. Rank-deficient and ill-conditioned `A_new`

The project implementation has no explicit rank threshold, condition-number
threshold, damping, ridge term, fallback, or rejection path. It always uses
the `lstsq` result. It does add `1e-8` to normalization and diagnostic
denominators, and logs the condition value. It does not inspect the returned
least-squares rank, residuals, or singular values.

Consequently, numerical rank handling is delegated entirely to the runtime
default behavior of `torch.linalg.lstsq` on the CPU `float64` inputs created by
the function; the project does not fix or record `driver` or `rcond`.

## Source-to-formula map

| Code | Mathematical role |
| --- | --- |
| `aggregate_up @ old_a / ||old_a||` | `M_old = G_old A_old_hat` |
| `new_a / ||new_a||` | `A_new_hat` |
| `lstsq(A_new_hat.T, M_old.T).solution.T` | `argmin_X ||X A_new_hat - M_old||_F^2` |
| `g_history + s * b / ||b||` | `G_t = Align(G_old) + s_t B_t_hat` |
| `_norm_live_a(x, A, G)` | historical forward `G A x / ||A||` |
| `svdvals(A_new_hat)` | condition-number diagnostic |
