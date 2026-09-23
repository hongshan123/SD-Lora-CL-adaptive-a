# Fixed-rank spectral allocation: offline feasibility check

Date: 2026-09-23. This is an offline mechanism diagnostic, not a CIL accuracy experiment.

## Setup

- Source states: completed Task 1 and Task 5 SBGC attribution checkpoints (rank 10) for CIFAR-100 seed 1993, ImageNet-R seed 1995, and CUB-200 seed 1. The next tasks analyzed are Task 2 and Task 6, respectively.
- Each next-task class contributes up to eight distinct **training** images to each of: prototype construction, gradient design, and gradient holdout. Test data are not used.
- The persisted Q/V operator in each ViT block is reconstructed as `M = G P`. A frozen pretrained ViT plus this operator produces features. A temporary cosine prototype head (temperature 20) on the next-task classes produces a cross-entropy gradient with respect to each effective Q/V weight. This head is diagnostic; it is not the deployed CIL classifier.
- For gradient collection, 256 evenly spaced tokens per batch approximate the full effective-weight gradient. A rank-32 sketch of the design gradient proposes candidate spaces. Candidate scoring uses the **full**, unsketched design and holdout gradients.
- Both history and design operators are normalized to unit Frobenius norm for the spectral search. The current/history weight grid is `{0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000}`. Among candidates with historical operator residual `<= 0.05`, the design-gradient residual chooses the candidate. Holdout data never choose a weight.
- No model parameter, classifier, or prototype is updated. Runs used cuda6 GPU 0 without interrupting the cuda7 SBGC queue.

For orthonormal row basis `V`, the reported history risk is `||M(I-V^T V)||_F^2 / ||M||_F^2`. The current coverage gain is the reduction in `||H(I-V^T V)||_F^2 / ||H||_F^2`. The stricter descent-alignment gain is

`(<H_holdout, H_design V^T V> - <H_holdout, H_design P_old^T P_old>) / (||H_holdout||_F ||H_design||_F)`.

Positive descent alignment is only a **first-order, local** indication that a projected design-gradient step may reduce the holdout surrogate loss. It is not an accuracy or forgetting bound.

## Results

All entries are medians across 24 Q/V branches, except positive branch counts and maximum history risk. Coverage/alignment values are fractions, not accuracy points.

| Dataset | Next task | Old off-space gradient | Sketch energy | Design coverage gain | Holdout coverage gain | Holdout descent gain | Positive descent branches | Max history risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 | 2 | 0.984 | 0.975 | 0.445 | 0.127 | 0.0231 | 18/24 | 0.0276 |
| CIFAR-100 | 6 | 0.982 | 0.992 | 0.638 | 0.112 | 0.0045 | 15/24 | 0.0442 |
| ImageNet-R | 2 | 0.984 | 0.866 | 0.237 | 0.087 | 0.0019 | 12/24 | 0.0314 |
| ImageNet-R | 6 | 0.983 | 0.872 | 0.218 | 0.119 | 0.0001 | 7/24 | 0.0255 |
| CUB-200 | 2 | 0.986 | 0.942 | 0.461 | 0.284 | 0.0374 | 22/24 | 0.0494 |
| CUB-200 | 6 | 0.985 | 0.952 | 0.482 | 0.264 | 0.0579 | 21/24 | 0.0458 |

Raw per-branch frontiers are in the six `*_descent.json` files. Earlier coverage-only JSON files are local intermediates and are not needed to reproduce the conclusions.

## Interpretation

The expected off-space fraction for a generic direction relative to a random rank-10 space in 768 dimensions is about `1 - 10/768 = 0.987`. Thus the observed `0.982-0.986` off-space fraction **alone is not evidence** of a useful new-task direction. The train/holdout gradient cosine is also small: median 0.019-0.110 across the six cases.

The constrained spectral search can improve **holdout coverage** while respecting the 5% historical operator budget. But an actual gradient step also needs design and holdout gradients to agree. CUB shows a reasonably consistent first-order signal; CIFAR-100 is weaker, particularly at Task 6. ImageNet-R provides no reliable positive descent signal under this estimator at either task. Training-set coverage substantially overstates holdout benefit.

Decision: **Do not launch a three-dataset formal training run with this gradient estimator yet.** The algebraic spectral objective is feasible, but these results do not validate improved Final/AAA or old-class retention. The next diagnostic should improve the current-demand estimator (more independent calibration samples or a trained auxiliary operator) and measure actual held-out new-class loss plus old-class predictions after operator reprojection. Compare against matched Frozen-A and Live-A checkpoints; these source states use SBGC, not pure Frozen additive.

## Reproduction

Run `scripts/offline_subspace_allocation.py` with `--config`, `--state`, and `--output`, using the matching T=10 task protocol. For example, the CIFAR Task 2 run used `sbgc_attribution_c100_t2_20260923.json` and `SBGC_ATTRIBUTION_C100_SEED1993_T2_20260923/sa_state.pt` with `--per-class=8 --sample-tokens=256 --batch-size=8`. The script infers the next task from `state['task_id']` and refuses non-SBGC fixed-P states. Tests: `python -m pytest tests/test_offline_subspace_allocation.py -q`.
