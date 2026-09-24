# Frozen-P Margin and G Consolidation Study

## Research question

Does the fixed-basis continual LoRA lose old-class accuracy because cumulative
`G` changes old-vs-new margins, because the deployed prototype cosine head is
miscalibrated, or through their interaction? The saved linear FC is a shadow
head; the deployed classifier is the prototype cosine head.

This study is gated. It does not change training or deploy a new rule unless
the current-task-data surrogate predicts the effect of candidate `G` writes
on held-out diagnostic old classes. Test labels are used only to audit that
surrogate, never to fit a bias or select a candidate within a run.

## P0: read-only state swaps

`scripts/counterfactual_g_head_swap.py` audits immutable post-merge task
snapshots. For fixed old-class test samples, it crosses old/target `G` with
old/target prototype rows while leaving later-class rows at the target state.
It also evaluates the saved FC separately and labels it shadow-only. See
`FROZEN_BRANCH_COUNTERFACTUAL_20260924/analysis.md` for all interpretation
limits and numeric examples.

On Task 0 -> 9, restoring old `G` while keeping final prototypes improves
Task-0 accuracy by 10.09 pp on CUB-200 and 5.50 pp on CIFAR-100. However,
adjacent old-`G` rollback has mixed or small effects. Consequently, a uniform
`G` freeze or cap is not supported by these swaps.

## P1: acquisition-only concentration and old-group bias

`backbone/margin_calibration.py` stores one class concentration `rho_c` at
acquisition and generates isotropic tangent-space synthetic old features.
At each later task, `scripts/evaluate_margin_calibration.py` uses only
current-task training features and synthetic old features in a stratified
two-fold fit of one old-group cosine-score offset. The per-class research
state is `rho_c` and accumulated `bias_c`; prediction arrays in the `.pt`
files are research diagnostics, not deployment state.

The completed T=10 read-only runs are negative:

| Dataset | Baseline Final | Bias Final | Baseline AAA | Bias AAA |
|---|---:|---:|---:|---:|
| CIFAR-100 seed 1993 | 88.17 | 87.55 | 92.38 | 91.90 |
| CUB-200 seed 1 | 84.40 | 79.67 | 89.60 | 87.26 |

At Task 9, synthetic old features have 0% predicted-new rate in both
datasets. Real old test samples have 1.20% and 1.85% predicted-new rate,
respectively. This is evidence that the isotropic pseudo-old distribution
misses boundary cases; it is not proof that it is the only source of the
accuracy drop. Bias-only fails the pre-screen. A joint `G`+head method has
not been accepted or evaluated, so the bias-only failure is not a proof that
every joint intervention fails.

The matched ImageNet-R seed 1995 Frozen-A snapshot run completed all ten
tasks with Final 78.75%, AAA 81.941%, and Forgetting 5.946%. The first two
failed launches are retained as failure artifacts. The initial postprocess
service failed because its systemd environment lacked `python` on `PATH`;
training and its snapshots were unaffected. After setting an explicit PATH,
`margin-inr-postprocess-20260924-r3.service` is running the read-only INR
Task 0 -> 9 and Task 8 -> 9 state swaps, calibration, and three G-proxy
transitions. The full three-dataset margin gate is pending that service.

## P2: G candidate surrogate, shadow only

`scripts/evaluate_margin_g_proxy.py` tests `alpha` in `{0.5, 0.75, 1.0}`
for the isolated cumulative `G` write. For each candidate it refits the
existing prototype transport on paired current-task training features,
checks that `alpha=1` reconstructs the saved deployed prototype state,
records current-data old-class KL and synthetic-old misrouting, then records
true old/new test accuracy as an audit only. It never selects a candidate for
deployment.

`scripts/summarize_margin_study.py` applies the gate: at least two
informative task transitions per dataset (candidate old Top-1 differs by at
least 0.20 pp), at least 65% inverse KL/old-Top1 pair agreement, and positive
mean inverse-KL Spearman. At least two datasets must pass before a `G`
selector may be designed. Uninformative comparisons are not counted as
successes. Stage-gate JSON is research output, not model state.

Across five CIFAR-100 and six CUB-200 transitions, the stage gate is not
met. CIFAR-100 has two informative tasks: only 1/3 candidate pairs are
ranked correctly, with mean inverse-KL Spearman -0.25. CUB has one
informative task, whose sole candidate pair is ranked backwards. The
remaining CUB task comparisons are below the preregistered 0.20 pp old
Top-1 difference and cannot count as supporting evidence.

Counterexamples are concrete. CIFAR-100 Task 1 moves opposite to the KL
proxy: `alpha=0.5` has KL 0.00287 / old Top-1 95.8%, whereas `alpha=1`
has KL 0.01002 / old Top-1 96.1%. CUB Task 8 likewise has KL
0.00036 / old Top-1 86.35% at `alpha=0.5`, versus KL 0.00145 / old Top-1
86.61% at `alpha=1`. The new-class Top-1 also rises from 82.50% to
83.02%. Thus the current-data old-class KL is not a trustworthy sign test
for the net old/new accuracy response to this `G` write. No automatic or
retrospective candidate selection is allowed.

## Decision rule

- Keep the Frozen-P additive implementation and all stored snapshots intact.
- Do not add head bias to the deployed classifier after the bias-only failures.
- Do not add a `G` decision rule until the preregistered two-dataset proxy
  gate passes. It currently does not. Report a mechanism-negative result
  rather than tuning a threshold on old test labels.
- The eventual joint single-seed experiment, if eligible, must report Final,
  AAA, old/new Top-1, adjacent and long-horizon forgetting, extra state,
  runtime, and paired predictions on C100/INR/CUB. Seeds are held for a
  subsequent confirmation stage.

## Reproduction

```bash
python -m pytest -q tests/test_counterfactual_g_head_swap.py tests/test_margin_calibration.py tests/test_margin_g_proxy.py tests/test_summarize_margin_study.py
python scripts/summarize_margin_study.py --calibration MARGIN_CALIBRATION_20260924/c100_t10.json --calibration MARGIN_CALIBRATION_20260924/cub_t10.json --proxy MARGIN_G_PROXY_20260924/cub_task1.json --proxy MARGIN_G_PROXY_20260924/cub_task6.json --proxy MARGIN_G_PROXY_20260924/c100_task1.json --output MARGIN_G_PROXY_20260924/stage_gate_partial.json
```

`MARGIN_G_PROXY_20260924/stage_gate_c100_cub.json` is the current
machine-readable two-dataset gate using all completed candidate runs.
