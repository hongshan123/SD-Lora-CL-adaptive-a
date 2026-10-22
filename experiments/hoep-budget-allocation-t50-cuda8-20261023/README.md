# HOEP branch coverage inside a fixed global energy budget

The original HOEP selector spends a global 5% historical operator-energy
budget on complete low-energy eigenspace groups. The optional `coverage` rule
first reserves the lowest complete group from every Q/V branch, then fills
the remaining global budget in ascending group-energy order. Groups that do
not fit are skipped. If the reservation is infeasible, the original global
selection is used. If the original selection already covers every branch,
its masks are retained. This greedy rule makes no optimality claim.
The original negligible-history and numerical tolerances are retained.

The global energy cap, spectral tie handling, rank10 deployment state,
orthogonal retraction, historical LS, prototype classifier and Coordinate
Transport are retained. NormCap is off. Coverage can select a different number
of A directions; it is not a count-matched location-only ablation.

The paired design crosses global/coverage allocation with A gradient
multiplier1/10. Existing global formal runs are reused. Only coverage1 and
coverage10 require new formal runs, on CUB seed1 and ImageNet-R seed1995,
T50, 20 epochs/task, single-GPU batch128. No performance claim follows from
the training-state probes or mathematical tests.

Eight three-task checks precede the formal runs. Global controls must match
the existing same-gain model and RNG hashes at all three boundaries. Coverage
checks require the same Task0, actual changed allocation and A updates,
unchanged protected rows and the original global energy cap. Short-check
accuracy never selects or blocks a numerically valid arm.

The queue consumes a frozen local manifest and shared GPU leases. The optional
GPU0 one-shot waiter starts only after the current owner releases its lease.
Runtime results belong in the parent canonical `RESEARCH_LOG.md`.

## Source publication

This directory contains source, tests, configuration templates and an input-only
`queue_plan.json`. Runtime manifests, journals, probe measurements, hashes,
model snapshots and results are excluded. Configuration output paths are
relative templates; set fresh absolute paths for a local run.

`build_suite.py` preserves the original preparation implementation. It requires
locally generated parent/A-step manifests, reference smoke snapshots and
training-state probes in the parent records directory. Those prerequisites
are not included in the publication. The original queue should be prepared
only after producing those local references; the input-only plan is not an
executable manifest. The parent and A-step README files describe their setup.

The allocation mathematics can be checked independently of datasets and saved
training records:

```bash
python -m unittest discover -s tests -v
```
