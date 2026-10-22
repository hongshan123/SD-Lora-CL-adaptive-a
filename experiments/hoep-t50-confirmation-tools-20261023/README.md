# Paired T50 confirmation input preparation

This tool creates input configurations only. It does not launch, resume or stop
training, and it does not change any of the three active suites.

The frozen parent protocol specifies CUB seeds 1, 2, 3 and ImageNet-R seeds
1995, 1996, 1997. A candidate keeps its original coefficients, rank, optimizer,
epochs, preprocessing, single-GPU batch128, LS, transport and NormCap settings.
Only the seed fields and output identifiers change. No scores are read or used
to omit seeds. The first seed configurations remain references to the original
suite. Frozen controls are shared across candidate arms.

Before preparing inputs, the tool verifies the existing frozen source and
configuration hashes and checks that the shared training core is byte-identical
across the parent, B-metric and A-step implementation families. It records the
original worker and source paths for each job; no training implementation is
copied or rewritten.

```bash
python prepare_confirmation.py \
  --base /path/to/experiment-parent \
  --destination /path/to/fresh-confirmation-inputs \
  --arms a_gain10
```

One arm normally produces eight new inputs: two additional seeds, two datasets,
and the candidate plus Frozen-A. Multiple arms share the four additional Frozen
inputs. Previously prepared, running, complete or failed configurations with
the same source and effective configuration become references instead of new
jobs. A status value is recorded as an unverified observation; it does not prove
that a process is live or that a result is valid.

The resulting `manifest.json` is explicitly marked `preparation_only`. It is
not an executable queue. Activate a prepared job only after checking process
ownership, fresh output paths, available leased GPUs, frozen file hashes and
the existing reference results. Failed references require an explicit audit
before any retry. Full results must be added to the canonical parent
`RESEARCH_LOG.md` and independently verified before quantitative claims.

CPU-only preparation checks:

```bash
python -m unittest discover -s tests -v
```
