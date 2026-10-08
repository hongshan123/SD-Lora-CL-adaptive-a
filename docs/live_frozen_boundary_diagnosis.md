# Live-A / Frozen-A boundary diagnosis

The only training reference is `nodual_core_threeway_results.md`: T=10,
rank 10, 20 epochs/task, single GPU with batch 128, no Dual-B. ImageNet-R
uses seeds 1995–1997, CIFAR-100 uses 1993–1995, and CUB-200 uses 1–3.
Both policies retain coordinate alignment, bounded NormCap absorption, and
prototype transport. Old results have no usable pre-merge boundary snapshots
on this host, so the new runs are protocol reproductions in separate paths.

Generate paired configs with `scripts/live_frozen_protocol.py`. The experiment
directory contains copied source configs, generated configs, run manifests,
logs, task snapshots, raw logits, and summary files. `run_manifest.json` in
each training directory records the Git commit and config digest. The current
source config files are copied from the three corresponding
`exps/*coordinate_stable_adaptive_a*` files; their Dual-B and Adaptive-A
fields are disabled and batch size is set to 128. If the historical source
configs or code revision cannot be independently recovered, do not label this
an exact replay of the September 2026 table.

Before formal runs, ImageNet-R seed 1995 trains Live-A and Frozen-A with
snapshots both off and on. `scripts/audit_live_frozen_smoke.py` requires equal
Task 0–9 model/RNG hashes and per-task accuracy curves. The snapshot-on pair
then runs `scripts/diagnose_live_frozen.py` for Task 1–9. Any failed audit or
logit reconstruction stops the queue before remaining seeds start.

For each method and task boundary, one head contains pre-transport old-class
prototypes and temporary new-class prototypes computed only from the current
task's training split with test preprocessing under S2. With that head and the
same test order, the script evaluates S0 (previous deployment), S1 (current A,
no B), S2 (trained pre-merge), S3 (LS-aligned history plus B), S4 (exact B
absorption without NormCap), S5 (NormCap deployment), and H_LS (LS-aligned
history, no B). S6 replaces only old prototypes with transported versions.
The final new prototypes stored by the training run are a separate head change.
S3/S4 and S5/reconstructed deployment logits must agree within max absolute
`1e-4` and relative L2 `1e-5`.

Per Q/V branch, `R_rec` is the historical operator's normalized residual
outside the current A row space. `R_weighted` applies that lost operator to
all old-class **training** images' S0 layer-input tokens, divided by the
historical response energy. These are offline diagnostics only; old images
are never replayed into training or used to build prototypes. Both metrics
are single-operator proxies, not classification-accuracy bounds.

`scripts/summarize_live_frozen_diagnostics.py` writes all seed×task results,
paired Live−Frozen differences, per-seed final metrics, and descriptive
Spearman correlations. The primary geometry comparison is residual versus
S0→H_LS old-class accuracy loss. The S0→final deployment relationship is
reported separately. Q/V branches are aggregated into one seed×task value
before correlation, so 24 branches do not inflate the sample count.
