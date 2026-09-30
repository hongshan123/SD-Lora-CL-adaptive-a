# Historical-only Seeds 4--13

## Registered Protocol

- User request on 2026-09-30: retain completed seeds 1--3; run ten NEW seeds 4--13, not 1--10.
- Three datasets: CIFAR-100, ImageNet-R and CUB-200; 30 new full-sequence runs.
- T=10, rank10, 20 epochs, SGD; dataset-specific schedules/LR and splits unchanged.
- Dual GPU, batch64 per process, effective batch128.
- Slots: physical GPU 0/1, 2/5 and 6/7; GPU 3/4 excluded.
- Run seed-major, dataset-interleaved; dispatch immediately when a slot completes.
- All training uses nohup under user-systemd; independent checks every 1800 seconds.

## Immutable Method

Train from detached independent clone at `c68e12a557bef0b197841a0a8ecb9a308c4b83cf`, the completed seeds 1--3 training commit. No algorithm changes.

Historical-only means `function_safe_pareto`, historical student branch for teacher consistency, cross-fit interval4. Preserve live-A coordinate alignment, bounded NormCap absorption, prototype transport and the original prototype classifier. Dual-B and HBD remain off.

Generate from the original registered per-dataset seed1 configuration. Formal configurations differ ONLY in seed, prefix and output filepath. Record configuration checksums, cached pretrained-weight checksum and environment in the manifest. Existing seeds1--3 must pass log, provenance and 24-branch artifact checks before reuse.

## Execution Gates

1. Focused scheduler/config tests, including exact preservation of source settings, seed coverage, no overwrite, exclusions, dynamic refill and no cherry-picking.
2. Three short seed4 Task0/1 smoke checks (one epoch per task), strictly excluded from performance tables.
3. Formal queue starts only after all three smoke artifacts/logs pass. All formal runs start afresh from Task0; interrupted outputs are archived, never treated as complete or resumed at an arbitrary task.
4. Completion requires status0, ten task evaluations and finite 24 Q/V branch artifacts with correct code/config provenance.
5. Summarize all seeds1--13 only when each dataset has all thirteen complete unique results. Report mean and sample standard deviation, all individual seeds and full curves; no selecting winning seeds.

## Metrics

Report Final, AAA and Forgetting. Keep raw logged Forgetting separately; recompute the reported Forgetting from numeric class-range identities in the original CNN dictionaries because the legacy trainer string-sorts two-digit/three-digit class ranges. This is read-only reporting, not a training-code change.

## Status

- Preparation: in progress. Existing seeds1--3 will not be overwritten or scheduled.
- Runtime: `.runtime_historical_only_seed4_13_20260930/`.
- Logs: `logs/<dataset>_historical_only_formal_seed<seed>_t10_i4_bs128.log`.
- Reports: `formal_results.json`, `formal_results.csv`; full summary withheld until complete.
- Launch via `bash run_historical_only_seed4_13_queue.sh formal` and independent `monitor` service after smoke passes.
