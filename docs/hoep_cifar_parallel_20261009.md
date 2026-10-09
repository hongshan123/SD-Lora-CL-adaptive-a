# Four-pair CIFAR queue handover

User approved four simultaneous two-GPU experiments on cuda7. Preserve all
15 authorized cells, exact configs, world_size2 and batch64 per rank. The
training sources remain pinned to the original experiment snapshot.

Pairs: 0,1 / 2,3 / 4,5 / 6,7. Existing Frozen1994 is complete. Existing
Live1994 on0,1 is adopted without stopping/restarting its torchrun/ranks.
New dispatch prioritizes low-HOEP1994/1995, then missing endpoint runs.
Random/high only start after the same-seed low-HOEP has completed and all
nine reference count vectors validate. Existing seed1993 counts are reused.

Handover pauses ONLY the serial scheduler parent using SIGSTOP, leaving
its torchrun and training children running. This prevents a second dispatch
from the serial scheduler. The new supervisor observes the adopted child's
unreaped /proc exit status before accepting its full log, then retires the
paused parent. Parent start ticks and command identities are checked.

Each new experiment has a worker recording its exit code and validated
metrics in an atomic result artifact. The coordinator can recover these
workers without overwriting outputs. Its systemd unit uses KillMode=process
so a coordinator failure does not kill healthy training workers. Any failure
stops new dispatch; already-running workers are allowed to finish.

Original source/config SHA256s are verified before every launch; the new
dispatcher has its own archived source and SHA256. Status/events/results
are saved in the original runtime's parallel directory, and aggregate status
and results are also written at the original runtime paths.

## Execution ledger

- Original experiment baseline: 395 tests passed before this change.
- Current server: cuda7, 96 logical CPUs; six unused GPUs at inspection.
- Dependency, reference, process-handover and recovered-worker tests: RED
  observed before implementation, then eight tests passed. Full suite: 403
  passed in 37.27s. Final startup recovery uses the same tested liveness
  helper; targeted eight tests passed again after that adjustment.
- Independent review identified recovered-worker disappearance monitoring;
  fixed and reviewed. Terminal proof is reread before reporting disappearance.
- Launched 2026-10-09 00:57 CST using
  `codex-hoep-cifar15-cuda7-parallel-20261009.service` (KillMode=process).
- GPU0,1: adopted Live1994 torchrun695244/ranks695408,695409 unchanged;
  only serial scheduler467330 is paused. Live continued Task3.
- GPU2,3: HOEP1994; GPU4,5: HOEP1995; GPU6,7: Frozen1995. All three
  entered Task0. Eight GPUs held 8480–8734MiB and showed 67–100% utilization
  in launch verification. Four active, one completed, ten pending; no failures.
- Frozen1994 and the three original seed1993 results are reused unchanged.
- Original training/config hashes and separate dispatcher hash passed after
  launch. Dispatcher SHA256:
  `c9a3c778d2272b42573cda0ed45522c86695298b84be20b60afe6ea3017a9dfc`.
- Runtime: `.runtime_hoep_cifar15_20261008_r2/parallel/`. Atomic status and
  results also remain available at the original runtime paths.
- When the adopted training exits and its exit status is recorded, the old
  serial scheduler is deliberately retired by a parent-only SIGKILL. Its old
  service may then show failed; experiment outcomes are the per-job proofs
  and the new parallel service/status, not that retired service status.
