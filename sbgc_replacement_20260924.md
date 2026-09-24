# Frozen-P NormCap-to-SBGC Replacement

## Question

Does Fisher-diagonal G consolidation improve the complete Frozen-P recipe when
prototype transport stays enabled and NormCap is removed?

## Protocol

- CIFAR-100 seed 1993, ImageNet-R seed 1995, CUB-200 seed 1.
- T=10, rank=10, 20 epochs/task, SGD and constant schedule from the existing
  complete Frozen-P configs. Two GPUs, batch 64/GPU, effective batch 128.
- GPU pairs: C100 0,1; INR 4,5; CUB 6,7. GPUs 2,3 are excluded.
- No Dual-B, HBD, Adaptive-A, or rehearsal memory.
- Fisher arm: `sensitivity_budgeted_g`, Fisher-diagonal sensitivity, 5% risk
  budget, QR-canonicalized frozen input basis, and prototype transport.
- Additive arm: identical SBGC implementation and calibration, but
  `sa_g_shadow_only=true`; it deploys additive G, not the Fisher candidate.
- Complete Frozen-P arm: existing `live_a_aggregate_b` with NormCap and
  prototype transport. Its previous complete runs are the external reference.

The Fisher-vs-additive pair isolates the G decision under the same QR basis and
transport. Comparison with complete Frozen-P is a system-level comparison:
Task-0 QR canonicalization and later optimizer coordinates differ as well as
the merge. Fixed P makes LS coordinate alignment unnecessary in the SBGC arms.
Do not attribute a Fisher-vs-Frozen-P difference solely to Fisher weighting.

## Status

- Code and config tests: 64 passed on 2026-09-24.
- First smoke launch invalid: deterministic CuBLAS requested without
  `CUBLAS_WORKSPACE_CONFIG`; no model result. Retained as a failed run.
- Task 0/1 smoke relaunch: all three `*_r2` runs passed. Task-0 QR
  operator relative error was 1.23e-7 to 1.25e-7. Task 1 activated the
  constraint in all 24 branches, with max risk 0.05. Rebuild, prototype
  transport, evaluation tensor hash, and RNG checks passed.
- Task-1 mean current-target distortion in the short smoke was 0.444 C100,
  0.566 INR, and 0.583 CUB. This is a pre-registered plasticity warning,
  not evidence of an accuracy benefit. Two-epoch accuracy is not comparable
  to the 20-epoch reference.
- Full T=10 Fisher then additive queues are the next step.

## Checks

For each dataset record Task-0 canonicalization error, nine transport gate
decisions, transition-wise Fisher risk (<=0.050001), target distortion,
Final/AAA/Forgetting and Top-1 curve. Verify artifacts after every task,
including a rebuilt fixed-P backbone. Report calibration time separately.
