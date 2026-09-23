# Frozen-A Current-Branch Intrusion Sequence Archive (2026-09-23)

## Question and protocol

Does the trained current `sBA` branch of the no-Dual Frozen-A main method
turn old-class samples into new-class predictions? This is a small causal
intervention, not a full T=10 performance comparison.

- Model: Frozen-A shared down-projection, CoordinateStable historical state,
  bounded NormCap absorption, prototype transport, prototype classifier; no
  Dual-B, HBD, or Adaptive-A.
- CUB-200: seed 1, Task 0/1/2, 20 classes per task, GPUs 4,5.
- CIFAR-100: seed 1993, Task 0/1/2, 10 classes per task, GPUs 6,7.
- Rank 10, SGD, 20 epochs per task, per-GPU batch 64, effective batch 128.
- Run script: `run_frozen_branch_intrusion_t3.sh`. The run manifest records
  launch HEAD `abd6140`; the snapshot/diagnostic source was uncommitted at
  launch and was subsequently committed as `84ff9b1`. No model-update logic
  was changed during the run; checksum manifest generation and diagnostic
  `.pt` export were added while the sequence was running.
- Environment: PyTorch 2.4.1+cu121, timm 1.0.9, CUDA toolkit 12.1.
- Frozen base model: `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`,
  Hugging Face revision `063c6c38a5d8510b2e57df480445e94b231dad2c`,
  `model.safetensors` SHA256
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`.
- Failed preflight directories without `_R2` are invalid: CUB lacked the
  deterministic cuBLAS environment variable; CIFAR used 32x32 inputs.
  Only `_R2` directories belong to this experiment.

## Saved state

Runs:

- `FROZEN_BRANCH_INTRUSION_CUB_T3_SEED1_R2/`
- `FROZEN_BRANCH_INTRUSION_C100_T3_SEED1993_R2/`

Each `task_snapshots/task_XXX/` stores:

- `pre_merge.pt`: all 24 Q/V branch down/current-up/historical-up tensors,
  current scale, classifier, class counts, and previous-task prototypes.
- `sa_state.pt`, `sa_merged_lora.pt`, `sa_prototypes.pt`: post-boundary fixed
  state, deployment LoRA, and class prototypes for that exact task.
- `CLs_weight*.pt`, `CLs_bias*.pt`, `config.json`, `complete.json` with SHA256
  checksums for the core files.
- For Task 1/2, `branch_intrusion.pt` contains labels and both sets of
  per-sample global logits, so confusion, margins, and accuracy can be
  recomputed without training or inference. The accompanying JSON contains
  the first analysis.

Each run directory also contains a copy of the training log, both diagnostic
logs, the queue log, and the original `run_manifest.json`. All three task
snapshots in each run passed the SHA256 audit.

The ViT base weights and datasets are external dependencies, but no previous
task LoRA checkpoint is required to analyze these saved states. These are
analysis snapshots, not optimizer/RNG checkpoints for resuming training.

To verify files and hashes:

```bash
python scripts/audit_sa_task_snapshots.py FROZEN_BRANCH_INTRUSION_CUB_T3_SEED1_R2
python scripts/audit_sa_task_snapshots.py FROZEN_BRANCH_INTRUSION_C100_T3_SEED1993_R2
```

To regenerate an inference diagnostic from a saved pre-merge state, without
retraining:

```bash
python scripts/diagnose_main_branch_intrusion.py \
  --run-dir FROZEN_BRANCH_INTRUSION_CUB_T3_SEED1_R2 --task 2 --device cuda:0
```

## Results

Post-boundary deployed Top-1 curves:

| Dataset | Task 0 | Task 1 | Task 2 |
| --- | ---: | ---: | ---: |
| CUB-200 | 96.87 | 93.18 | 92.00 |
| CIFAR-100 | 96.90 | 95.80 | 95.10 |

The read-only branch intervention gives:

| Dataset/task | Old Top-1: history to full | Correct old to new error | New Top-1: history to full |
| --- | ---: | ---: | ---: |
| CUB Task 1 | 90.96 to 90.61 | 0.35% | 94.85 to 95.71 |
| CUB Task 2 | 91.88 to 91.62 | 0.35% | 90.54 to 92.57 |
| CIFAR-100 Task 1 | 96.00 to 94.70 | 2.30% | 88.20 to 97.20 |
| CIFAR-100 Task 2 | 94.85 to 93.50 | 1.55% | 91.90 to 98.00 |

The correct-old-to-new denominator is **all old test samples**, not only
historically correct samples. The old-class restricted Top-1 does not fall in
any of these four interventions; this suggests cross-task competition,
not lost within-old-class discrimination. The current branch helps the new
classes substantially, particularly on CIFAR-100. This short sequence does
not establish behavior at later tasks or across seeds.

### Interpretation boundary

All comparisons use the same pre-merge model, classifier, and prototypes;
only the current branch is disabled. They therefore isolate training-time
current-branch intrusion. They do not isolate post-merge absorption or
prototype transport, and Task 0 has no old classes for this diagnostic.
