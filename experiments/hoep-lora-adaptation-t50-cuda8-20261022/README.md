# Paired LoRA adaptation experiments

This directory preserves the isolated implementation used to compare four
training objectives: the HOEP baseline, historical-input-weighted total-update
regularization, all-seen-class prototype training, and their combination.
No measured performance or execution records are included.

## Protocol

- CIFAR-100 seed1993 and ImageNet-R seed1995; T50, rank10, 20 epochs per task.
- One independent job per GPU, physical GPUs 0–3, batch128, no accumulation.
- HOEP energy budget 5%, historical LS on, NormCap off, Coordinate Transport on.
- Historical response penalty weight 1.0; prototype penalty weight 0.1,
  temperature 0.05, new-class EMA coefficient 0.1. Task0 uses neither penalty.
- Seven two-task checks precede eight formal jobs. These validate disabled-loss
  equivalence, common Task0 hashes, finite/nonzero enabled losses, snapshot
  integrity and the single-GPU path. Formal runs require all checks to pass.

`source/` is a self-contained code snapshot with optional routing in
`utils/factory.py` and the added learner in `models/hoep_adaptation.py`.
Historical input moments use fixed training memory; deployment adapter shapes
do not grow. Moments can become stale and are not a forgetting bound.

## Prepare and run

Install the repository dependencies and cache the pretrained ViT weights.
From this directory, prepare local paths and a dependency on a preceding
training/replay queue:

```bash
python prepare.py --data-root /path/to/data --wait-root /path/to/previous-queue
python run_queue.py
```

The previous queue must provide `status.json` and `replay_status.json`; the new
queue waits until both states are `complete`, then acquires shared GPU leases
and checks available memory. It never terminates the preceding jobs.

Preparation generates ignored machine-local configuration and manifest files.
Tracked `configs/` and `queue_plan.json` are input templates. For an individual
experiment, copy a template, provide a fresh absolute output path and run its
`source/main.py` with the configuration; that entrypoint does not depend on a
preceding queue.

Mathematical checks can run without a GPU:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

Runtime logs, status, metrics, hashes, weights, prototype caches and task
snapshots stay in ignored local directories. Other snapshot/replay utilities
in the parent `experiments/` directory require their locally generated runtime
inputs; those inputs are intentionally not distributed.
