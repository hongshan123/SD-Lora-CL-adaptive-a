# Paired CUB / ImageNet-R LoRA adaptation

This directory preserves training source, mathematical tests, configuration
templates and queue tools. It includes no measured results, execution history,
weights, state files or completed-run manifests.

The paired arms are Frozen-A, HOEP, HOEP plus historical-input-weighted total
operator regularization, HOEP plus all-seen prototype CE, and their combination.
Formal templates use T50, rank10, 20 epochs per task and single-GPU batch128;
NormCap is off, historical LS and Coordinate Transport are on. New objectives
are disabled on Task0. The three-task checks use 20 epochs on Task0 and two
epochs on later tasks; they gate formal training through model/RNG equality,
finite active losses and snapshot integrity.

From the repository root, prepare a fresh copy of this directory:

```bash
python scripts/prepare_hoep_queue.py \
  --suite experiments/hoep-t50-improvement-cuda8-20261023 \
  --data-root /path/to/data --gpus 0 1 2 3
cd experiments/hoep-t50-improvement-cuda8-20261023
python run_queue.py
```

The data root must contain `cub/{train,test}` and
`imagenet-r/{train,test}`. Install the root dependencies and cache the ViT
weights before launching; the queue uses offline model loading. Preparation
only writes ignored machine-local inputs. Reuse a fresh copy for a new queue.

For a single job, copy a template, set a fresh absolute `filepath`, prepare
`source/data`, then run `python run_job.py --config /path/to/config.json` from
`source/`. Queue execution acquires shared GPU leases and waits for free memory.
`sync_records.py` and `sync_research_log.py` retain machine-specific SSH defaults;
adjust these separately if record mirroring is needed.

The source snapshot preserves the training implementation independently of the
repository root implementation. Publishing an experimental mechanism makes no
claim of a verified accuracy improvement.
