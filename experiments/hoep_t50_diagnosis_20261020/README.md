# HOEP-A T50 diagnostic experiment (2026-10-20)

This directory archives the exact launch configuration and coordinator used on cuda7.
Training is still in progress; these files do not claim a completed T50 result.

## Protocol

CIFAR-100 (`cifar224`), seed 1993, T=50, rank 10, 20 epochs per task,
effective batch 128 with microbatch 64, FP32, HOEP low-energy budget 5%,
historical LS and prototype transport enabled, NormCap disabled.
GPU 6 runs one training trajectory. GPU 7 evaluates completed task boundaries;
after training, both GPUs can drain the remaining diagnostics. This is not DDP.

## Code and provenance

The diagnostic implementation is tracked in `models/`, `utils/`, `scripts/`,
and `tests/`. `source_origin.json` identifies the original base commit;
`source_provenance.json` records hashes of the isolated source used by the run.
The two verification JSON files record previous snapshot and optimization checks.
They are verification records, not final performance results.

`formal.json`, `launch_gpu67.py`, and `verify_optimization.py` are exact archival
copies, including original machine paths. The coordinator expects a sibling
`source/` checkout, `configs/formal.json`, a Python environment, and the prior
smoke-test snapshots. It deliberately refuses an existing output directory.
Do not run the archived coordinator without preparing those dependencies.

## Reproduce without the archived coordinator

Install the project dependencies and configure the dataset and pretrained ViT.
Copy `formal.json` to a local configuration and change `filepath` to a fresh
output directory. From the repository root, use the same launch environment:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=6 python main.py --config /path/to/local-formal.json
```

For a completed boundary (both it and the previous boundary must have
`complete.json`), run diagnostics in a separate process:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/diagnose_hoep.py \
  --run-dir /path/to/fresh-output --task 1 \
  --output-dir /path/to/fresh-output/diagnosis --batch-size 64
python -m pytest -q tests/test_hoep_diagnostic.py
```

The experiment evaluates S0–S6, exact absorption and deployed-model checks,
prototype counterfactuals, and ordinary/input-weighted recoverability residuals.
Old training samples used by the offline diagnostic are not training replay.
Datasets, checkpoints, per-sample logits, and mutable run logs are not included.
