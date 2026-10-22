#!/usr/bin/env bash
# T=5/10/20/50 Frozen-A, Live-A, and Adaptive-A on C100, INR, CUB, DomainNet.
# Three independent two-GPU slots: 0,1; 4,5; 6,7. GPUs 2 and 3 are excluded.
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sdlora}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.runtime_tasklen_fla_2gpu_$RUN_TAG}"
QUEUE_LOG="${QUEUE_LOG:-$PROJECT_ROOT/tasklen_fla_2gpu_${RUN_TAG}.log}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python scripts/tasklen_fla_queue.py \
  --project-root "$PROJECT_ROOT" \
  --runtime-dir "$RUNTIME_DIR" \
  --run-tag "$RUN_TAG" \
  --batch-size 64 \
  --world-size 2 \
  --gpu-pairs '0,1;4,5;6,7' \
  > "$QUEUE_LOG" 2>&1
