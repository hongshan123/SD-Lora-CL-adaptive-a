#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sdlora}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
NPROC="${NPROC:-$(awk -F',' '{print NF}' <<< "$GPU_IDS")}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CONFIG="exps/sa_sdlora_operator_stability_smoke_inr_seed1995.json"

cd "$PROJECT_ROOT"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_ENDPOINT
export PYTHONUNBUFFERED=1

name="$(basename "$CONFIG" .json)"
echo "===== $(date '+%F %T') START $name GPUs=$GPU_IDS ====="
torchrun --standalone --nproc_per_node="$NPROC" main.py --config="./$CONFIG" \
  > "${name}.log" 2>&1
status=$?
echo "===== $(date '+%F %T') END $name status=$status ====="
exit "$status"
