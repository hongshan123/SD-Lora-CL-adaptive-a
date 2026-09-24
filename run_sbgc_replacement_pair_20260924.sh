#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: $0 {c100|inr|cub} GPU_IDS}"
gpu_ids="${2:?usage: $0 {c100|inr|cub} GPU_IDS}"
case "$dataset" in
  c100|inr|cub) ;;
  *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
esac
case "$gpu_ids" in
  0,1|4,5|6,7) ;;
  *) echo "GPU_IDS must be 0,1, 4,5 or 6,7" >&2; exit 2 ;;
esac

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
torchrun_bin=/home/hongzhijun/miniconda3/envs/sdlora/bin/torchrun

for arm in fisher additive; do
  config="exps/sbgc_replace_${dataset}_${arm}_20260924.json"
  log="sbgc_replace_${dataset}_${arm}_20260924.log"
  printf '%s START %s GPUs=%s\n' "$(date '+%F %T')" "$config" "$gpu_ids"
  "$torchrun_bin" --standalone --nproc_per_node=2 main.py \
    --config="$config" > "$log" 2>&1
  printf '%s END %s\n' "$(date '+%F %T')" "$config"
done
