#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/hongzhijun/causal_cub_20260923}
REPO=${REPO:-/home/hongzhijun/SD-Lora-CL-adaptive-a}
TORCHRUN=${TORCHRUN:-/home/hongzhijun/miniconda3/envs/sdlora/bin/torchrun}
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${GPU_IDS:-0,1}
export PYTHONPATH="$REPO"
cd "$REPO"

for task in 2 6; do
  for mode in stable raw frozen live; do
    result="$ROOT/results/t${task}_${mode}.json"
    log="$ROOT/results/t${task}_${mode}.log"
    if [[ -e "$result" ]]; then
      echo "Refusing to overwrite $result" >&2
      exit 1
    fi
    echo "$(date -Is) START task=$task mode=$mode" | tee -a "$ROOT/queue.log"
    "$TORCHRUN" --standalone --nproc_per_node=2 \
      "$ROOT/causal_subspace_intervention.py" \
      --config "$ROOT/sbgc_attribution_cub_t2_20260923.json" \
      --source "$ROOT/source_t${task}" \
      --demand "$ROOT/demand_t${task}.pt" \
      --output "$result" --mode "$mode" > "$log" 2>&1
    echo "$(date -Is) END task=$task mode=$mode" | tee -a "$ROOT/queue.log"
  done
done
