#!/usr/bin/env bash
set +e

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL"
CONDA_SH="/home/zhaoyang/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sdlora"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
NPROC="${NPROC:-$(awk -F',' '{print NF}' <<< "$GPU_IDS")}" 
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$PROJECT_ROOT" || exit 1
source "$CONDA_SH" || exit 1
conda activate "$CONDA_ENV" || exit 1

export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1

VISIBLE_CUDA_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "$VISIBLE_CUDA_COUNT" -lt 1 ]]; then
  echo "No usable CUDA devices visible under CUDA_VISIBLE_DEVICES=${GPU_IDS}."
  exit 1
fi
if [[ "$NPROC" -gt "$VISIBLE_CUDA_COUNT" ]]; then
  NPROC="$VISIBLE_CUDA_COUNT"
fi

configs=(
  exps/sdlora_inr_seed1995_proto_baseline.json
  exps/proto_routed_sdlora_inr_seed1995.json
  exps/sdlora_c100_seed1993_proto_baseline.json
  exps/proto_routed_sdlora_c100_seed1993.json
)

overall_status=0
for cfg in "${configs[@]}"; do
  name=$(basename "$cfg" .json)
  echo "===== $(date '+%F %T') START $name ====="
  torchrun --standalone --nproc_per_node="$NPROC" main.py --config="./$cfg" > "${name}.log" 2>&1
  status=$?
  echo "===== $(date '+%F %T') END $name status=$status ====="
  if [[ $status -ne 0 && $overall_status -eq 0 ]]; then
    overall_status=$status
  fi
  sleep 5
done

exit "$overall_status"
