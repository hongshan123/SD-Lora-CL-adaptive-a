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

# Stage A: capacity-matched comparison on ImageNet-R seed1995.
# union_svd_r4 vs gauge_r4, union_svd_r8 vs gauge_r8, then EXP-009/rank10
# gauge reference already exists (sa_cumulative_inr_seed1995_gauge.log).
configs=(
  exps/sa_cumulative_union_svd_r4_inr_seed1995.json
  exps/sa_cumulative_gauge_r4_inr_seed1995.json
  exps/sa_cumulative_union_svd_r8_inr_seed1995.json
  exps/sa_cumulative_gauge_r8_inr_seed1995.json
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
