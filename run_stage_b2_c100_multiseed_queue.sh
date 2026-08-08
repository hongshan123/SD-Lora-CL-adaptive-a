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

# Stage B2 Schedule B: C100 seed1993 then paired multi-seed seeds 1/2/3.
configs=(
  exps/live_a_dual_head_c100_seed1993_b.json
  exps/live_a_dual_head_inr_seed1_b.json
  exps/live_a_dual_head_inr_seed2_b.json
  exps/live_a_dual_head_inr_seed3_b.json
  exps/live_a_dual_head_c100_seed1_b.json
  exps/live_a_dual_head_c100_seed2_b.json
  exps/live_a_dual_head_c100_seed3_b.json
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
