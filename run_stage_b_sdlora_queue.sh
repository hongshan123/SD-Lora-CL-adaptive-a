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

# Original SD-LoRA paired baselines (rerun2 output dirs; rerun1 logs kept as
# .failed.log after the live_a_gradient_diagnostics guard bug was fixed).
configs=(
  exps/sdlora_inr_seed1995_proto_baseline_paired_rerun.json
  exps/sdlora_inr_seed1_proto_baseline_paired_rerun.json
  exps/sdlora_inr_seed2_proto_baseline_paired_rerun.json
  exps/sdlora_inr_seed3_proto_baseline_paired_rerun.json
  exps/sdlora_c100_seed1993_proto_baseline_paired_rerun.json
  exps/sdlora_c100_seed1_proto_baseline_paired_rerun.json
  exps/sdlora_c100_seed2_proto_baseline_paired_rerun.json
  exps/sdlora_c100_seed3_proto_baseline_paired_rerun.json
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
