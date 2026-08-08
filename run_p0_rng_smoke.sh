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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

control=exps/live_a_rng_smoke_control_c100_seed1.json
dual=exps/live_a_rng_smoke_dual_b_c100_seed1.json
control_log=live_a_rng_smoke_control.log
dual_log=live_a_rng_smoke_dual_b.log

echo "===== $(date '+%F %T') START control ====="
torchrun --standalone --nproc_per_node="$NPROC" main.py --config="./$control" > "$control_log" 2>&1
control_status=$?
echo "===== $(date '+%F %T') END control status=$control_status ====="

if [[ $control_status -eq 0 ]]; then
  echo "===== $(date '+%F %T') START dual-b ====="
  torchrun --standalone --nproc_per_node="$NPROC" main.py --config="./$dual" > "$dual_log" 2>&1
  dual_status=$?
  echo "===== $(date '+%F %T') END dual-b status=$dual_status ====="
else
  dual_status=99
fi

if [[ $control_status -ne 0 || $dual_status -ne 0 ]]; then
  echo "SMOKE FAIL: control=$control_status dual=$dual_status"
  exit 1
fi

control_hashes=$(grep '\[TrajectoryHash\]' "$control_log" | sed 's/.*hash=//' | sort)
dual_hashes=$(grep '\[TrajectoryHash\]' "$dual_log" | sed 's/.*hash=//' | sort)
echo "control hashes:"
echo "$control_hashes"
echo "dual hashes:"
echo "$dual_hashes"

if [[ "$control_hashes" != "$dual_hashes" ]]; then
  echo "SMOKE FAIL: trajectory hashes differ"
  exit 1
fi

if ! grep -q '\[DualHead\] task 0' "$dual_log" || ! grep -q '\[DualHead\] task 1' "$dual_log"; then
  echo "SMOKE FAIL: DualHead eval missing"
  exit 1
fi

echo "SMOKE PASS"
