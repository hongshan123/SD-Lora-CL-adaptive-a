#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL-coordinate"
CONDA_SH="/home/zhaoyang/miniconda3/etc/profile.d/conda.sh"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC="${NPROC:-4}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MAIN_PID_FILE="$PROJECT_ROOT/coordinate_stable_opabsorb_main_queue.pid"
MAIN_LOG="$PROJECT_ROOT/coordinate_stable_opabsorb_main_queue.log"

cd "$PROJECT_ROOT"
source "$CONDA_SH"
conda activate sdlora
export HF_ENDPOINT CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

if [ ! -f "$MAIN_PID_FILE" ]; then
  echo "follow-up FAIL: missing main queue PID file"
  exit 1
fi
main_pid="$(tr -d '[:space:]' < "$MAIN_PID_FILE")"
if ! [[ "$main_pid" =~ ^[0-9]+$ ]]; then
  echo "follow-up FAIL: invalid main queue PID: $main_pid"
  exit 1
fi

while kill -0 "$main_pid" 2>/dev/null; do
  echo "===== $(date '+%F %T') WAIT main queue pid=$main_pid ====="
  sleep 300
done

if ! grep -q "OPERATOR-PRESERVING MAIN QUEUE DONE" "$MAIN_LOG"; then
  echo "follow-up FAIL: main queue exited without completion marker"
  exit 1
fi

make_config() {
  local base="$1" seed="$2" prefix="$3" filepath="$4" output="$5"
  jq --arg prefix "$prefix" --arg filepath "$filepath" --argjson seed "$seed" '
    .prefix = $prefix
    | .seed = [$seed]
    | .filepath = $filepath
    | .device = ["0"]
    | .dist_backend = "nccl"
    | .sa_live_a_absorb_mode = "operator_preserving_absorb"
    | .sa_resume = false
  ' "$base" > "$output"
}

run_one() {
  local dataset="$1" seed="$2" base="$3"
  local prefix="${dataset}_coordinate_stable_opabsorb_seed${seed}_nccl"
  local filepath="./${dataset^^}_COORDINATE_STABLE_OPABSORB_SEED${seed}_NCCL/"
  local output_dir="${filepath%/}"
  local config

  if [ -e "$output_dir" ]; then
    echo "follow-up FAIL: output exists for $prefix: $output_dir"
    return 1
  fi
  config="$(mktemp "/tmp/${prefix}.XXXXXX.json")"
  make_config "$base" "$seed" "$prefix" "$filepath" "$config"

  echo "===== $(date '+%F %T') START $prefix ====="
  echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$config" | awk '{print $1}')"
  if torchrun --standalone --nproc_per_node="$NPROC" main.py \
    --config="$config" > "${prefix}.log" 2>&1; then
    status=0
  else
    status=$?
  fi
  rm -f "$config"
  echo "===== $(date '+%F %T') END $prefix status=$status ====="
  return "$status"
}

for seed in 1 2 3; do
  run_one c100 "$seed" \
    exps/c100_coordinate_stable_opabsorb_seed1993_nccl.json
  sleep 5
done

for seed in 1 2 3; do
  run_one inr "$seed" \
    exps/inr_coordinate_stable_opabsorb_seed1995_nccl.json
  sleep 5
done

for seed in 2 3; do
  run_one cub "$seed" \
    exps/cub_coordinate_stable_opabsorb_seed1_nccl.json
  sleep 5
done

echo "===== $(date '+%F %T') OPABSORB MULTISEED FOLLOW-UP DONE ====="
