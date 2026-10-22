#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL-coordinate"
CONDA_SH="/home/zhaoyang/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sdlora"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC="${NPROC:-4}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$PROJECT_ROOT"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

if [ -n "$(git status --porcelain)" ]; then
  echo "CoordinateStable CUB-SGD FAIL: working tree is not clean"
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "CoordinateStable CUB-SGD FAIL: jq is required"
  exit 1
fi

make_config() {
  local base="$1" seed="$2" prefix="$3" filepath="$4" output="$5"
  jq --arg prefix "$prefix" --arg filepath "$filepath" --argjson seed "$seed" '.prefix = $prefix
     | .seed = [$seed]
     | .filepath = $filepath
     | .device = ["0"]
     | .dist_backend = "nccl"
     | .sa_cumulative_state = true
     | .sa_cumulative_merge = "live_a_aggregate_b"
     | .sa_live_a_history_groups = 1
     | .sa_live_a_coordinate_align = true
     | .sa_coordinate_stable_transport = true
     | .sa_coordinate_transport_rank = .lora_rank
     | .sa_coordinate_transport_reg = 0.0001
     | .sa_coordinate_transport_min_gain = 0.0
     | .sa_deterministic_training = true
     | .sa_dual_head = true
     | .sa_dual_head_schedule = "B"
     | .sa_resume = false' "$base" > "$output"
}

run_one() {
  local seed="$1" base="$2" prefix="$3" filepath="$4"
  local logfile="${prefix}.log"
  local output_dir="${filepath%/}"
  local tmp_config status

  if [ -e "$output_dir" ] || [ -e "${output_dir}.lock" ]; then
    echo "CoordinateStable CUB-SGD FAIL: output exists for $prefix"
    return 1
  fi

  tmp_config="$(mktemp "/tmp/${prefix}.XXXXXX.json")"
  make_config "$base" "$seed" "$prefix" "$filepath" "$tmp_config"

  echo "===== $(date '+%F %T') START $prefix seed=$seed ====="
  echo "base=$base filepath=$filepath gpus=$GPU_IDS nproc=$NPROC"
  echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$tmp_config" | awk '{print $1}')"

  if torchrun --standalone --nproc_per_node="$NPROC" main.py --config="$tmp_config" > "$logfile" 2>&1; then
    status=0
  else
    status=$?
  fi

  rm -f "$tmp_config"
  echo "===== $(date '+%F %T') END $prefix status=$status ====="
  return "$status"
}

echo "CoordinateStable CUB-SGD queue start: $(date '+%F %T')"
echo "Protocol: frozen P5 SGD config plus CoordinateStable only"

for seed in 1 2 3; do
  run_one "$seed" "exps/p5_cub_livea_dual_b_seed${seed}_nccl.json" \
    "cub_coordinate_stable_sgd_seed${seed}_nccl" \
    "./CUB_COORDINATE_STABLE_SGD_SEED${seed}_NCCL/"
  sleep 5
done

echo "CoordinateStable CUB-SGD queue DONE: $(date '+%F %T')"
