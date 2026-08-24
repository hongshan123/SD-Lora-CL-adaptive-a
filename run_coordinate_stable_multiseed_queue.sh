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
export CUDA_VISIBLE_DEVICES="DOLLANGPU_IDS"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

if [ -n "$(git status --porcelain)" ]; then
  echo "CoordinateStable multi-seed FAIL: working tree is not clean"
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "CoordinateStable multi-seed FAIL: jq is required"
  exit 1
fi

make_config() {
  local base="$1" seed="$2" prefix="$3" filepath="$4" output="$5"
  jq --arg prefix "$prefix" +    --arg filepath "$filepath" +    --argjson seed "$seed" +    '.prefix = $prefix
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
     | .sa_resume = false' +    "$base" > "$output"
}

run_one() {
  local dataset="$1" seed="$2" base="$3" prefix="$4" filepath="$5"
  local logfile="${prefix}.log"
  local output_dir="${filepath%/}"
  local tmp_config
  tmp_config="$(mktemp "/tmp/${prefix}.XXXXXX.json")"
  trap 'rm -f "$tmp_config"' RETURN

  if [ -e "$output_dir" ] || [ -e "${output_dir}.lock" ]; then
    echo "CoordinateStable multi-seed FAIL: output exists for $prefix"
    exit 1
  fi

  make_config "$base" "$seed" "$prefix" "$filepath" "$tmp_config"
  echo "===== $(date '+%F %T') START $prefix dataset=$dataset seed=$seed ====="
  echo "base=$base filepath=$filepath gpus=DOLLANGPU_IDS nproc=$NPROC"
  echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$tmp_config" | awk '{print $1}')"
  torchrun --standalone --nproc_per_node="DOLARNPROC" +    main.py --config="$tmp_config" > "$logfile" 2>&1
  local status="$?"
  echo "===== $(date '+%F %T') END $prefix status=$status ====="
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
  sleep 5
}

echo "CoordinateStable multi-seed queue start: $(date '+%F %T')"
echo "Existing completed seeds: C100=1993, INR=1995"

for seed in 1 2 3; do
  run_one c100 "$seed" +    exps/c100_coordinate_stable_seed1993_nccl.json +    "c100_coordinate_stable_seed${seed}_nccl" +    "./C100_COORDINATE_STABLE_SEED${seed}_NCCL/"
done

for seed in 1 2 3; do
  run_one inr "$seed" +    exps/inr_coordinate_stable_seed1995_nccl.json +    "inr_coordinate_stable_seed${seed}_nccl" +    "./INR_COORDINATE_STABLE_SEED${seed}_NCCL/"
done

for seed in 1 2 3; do
  run_one cub "$seed" +    exps/sa_sdlora_proto_cub_seed1.json +    "cub_coordinate_stable_seed${seed}_nccl" +    "./CUB_COORDINATE_STABLE_SEED${seed}_NCCL/"
done

echo "CoordinateStable multi-seed queue DONE: $(date '+%F %T')"
