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
  echo "Core ablation queue FAIL: working tree is not clean"
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "Core ablation queue FAIL: jq is required"
  exit 1
fi

make_config() {
  local base="$1" seed="$2" prefix="$3" filepath="$4"
  local coordinate_align="$5" prototype_transport="$6" output="$7"
  jq --arg prefix "$prefix" --arg filepath "$filepath" \
     --argjson seed "$seed" --argjson coordinate_align "$coordinate_align" \
     --argjson prototype_transport "$prototype_transport" \
     '.prefix = $prefix
      | .seed = [$seed]
      | .filepath = $filepath
      | .device = ["0"]
      | .dist_backend = "nccl"
      | .sa_cumulative_state = true
      | .sa_cumulative_merge = "live_a_aggregate_b"
      | .sa_live_a_history_groups = 1
      | .sa_live_a_coordinate_align = $coordinate_align
      | .sa_coordinate_stable_transport = $prototype_transport
      | .sa_coordinate_transport_rank = .lora_rank
      | .sa_coordinate_transport_reg = 0.0001
      | .sa_coordinate_transport_min_gain = 0.0
      | .sa_deterministic_training = true
      | .sa_dual_head = true
      | .sa_dual_head_schedule = "B"
      | .sa_resume = false' "$base" > "$output"
}

is_complete() {
  local output_dir="$1" logfile="$2"
  [ -f "$output_dir/run_manifest.json" ] \
    && [ -f "$logfile" ] \
    && grep -q "Forgetting (CNN):" "$logfile"
}

run_one() {
  local variant="$1" dataset="$2" seed="$3" base="$4"
  local coordinate_align="$5" prototype_transport="$6"
  local dataset_upper prefix output_dir filepath logfile tmp_config status
  dataset_upper="$(printf '%s' "$dataset" | tr '[:lower:]' '[:upper:]')"
  prefix="${dataset}_abl_${variant}_seed${seed}_nccl"
  output_dir="${dataset_upper}_ABL_${variant^^}_SEED${seed}_NCCL"
  filepath="./${output_dir}/"
  logfile="${prefix}.log"

  if is_complete "$output_dir" "$logfile"; then
    echo "===== $(date '+%F %T') SKIP complete $prefix ====="
    return 0
  fi
  if [ -e "$output_dir" ] || [ -e "${output_dir}.lock" ] || [ -e "$logfile" ]; then
    echo "Core ablation queue FAIL: incomplete artifact exists for $prefix"
    return 1
  fi

  tmp_config="$(mktemp "/tmp/${prefix}.XXXXXX.json")"
  make_config "$base" "$seed" "$prefix" "$filepath" \
    "$coordinate_align" "$prototype_transport" "$tmp_config"
  echo "===== $(date '+%F %T') START $prefix variant=$variant dataset=$dataset seed=$seed ====="
  echo "base=$base filepath=$filepath gpus=$GPU_IDS nproc=$NPROC"
  echo "coordinate_align=$coordinate_align prototype_transport=$prototype_transport"
  echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$tmp_config" | awk '{print $1}')"

  if torchrun --standalone --nproc_per_node="$NPROC" \
      main.py --config="$tmp_config" > "$logfile" 2>&1; then
    status=0
  else
    status=$?
  fi
  if [ -d "$output_dir" ]; then
    cp "$tmp_config" "$output_dir/effective_config.json"
  fi
  rm -f "$tmp_config"
  echo "===== $(date '+%F %T') END $prefix status=$status ====="
  return "$status"
}

base_config() {
  local dataset="$1" seed="$2"
  case "$dataset" in
    cub) echo "exps/p5_cub_livea_dual_b_seed${seed}_nccl.json" ;;
    inr) echo "exps/p3_inr_livea_dual_b_seed${seed}_nccl.json" ;;
    c100) echo "exps/p3_c100_livea_dual_b_seed${seed}_nccl.json" ;;
    *) echo "unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

run_pair() {
  local dataset="$1" seed="$2" base
  base="$(base_config "$dataset" "$seed")"
  run_one coordinate_only "$dataset" "$seed" "$base" true false
  sleep 5
  run_one transport_only "$dataset" "$seed" "$base" false true
  sleep 5
}

echo "Core CoordinateStable ablation queue start: $(date '+%F %T')"
echo "A0 and A3 already complete; running only A1 coordinate-only and A2 transport-only"
echo "Protocol: frozen P3/P5 configs, seeds 1-3, GPU $GPU_IDS, $NPROC-rank NCCL"

run_pair cub 1
for seed in 2 3; do
  run_pair cub "$seed"
done
for seed in 1 2 3; do
  run_pair inr "$seed"
done
for seed in 1 2 3; do
  run_pair c100 "$seed"
done

echo "Core CoordinateStable ablation queue DONE: $(date '+%F %T')"
