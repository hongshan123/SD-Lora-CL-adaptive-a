#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_functional_halfspace_t10_2gpu"
QUEUE_LOG="$PROJECT_ROOT/functional_halfspace_t10_3datasets_2gpu_queue.log"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

if [ ! -f "$CONDA_SH" ]; then
    echo "Conda initialization script not found: $CONDA_SH" >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to generate runtime configs" >&2
    exit 1
fi
JQ_BIN="$(command -v jq)"

source "$CONDA_SH"
conda activate sdlora
cd "$PROJECT_ROOT"
mkdir -p "$RUNTIME_DIR"

make_config() {
    local source="$1"
    local name="$2"
    local output="$RUNTIME_DIR/${name}.json"
    local temporary="${output}.tmp"

    "$JQ_BIN" --arg name "$name" --arg filepath "./${name^^}/" '
        .prefix = $name
        | .filepath = $filepath
        | .device = ["0", "1"]
        | .batch_size = 64
        | .max_tasks = 10
        | .sa_dual_head = false
        | .sa_hbd_enabled = false
        | .sa_adaptive_a_enabled = true
        | .sa_adaptive_a_strategy = "functional_halfspace"
        | .sa_functional_conflict_tol = 1e-12
        | .sa_functional_normal_tol = 1e-12
        | .sa_functional_min_normal_fraction = 1e-4
        | .sa_resume = false
        | del(.sa_dual_head_schedule, .sa_adaptive_a_crossfit_interval)
    ' "$source" > "$temporary"
    mv "$temporary" "$output"
}

make_config \
    "exps/c100_coordinate_stable_adaptive_a_seed1993_nccl.json" \
    "c100_functional_halfspace_seed1993_t10_bs128_2gpu"
make_config \
    "exps/inr_coordinate_stable_adaptive_a_seed1995_nccl.json" \
    "inr_functional_halfspace_seed1995_t10_bs128_2gpu"
make_config \
    "exps/cub_coordinate_stable_adaptive_a_seed1_nccl.json" \
    "cub_functional_halfspace_seed1_t10_bs128_2gpu"

experiments=(
    "c100|1993|0,1"
    "inr|1995|2,3"
    "cub|1|4,5"
)

preflight_one() {
    local dataset="$1"
    local seed="$2"
    local gpu_ids="$3"
    local name="${dataset}_functional_halfspace_seed${seed}_t10_bs128_2gpu"
    local config="$RUNTIME_DIR/${name}.json"
    local log="$PROJECT_ROOT/${name}.log"
    local output_dir="$PROJECT_ROOT/${name^^}"

    if [ -e "$log" ] || [ -e "$output_dir" ]; then
        echo "Refusing to overwrite existing output for $name" >&2
        return 2
    fi
    if pgrep -f "[t]orchrun.*${config}" >/dev/null; then
        echo "Duplicate experiment process exists for $name" >&2
        return 2
    fi
    "$JQ_BIN" -e --argjson seed "$seed" '
        .seed == [$seed]
        and .batch_size == 64
        and .max_tasks == 10
        and .epochs == 20
        and .optimizer == "sgd"
        and .sa_dual_head == false
        and .sa_hbd_enabled == false
        and .sa_adaptive_a_enabled == true
        and .sa_adaptive_a_strategy == "functional_halfspace"
        and .sa_train_a_all_tasks == true
        and .sa_cumulative_state == true
        and .sa_cumulative_merge == "live_a_aggregate_b"
        and .sa_live_a_coordinate_align == true
        and .sa_live_a_absorb_mode == "bounded_norm_calibrated_absorb"
        and .sa_coordinate_stable_transport == true
    ' "$config" >/dev/null
    echo "preflight PASS dataset=$dataset GPUs=$gpu_ids config=$config"
}

for experiment in "${experiments[@]}"; do
    IFS='|' read -r dataset seed gpu_ids <<< "$experiment"
    preflight_one "$dataset" "$seed" "$gpu_ids"
done

if [ "$PREPARE_ONLY" = "1" ]; then
    echo "PREPARE_ONLY complete"
    exit 0
fi

export HF_ENDPOINT PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

run_one() {
    local dataset="$1"
    local seed="$2"
    local gpu_ids="$3"
    local name="${dataset}_functional_halfspace_seed${seed}_t10_bs128_2gpu"
    local config="$RUNTIME_DIR/${name}.json"
    local log="$PROJECT_ROOT/${name}.log"
    local status

    {
        echo "===== $(date '+%F %T') START $name GPUs=$gpu_ids commit=$(git rev-parse HEAD) ====="
        echo "config_sha=$(sha256sum "$config" | awk '{print $1}')"
        if CUDA_VISIBLE_DEVICES="$gpu_ids" torchrun --standalone \
            --nproc_per_node=2 main.py --config="$config"; then
            status=0
        else
            status=$?
        fi
        echo "===== $(date '+%F %T') END $name GPUs=$gpu_ids status=$status ====="
        return "$status"
    } > "$log" 2>&1
}

echo "===== $(date '+%F %T') QUEUE START commit=$(git rev-parse HEAD) =====" > "$QUEUE_LOG"
pids=()
for experiment in "${experiments[@]}"; do
    IFS='|' read -r dataset seed gpu_ids <<< "$experiment"
    run_one "$dataset" "$seed" "$gpu_ids" >> "$QUEUE_LOG" 2>&1 &
    pids+=("$!")
    echo "launched dataset=$dataset GPUs=$gpu_ids pid=${pids[-1]}" >> "$QUEUE_LOG"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
echo "===== $(date '+%F %T') QUEUE END status=$status =====" >> "$QUEUE_LOG"
exit "$status"
