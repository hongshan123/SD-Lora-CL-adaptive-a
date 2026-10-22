#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_function_safe_pareto_t10_2gpu"
QUEUE_LOG="$PROJECT_ROOT/function_safe_pareto_t10_3datasets_2gpu_queue.log"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
C100_GPU_IDS="${C100_GPU_IDS:-0,2}"
INR_GPU_IDS="${INR_GPU_IDS:-3,4}"
CUB_GPU_IDS="${CUB_GPU_IDS:-5,6}"

if [ ! -f "$CONDA_SH" ]; then
    echo "Conda initialization script not found: $CONDA_SH" >&2
    exit 1
fi

source "$CONDA_SH"
conda activate sdlora
cd "$PROJECT_ROOT"
mkdir -p "$RUNTIME_DIR"

python - "$PROJECT_ROOT" "$RUNTIME_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime = Path(sys.argv[2])
specs = (
    ("c100", "c100_coordinate_stable_adaptive_a_seed1993_nccl.json", 1993),
    ("inr", "inr_coordinate_stable_adaptive_a_seed1995_nccl.json", 1995),
    ("cub", "cub_coordinate_stable_adaptive_a_seed1_nccl.json", 1),
)

for dataset, source_name, seed in specs:
    name = f"{dataset}_function_safe_pareto_seed{seed}_t10_i4_bs128_2gpu"
    config = json.loads((root / "exps" / source_name).read_text())
    config.update(
        {
            "prefix": name,
            "filepath": f"./{name.upper()}/",
            "device": ["0", "1"],
            "seed": [seed],
            "batch_size": 64,
            "max_tasks": 10,
            "sa_dual_head": False,
            "sa_hbd_enabled": False,
            "sa_adaptive_a_enabled": True,
            "sa_adaptive_a_strategy": "function_safe_pareto",
            "sa_adaptive_a_crossfit_interval": 4,
            "sa_functional_conflict_cosine": 0.05,
            "sa_functional_temperature": 2.0,
            "sa_resume": False,
        }
    )
    config.pop("sa_dual_head_schedule", None)
    (runtime / f"{name}.json").write_text(
        json.dumps(config, indent=4) + "\n"
    )
PY

experiments=(
    "c100|1993|$C100_GPU_IDS"
    "inr|1995|$INR_GPU_IDS"
    "cub|1|$CUB_GPU_IDS"
)

preflight_one() {
    local dataset="$1"
    local seed="$2"
    local gpu_ids="$3"
    local name="${dataset}_function_safe_pareto_seed${seed}_t10_i4_bs128_2gpu"
    local config="$RUNTIME_DIR/${name}.json"
    local log="$PROJECT_ROOT/${name}.log"
    local output_dir="$PROJECT_ROOT/${name^^}"
    local gpu_count

    IFS=',' read -r -a gpu_list <<< "$gpu_ids"
    gpu_count="${#gpu_list[@]}"
    if [ "$gpu_count" -ne 2 ]; then
        echo "$dataset requires exactly two GPUs; got $gpu_ids" >&2
        return 2
    fi
    if [ -e "$log" ] || [ -e "$output_dir" ]; then
        echo "Refusing to overwrite existing output for $name" >&2
        return 2
    fi
    if pgrep -f "[t]orchrun.*${name}.json" >/dev/null; then
        echo "Duplicate experiment process exists for $name" >&2
        return 2
    fi
    python - "$config" "$seed" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
seed = int(sys.argv[2])
assert config["seed"] == [seed]
assert config["batch_size"] == 64
assert config["max_tasks"] == 10
assert config["epochs"] == 20
assert config["optimizer"] == "sgd"
assert config["sa_dual_head"] is False
assert config["sa_hbd_enabled"] is False
assert config["sa_adaptive_a_enabled"] is True
assert config["sa_adaptive_a_strategy"] == "function_safe_pareto"
assert config["sa_adaptive_a_crossfit_interval"] == 4
assert config["sa_functional_conflict_cosine"] == 0.05
assert config["sa_functional_temperature"] == 2.0
assert config["sa_train_a_all_tasks"] is True
assert config["sa_cumulative_state"] is True
assert config["sa_cumulative_merge"] == "live_a_aggregate_b"
assert config["sa_live_a_coordinate_align"] is True
assert config["sa_live_a_absorb_mode"] == "bounded_norm_calibrated_absorb"
assert config["sa_coordinate_stable_transport"] is True
PY
    echo "preflight PASS dataset=$dataset GPUs=$gpu_ids config=$config"
}

for experiment in "${experiments[@]}"; do
    IFS='|' read -r dataset seed gpu_ids <<< "$experiment"
    preflight_one "$dataset" "$seed" "$gpu_ids" || exit $?
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
    local name="${dataset}_function_safe_pareto_seed${seed}_t10_i4_bs128_2gpu"
    local config="$RUNTIME_DIR/${name}.json"
    local log="$PROJECT_ROOT/${name}.log"
    local status

    {
        echo "===== $(date '+%F %T') START $name GPUs=$gpu_ids commit=$(git rev-parse HEAD) ====="
        echo "config_sha=$(sha256sum "$config" | awk '{print $1}')"
        CUDA_VISIBLE_DEVICES="$gpu_ids" torchrun --standalone \
            --nproc_per_node=2 main.py --config="$config"
        status=$?
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
