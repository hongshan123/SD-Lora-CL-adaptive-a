#!/usr/bin/env bash
# Three-dataset, three-seed CUO queue: one deterministic two-GPU worker per dataset.
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_cuo_lowrank_multiseed/$RUN_TAG"
QUEUE_LOG="$PROJECT_ROOT/cuo_lowrank_r10_multiseed_3datasets_${RUN_TAG}.log"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

SEEDS=(1 2 3)
DATASETS=(c100 inr cub)
GPU_PAIRS=(0,1 4,5 6,7)

if [ ! -f "$CONDA_SH" ]; then
    echo "Conda initialization script not found: $CONDA_SH" >&2
    exit 2
fi

source "$CONDA_SH"
conda activate sdlora
cd "$PROJECT_ROOT"
mkdir -p "$RUNTIME_DIR"

if ! python - "$PROJECT_ROOT" "$RUNTIME_DIR" "$RUN_TAG" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime = Path(sys.argv[2])
run_tag = sys.argv[3]
sources = {
    "c100": "cuo_lowrank_r10_c100_seed1993.json",
    "inr": "cuo_lowrank_r10_inr_seed1995.json",
    "cub": "cuo_lowrank_r10_cub_seed1.json",
}

for dataset, source_name in sources.items():
    source = json.loads((root / "exps" / source_name).read_text())
    for seed in (1, 2, 3):
        name = f"cuo_lowrank_r10_{dataset}_seed{seed}_t10_bs128_2gpu_{run_tag}"
        config = dict(source)
        config.update(
            {
                "prefix": name,
                "filepath": f"./{name.upper()}/",
                "device": ["0", "1"],
                "seed": [seed],
                "batch_size": 64,
                "sa_resume": False,
            }
        )
        (runtime / f"{name}.json").write_text(json.dumps(config, indent=4) + "\n")
PY
then
    echo "Failed to generate CUO runtime configs" >&2
    exit 2
fi

experiment_name() {
    local dataset="$1" seed="$2"
    printf 'cuo_lowrank_r10_%s_seed%s_t10_bs128_2gpu_%s' "$dataset" "$seed" "$RUN_TAG"
}

preflight_one() {
    local dataset="$1" seed="$2" gpu_ids="$3"
    local name config log output_dir
    name="$(experiment_name "$dataset" "$seed")"
    config="$RUNTIME_DIR/${name}.json"
    log="$PROJECT_ROOT/${name}.log"
    output_dir="$PROJECT_ROOT/${name^^}"
    if [ -e "$log" ] || [ -e "$output_dir" ]; then
        echo "Refusing to overwrite existing output for $name" >&2
        return 2
    fi
    if pgrep -f "[t]orchrun.*${name}.json" >/dev/null; then
        echo "Duplicate experiment process exists for $name" >&2
        return 2
    fi
    if ! python - "$config" "$seed" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
seed = int(sys.argv[2])
assert config["seed"] == [seed]
assert config["batch_size"] == 64
assert config["epochs"] == 20
assert config["optimizer"] == "sgd"
assert config["lora_rank"] == config["sa_cumulative_rank"] == 10
assert config["sa_cumulative_state"] is True
assert config["sa_cumulative_merge"] == "cuo_lowrank"
assert config["sa_cuo_lambda"] == 1e-5
assert config["sa_use_prototype_classifier"] is True
assert config.get("sa_dual_head", False) is False
assert config.get("sa_hbd_enabled", False) is False
assert config.get("sa_coordinate_stable_transport", False) is False
assert config.get("sa_adaptive_a_enabled", False) is False
PY
    then
        echo "Protocol validation failed for $name" >&2
        return 2
    fi
    echo "preflight PASS name=$name GPUs=$gpu_ids config=$config"
}

for index in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$index]}"
    gpu_ids="${GPU_PAIRS[$index]}"
    for seed in "${SEEDS[@]}"; do
        preflight_one "$dataset" "$seed" "$gpu_ids" || exit $?
    done
done

if [ "$PREPARE_ONLY" = "1" ]; then
    echo "PREPARE_ONLY complete tag=$RUN_TAG"
    exit 0
fi

export HF_ENDPOINT PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

run_one() {
    local dataset="$1" seed="$2" gpu_ids="$3"
    local name config log status pid
    name="$(experiment_name "$dataset" "$seed")"
    config="$RUNTIME_DIR/${name}.json"
    log="$PROJECT_ROOT/${name}.log"
    {
        echo "===== $(date '+%F %T') START $name GPUs=$gpu_ids commit=$(git rev-parse HEAD) ====="
        echo "config_sha=$(sha256sum "$config" | awk '{print $1}')"
    } > "$log"
    nohup env CUDA_VISIBLE_DEVICES="$gpu_ids" torchrun --standalone --nproc_per_node=2 \
        main.py --config="$config" >> "$log" 2>&1 &
    pid=$!
    echo "launched name=$name GPUs=$gpu_ids pid=$pid"
    if wait "$pid"; then
        status=0
    else
        status=$?
    fi
    echo "===== $(date '+%F %T') END $name GPUs=$gpu_ids status=$status =====" >> "$log"
    return "$status"
}

run_dataset() {
    local dataset="$1" gpu_ids="$2"
    local seed
    for seed in "${SEEDS[@]}"; do
        run_one "$dataset" "$seed" "$gpu_ids" || return $?
    done
}

echo "===== $(date '+%F %T') QUEUE START tag=$RUN_TAG commit=$(git rev-parse HEAD) =====" > "$QUEUE_LOG"
pids=()
for index in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$index]}"
    gpu_ids="${GPU_PAIRS[$index]}"
    run_dataset "$dataset" "$gpu_ids" >> "$QUEUE_LOG" 2>&1 &
    pids+=("$!")
    echo "launched dataset=$dataset GPUs=$gpu_ids pid=${pids[-1]}" >> "$QUEUE_LOG"
done

overall_status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        status=0
    else
        status=$?
    fi
    echo "===== $(date '+%F %T') DATASET END ${DATASETS[$index]} status=$status =====" >> "$QUEUE_LOG"
    if [ "$status" -ne 0 ]; then
        overall_status=1
    fi
done
echo "===== $(date '+%F %T') QUEUE END status=$overall_status =====" >> "$QUEUE_LOG"
exit "$overall_status"
