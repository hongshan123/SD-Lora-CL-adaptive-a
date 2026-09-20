#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-/home/hongzhijun/miniconda3/envs/sdlora/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname -- "$PYTHON_BIN")/torchrun}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.runtime_recoverability_formal_$RUN_TAG}"
QUEUE_LOG="${QUEUE_LOG:-$PROJECT_ROOT/recoverability_formal_${RUN_TAG}_queue.log}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
C100_GPUS="${C100_GPUS:-0,1}"
INR_GPUS="${INR_GPUS:-4,5}"
CUB_GPUS="${CUB_GPUS:-6,7}"
DATASETS="${DATASETS:-c100 inr cub}"

export C100_GPUS INR_GPUS CUB_GPUS DATASETS

if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$TORCHRUN_BIN" ]; then
    echo "Missing sdlora Python or torchrun: $PYTHON_BIN $TORCHRUN_BIN" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$RUNTIME_DIR/configs"

"$PYTHON_BIN" - "$PROJECT_ROOT" "$RUNTIME_DIR" "$RUN_TAG" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime = Path(sys.argv[2])
run_tag = sys.argv[3]
sys.path.insert(0, str(root))

from scripts.make_recoverability_validation_configs import build_stage_configs

available_datasets = (
    (
        "c100",
        "c100_coordinate_stable_adaptive_a_seed1993_nccl.json",
        1993,
        os.environ["C100_GPUS"],
    ),
    (
        "inr",
        "inr_coordinate_stable_adaptive_a_seed1995_nccl.json",
        1995,
        os.environ["INR_GPUS"],
    ),
    (
        "cub",
        "cub_coordinate_stable_adaptive_a_seed1_nccl.json",
        1,
        os.environ["CUB_GPUS"],
    ),
)
selected = os.environ["DATASETS"].split()
known = {entry[0] for entry in available_datasets}
if not selected or len(selected) != len(set(selected)) or set(selected) - known:
    raise ValueError("DATASETS must contain unique values from c100/inr/cub")
datasets = [entry for entry in available_datasets if entry[0] in selected]
manifest = []
for dataset, source_name, seed, gpu_ids in datasets:
    world_size = len(gpu_ids.split(","))
    if world_size not in (1, 2, 4) or 128 % world_size:
        raise ValueError(f"unsupported GPU list for {dataset}: {gpu_ids}")
    batch_size = 128 // world_size
    source = json.loads((root / "exps" / source_name).read_text())
    configs = build_stage_configs(
        source,
        budget=0.01,
        step_size=0.1,
        interval=4,
        sketch_rank=16,
    )
    for stage_index, (stage, config) in enumerate(configs.items(), start=1):
        name = (
            f"rga_{dataset}_s{stage_index}_{stage}_seed{seed}"
            f"_t10_bs128_{world_size}gpu_{run_tag}"
        )
        config.update(
            {
                "prefix": name,
                "filepath": f"./{name.upper()}/",
                "device": [str(index) for index in range(world_size)],
                "seed": [seed],
                "batch_size": batch_size,
                "max_tasks": 10,
                "sa_resume": False,
            }
        )
        config_path = runtime / "configs" / f"{name}.json"
        config_path.write_text(json.dumps(config, indent=4) + "\n")
        manifest.append(
            {
                "dataset": dataset,
                "stage_index": stage_index,
                "stage": stage,
                "seed": seed,
                "gpu_ids": gpu_ids,
                "world_size": world_size,
                "batch_size": batch_size,
                "name": name,
                "config": str(config_path),
                "output": str(root / name.upper()),
            }
        )
(runtime / "manifest.json").write_text(json.dumps(manifest, indent=4) + "\n")
PY

mapfile -t experiments < <(
    "$PYTHON_BIN" - "$RUNTIME_DIR/manifest.json" <<'PY'
import json
import sys

for entry in json.load(open(sys.argv[1])):
    print(
        "{dataset}|{stage_index}|{stage}|{seed}|{gpu_ids}|{world_size}|{batch_size}|{name}|{config}|{output}".format(
            **entry
        )
    )
PY
)

validate_config() {
    local config="$1" stage="$2" seed="$3" batch_size="$4"
    "$PYTHON_BIN" - "$config" "$stage" "$seed" "$batch_size" <<'PY'
import json
import sys

c = json.load(open(sys.argv[1]))
stage = sys.argv[2]
seed = int(sys.argv[3])
batch_size = int(sys.argv[4])
assert c["seed"] == [seed]
assert c["batch_size"] == batch_size
assert c["max_tasks"] == 10
assert c["init_epoch"] == 20 and c["epochs"] == 20
assert c["optimizer"] == "sgd" and c["lora_rank"] == 10
assert c["sa_adaptive_a_enabled"] is True
assert c["sa_adaptive_a_strategy"] == "recoverability"
assert c["sa_recoverability_stage"] == stage
assert c["sa_recoverability_budget"] == 0.01
assert c["sa_recoverability_step_size"] == 0.1
assert c["sa_recoverability_interval"] == 4
assert c["sa_recoverability_sketch_rank"] == 16
assert c["sa_recoverability_gammas"] == [0.0, 0.25, 0.5, 0.75, 1.0]
assert c["sa_train_a_all_tasks"] is True
assert c["sa_cumulative_state"] is True
assert c["sa_cumulative_merge"] == "live_a_aggregate_b"
assert c["sa_live_a_coordinate_align"] is True
assert c["sa_resume"] is False
PY
}

for experiment in "${experiments[@]}"; do
    IFS='|' read -r dataset stage_index stage seed gpu_ids world_size batch_size name config output <<< "$experiment"
    log="$PROJECT_ROOT/${name}.log"
    if [ -e "$log" ] || [ -e "$output" ]; then
        echo "Refusing to overwrite existing output for $name" >&2
        exit 2
    fi
    if pgrep -f "[t]orchrun.*${config}" >/dev/null; then
        echo "Duplicate experiment process exists for $name" >&2
        exit 2
    fi
    validate_config "$config" "$stage" "$seed" "$batch_size" || exit 2
    echo "preflight PASS dataset=$dataset stage=$stage_index:$stage GPUs=$gpu_ids world_size=$world_size batch=$batch_size"
done

if [ "$PREPARE_ONLY" = "1" ]; then
    echo "PREPARE_ONLY complete: $RUNTIME_DIR/manifest.json"
    exit 0
fi
if [ -e "$QUEUE_LOG" ]; then
    echo "Refusing to overwrite queue log: $QUEUE_LOG" >&2
    exit 2
fi

export HF_ENDPOINT PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

run_one() {
    local dataset="$1" stage_index="$2" stage="$3" seed="$4"
    local gpu_ids="$5" world_size="$6" name="$7" config="$8"
    local log="$PROJECT_ROOT/${name}.log"
    local status

    {
        echo "===== $(date '+%F %T') START $name dataset=$dataset stage=$stage_index:$stage GPUs=$gpu_ids commit=$(git rev-parse HEAD) ====="
        echo "config=$config config_sha=$(sha256sum "$config" | awk '{print $1}')"
        if CUDA_VISIBLE_DEVICES="$gpu_ids" "$TORCHRUN_BIN" --standalone \
            --nproc_per_node="$world_size" main.py --config="$config"; then
            status=0
        else
            status=$?
        fi
        echo "===== $(date '+%F %T') END $name status=$status ====="
        return "$status"
    } > "$log" 2>&1
}

run_dataset() {
    local wanted_dataset="$1"
    local status=0
    local experiment
    for experiment in "${experiments[@]}"; do
        IFS='|' read -r dataset stage_index stage seed gpu_ids world_size batch_size name config output <<< "$experiment"
        if [ "$dataset" != "$wanted_dataset" ]; then
            continue
        fi
        echo "===== $(date '+%F %T') DISPATCH dataset=$dataset stage=$stage_index:$stage GPUs=$gpu_ids ====="
        if ! run_one "$dataset" "$stage_index" "$stage" "$seed" "$gpu_ids" "$world_size" "$name" "$config"; then
            status=1
            echo "===== $(date '+%F %T') STOP dataset=$dataset failed_stage=$stage_index:$stage ====="
            break
        fi
    done
    return "$status"
}

echo "===== $(date '+%F %T') QUEUE START commit=$(git rev-parse HEAD) runtime=$RUNTIME_DIR =====" > "$QUEUE_LOG"
pids=()
for dataset in $DATASETS; do
    run_dataset "$dataset" >> "$QUEUE_LOG" 2>&1 &
    pids+=("$!")
    echo "launched dataset=$dataset pid=${pids[-1]}" >> "$QUEUE_LOG"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
echo "===== $(date '+%F %T') QUEUE END status=$status =====" >> "$QUEUE_LOG"
exit "$status"
