#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-/home/hongzhijun/miniconda3/envs/sdlora/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname -- "$PYTHON_BIN")/torchrun}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')_retry}"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.runtime_momentum_adaptive_a_missing_$RUN_TAG}"
QUEUE_LOG="${QUEUE_LOG:-$PROJECT_ROOT/momentum_adaptive_a_missing_${RUN_TAG}_queue.log}"
TASK_SPECS="${TASK_SPECS:-c100:20,c100:50,inr:10,inr:20,inr:50}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" - "$PROJECT_ROOT" "$RUNTIME_DIR" "$RUN_TAG" "$TASK_SPECS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
runtime = Path(sys.argv[2]).resolve()
run_tag = sys.argv[3]
specs = sys.argv[4].split(",")

sys.path.insert(0, str(root))
from scripts.generate_momentum_adaptive_a_configs import DATASETS, build_config

requested = []
for item in specs:
    dataset, raw_tasks = item.split(":", 1)
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset: {dataset}")
    tasks = int(raw_tasks)
    if tasks not in (5, 10, 20, 50):
        raise SystemExit(f"unsupported task count: {tasks}")
    requested.append((dataset, tasks))

if len(set(requested)) != len(requested):
    raise SystemExit("duplicate task specification")

runtime.joinpath("configs").mkdir(parents=True, exist_ok=True)
runtime.joinpath("results").mkdir(parents=True, exist_ok=True)
manifest = []
for dataset_key, task_count in requested:
    spec = DATASETS[dataset_key]
    source = json.loads((root / "exps" / spec["source"]).read_text())
    name, config = build_config(
        source, dataset_key, runtime, run_tag, task_count=task_count
    )
    config_path = runtime / "configs" / f"{name}.json"
    config_path.write_text(json.dumps(config, indent=4) + "\n")
    manifest.append(
        {
            "name": name,
            "dataset": dataset_key,
            "tasks": task_count,
            "seed": spec["seed"],
            "gpu_ids": spec["gpu_ids"],
            "config": str(config_path),
            "output": config["filepath"],
        }
    )
runtime.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"generated {len(manifest)} missing-task configs in {runtime}")
PY

if [ "$PREPARE_ONLY" = "1" ]; then
    echo "PREPARE_ONLY complete: $RUNTIME_DIR/manifest.json"
    exit 0
fi

if [ ! -x "$TORCHRUN_BIN" ]; then
    echo "torchrun not executable: $TORCHRUN_BIN" >&2
    exit 1
fi

validate_config() {
    local config="$1"
    "$PYTHON_BIN" - "$config" <<'PY'
import json
import sys

c = json.load(open(sys.argv[1]))
assert c["batch_size"] == 64
assert c["max_tasks"] in (5, 10, 20, 50)
assert c["init_epoch"] == 20 and c["epochs"] == 20
assert c["optimizer"] == "sgd" and c["lora_rank"] == 10
assert c["sa_adaptive_a_enabled"] is True
assert c["sa_adaptive_a_strategy"] == "impact_ratio"
assert c["sa_adaptive_a_gate_formula"] == "ratio"
assert c["sa_adaptive_a_gate_floor"] == 0.0
assert c["sa_adaptive_a_gate_momentum"] == 0.0
assert c["sa_live_a_absorb_mode"] == "operator_preserving_absorb"
assert c["sa_dual_head"] is False
assert c["sa_coordinate_stable_transport"] is False
PY
}

mapfile -t experiments < <(
    "$PYTHON_BIN" - "$RUNTIME_DIR/manifest.json" <<'PY'
import json
import sys

for entry in json.load(open(sys.argv[1])):
    print("{dataset}|{tasks}|{name}|{config}|{gpu_ids}|{output}".format(**entry))
PY
)

for experiment in "${experiments[@]}"; do
    IFS='|' read -r dataset tasks name config gpu_ids output <<< "$experiment"
    log="$PROJECT_ROOT/${name}.log"
    if [ -e "$log" ] || [ -e "$output" ]; then
        echo "Refusing to overwrite existing output for $name" >&2
        exit 2
    fi
    validate_config "$config"
    echo "preflight PASS dataset=$dataset tasks=$tasks GPUs=$gpu_ids"
done

export HF_ENDPOINT PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

run_one() {
    local dataset="$1"
    local tasks="$2"
    local name="$3"
    local config="$4"
    local gpu_ids="$5"
    local log="$PROJECT_ROOT/${name}.log"
    local status

    {
        echo "===== $(date '+%F %T') START $name dataset=$dataset tasks=$tasks GPUs=$gpu_ids commit=$(git rev-parse HEAD) ====="
        echo "config=$config config_sha=$(sha256sum "$config" | awk '{print $1}')"
        if CUDA_VISIBLE_DEVICES="$gpu_ids" "$TORCHRUN_BIN" --standalone \
            --nproc_per_node=2 main.py --config="$config"; then
            status=0
        else
            status=$?
        fi
        echo "===== $(date '+%F %T') END $name dataset=$dataset tasks=$tasks GPUs=$gpu_ids status=$status ====="
        return "$status"
    } > "$log" 2>&1
}

run_dataset() {
    local wanted_dataset="$1"
    local status=0
    for experiment in "${experiments[@]}"; do
        IFS='|' read -r dataset tasks name config gpu_ids output <<< "$experiment"
        if [ "$dataset" != "$wanted_dataset" ]; then
            continue
        fi
        if ! run_one "$dataset" "$tasks" "$name" "$config" "$gpu_ids"; then
            status=1
            break
        fi
    done
    return "$status"
}

echo "===== $(date '+%F %T') QUEUE START commit=$(git rev-parse HEAD) specs=$TASK_SPECS =====" > "$QUEUE_LOG"
pids=()
for dataset in c100 inr cub; do
    if printf '%s\n' "${experiments[@]}" | grep -q "^${dataset}|"; then
        run_dataset "$dataset" >> "$QUEUE_LOG" 2>&1 &
        pids+=("$!")
        echo "launched dataset=$dataset pid=${pids[-1]}" >> "$QUEUE_LOG"
    fi
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
echo "===== $(date '+%F %T') QUEUE END status=$status =====" >> "$QUEUE_LOG"
exit "$status"
