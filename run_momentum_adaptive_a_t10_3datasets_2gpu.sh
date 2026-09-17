#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-/home/hongzhijun/miniconda3/envs/sdlora/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname -- "$PYTHON_BIN")/torchrun}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.runtime_momentum_adaptive_a_t10_$RUN_TAG}"
QUEUE_LOG="${QUEUE_LOG:-$PROJECT_ROOT/momentum_adaptive_a_t10_${RUN_TAG}_queue.log}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/generate_momentum_adaptive_a_configs.py \
    "$PROJECT_ROOT" "$RUNTIME_DIR" "$RUN_TAG"

if [ "$PREPARE_ONLY" = "1" ]; then
    echo "PREPARE_ONLY complete: $RUNTIME_DIR/manifest.json"
    exit 0
fi
if [ ! -x "$TORCHRUN_BIN" ]; then
    echo "torchrun not executable: $TORCHRUN_BIN" >&2
    exit 1
fi

mapfile -t experiments < <(
    "$PYTHON_BIN" - "$RUNTIME_DIR/manifest.json" <<'PY'
import json
import sys

for entry in json.load(open(sys.argv[1])):
    print("{name}|{config}|{gpu_ids}|{output}".format(**entry))
PY
)

for experiment in "${experiments[@]}"; do
    IFS='|' read -r name config gpu_ids output <<< "$experiment"
    log="$PROJECT_ROOT/${name}.log"
    if [ -e "$log" ] || [ -e "$output" ]; then
        echo "Refusing to overwrite existing output for $name" >&2
        exit 2
    fi
    if pgrep -f "[t]orchrun.*${config}" >/dev/null; then
        echo "Duplicate experiment process exists for $name" >&2
        exit 2
    fi
    "$PYTHON_BIN" - "$config" <<'PY'
import json
import sys

c = json.load(open(sys.argv[1]))
assert c["batch_size"] == 64 and c["max_tasks"] == 10
assert c["init_epoch"] == 20 and c["epochs"] == 20
assert c["optimizer"] == "sgd" and c["lora_rank"] == 10
assert c["sa_adaptive_a_strategy"] == "impact_ratio"
assert c["sa_adaptive_a_gate_formula"] == "ratio"
assert c["sa_adaptive_a_gate_floor"] == 0.0
assert c["sa_adaptive_a_gate_momentum"] == 0.0
assert c["sa_live_a_absorb_mode"] == "operator_preserving_absorb"
assert c["sa_dual_head"] is False
assert c["sa_coordinate_stable_transport"] is False
PY
    echo "preflight PASS name=$name GPUs=$gpu_ids"
done

export HF_ENDPOINT PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

run_one() {
    local name="$1"
    local config="$2"
    local gpu_ids="$3"
    local log="$PROJECT_ROOT/${name}.log"
    local status

    {
        echo "===== $(date '+%F %T') START $name GPUs=$gpu_ids commit=$(git rev-parse HEAD) ====="
        echo "config=$config config_sha=$(sha256sum "$config" | awk '{print $1}')"
        if CUDA_VISIBLE_DEVICES="$gpu_ids" "$TORCHRUN_BIN" --standalone \
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
    IFS='|' read -r name config gpu_ids output <<< "$experiment"
    run_one "$name" "$config" "$gpu_ids" &
    pids+=("$!")
    echo "launched name=$name GPUs=$gpu_ids pid=${pids[-1]}" >> "$QUEUE_LOG"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
echo "===== $(date '+%F %T') QUEUE END status=$status =====" >> "$QUEUE_LOG"
exit "$status"
