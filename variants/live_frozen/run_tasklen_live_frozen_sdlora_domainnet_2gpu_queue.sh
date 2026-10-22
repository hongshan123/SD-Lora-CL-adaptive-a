#!/usr/bin/env bash
# T=5/10/20/40 Live-A vs Frozen-A vs SD-LoRA on four datasets.
# Each run uses 2 GPUs with per-GPU batch size 64 (effective batch size 128).
# Physical GPUs 2 and 3 are intentionally excluded.
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sdlora}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.runtime_tasklen_pair_2gpu_$RUN_TAG}"
QUEUE_LOG="${QUEUE_LOG:-$PROJECT_ROOT/tasklen_pair_2gpu_${RUN_TAG}_queue.log}"

# Three independent two-GPU slots, excluding unstable GPUs 2 and 3.
GPU_PAIRS=("0,1" "4,5" "6,7")
WORLD_SIZE=2
BATCH_SIZE=64

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
mkdir -p "$RUNTIME_DIR"

python scripts/generate_tasklen_pair_configs.py \
    "$PROJECT_ROOT" "$RUNTIME_DIR" "$BATCH_SIZE" "$WORLD_SIZE"

mapfile -t JOBS < <(python - "$RUNTIME_DIR/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
for item in manifest:
    print("{}|{}|{}|{}".format(item["name"], item["dataset"], item["tasks"], item["config"]))
PY
)

if [ "${#JOBS[@]}" -ne 48 ]; then
    echo "expected 48 jobs, got ${#JOBS[@]}" >&2
    exit 2
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

run_one() {
    local name="$1" config="$2" gpu_ids="$3" log="$PROJECT_ROOT/${name}.log"
    {
        echo "===== $(date '+%F %T') START $name GPUs=$gpu_ids commit=$(git rev-parse HEAD) ====="
        echo "config=$config config_sha=$(sha256sum "$config" | awk '{print $1}')"
        CUDA_VISIBLE_DEVICES="$gpu_ids" torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
            main.py --config="$config"
        local status=$?
        echo "===== $(date '+%F %T') END $name GPUs=$gpu_ids status=$status ====="
        return "$status"
    } > "$log" 2>&1
}

echo "===== $(date '+%F %T') QUEUE START commit=$(git rev-parse HEAD) jobs=${#JOBS[@]} world_size=$WORLD_SIZE batch_size=$BATCH_SIZE gpu_pairs=${GPU_PAIRS[*]} =====" > "$QUEUE_LOG"
overall_status=0
next=0
wave=0
while [ "$next" -lt "${#JOBS[@]}" ]; do
    pids=()
    slot=0
    while [ "$slot" -lt "${#GPU_PAIRS[@]}" ] && [ "$next" -lt "${#JOBS[@]}" ]; do
        IFS='|' read -r name dataset tasks config <<< "${JOBS[$next]}"
        gpu_ids="${GPU_PAIRS[$slot]}"
        run_one "$name" "$config" "$gpu_ids" &
        pid=$!
        pids+=("$pid")
        echo "launched wave=$wave name=$name dataset=$dataset tasks=$tasks GPUs=$gpu_ids pid=$pid" >> "$QUEUE_LOG"
        next=$((next + 1))
        slot=$((slot + 1))
    done
    wave_status=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            wave_status=1
        fi
    done
    echo "===== $(date '+%F %T') WAVE END wave=$wave status=$wave_status =====" >> "$QUEUE_LOG"
    if [ "$wave_status" -ne 0 ]; then
        overall_status=1
        echo "wave $wave failed; continuing remaining jobs" >> "$QUEUE_LOG"
    fi
    wave=$((wave + 1))
done
echo "===== $(date '+%F %T') QUEUE END status=$overall_status =====" >> "$QUEUE_LOG"
exit "$overall_status"
