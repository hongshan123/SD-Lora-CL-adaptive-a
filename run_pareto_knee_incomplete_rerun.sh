#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_pareto_knee_incomplete_rerun"
QUEUE_LOG="$PROJECT_ROOT/pareto_knee_incomplete_rerun_queue.log"

source "$CONDA_SH"
conda activate sdlora
cd "$PROJECT_ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1
mkdir -p "$RUNTIME_DIR"

python - "$PROJECT_ROOT" "$RUNTIME_DIR" > "$QUEUE_LOG" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime = Path(sys.argv[2])
items = [
    ("c100", "c100_coordinate_stable_adaptive_a_seed1993_nccl.json", 100, 1993, 5, 3),
    ("c100", "c100_coordinate_stable_adaptive_a_seed1993_nccl.json", 100, 1993, 10, 4),
    ("c100", "c100_coordinate_stable_adaptive_a_seed1993_nccl.json", 100, 1993, 20, 6),
    ("inr", "inr_coordinate_stable_adaptive_a_seed1995_nccl.json", 200, 1995, 5, 7),
    ("inr", "inr_coordinate_stable_adaptive_a_seed1995_nccl.json", 200, 1995, 10, 6),
    ("inr", "inr_coordinate_stable_adaptive_a_seed1995_nccl.json", 200, 1995, 20, 3),
    ("cub", "cub_coordinate_stable_adaptive_a_seed1_nccl.json", 200, 1, 20, 4),
]

for dataset, source_name, class_count, seed, task_count, gpu in items:
    increment = class_count // task_count
    name = f"{dataset}_pareto_knee_seed{seed}_t{task_count}_i4_bs128_rerun"
    config = json.loads((root / "exps" / source_name).read_text())
    config.update(
        {
            "prefix": name,
            "filepath": f"./{name.upper()}/",
            "device": ["0"],
            "seed": [seed],
            "init_cls": increment,
            "increment": increment,
            "batch_size": 128,
            "max_tasks": task_count,
            "sa_dual_head": False,
            "sa_adaptive_a_enabled": True,
            "sa_adaptive_a_strategy": "pareto_knee",
            "sa_adaptive_a_crossfit_interval": 4,
            "sa_resume": False,
        }
    )
    config.pop("sa_dual_head_schedule", None)
    path = runtime / f"{name}.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite config: {path}")
    path.write_text(json.dumps(config, indent=4) + "\n")
    print(f"generated {name} GPU={gpu} config={path}", flush=True)
PY

run_one() {
    local config="$1"
    local gpu="$2"
    local name
    name="$(basename "$config" .json)"
    {
        printf "===== %s START %s GPU=%s =====\n" "$(date "+%F %T")" "$name" "$gpu"
        printf "python=%s torchrun=%s timm=" "$(command -v python)" "$(command -v torchrun)"
        python -c "import timm; print(timm.__version__)"
        env CUDA_VISIBLE_DEVICES="$gpu" \
            torchrun --standalone --nproc_per_node=1 \
            main.py --config="$RUNTIME_DIR/${name}.json"
        local status=$?
        printf "===== %s END %s GPU=%s status=%s =====\n" "$(date "+%F %T")" "$name" "$gpu" "$status"
        return "$status"
    } > "$PROJECT_ROOT/${name}.log" 2>&1
}

run_worker() {
    local worker_gpu="$1"
    local config gpu name worker_status=0
    for config in "$RUNTIME_DIR"/*.json; do
        name="$(basename "$config" .json)"
        case "$name" in
            c100_pareto_knee_seed1993_t5_i4_bs128_rerun) gpu=3 ;;
            c100_pareto_knee_seed1993_t10_i4_bs128_rerun) gpu=4 ;;
            c100_pareto_knee_seed1993_t20_i4_bs128_rerun) gpu=6 ;;
            inr_pareto_knee_seed1995_t5_i4_bs128_rerun) gpu=7 ;;
            inr_pareto_knee_seed1995_t10_i4_bs128_rerun) gpu=6 ;;
            inr_pareto_knee_seed1995_t20_i4_bs128_rerun) gpu=3 ;;
            cub_pareto_knee_seed1_t20_i4_bs128_rerun) gpu=4 ;;
            *) continue ;;
        esac
        if [ "$gpu" = "$worker_gpu" ]; then
            if ! run_one "$config" "$gpu"; then
                worker_status=1
            fi
        fi
    done
    return "$worker_status"
}

printf "===== %s QUEUE START =====\n" "$(date "+%F %T")" >> "$QUEUE_LOG"
printf "environment python=%s torchrun=%s\n" "$(command -v python)" "$(command -v torchrun)" >> "$QUEUE_LOG"
worker_pids=()
for gpu in 3 4 6 7; do
    run_worker "$gpu" >> "$QUEUE_LOG" 2>&1 &
    worker_pids+=("$!")
done

queue_status=0
for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
        queue_status=1
    fi
done
printf "===== %s QUEUE END status=%s =====\n" "$(date "+%F %T")" "$queue_status" >> "$QUEUE_LOG"
exit "$queue_status"
