#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_pareto_knee_rmsfix_t10"
QUEUE_LOG="$PROJECT_ROOT/pareto_knee_rmsfix_t10_queue.log"

source "$CONDA_SH"
conda activate sdlora
cd "$PROJECT_ROOT"
export HF_ENDPOINT PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1
mkdir -p "$RUNTIME_DIR"

python - "$PROJECT_ROOT" "$RUNTIME_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime = Path(sys.argv[2])
experiments = (
    ("c100", "c100_coordinate_stable_adaptive_a_seed1993_nccl.json", 1993, 10),
    ("inr", "inr_coordinate_stable_adaptive_a_seed1995_nccl.json", 1995, 20),
    ("cub", "cub_coordinate_stable_adaptive_a_seed1_nccl.json", 1, 20),
)

for dataset, source_name, seed, increment in experiments:
    name = f"{dataset}_pareto_knee_rmsfix_seed{seed}_t10_i4_bs128"
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
            "max_tasks": 10,
            "sa_dual_head": False,
            "sa_adaptive_a_enabled": True,
            "sa_adaptive_a_strategy": "pareto_knee",
            "sa_adaptive_a_crossfit_interval": 4,
            "sa_resume": False,
        }
    )
    config.pop("sa_dual_head_schedule", None)
    output = runtime / f"{name}.json"
    output.write_text(json.dumps(config, indent=4) + "\n")
    print(output)
PY

run_one() {
    local dataset="$1"
    local seed="$2"
    local gpu="$3"
    local name="${dataset}_pareto_knee_rmsfix_seed${seed}_t10_i4_bs128"
    local config="$RUNTIME_DIR/${name}.json"
    local log="$PROJECT_ROOT/${name}.log"
    local output_dir="$PROJECT_ROOT/${name^^}"

    if [ -e "$log" ] || [ -e "$output_dir" ]; then
        printf "Refusing to overwrite existing output for %s\n" "$name" >&2
        return 2
    fi

    {
        printf "===== %s START %s GPU=%s commit=%s =====\n" \
            "$(date '+%F %T')" "$name" "$gpu" "$(git rev-parse --short HEAD)"
        CUDA_VISIBLE_DEVICES="$gpu" torchrun --standalone --nproc_per_node=1 \
            main.py --config="$config"
        status=$?
        printf "===== %s END %s GPU=%s status=%s =====\n" \
            "$(date '+%F %T')" "$name" "$gpu" "$status"
        return "$status"
    } > "$log" 2>&1
}

printf "===== %s QUEUE START commit=%s =====\n" \
    "$(date '+%F %T')" "$(git rev-parse --short HEAD)" > "$QUEUE_LOG"

run_one c100 1993 3 >> "$QUEUE_LOG" 2>&1 &
pid_c100=$!
run_one inr 1995 4 >> "$QUEUE_LOG" 2>&1 &
pid_inr=$!
run_one cub 1 6 >> "$QUEUE_LOG" 2>&1 &
pid_cub=$!

status=0
for pid in "$pid_c100" "$pid_inr" "$pid_cub"; do
    if ! wait "$pid"; then
        status=1
    fi
done
printf "===== %s QUEUE END status=%s =====\n" "$(date '+%F %T')" "$status" >> "$QUEUE_LOG"
exit "$status"
