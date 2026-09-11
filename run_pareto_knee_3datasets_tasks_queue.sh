#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_pareto_knee_3datasets_tasks_i4_bs128"

IFS=',' read -r -a gpu_id_list <<< "$GPU_IDS"
GPU_COUNT="${#gpu_id_list[@]}"
if [ "$GPU_COUNT" -lt 1 ]; then
  echo "At least one GPU ID is required" >&2
  exit 2
fi
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
    ("c100", "c100_coordinate_stable_adaptive_a_seed1993_nccl.json", 100, 1993),
    ("inr", "inr_coordinate_stable_adaptive_a_seed1995_nccl.json", 200, 1995),
    ("cub", "cub_coordinate_stable_adaptive_a_seed1_nccl.json", 200, 1),
)

configs = []
for dataset, source_name, class_count, seed in specs:
    source = json.loads((root / "exps" / source_name).read_text())
    for task_count in (5, 10, 20):
        increment = class_count // task_count
        name = f"{dataset}_pareto_knee_seed{seed}_t{task_count}_i4_bs128"
        config = dict(source)
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
        path.write_text(json.dumps(config, indent=4) + "\n")
        configs.append(path)

(runtime / "configs.txt").write_text("\n".join(map(str, configs)) + "\n")
print(f"generated {len(configs)} configs", flush=True)
PY

mapfile -t configs < "$RUNTIME_DIR/configs.txt"
for config in "${configs[@]}"; do
  name="$(basename "$config" .json)"
  if [ -e "$PROJECT_ROOT/${name^^}" ] || [ -e "$PROJECT_ROOT/${name}.log" ]; then
    echo "Output or log already exists for $name; refusing to overwrite" >&2
    exit 3
  fi
  if pgrep -f "[t]orchrun.*${name}.json" >/dev/null; then
    echo "Existing process found for $name" >&2
    exit 3
  fi
done

export HF_ENDPOINT PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

run_one() {
  local config="$1"
  local gpu_id="$2"
  local name log_file status
  name="$(basename "$config" .json)"
  log_file="$PROJECT_ROOT/${name}.log"
  {
    echo "===== $(date '+%F %T') START $name GPU=$gpu_id ====="
    env CUDA_VISIBLE_DEVICES="$gpu_id" \
      torchrun --standalone --nproc_per_node=1 \
      main.py --config="$RUNTIME_DIR/${name}.json"
    status=$?
    echo "===== $(date '+%F %T') END $name GPU=$gpu_id status=$status ====="
  } > "$log_file" 2>&1
  return "$status"
}

worker() {
  local worker_index="$1"
  local worker_status=0
  local index
  for ((index = worker_index; index < ${#configs[@]}; index += GPU_COUNT)); do
    if ! run_one "${configs[$index]}" "${gpu_id_list[$worker_index]}"; then
      worker_status=1
    fi
  done
  return "$worker_status"
}

worker_pids=()
for ((worker_index = 0; worker_index < GPU_COUNT; worker_index++)); do
  worker "$worker_index" &
  worker_pids+=("$!")
done

queue_status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    queue_status=1
  fi
done

echo "===== $(date '+%F %T') 3-DATASET TASK QUEUE status=$queue_status ====="
exit "$queue_status"
