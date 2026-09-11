#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
GPU_IDS="${GPU_IDS:-0,1}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_pareto_knee_calibration_bs128"

IFS=',' read -r -a gpu_id_list <<< "$GPU_IDS"
if [ "${#gpu_id_list[@]}" -ne 2 ]; then
  echo "This Pareto-knee queue requires exactly 2 GPU IDs; got: $GPU_IDS" >&2
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
sources = (
    ("c100", "c100_coordinate_stable_adaptive_a_seed1993_nccl.json", 1993),
    ("cub", "cub_coordinate_stable_adaptive_a_seed1_nccl.json", 1),
)

configs = []
for dataset, source_name, seed in sources:
    name = f"{dataset}_pareto_knee_seed{seed}_single_bs128_t3"
    config = json.loads((root / "exps" / source_name).read_text())
    config.update(
        {
            "prefix": name,
            "filepath": f"./{name.upper()}/",
            "device": ["0"],
            "batch_size": 128,
            "max_tasks": 3,
            "sa_dual_head": False,
            "sa_adaptive_a_enabled": True,
            "sa_adaptive_a_strategy": "pareto_knee",
            "sa_resume": False,
        }
    )
    config.pop("sa_dual_head_schedule", None)
    path = runtime / f"{name}.json"
    path.write_text(json.dumps(config, indent=4) + "\n")
    configs.append(path)

(runtime / "configs.txt").write_text("\n".join(map(str, configs)) + "\n")
print(f"generated {len(configs)} Pareto-knee calibration configs", flush=True)
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
      main.py --config="./.runtime_pareto_knee_calibration_bs128/${name}.json"
    status=$?
    echo "===== $(date '+%F %T') END $name GPU=$gpu_id status=$status ====="
    exit "$status"
  } > "$log_file" 2>&1
}

run_pids=()
for index in "${!configs[@]}"; do
  run_one "${configs[$index]}" "${gpu_id_list[$index]}" &
  run_pids+=("$!")
done

queue_status=0
for pid in "${run_pids[@]}"; do
  if ! wait "$pid"; then
    queue_status=1
  fi
done

echo "===== $(date '+%F %T') PARETO-KNEE CALIBRATION status=$queue_status ====="
exit "$queue_status"
