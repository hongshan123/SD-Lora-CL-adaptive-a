#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
USER_HOME_DIR="${USER_HOME_DIR:-/home/$(id -un)}"
CONDA_SH="${CONDA_SH:-$USER_HOME_DIR/miniconda3/etc/profile.d/conda.sh}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_risk_budget_calibration_bs128"

IFS=',' read -r -a gpu_id_list <<< "$GPU_IDS"
if [ "${#gpu_id_list[@]}" -ne 8 ]; then
  echo "This calibration queue requires exactly 8 GPU IDs; got: $GPU_IDS" >&2
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
budgets = (1e-5, 3e-5, 1e-4, 3e-4)
sources = (
    ("c100", "c100_coordinate_stable_adaptive_a_seed1993_nccl.json", 1993),
    ("cub", "cub_coordinate_stable_adaptive_a_seed1_nccl.json", 1),
)

configs = []
for dataset, source_name, seed in sources:
    source = root / "exps" / source_name
    base = json.loads(source.read_text())
    for budget in budgets:
        label = format(budget, ".0e").replace("+", "")
        name = (
            f"{dataset}_risk_calibration_b{label}_seed{seed}"
            f"_single_bs128_t3"
        )
        config = dict(base)
        config.update(
            {
                "prefix": name,
                "filepath": f"./{name.upper()}/",
                "device": ["0"],
                "batch_size": 128,
                "max_tasks": 3,
                "sa_dual_head": False,
                "sa_adaptive_a_enabled": True,
                "sa_adaptive_a_strategy": "risk_budgeted",
                "sa_adaptive_a_risk_budget": budget,
                "sa_resume": False,
            }
        )
        config.pop("sa_dual_head_schedule", None)
        path = runtime / f"{name}.json"
        path.write_text(json.dumps(config, indent=4) + "\n")
        configs.append(path)

(runtime / "configs.txt").write_text("\n".join(map(str, configs)) + "\n")
print(f"generated {len(configs)} calibration configs")
PY

mapfile -t configs < "$RUNTIME_DIR/configs.txt"
if [ "${#configs[@]}" -ne 8 ]; then
  echo "Expected 8 generated configs; got ${#configs[@]}" >&2
  exit 1
fi

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

if [ "$GENERATE_ONLY" = "1" ]; then
  echo "Generation and preflight completed; GENERATE_ONLY=1"
  exit 0
fi

export HF_ENDPOINT PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

run_one() {
  local config="$1"
  local gpu_id="$2"
  local name log_file status pid
  name="$(basename "$config" .json)"
  log_file="$PROJECT_ROOT/${name}.log"
  echo "===== $(date '+%F %T') START $name GPU=$gpu_id ====="
  nohup env CUDA_VISIBLE_DEVICES="$gpu_id" \
    torchrun --standalone --nproc_per_node=1 \
    main.py --config="./.runtime_risk_budget_calibration_bs128/${name}.json" \
    > "$log_file" 2>&1 &
  pid=$!
  if wait "$pid"; then
    status=0
  else
    status=$?
  fi
  echo "===== $(date '+%F %T') END $name GPU=$gpu_id status=$status ====="
  return "$status"
}

queue_status=0
run_pids=()
for index in "${!configs[@]}"; do
  run_one "${configs[$index]}" "${gpu_id_list[$index]}" &
  run_pids+=("$!")
done

for pid in "${run_pids[@]}"; do
  if ! wait "$pid"; then
    queue_status=1
  fi
done

echo "===== $(date '+%F %T') RISK BUDGET CALIBRATION QUEUE status=$queue_status ====="
exit "$queue_status"
