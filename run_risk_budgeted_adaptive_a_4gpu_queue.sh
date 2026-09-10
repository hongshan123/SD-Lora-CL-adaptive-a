#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
USER_HOME_DIR="${USER_HOME_DIR:-/home/$(id -un)}"
CONDA_SH="${CONDA_SH:-$USER_HOME_DIR/miniconda3/etc/profile.d/conda.sh}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RISK_BUDGET="${RISK_BUDGET:-0.05}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_risk_budgeted_adaptive_a_4gpu"

IFS=',' read -r -a gpu_id_list <<< "$GPU_IDS"
if [ "${#gpu_id_list[@]}" -ne 4 ]; then
  echo "This queue requires exactly 4 GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi
for gpu_id in "${gpu_id_list[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "GPU IDs must be numeric; got: $GPU_IDS" >&2
    exit 2
  fi
done
if [ ! -f "$CONDA_SH" ]; then
  echo "Conda initialization script not found: $CONDA_SH" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate sdlora
cd "$PROJECT_ROOT"
mkdir -p "$RUNTIME_DIR"

python - "$PROJECT_ROOT" "$RUNTIME_DIR" "$RISK_BUDGET" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime = Path(sys.argv[2])
risk_budget = float(sys.argv[3])
sources = {
    "c100": ("c100_coordinate_stable_adaptive_a_seed1993_nccl.json", 1993),
    "inr": ("inr_coordinate_stable_adaptive_a_seed1995_nccl.json", 1995),
}

for short_name, (source_name, seed) in sources.items():
    source = root / "exps" / source_name
    config = json.loads(source.read_text())
    name = f"{short_name}_coordinate_stable_risk_budgeted_adaptive_a_seed{seed}_4gpu"
    config.update(
        {
            "prefix": name,
            "filepath": f"./{name.upper()}/",
            "batch_size": 32,
            "sa_dual_head": False,
            "sa_adaptive_a_enabled": True,
            "sa_adaptive_a_strategy": "risk_budgeted",
            "sa_adaptive_a_risk_budget": risk_budget,
            "sa_resume": False,
        }
    )
    config.pop("sa_dual_head_schedule", None)
    (runtime / f"{name}.json").write_text(
        json.dumps(config, indent=4) + "\n"
    )
PY

configs=(
  ".runtime_risk_budgeted_adaptive_a_4gpu/c100_coordinate_stable_risk_budgeted_adaptive_a_seed1993_4gpu.json"
  ".runtime_risk_budgeted_adaptive_a_4gpu/inr_coordinate_stable_risk_budgeted_adaptive_a_seed1995_4gpu.json"
)

for config in "${configs[@]}"; do
  name="$(basename "$config" .json)"
  if [ -e "$PROJECT_ROOT/${name^^}" ] || [ -e "$PROJECT_ROOT/${name}.log" ]; then
    echo "Output or log already exists for $name; refusing to overwrite" >&2
    exit 3
  fi
done

export HF_ENDPOINT CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

for config in "${configs[@]}"; do
  name="$(basename "$config" .json)"
  log_file="$PROJECT_ROOT/${name}.log"
  echo "===== $(date '+%F %T') START $name GPUs=$GPU_IDS ====="
  {
    echo "===== $(date '+%F %T') START $name GPUs=$GPU_IDS ====="
    echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$config" | awk '{print $1}')"
    torchrun --standalone --nproc_per_node=4 main.py --config="./$config"
    status=$?
    echo "===== $(date '+%F %T') END $name status=$status ====="
    exit "$status"
  } > "$log_file" 2>&1
  echo "===== $(date '+%F %T') END $name log=$log_file ====="
done

echo "===== $(date '+%F %T') RISK-BUDGETED ADAPTIVE-A 4GPU QUEUE DONE ====="
