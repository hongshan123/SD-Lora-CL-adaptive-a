#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
HOME="${HOME:-/home/$(id -un)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
NPROC=$(awk -F',' '{print NF}' <<< "$GPU_IDS")

IFS=',' read -r -a gpu_id_list <<< "$GPU_IDS"
if [ "$NPROC" -ne 4 ]; then
  echo "Adaptive-A queue requires exactly 4 GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi
if [ "${#gpu_id_list[@]}" -ne 4 ]; then
  echo "Adaptive-A queue requires exactly 4 GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi
for gpu_id in "${gpu_id_list[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "Adaptive-A queue requires comma-separated numeric GPU IDs; got: $GPU_IDS" >&2
    exit 2
  fi
done

cd "$PROJECT_ROOT"

configs=(
  "exps/c100_coordinate_stable_adaptive_a_seed1993_nccl.json"
  "exps/inr_coordinate_stable_adaptive_a_seed1995_nccl.json"
  "exps/cub_coordinate_stable_adaptive_a_seed1_nccl.json"
)

preflight_one() {
  local config="$1"
  local name output_dir log_file
  name="$(basename "$config" .json)"
  output_dir="$PROJECT_ROOT/${name^^}"
  log_file="$PROJECT_ROOT/${name}.log"

  if [ -e "$output_dir" ] || [ -e "$log_file" ]; then
    echo "preflight FAIL: output or log already exists for $name" >&2
    return 1
  fi
  if pgrep -f "[t]orchrun.*${config}" >/dev/null; then
    echo "preflight FAIL: duplicate experiment process exists for $name" >&2
    return 1
  fi
}

for config in "${configs[@]}"; do
  preflight_one "$config"
done

python - "$PROJECT_ROOT" "${configs[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in sys.argv[2:]:
    path = root / name
    config = json.loads(path.read_text())
    required = {
        "batch_size": 32,
        "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
        "sa_adaptive_a_enabled": True,
        "sa_adaptive_a_stability_weight": 1.0,
        "sa_adaptive_a_gate_floor": 0.05,
        "sa_adaptive_a_gate_momentum": 0.9,
        "sa_adaptive_a_eps": 1e-8,
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise SystemExit(f"invalid Adaptive-A protocol config: {path}")
    if any("aob" in key.lower() for key in config):
        raise SystemExit(f"AOB field found in config: {path}")
    if "sa_adaptive_operator_budget" in config or "sa_adaptive_normcap" in config:
        raise SystemExit(f"forbidden adaptive budget field found: {path}")
print("Adaptive-A queue preflight PASS")
PY

if [ ! -f "$CONDA_SH" ]; then
  echo "Conda initialization script not found: $CONDA_SH" >&2
  exit 1
fi
source "$CONDA_SH"
conda activate sdlora
export HF_ENDPOINT CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

run_one() {
  local config="$1"
  local name log_file status
  name="$(basename "$config" .json)"
  log_file="$PROJECT_ROOT/${name}.log"
  echo "===== $(date '+%F %T') START $name GPUs=$GPU_IDS log=$log_file ====="
  {
    echo "===== $(date '+%F %T') START $name GPUs=$GPU_IDS ====="
    echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$config" | awk '{print $1}')"
    if torchrun --standalone --nproc_per_node="$NPROC" main.py \
      --config="./$config"; then
      status=0
    else
      status=$?
    fi
    echo "===== $(date '+%F %T') END $name status=$status ====="
  } > "$log_file" 2>&1
  echo "===== $(date '+%F %T') END $name status=$status ====="
  return "$status"
}

for config in "${configs[@]}"; do
  run_one "$config"
done

echo "===== $(date '+%F %T') ADAPTIVE-A QUEUE DONE ====="
