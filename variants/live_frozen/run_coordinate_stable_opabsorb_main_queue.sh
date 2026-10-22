#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL-coordinate"
CONDA_SH="/home/zhaoyang/miniconda3/etc/profile.d/conda.sh"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC="${NPROC:-4}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$PROJECT_ROOT"
source "$CONDA_SH"
conda activate sdlora
export HF_ENDPOINT CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

configs=(
  "exps/c100_coordinate_stable_opabsorb_seed1993_nccl.json"
  "exps/inr_coordinate_stable_opabsorb_seed1995_nccl.json"
  "exps/cub_coordinate_stable_opabsorb_seed1_nccl.json"
)

python - "${configs[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path("/home/zhaoyang/SD-Lora-CL-coordinate")
for name in sys.argv[1:]:
    path = root / name
    config = json.loads(path.read_text())
    if config.get("sa_live_a_absorb_mode") != "operator_preserving_absorb":
        raise SystemExit(f"invalid absorption mode: {path}")
    if not config.get("sa_live_a_coordinate_align"):
        raise SystemExit(f"alignment disabled: {path}")
    if not config.get("sa_coordinate_stable_transport"):
        raise SystemExit(f"transport disabled: {path}")
    output = root / config["filepath"]
    if output.exists():
        raise SystemExit(f"fresh output required: {output}")
print("operator-preserving main queue preflight PASS")
PY

run_one() {
  local config="$1"
  local name
  name="$(basename "$config" .json)"
  echo "===== $(date '+%F %T') START $name ====="
  echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$config" | awk '{print $1}')"
  torchrun --standalone --nproc_per_node="$NPROC" main.py \
    --config="./$config" > "${name}.log" 2>&1
  echo "===== $(date '+%F %T') END $name status=$? ====="
}

for config in "${configs[@]}"; do
  run_one "$config"
  sleep 5
done

echo "===== $(date '+%F %T') OPERATOR-PRESERVING MAIN QUEUE DONE ====="
