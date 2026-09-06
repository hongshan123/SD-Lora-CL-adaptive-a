#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
USER_HOME_DIR="${USER_HOME_DIR:-/home/$(id -un)}"
CONDA_SH="${CONDA_SH:-$USER_HOME_DIR/miniconda3/etc/profile.d/conda.sh}"
GPU_IDS="${GPU_IDS:-2,3}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
NPROC=$(awk -F',' '{print NF}' <<< "$GPU_IDS")
RUNTIME_DIR="$PROJECT_ROOT/.runtime_adaptive_a_three_way_2gpu"

IFS=',' read -r -a gpu_id_list <<< "$GPU_IDS"
if [ "$NPROC" -ne 2 ] || [ "${#gpu_id_list[@]}" -ne 2 ]; then
  echo "Adaptive-A three-way queue requires exactly 2 GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi
for gpu_id in "${gpu_id_list[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "Adaptive-A three-way queue requires numeric GPU IDs; got: $GPU_IDS" >&2
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

python - "$PROJECT_ROOT" "$RUNTIME_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime_dir = Path(sys.argv[2])
datasets = (("c100", 1993), ("inr", 1995), ("cub", 1))
groups = (
    ("live_a", True, False),
    ("frozen_a", False, False),
    ("adaptive_a", True, True),
)

for dataset, seed in datasets:
    source = root / "exps" / (
        f"{dataset}_coordinate_stable_adaptive_a_seed{seed}_nccl.json"
    )
    base = json.loads(source.read_text())
    required = {
        "batch_size": 32,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
        "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
        "sa_coordinate_stable_transport": True,
        "sa_dual_head": True,
        "sa_adaptive_a_enabled": True,
    }
    if any(base.get(key) != value for key, value in required.items()):
        raise SystemExit(f"unexpected source protocol: {source}")
    if any("aob" in key.lower() for key in base):
        raise SystemExit(f"AOB field found in source protocol: {source}")

    for group, train_a, adaptive_a in groups:
        name = f"{dataset}_coordinate_stable_{group}_seed{seed}_2gpu_bs64"
        config = dict(base)
        config.update(
            {
                "prefix": name,
                "filepath": f"./{name.upper()}/",
                "batch_size": 64,
                "sa_train_a_all_tasks": train_a,
                "sa_adaptive_a_enabled": adaptive_a,
                "sa_resume": False,
            }
        )
        (runtime_dir / f"{name}.json").write_text(
            json.dumps(config, indent=4) + "\n"
        )
PY

configs=(
  ".runtime_adaptive_a_three_way_2gpu/c100_coordinate_stable_live_a_seed1993_2gpu_bs64.json"
  ".runtime_adaptive_a_three_way_2gpu/c100_coordinate_stable_frozen_a_seed1993_2gpu_bs64.json"
  ".runtime_adaptive_a_three_way_2gpu/c100_coordinate_stable_adaptive_a_seed1993_2gpu_bs64.json"
  ".runtime_adaptive_a_three_way_2gpu/inr_coordinate_stable_live_a_seed1995_2gpu_bs64.json"
  ".runtime_adaptive_a_three_way_2gpu/inr_coordinate_stable_frozen_a_seed1995_2gpu_bs64.json"
  ".runtime_adaptive_a_three_way_2gpu/inr_coordinate_stable_adaptive_a_seed1995_2gpu_bs64.json"
  ".runtime_adaptive_a_three_way_2gpu/cub_coordinate_stable_live_a_seed1_2gpu_bs64.json"
  ".runtime_adaptive_a_three_way_2gpu/cub_coordinate_stable_frozen_a_seed1_2gpu_bs64.json"
  ".runtime_adaptive_a_three_way_2gpu/cub_coordinate_stable_adaptive_a_seed1_2gpu_bs64.json"
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

echo "===== $(date '+%F %T') ADAPTIVE-A THREE-WAY QUEUE DONE ====="
