#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_ROOT}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sdlora}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC="${NPROC:-4}"
DATASET="${1:-inr}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

case "$DATASET" in
  inr)
    CONFIG="exps/sdlora_rank6_inr_seed1995.json"
    OUTPUT_DIR="ImageNetR_SDLORA_RANK6_SEED1995"
    LOG="sdlora_rank6_inr_seed1995.log"
    ;;
  c100)
    CONFIG="exps/sdlora_rank6_c100_seed1993.json"
    OUTPUT_DIR="CF100_SDLORA_RANK6_SEED1993"
    LOG="sdlora_rank6_c100_seed1993.log"
    ;;
  *)
    echo "Usage: $0 {inr|c100}" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

if find "$OUTPUT_DIR" -maxdepth 1 -type f \( -name 'lora_w_a_*.pt' -o -name 'CLs_weight*.pt' \) -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing to overwrite existing experiment artifacts in $OUTPUT_DIR" >&2
  exit 3
fi

VISIBLE_COUNT=$(CUDA_VISIBLE_DEVICES="$GPU_IDS" python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "$VISIBLE_COUNT" -ne "$NPROC" ]]; then
  echo "Expected $NPROC visible GPUs, found $VISIBLE_COUNT for GPU_IDS=$GPU_IDS" >&2
  exit 4
fi

mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_ENDPOINT
export PYTHONUNBUFFERED=1

echo "[$(date '+%F %T')] START dataset=$DATASET config=$CONFIG GPUs=$GPU_IDS commit=${EXPERIMENT_COMMIT:-unknown}"
set +e
torchrun --standalone --nproc_per_node="$NPROC" main.py --config="./$CONFIG" > "$LOG" 2>&1
status=$?
set -e
echo "[$(date '+%F %T')] END dataset=$DATASET status=$status log=$LOG"
exit "$status"
