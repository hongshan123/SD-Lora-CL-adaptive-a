#!/usr/bin/env bash
# P5 CUB-200 remaining multi-seed runs (EXP-009 seed1-3, SD-LoRA seed1-3)
# after the earlier queue aborted on an EXP-009 OOM caused by concurrent
# external-baseline jobs.  Run only when GPUs 0-3 are otherwise idle.
set -uo pipefail

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL"
CONDA_SH="/home/zhaoyang/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sdlora"

cd "$PROJECT_ROOT" || exit 1
source "$CONDA_SH" || exit 1
conda activate "$CONDA_ENV" || exit 1
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

RUNS=(
  "exps/p5_cub_exp009_seed1_nccl.json|./CUB_P5_EXP009_SEED1_NCCL/|p5_cub_exp009_seed1_nccl.log"
  "exps/p5_cub_exp009_seed2_nccl.json|./CUB_P5_EXP009_SEED2_NCCL/|p5_cub_exp009_seed2_nccl.log"
  "exps/p5_cub_exp009_seed3_nccl.json|./CUB_P5_EXP009_SEED3_NCCL/|p5_cub_exp009_seed3_nccl.log"
  "exps/p5_cub_sdlora_seed1_nccl.json|./CUB_P5_SDLORA_SEED1_NCCL/|p5_cub_sdlora_seed1_nccl.log"
  "exps/p5_cub_sdlora_seed2_nccl.json|./CUB_P5_SDLORA_SEED2_NCCL/|p5_cub_sdlora_seed2_nccl.log"
  "exps/p5_cub_sdlora_seed3_nccl.json|./CUB_P5_SDLORA_SEED3_NCCL/|p5_cub_sdlora_seed3_nccl.log"
)

if [ -n "$(git status --porcelain)" ]; then
  echo "P5-CUB-REMAIN FAIL: working tree is not clean; commit before launching"
  exit 1
fi
if pgrep -f 'torchrun.*p5_cub' > /dev/null 2>&1; then
  echo "P5-CUB-REMAIN FAIL: p5_cub torchrun already running"
  exit 1
fi
if pgrep -f 'bestformer/bin/python main.py' > /dev/null 2>&1; then
  echo "P5-CUB-REMAIN FAIL: external-baseline jobs still running; GPUs not idle"
  exit 1
fi

commit=$(git rev-parse HEAD)
echo "P5-CUB-REMAIN start: commit=$commit $(date +%F_%T)"

for entry in "${RUNS[@]}"; do
  config="${entry%%|*}"
  rest="${entry#*|}"
  outdir="${rest%%|*}"
  runlog="${rest#*|}"

  if [ -e "$outdir" ]; then
    echo "FAIL: output directory already exists: $outdir"
    exit 1
  fi
  config_sha=$(sha256sum "$config" | awk '{print $1}')
  echo "===== $(date +%F_%T) START $(basename "$config") ====="

  torchrun --standalone --nproc_per_node=4 main.py \
    --config=./"$config" \
    > "$runlog" 2>&1
  status=$?
  echo "===== $(date +%F_%T) END $(basename "$config") status=$status config_sha=$config_sha ====="
  if [ "$status" -ne 0 ]; then
    echo "P5-CUB-REMAIN ABORT: $(basename "$config") failed with status $status"
    exit "$status"
  fi
done

echo "P5-CUB-REMAIN ALL DONE $(date +%F_%T)"
