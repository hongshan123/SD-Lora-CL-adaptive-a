#!/usr/bin/env bash
# P5 third-dataset (CUB-200) run for the frozen Live-A Dual-B method.
set -uo pipefail

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL"
CONDA_SH="/home/zhaoyang/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sdlora"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC="${NPROC:-4}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$PROJECT_ROOT" || exit 1
source "$CONDA_SH" || exit 1
conda activate "$CONDA_ENV" || exit 1

config=exps/p5_cub_livea_dual_b_seed1_nccl.json
outdir=./CUB_P5_LIVEA_DUALB_SEED1_NCCL/
runlog=p5_cub_livea_dual_b_seed1_nccl.log
queuelog=p5_cub_queue.log
pidfile=p5_cub_queue.pid
lock="${outdir%/}.lock"

if [ -e "$outdir" ]; then
  echo "P5-CUB FAIL: output directory already exists: $outdir"
  exit 1
fi
if ! flock -n "$lock" true 2>/dev/null; then
  echo "P5-CUB FAIL: filepath lock is held: $lock"
  exit 1
fi
if pgrep -f 'torchrun.*p5_cub' > /dev/null 2>&1; then
  echo "P5-CUB FAIL: duplicate p5_cub torchrun process exists"
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "P5-CUB FAIL: working tree is not clean; commit before launching"
  exit 1
fi

commit=$(git rev-parse HEAD)
config_sha=$(sha256sum "$config" | awk '{print $1}')

setsid nohup bash -lc '
set -uo pipefail
cd /home/zhaoyang/SD-Lora-CL
source /home/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate sdlora
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

echo "===== $(date +%F_%T) START p5_cub_livea_dual_b_seed1 ====="
torchrun --standalone --nproc_per_node=4 main.py \
  --config=./exps/p5_cub_livea_dual_b_seed1_nccl.json \
  > p5_cub_livea_dual_b_seed1_nccl.log 2>&1
status=$?
echo "===== $(date +%F_%T) END p5_cub_livea_dual_b_seed1 status=$status ====="
exit "$status"
' > "$queuelog" 2>&1 < /dev/null &

queue_pid=$!
echo "$queue_pid" > "$pidfile"
echo "P5 CUB launched: queue_pid=$queue_pid gpus=$GPU_IDS nproc=$NPROC"
echo "commit=$commit config_sha=$config_sha"
