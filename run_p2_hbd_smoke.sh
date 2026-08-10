#!/usr/bin/env bash
# Launch the HBD two-task NCCL DDP smoke with a single outer nohup wrapper
# (guide section 4.1). Queue log and run log are separate; PID is recorded.
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

config=exps/hbd_smoke_c100_seed1.json
outdir=./C100_HBD_SMOKE_NCCL/
runlog=hbd_smoke_c100_seed1_nccl.log
queuelog=hbd_smoke_queue.log
pidfile=hbd_smoke_queue.pid
lock="${outdir%/}.lock"

if [ -e "$outdir" ]; then
  echo "HBD SMOKE FAIL: output directory already exists: $outdir"
  exit 1
fi
if ! flock -n "$lock" true 2>/dev/null; then
  echo "HBD SMOKE FAIL: filepath lock is held: $lock"
  exit 1
fi
if pgrep -f 'torchrun.*hbd_smoke' > /dev/null 2>&1; then
  echo "HBD SMOKE FAIL: duplicate hbd_smoke torchrun process exists"
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "HBD SMOKE FAIL: working tree is not clean; commit before launching"
  exit 1
fi

commit=$(git rev-parse HEAD)
config_sha=$(sha256sum "$config" | awk '{print $1}')

nohup bash -lc '
set -uo pipefail
cd /home/zhaoyang/SD-Lora-CL
source /home/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate sdlora
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

echo "===== $(date +%F_%T) START hbd_smoke_c100_seed1 ====="
torchrun --standalone --nproc_per_node=4 main.py \
  --config=./exps/hbd_smoke_c100_seed1.json \
  > hbd_smoke_c100_seed1_nccl.log 2>&1
status=$?
echo "===== $(date +%F_%T) END hbd_smoke_c100_seed1 status=$status ====="
exit "$status"
' > "$queuelog" 2>&1 < /dev/null &

queue_pid=$!
echo "$queue_pid" > "$pidfile"
echo "HBD SMOKE launched: queue_pid=$queue_pid gpus=$GPU_IDS nproc=$NPROC"
echo "commit=$commit config_sha=$config_sha"
echo "queuelog=$queuelog runlog=$runlog outdir=$outdir"
