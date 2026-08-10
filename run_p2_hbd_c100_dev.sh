#!/usr/bin/env bash
# Launch the P2 development run: CIFAR-100 seed1993 with HBD
# (single outer nohup wrapper, guide section 4.1).
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

config=exps/c100_p2_hbd_seed1993_nccl.json
outdir=./C100_P2_HBD_SEED1993_NCCL/
runlog=c100_p2_hbd_seed1993_nccl.log
queuelog=hbd_c100_dev_queue.log
pidfile=hbd_c100_dev_queue.pid
lock="${outdir%/}.lock"

if [ -e "$outdir" ]; then
  echo "HBD C100 FAIL: output directory already exists: $outdir"
  exit 1
fi
if ! flock -n "$lock" true 2>/dev/null; then
  echo "HBD C100 FAIL: filepath lock is held: $lock"
  exit 1
fi
if pgrep -f 'torchrun.*c100_p2_hbd' > /dev/null 2>&1; then
  echo "HBD C100 FAIL: duplicate c100_p2_hbd torchrun process exists"
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "HBD C100 FAIL: working tree is not clean; commit before launching"
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

echo "===== $(date +%F_%T) START c100_p2_hbd_seed1993 ====="
torchrun --standalone --nproc_per_node=4 main.py \
  --config=./exps/c100_p2_hbd_seed1993_nccl.json \
  > c100_p2_hbd_seed1993_nccl.log 2>&1
status=$?
echo "===== $(date +%F_%T) END c100_p2_hbd_seed1993 status=$status ====="
exit "$status"
' > "$queuelog" 2>&1 < /dev/null &

queue_pid=$!
echo "$queue_pid" > "$pidfile"
echo "HBD C100 dev launched: queue_pid=$queue_pid gpus=$GPU_IDS nproc=$NPROC"
echo "commit=$commit config_sha=$config_sha"
echo "queuelog=$queuelog runlog=$runlog outdir=$outdir"
