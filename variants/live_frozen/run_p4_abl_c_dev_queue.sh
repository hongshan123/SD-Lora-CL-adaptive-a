#!/usr/bin/env bash
# P4 ablation C (freeze Live-A shared A) development queue:
# C100 seed1993 -> INR seed1995, single outer nohup, stop on failure.
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

queuelog=p4_abl_c_dev_queue.log
pidfile=p4_abl_c_dev_queue.pid

if pgrep -f 'torchrun.*abl_c_frozen_a' > /dev/null 2>&1; then
  echo "P4-C FAIL: duplicate abl_c torchrun process exists"
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "P4-C FAIL: working tree is not clean; commit before launching"
  exit 1
fi

commit=$(git rev-parse HEAD)

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

run_one() {
  local name="$1"
  local config="$2"
  echo "===== $(date +%F_%T) START $name ====="
  echo "config=$config outdir=$(python -c "import json;print(json.load(open(\"$config\"))[\"filepath\"])") gpus=0,1,2,3 nproc=4"
  echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$config" | awk "{print \$1}")"
  torchrun --standalone --nproc_per_node=4 main.py \
    --config=./"$config" > "$name.log" 2>&1
  local status=$?
  echo "===== $(date +%F_%T) END $name status=$status ====="
  return "$status"
}

run_one c100_p2_abl_c_frozen_a_seed1993_nccl exps/c100_p2_abl_c_frozen_a_seed1993_nccl.json
status=$?
if [ "$status" -ne 0 ]; then
  echo "P4-C QUEUE ABORT after C100 (status=$status)"
  exit "$status"
fi
run_one inr_p1_abl_c_frozen_a_seed1995_nccl exps/inr_p1_abl_c_frozen_a_seed1995_nccl.json
status=$?
echo "P4-C QUEUE DONE (final status=$status)"
exit "$status"
' > "$queuelog" 2>&1 < /dev/null &

queue_pid=$!
echo "$queue_pid" > "$pidfile"
echo "P4-C dev queue launched: queue_pid=$queue_pid gpus=$GPU_IDS nproc=$NPROC"
echo "commit=$commit queuelog=$queuelog"
