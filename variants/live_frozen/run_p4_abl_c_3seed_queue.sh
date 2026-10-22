#!/usr/bin/env bash
# P4 ablation C (frozen A) 3-seed queue: C100 seeds 1-3 then INR seeds 1-3.
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

queuelog=p4_abl_c_3seed_queue.log
pidfile=p4_abl_c_3seed_queue.pid

if pgrep -f 'torchrun.*p4_abl_c' > /dev/null 2>&1; then
  echo "P4-C3 FAIL: duplicate abl_c torchrun process exists"
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "P4-C3 FAIL: working tree is not clean; commit before launching"
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

for seed in 1 2 3; do
  run_one p4_abl_c_cifar224_seed${seed}_nccl exps/p4_abl_c_cifar224_seed${seed}_nccl.json || { echo "P4-C3 QUEUE ABORT at C100 seed $seed"; exit 1; }
done
for seed in 1 2 3; do
  run_one p4_abl_c_imagenetr_seed${seed}_nccl exps/p4_abl_c_imagenetr_seed${seed}_nccl.json || { echo "P4-C3 QUEUE ABORT at INR seed $seed"; exit 1; }
done
echo "P4-C3 QUEUE DONE"
' > "$queuelog" 2>&1 < /dev/null &

queue_pid=$!
echo "$queue_pid" > "$pidfile"
echo "P4-C 3-seed queue launched: queue_pid=$queue_pid gpus=$GPU_IDS nproc=$NPROC"
echo "commit=$commit queuelog=$queuelog"
