#!/usr/bin/env bash
# P5 task-length queue (T5/T20, dev seeds) for the frozen protocol:
# full method, SD-LoRA, EXP-009, rank-1 SD-LoRA+Dual-B on C100/INR.
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

queuelog=p5_tasklen_queue.log
pidfile=p5_tasklen_queue.pid

if pgrep -f 'torchrun.*p5_' > /dev/null 2>&1; then
  echo "P5 FAIL: duplicate p5 torchrun process exists"
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "P5 FAIL: working tree is not clean; commit before launching"
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

for method in full sdlora exp009 rank1; do
  for ds in c100 inr; do
    for t in 5 20; do
      run_one p5_${method}_t${t}_${ds}_nccl exps/p5_${method}_t${t}_${ds}_nccl.json \
        || { echo "P5 QUEUE ABORT at ${method}/${ds}/T${t}"; exit 1; }
    done
  done
done
echo "P5 QUEUE DONE"
' > "$queuelog" 2>&1 < /dev/null &

queue_pid=$!
echo "$queue_pid" > "$pidfile"
echo "P5 task-length queue launched: queue_pid=$queue_pid gpus=$GPU_IDS nproc=$NPROC"
echo "commit=$commit queuelog=$queuelog"
