#!/usr/bin/env bash
# Wait for the frozen P7 queue, then run coordinate-stable development seeds.
set -uo pipefail

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL-coordinate"
CANONICAL_ROOT="/home/zhaoyang/SD-Lora-CL"
CONDA_SH="/home/zhaoyang/miniconda3/etc/profile.d/conda.sh"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC="${NPROC:-4}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$PROJECT_ROOT" || exit 1
source "$CONDA_SH" || exit 1
conda activate sdlora || exit 1

queue_log="coordinate_stable_single_seed_queue.log"
queue_pid_file="coordinate_stable_single_seed_queue.pid"
monitor_log="monitor_coordinate_stable_single_seed.log"
monitor_pid_file="monitor_coordinate_stable_single_seed.pid"

if [ -n "$(git status --porcelain)" ]; then
  echo "CoordinateStable FAIL: worktree must be clean"
  exit 1
fi
if pgrep -f 'torchrun.*coordinate_stable_seed' > /dev/null 2>&1; then
  echo "CoordinateStable FAIL: duplicate experiment process exists"
  exit 1
fi

python - <<'PY'
import json
from pathlib import Path

root = Path("/home/zhaoyang/SD-Lora-CL-coordinate")
configs = [
    root / "exps/c100_coordinate_stable_seed1993_nccl.json",
    root / "exps/inr_coordinate_stable_seed1995_nccl.json",
]
for path in configs:
    config = json.loads(path.read_text())
    if config.get("sa_resume") is not False:
        raise SystemExit(f"resume must be false: {path}")
    if not config.get("sa_live_a_coordinate_align"):
        raise SystemExit(f"operator alignment missing: {path}")
    if not config.get("sa_coordinate_stable_transport"):
        raise SystemExit(f"prototype transport missing: {path}")
    output = root / config["filepath"]
    if output.exists():
        raise SystemExit(f"fresh output required: {output}")
print("CoordinateStable preflight PASS")
PY

setsid nohup env \
  PROJECT_ROOT="$PROJECT_ROOT" \
  CANONICAL_ROOT="$CANONICAL_ROOT" \
  GPU_IDS="$GPU_IDS" \
  NPROC="$NPROC" \
  HF_ENDPOINT="$HF_ENDPOINT" \
  bash -lc '
set -uo pipefail
cd "$PROJECT_ROOT"
source /home/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate sdlora
export HF_ENDPOINT="$HF_ENDPOINT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

p7_pid=""
if [ -f "$CANONICAL_ROOT/p7_t20_multiseed_queue.pid" ]; then
  p7_pid=$(cat "$CANONICAL_ROOT/p7_t20_multiseed_queue.pid")
fi
while [ -n "$p7_pid" ] && kill -0 "$p7_pid" 2>/dev/null; do
  echo "===== $(date +%F_%T) WAITING for P7 queue pid=$p7_pid ====="
  sleep 300
done
sleep 30

run_one() {
  local name="$1"
  local config="$2"
  echo "===== $(date +%F_%T) START $name ====="
  echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$config" | awk "{print \$1}")"
  torchrun --standalone --nproc_per_node="$NPROC" main.py \
    --config="./$config" > "$name.log" 2>&1
  local status=$?
  echo "===== $(date +%F_%T) END $name status=$status ====="
  return "$status"
}

run_one c100_coordinate_stable_seed1993_nccl \
  exps/c100_coordinate_stable_seed1993_nccl.json || exit 1
sleep 5
run_one inr_coordinate_stable_seed1995_nccl \
  exps/inr_coordinate_stable_seed1995_nccl.json || exit 1
echo "COORDINATE STABLE SINGLE-SEED QUEUE DONE"
' > "$queue_log" 2>&1 < /dev/null &

queue_pid=$!
echo "$queue_pid" > "$queue_pid_file"
setsid nohup bash "$PROJECT_ROOT/monitor_coordinate_stable_single_seed.sh" \
  "$queue_pid" > "$monitor_log" 2>&1 < /dev/null &
monitor_pid=$!
echo "$monitor_pid" > "$monitor_pid_file"

echo "Coordinate-stable queue launched"
echo "queue_pid=$queue_pid monitor_pid=$monitor_pid"
echo "queue_log=$queue_log monitor_log=$monitor_log"
