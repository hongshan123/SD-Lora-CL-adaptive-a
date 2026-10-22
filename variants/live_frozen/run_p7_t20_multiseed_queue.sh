#!/usr/bin/env bash
# P7 confirmation queue: T=20, seeds 1-3, full/SD-LoRA/EXP-009,
# CIFAR-100 and ImageNet-R. All runs use the frozen P5 protocol.
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

queue_log="p7_t20_multiseed_queue.log"
pid_file="p7_t20_multiseed_queue.pid"
monitor_log="monitor_p7_t20_multiseed.log"
monitor_pid_file="monitor_p7_t20_multiseed.pid"

if pgrep -f 'torchrun.*p7_t20_' > /dev/null 2>&1; then
  echo "P7 FAIL: duplicate P7 T20 torchrun process exists"
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "P7 FAIL: working tree is not clean; commit before launching"
  exit 1
fi

python - <<'PY'
import json
from pathlib import Path

root = Path("/home/zhaoyang/SD-Lora-CL")
configs = sorted((root / "exps").glob("p7_t20_*_seed*_nccl.json"))
if len(configs) != 18:
    raise SystemExit(f"P7 FAIL: expected 18 configs, found {len(configs)}")
outdirs = []
for path in configs:
    config = json.loads(path.read_text())
    if config.get("sa_resume") is not False:
        raise SystemExit(f"P7 FAIL: sa_resume must be false in {path}")
    outdir = root / config["filepath"]
    if outdir.exists():
        raise SystemExit(f"P7 FAIL: output already exists: {outdir}")
    outdirs.append(str(outdir.resolve()))
if len(set(outdirs)) != len(outdirs):
    raise SystemExit("P7 FAIL: duplicate output directories")
print("P7 preflight PASS: 18 fresh configs")
PY

commit=$(git rev-parse HEAD)

setsid nohup env \
  GPU_IDS="$GPU_IDS" \
  NPROC="$NPROC" \
  HF_ENDPOINT="$HF_ENDPOINT" \
  bash -lc '
set -uo pipefail
cd /home/zhaoyang/SD-Lora-CL
source /home/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate sdlora
export HF_ENDPOINT="$HF_ENDPOINT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

run_one() {
  local name="$1"
  local config="$2"
  echo "===== $(date +%F_%T) START $name ====="
  echo "config=$config outdir=$(python -c "import json;print(json.load(open(\"$config\"))[\"filepath\"])") gpus=$GPU_IDS nproc=$NPROC"
  echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$config" | awk "{print \$1}")"
  torchrun --standalone --nproc_per_node="$NPROC" main.py \
    --config="./$config" > "$name.log" 2>&1
  local status=$?
  echo "===== $(date +%F_%T) END $name status=$status ====="
  return "$status"
}

for seed in 1 2 3; do
  for dataset in c100 inr; do
    for method in full sdlora exp009; do
      name="p7_t20_${dataset}_${method}_seed${seed}_nccl"
      config="exps/${name}.json"
      run_one "$name" "$config" || {
        echo "P7 QUEUE ABORT at seed=$seed dataset=$dataset method=$method"
        exit 1
      }
      sleep 5
    done
  done
done
echo "P7 T20 MULTISEED QUEUE DONE"
' > "$queue_log" 2>&1 < /dev/null &

queue_pid=$!
echo "$queue_pid" > "$pid_file"

setsid nohup bash "$PROJECT_ROOT/monitor_p7_t20_multiseed.sh" \
  "$queue_pid" > "$monitor_log" 2>&1 < /dev/null &
monitor_pid=$!
echo "$monitor_pid" > "$monitor_pid_file"

echo "P7 T20 multiseed queue launched"
echo "queue_pid=$queue_pid monitor_pid=$monitor_pid"
echo "gpus=$GPU_IDS nproc=$NPROC commit=$commit"
echo "queue_log=$queue_log monitor_log=$monitor_log"
