#!/usr/bin/env bash
set +e

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL"
CONDA_SH="/home/zhaoyang/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sdlora"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC="${NPROC:-4}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$PROJECT_ROOT" || exit 1
source "$CONDA_SH" || exit 1
conda activate "$CONDA_ENV" || exit 1
export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

if [ -n "$(git status --porcelain)" ]; then
  echo "P0.5 FAIL: working tree is not clean; commit before launching"
  exit 1
fi

config=exps/live_a_rng_smoke_dual_b_nccl_c100_seed1.json
outdir=./CF100_P0_5_NCCL_DUAL_B_SMOKE/
logfile=live_a_rng_smoke_dual_b_nccl.log
lock="${outdir%/}.lock"

if [ -e "$outdir" ]; then
  echo "P0.5 FAIL: output directory already exists: $outdir"
  exit 1
fi
if ! flock -n "$lock" true 2>/dev/null; then
  echo "P0.5 FAIL: filepath lock is held: $lock"
  exit 1
fi

echo "===== $(date '+%F %T') START P0.5 NCCL Dual-B smoke ====="
echo "config=$config outdir=$outdir gpus=$GPU_IDS nproc=$NPROC"
echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$config" | awk '{print $1}')"
torchrun --standalone --nproc_per_node="$NPROC" \
  main.py --config="./$config" > "$logfile" 2>&1
status=$?
echo "===== $(date '+%F %T') END P0.5 NCCL Dual-B smoke status=$status ====="
if [ "$status" -ne 0 ]; then
  echo "P0.5 FAIL: torchrun exited with status $status"
  exit 1
fi
if grep -qiE 'non-determin|warn_only=True|memory efficient attention defaults to a non' "$logfile"; then
  echo "P0.5 FAIL: determinism warning found"
  grep -niE 'non-determin|warn_only=True|memory efficient attention defaults' "$logfile" | head -20
  exit 1
fi

echo "===== P0.5 verification ====="
for task in 0 1; do
  if ! grep -q "\[PostTrainHash\] task $task " "$logfile"; then
    echo "P0.5 FAIL: missing PostTrainHash task $task"
    exit 1
  fi
done
for pattern in \
  '\[RNGHash\] task .* calibration .* PASS' \
  '\[RNGHash\] task .* eval .* PASS' \
  '\[PrototypeSync\] rank consistency PASS' \
  '\[DualHead\] calibration parameter invariance PASS' \
  '\[DualHead\] task 1 fused_proto_max_diff=.* PASS'; do
  if ! grep -qE "$pattern" "$logfile"; then
    echo "P0.5 FAIL: missing $pattern"
    exit 1
  fi
done

for task in 0 1; do
  values=$(grep "\[DualHead\] rank .* task $task " "$logfile" | sed 's/.*task [0-9]* //' | sort -u)
  count=$(printf '%s\n' "$values" | sed '/^$/d' | wc -l)
  if [ "$count" -ne 1 ]; then
    echo "P0.5 FAIL: task $task rank calibration values differ:"
    echo "$values"
    exit 1
  fi
done

echo "P0.5 SMOKE PASS"
