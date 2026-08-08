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
  echo "P1 FAIL: working tree is not clean; commit before launching"
  exit 1
fi

run_one() {
  local name="$1" config="$2" outdir="$3" logfile="$4"
  local lock="${outdir%/}.lock"
  if [ -e "$outdir" ]; then
    echo "P1 FAIL: $name output directory already exists: $outdir"
    exit 1
  fi
  if ! flock -n "$lock" true 2>/dev/null; then
    echo "P1 FAIL: $name filepath lock is held: $lock"
    exit 1
  fi
  echo "===== $(date '+%F %T') START $name ====="
  echo "config=$config outdir=$outdir gpus=$GPU_IDS nproc=$NPROC"
  echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$config" | awk '{print $1}')"
  torchrun --standalone --nproc_per_node="$NPROC" \
    main.py --config="./$config" > "$logfile" 2>&1
  local status=$?
  echo "===== $(date '+%F %T') END $name status=$status ====="
  if [ "$status" -ne 0 ]; then
    echo "P1 FAIL: $name exited with status $status"
    exit 1
  fi
  if grep -qiE 'non-determin|warn_only=True|memory efficient attention defaults to a non' "$logfile"; then
    echo "P1 FAIL: $name log contains a determinism warning"
    grep -niE 'non-determin|warn_only=True|memory efficient attention defaults' "$logfile" | head -20
    exit 1
  fi
}

run_one LIVE_A_CONTROL exps/inr_p1_livea_control_seed1995_nccl.json \
  ./INR_P1_LIVEA_CONTROL_SEED1995_NCCL/ inr_p1_livea_control_seed1995_nccl.log
run_one LIVE_A_DUAL_B exps/inr_p1_livea_dual_b_seed1995_nccl.json \
  ./INR_P1_LIVEA_DUALB_SEED1995_NCCL/ inr_p1_livea_dual_b_seed1995_nccl.log
run_one SDLORA exps/inr_p1_sdlora_seed1995_nccl.json \
  ./INR_P1_SDLORA_SEED1995_NCCL/ inr_p1_sdlora_seed1995_nccl.log
run_one EXP009 exps/inr_p1_exp009_seed1995_nccl.json \
  ./INR_P1_EXP009_SEED1995_NCCL/ inr_p1_exp009_seed1995_nccl.log

python scripts/verify_p1_pair.py \
  --control-log inr_p1_livea_control_seed1995_nccl.log \
  --dual-log inr_p1_livea_dual_b_seed1995_nccl.log \
  --sdlora-log inr_p1_sdlora_seed1995_nccl.log \
  --exp009-log inr_p1_exp009_seed1995_nccl.log \
  --summary p1_inr_seed1995_nccl_summary.json
status=$?
if [ "$status" -ne 0 ]; then
  echo "P1 FAIL: verification failed"
  exit 1
fi

echo "P1 INR SEED1995 PASS"
