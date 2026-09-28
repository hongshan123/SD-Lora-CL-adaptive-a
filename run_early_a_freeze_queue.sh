#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DATASET="${1:?dataset must be cub or inr}"
PHASE="${2:-formal}"
case "$DATASET" in
  cub) SEED=1; DEFAULT_GPU_IDS=4,5 ;;
  inr) SEED=1995; DEFAULT_GPU_IDS=6,7 ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
case "$PHASE" in
  formal) SUFFIX=20260928 ;;
  smoke) SUFFIX=smoke_20260928 ;;
  *) echo "Unknown phase: $PHASE" >&2; exit 2 ;;
esac
GPU_IDS="${GPU_IDS:-$DEFAULT_GPU_IDS}"
if [[ ! "$GPU_IDS" =~ ^[0-7],[0-7]$ ]] || [[ "$GPU_IDS" == *2* || "$GPU_IDS" == *3* ]] || [[ "${GPU_IDS%,*}" == "${GPU_IDS#*,}" ]]; then
  echo "Use two distinct GPUs, excluding 2 and 3: $GPU_IDS" >&2
  exit 2
fi
export PATH="/home/hongzhijun/miniconda3/envs/sdlora/bin:$PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
cd "$ROOT"
trap 'status=$?; echo "$(date -Is) QUEUE_EXIT dataset=$DATASET phase=$PHASE status=$status"' EXIT

run_one() {
  local name="$1" output="${1^^}"
  if [[ -e "$output" || -e "${name}.log" ]]; then
    echo "Refusing to overwrite existing run: $name" >&2
    return 1
  fi
  echo "$(date -Is) START $name GPUs=$GPU_IDS"
  nohup torchrun --standalone --nproc_per_node=2 main.py \
    --config="exps/${name}.json" > "${name}.log" 2>&1
  python scripts/audit_sa_task_snapshots.py "$output" \
    > "${name}_snapshot_audit.json"
  echo "$(date -Is) END $name status=0"
}

BASE="early_a_${DATASET}_seed${SEED}"
run_one "${BASE}_prefix_t3_${SUFFIX}"
for arm in freeze live; do
  name="${BASE}_${arm}_t10_${SUFFIX}"
  run_one "$name"
  python scripts/verify_early_a_fork.py "${name^^}" \
    > "${name}_fork_audit.json"
  echo "$(date -Is) FORK_VERIFIED $name"
done
