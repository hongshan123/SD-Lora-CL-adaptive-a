#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source /home/hongzhijun/miniconda3/etc/profile.d/conda.sh
conda activate sdlora
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
cd "$ROOT"

run_one() {
  local name="$1" gpu_pair="$2" config="$3" output="$4"
  local first_gpu="${gpu_pair%%,*}"
  local task
  if [ -e "$output" ] || [ -e "${name}.log" ]; then
    echo "Refusing to overwrite existing run: $name" >&2
    return 1
  fi
  echo "$(date -Is) START $name on $gpu_pair"
  CUDA_VISIBLE_DEVICES="$gpu_pair" torchrun --standalone --nproc_per_node=2 \
    main.py --config="$config" > "${name}.log" 2>&1
  echo "$(date -Is) TRAINING_DONE $name"
  for task in $(seq 1 9); do
    CUDA_VISIBLE_DEVICES="$first_gpu" python scripts/diagnose_main_branch_intrusion.py \
      --run-dir "$output" --task "$task" --device cuda:0 \
      > "${name}_task${task}_diagnostic.log" 2>&1
    echo "$(date -Is) DIAGNOSTIC_DONE $name task=$task"
  done
  python scripts/audit_sa_task_snapshots.py "$output" > "${name}_snapshot_audit.json"
  echo "$(date -Is) END $name"
}

run_one frozen_branch_intrusion_cub_t10_seed1_20260923 4,5 \
  exps/frozen_branch_intrusion_cub_t10_seed1.json \
  FROZEN_BRANCH_INTRUSION_CUB_T10_SEED1_20260923 &
cub_pid=$!
run_one frozen_branch_intrusion_c100_t10_seed1993_20260923 6,7 \
  exps/frozen_branch_intrusion_c100_t10_seed1993.json \
  FROZEN_BRANCH_INTRUSION_C100_T10_SEED1993_20260923 &
c100_pid=$!

status=0
wait "$cub_pid" || status=1
wait "$c100_pid" || status=1
exit "$status"
