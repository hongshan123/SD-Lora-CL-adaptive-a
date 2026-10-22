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
  for task in 1 2; do
    CUDA_VISIBLE_DEVICES="$first_gpu" python scripts/diagnose_main_branch_intrusion.py \
      --run-dir "$output" --task "$task" --device cuda:0 \
      > "${name}_task${task}_diagnostic.log" 2>&1
  done
  echo "$(date -Is) END $name"
}

run_one frozen_branch_intrusion_cub_t3_seed1_r2 4,5 \
  exps/frozen_branch_intrusion_cub_t3_seed1.json \
  FROZEN_BRANCH_INTRUSION_CUB_T3_SEED1_R2 &
cub_pid=$!
run_one frozen_branch_intrusion_c100_t3_seed1993_r2 6,7 \
  exps/frozen_branch_intrusion_c100_t3_seed1993.json \
  FROZEN_BRANCH_INTRUSION_C100_T3_SEED1993_R2 &
c100_pid=$!

status=0
wait "$cub_pid" || status=1
wait "$c100_pid" || status=1
exit "$status"
