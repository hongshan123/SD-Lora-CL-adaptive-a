#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

torchrun_bin="${TORCHRUN_BIN:-torchrun}"
python_bin="${PYTHON_BIN:-python}"

for dataset in c100 inr cub; do
  name="sbgc_trainproj_${dataset}_t10_20260925"
  config="exps/${name}.json"
  output="SBGC_TRAINPROJ_${dataset^^}_T10_20260925"
  if [[ -e "$output" || -e "${name}.log" ]]; then
    printf 'Refusing to overwrite existing run: %s\n' "$name" >&2
    exit 1
  fi
  printf '%s START %s GPUs=%s\n' "$(date -u '+%F %T UTC')" "$name" "$CUDA_VISIBLE_DEVICES"
  "$torchrun_bin" --standalone --nproc_per_node=2 main.py --config="./$config" \
    > "${name}.log" 2>&1
  "$python_bin" scripts/audit_sa_task_snapshots.py "$output" > "${name}_audit.json"
  "$python_bin" -c 'import json,sys; a=json.load(open(sys.argv[1])); assert len(a)==10 and all(x["verified"] for x in a)' \
    "${name}_audit.json"
  printf '%s PASS %s\n' "$(date -u '+%F %T UTC')" "$name"
done
