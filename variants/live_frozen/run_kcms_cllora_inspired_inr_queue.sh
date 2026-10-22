#!/usr/bin/env bash
set +e

PROJECT_ROOT="/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL"
CONDA_SH="/home/hongzhijun/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sdlora"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
NPROC="${NPROC:-$(awk -F',' '{print NF}' <<< "$GPU_IDS")}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$PROJECT_ROOT" || exit 1
source "$CONDA_SH" || exit 1
conda activate "$CONDA_ENV" || exit 1

export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

echo "Python: $(which python)"
python -V
echo "Project root: $(pwd)"
echo "GPU_IDS: ${GPU_IDS}"
echo "NPROC: ${NPROC}"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"

configs=(
  exps/seed_1995_k4_hardcap_inr.json
  exps/seed_1995_k4_anchor_inr.json
  exps/seed_1995_k4_bwscale_inr.json
  exps/seed_1995_k4_hardcap_bwscale_inr.json
  exps/seed_1995_k4_fixedortho6_inr.json
  exps/seed_1995_k4_hardcap_anchor_bwscale_inr.json
)

overall_status=0

for cfg in "${configs[@]}"; do
  name=$(basename "$cfg" .json)
  if [[ ! -f "$cfg" ]]; then
    echo "===== $(date '+%F %T') SKIP $name: config not found: $cfg ====="
    overall_status=1
    continue
  fi

  out_dir=$(python - "$cfg" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("filepath", ""))
PY
)
  if [[ -n "$out_dir" ]]; then
    mkdir -p "$out_dir"
  fi

  echo "===== $(date '+%F %T') START $name ====="
  echo "Config: ./${cfg}"
  echo "Output dir: ${out_dir:-<none>}"
  torchrun --standalone --nproc_per_node="$NPROC" main.py --config="./$cfg" > "${name}.log" 2>&1
  status=$?
  echo "===== $(date '+%F %T') END $name status=${status} ====="
  if [[ $status -ne 0 && $overall_status -eq 0 ]]; then
    overall_status=$status
  fi
  sleep 5
done

exit "$overall_status"
