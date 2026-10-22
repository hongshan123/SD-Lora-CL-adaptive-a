#!/usr/bin/env bash
set +e

PROJECT_ROOT="/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL"
CONDA_SH="/home/hongzhijun/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sdlora"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
NPROC="${NPROC:-$(awk -F',' '{print NF}' <<< "$GPU_IDS")}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$PROJECT_ROOT" || exit 1
source "$CONDA_SH" || exit 1
conda activate "$CONDA_ENV" || exit 1

echo "Python: $(which python)"
python -V
echo "Project root: $(pwd)"
echo "GPU_IDS: ${GPU_IDS}"
echo "NPROC: ${NPROC}"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"

CONFIGS=(
  ./exps/seed_1995_k3_bal0005_inr.json
  ./exps/seed_1995_k3_bal001_inr.json
  ./exps/seed_1995_k3_bal002_inr.json
  ./exps/seed_1995_k3_lowsim1_inr.json
  ./exps/seed_1995_k3_bal001_lowsim1_inr.json
  ./exps/seed_1995_k3_bal001_age001_inr.json
  ./exps/seed_1995_k3_realloc_inr.json
  ./exps/seed_1995_k4_bal001_inr.json
)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .json)
  if [[ ! -f "$cfg" ]]; then
    echo "[$(date '+%F %T')] SKIP ${name}: config not found: ${cfg}"
    echo
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

  echo "[$(date '+%F %T')] START ${name}"
  echo "Config: ${cfg}"
  echo "Output dir: ${out_dir:-<none>}"

  HF_ENDPOINT="$HF_ENDPOINT" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    torchrun --standalone --nproc_per_node="$NPROC" main.py --config="$cfg" \
    > "${name}.log" 2>&1
  status=$?
  echo "[$(date '+%F %T')] END ${name} status=${status}"
  echo
  sleep 5
done
