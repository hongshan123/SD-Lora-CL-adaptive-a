#!/usr/bin/env bash
set -u

cd /home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL || exit 1

source /home/hongzhijun/miniconda3/etc/profile.d/conda.sh
conda activate sdlora

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3}"
export PYTHONUNBUFFERED=1

configs=(
  "exps/seed_1995_k4_clshared4_bwscale_inr.json"
  "exps/k4_clshared4_hardcap_bwscale_c100.json"
  "exps/seed_1995_k4_clshared6_t0freeze_bwscale_inr.json"
  "exps/k4_clshared6_t0freeze_hardcap_bwscale_c100.json"
)

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
runtime_dir=".runtime_clshared4_t0freeze"
mkdir -p "${runtime_dir}"
echo "Using DataParallel device ids: ${CUDA_VISIBLE_DEVICES}"

for cfg in "${configs[@]}"; do
  name="$(basename "${cfg}" .json)"
  runtime_cfg="${runtime_dir}/${name}_dp.json"
  python - "${cfg}" "${runtime_cfg}" "${#visible_gpu_ids[@]}" <<'PY'
import json
import sys

src, dst, num_gpus = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    cfg = json.load(f)

base_batch = int(cfg.get("batch_size", 32))
cfg["device"] = [str(i) for i in range(num_gpus)]
cfg["batch_size"] = base_batch * max(1, num_gpus)

with open(dst, "w") as f:
    json.dump(cfg, f, indent=4)
    f.write("\n")
PY
  echo "===== $(date '+%F %T') START ${name} ====="
  python main.py --config="./${runtime_cfg}" > "${name}.log" 2>&1
  status=$?
  echo "===== $(date '+%F %T') END ${name} status=${status} ====="
  if [ "${status}" -ne 0 ]; then
    echo "Queue stopped because ${name} failed. Check ${name}.log"
    exit "${status}"
  fi
  sleep 5
done

echo "===== $(date '+%F %T') ALL DONE ====="
