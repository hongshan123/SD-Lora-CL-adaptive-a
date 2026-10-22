#!/usr/bin/env bash
set +e

PROJECT_ROOT="/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL"
export GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export NPROC="${NPROC:-$(awk -F',' '{print NF}' <<< "$GPU_IDS")}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$PROJECT_ROOT" || exit 1

overall_status=0

echo "===== $(date '+%F %T') START ImageNet-R queue ====="
bash ./run_kcms_cllora_inspired_inr_queue.sh
status=$?
if [[ $status -ne 0 ]]; then
  overall_status=$status
fi
echo "===== $(date '+%F %T') END ImageNet-R queue ====="

echo "===== $(date '+%F %T') START CIFAR-100 queue ====="
bash ./run_kcms_cllora_inspired_c100_queue.sh
status=$?
if [[ $status -ne 0 && $overall_status -eq 0 ]]; then
  overall_status=$status
fi
echo "===== $(date '+%F %T') END CIFAR-100 queue ====="

exit "$overall_status"
