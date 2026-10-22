#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHRUN="/home/hongzhijun/miniconda3/envs/sdlora/bin/torchrun"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

run_job() {
    local pair="$1"
    local config="$2"
    local output_dir="$3"
    local log_file="$4"
    local name
    name="$(basename "${config}" .json)"
    if [[ -e "${ROOT}/${output_dir}" ]]; then
        echo "[$(date '+%F %T')] REFUSE ${name}: ${output_dir} already exists"
        return 2
    fi
    echo "[$(date '+%F %T')] START ${name} GPUs=${pair}"
    nohup env \
        HF_ENDPOINT="${HF_ENDPOINT}" \
        CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG}" \
        CUDA_VISIBLE_DEVICES="${pair}" \
        "${TORCHRUN}" --standalone --nproc_per_node=2 \
        main.py --config="./${config}" > "${ROOT}/${log_file}" 2>&1
    local status=$?
    echo "[$(date '+%F %T')] END ${name} status=${status}"
    return "${status}"
}

monitor_jobs() {
    local pids=("$@")
    local logs=(
        "sbgc_offline_t1_c100_seed1993_r10.log"
        "sbgc_offline_t1_inr_seed1995_r10.log"
        "sbgc_offline_t1_cub_seed1_r10.log"
    )
    while true; do
        local alive=0
        local pid
        for pid in "${pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                alive=1
            fi
        done
        if (( alive == 0 )); then
            return
        fi
        echo "[$(date '+%F %T')] MONITOR"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader || true
        local log
        for log in "${logs[@]}"; do
            if [[ -f "${ROOT}/${log}" ]]; then
                echo "--- ${log} ---"
                grep -aE "Learning on|Epoch (1|6|11|16|20)/20|CNN top1 curve|Average Accuracy|Traceback|RuntimeError|NCCL" \
                    "${ROOT}/${log}" | tail -n 6 || true
            fi
        done
        sleep 1800
    done
}

cd "${ROOT}"
run_job "0,1" \
    "exps/sbgc_offline_t1_c100_seed1993_r10.json" \
    "SBGC_OFFLINE_T1_C100_SEED1993_R10_20260923" \
    "sbgc_offline_t1_c100_seed1993_r10.log" \
    > sbgc_offline_t1_c100_queue.log 2>&1 &
c100_pid=$!
run_job "4,5" \
    "exps/sbgc_offline_t1_inr_seed1995_r10.json" \
    "SBGC_OFFLINE_T1_INR_SEED1995_R10_20260923" \
    "sbgc_offline_t1_inr_seed1995_r10.log" \
    > sbgc_offline_t1_inr_queue.log 2>&1 &
inr_pid=$!
run_job "6,7" \
    "exps/sbgc_offline_t1_cub_seed1_r10.json" \
    "SBGC_OFFLINE_T1_CUB_SEED1_R10_20260923" \
    "sbgc_offline_t1_cub_seed1_r10.log" \
    > sbgc_offline_t1_cub_queue.log 2>&1 &
cub_pid=$!

monitor_jobs "${c100_pid}" "${inr_pid}" "${cub_pid}" &
monitor_pid=$!
status=0
wait "${c100_pid}" || status=1
wait "${inr_pid}" || status=1
wait "${cub_pid}" || status=1
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
exit "${status}"
