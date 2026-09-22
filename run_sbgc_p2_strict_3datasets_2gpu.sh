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

run_c100_queue() {
    run_job "0,1" \
        "exps/sbgc_p2_fisher_c100_seed1993_t10.json" \
        "SBGC_P2_FISHER_C100_SEED1993_T10_20260922" \
        "sbgc_p2_fisher_c100_seed1993_t10.log" || return $?
    run_job "0,1" \
        "exps/sbgc_p2_uniform_c100_seed1993_t10.json" \
        "SBGC_P2_UNIFORM_C100_SEED1993_T10_20260922" \
        "sbgc_p2_uniform_c100_seed1993_t10.log" || return $?
    run_job "0,1" \
        "exps/sbgc_p2_cuo_r10_c100_seed1993_t10.json" \
        "SBGC_P2_CUO_R10_C100_SEED1993_T10_20260922" \
        "sbgc_p2_cuo_r10_c100_seed1993_t10.log"
}

run_inr_queue() {
    run_job "4,5" \
        "exps/sbgc_p2_fisher_inr_seed1995_t10.json" \
        "SBGC_P2_FISHER_INR_SEED1995_T10_20260922" \
        "sbgc_p2_fisher_inr_seed1995_t10.log" || return $?
    run_job "4,5" \
        "exps/sbgc_p2_uniform_inr_seed1995_t10.json" \
        "SBGC_P2_UNIFORM_INR_SEED1995_T10_20260922" \
        "sbgc_p2_uniform_inr_seed1995_t10.log" || return $?
    run_job "4,5" \
        "exps/sbgc_p2_cuo_r10_inr_seed1995_t10.json" \
        "SBGC_P2_CUO_R10_INR_SEED1995_T10_20260922" \
        "sbgc_p2_cuo_r10_inr_seed1995_t10.log"
}

run_cub_queue() {
    run_job "6,7" \
        "exps/sbgc_p2_fisher_cub_seed1_t10.json" \
        "SBGC_P2_FISHER_CUB_SEED1_T10_20260922" \
        "sbgc_p2_fisher_cub_seed1_t10.log" || return $?
    run_job "6,7" \
        "exps/sbgc_p2_uniform_cub_seed1_t10.json" \
        "SBGC_P2_UNIFORM_CUB_SEED1_T10_20260922" \
        "sbgc_p2_uniform_cub_seed1_t10.log" || return $?
    run_job "6,7" \
        "exps/sbgc_p2_cuo_r10_cub_seed1_t10.json" \
        "SBGC_P2_CUO_R10_CUB_SEED1_T10_20260922" \
        "sbgc_p2_cuo_r10_cub_seed1_t10.log"
}

monitor_queues() {
    local pids=("$@")
    local logs=(
        "sbgc_p2_fisher_c100_seed1993_t10.log"
        "sbgc_p2_uniform_c100_seed1993_t10.log"
        "sbgc_p2_cuo_r10_c100_seed1993_t10.log"
        "sbgc_p2_fisher_inr_seed1995_t10.log"
        "sbgc_p2_uniform_inr_seed1995_t10.log"
        "sbgc_p2_cuo_r10_inr_seed1995_t10.log"
        "sbgc_p2_fisher_cub_seed1_t10.log"
        "sbgc_p2_uniform_cub_seed1_t10.log"
        "sbgc_p2_cuo_r10_cub_seed1_t10.log"
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
                grep -aE "Learning on|CNN top1 curve|Average Accuracy|Forgetting|Traceback|RuntimeError|NCCL" \
                    "${ROOT}/${log}" | tail -n 6 || true
            fi
        done
        sleep 1800
    done
}

cd "${ROOT}"
run_c100_queue > sbgc_p2_c100_queue.log 2>&1 &
c100_pid=$!
run_inr_queue > sbgc_p2_inr_queue.log 2>&1 &
inr_pid=$!
run_cub_queue > sbgc_p2_cub_queue.log 2>&1 &
cub_pid=$!

monitor_queues "${c100_pid}" "${inr_pid}" "${cub_pid}" &
monitor_pid=$!

status=0
wait "${c100_pid}" || status=1
wait "${inr_pid}" || status=1
wait "${cub_pid}" || status=1
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
exit "${status}"
