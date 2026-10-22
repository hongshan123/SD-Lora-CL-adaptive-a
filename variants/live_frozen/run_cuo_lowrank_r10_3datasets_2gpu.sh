#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$GPU_IDS" ]]; then
    echo "Set GPU_IDS or CUDA_VISIBLE_DEVICES to at least four comma-separated GPU ids." >&2
    exit 2
fi
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} < 4 )); then
    echo "Need at least four GPUs for two concurrent CUO runs; got: $GPU_IDS" >&2
    exit 2
fi

stamp() { date '+%F %T'; }
run_one() {
    local name="$1" config="$2" visible="$3"
    local log="${name}.log"
    echo "===== $(stamp) START ${name} GPUs=${visible} =====" | tee -a "$log" >&2
    nohup env CUDA_VISIBLE_DEVICES="$visible" torchrun --standalone --nproc_per_node=2 \
        main.py --config="$config" >> "$log" 2>&1 &
    echo $!
}

wait_status() {
    local name="$1" pid="$2"
    local status=0
    if wait "$pid"; then
        status=0
    else
        status=$?
    fi
    echo "===== $(stamp) END ${name} status=${status} =====" | tee -a "${name}.log"
    return "$status"
}

C100_PID="$(run_one cuo_lowrank_r10_c100_seed1993 exps/cuo_lowrank_r10_c100_seed1993.json "${GPUS[0]},${GPUS[1]}")"
INR_PID="$(run_one cuo_lowrank_r10_inr_seed1995 exps/cuo_lowrank_r10_inr_seed1995.json "${GPUS[2]},${GPUS[3]}")"

wait_status cuo_lowrank_r10_c100_seed1993 "$C100_PID"
CUB_PID="$(run_one cuo_lowrank_r10_cub_seed1 exps/cuo_lowrank_r10_cub_seed1.json "${GPUS[0]},${GPUS[1]}")"
wait_status cuo_lowrank_r10_inr_seed1995 "$INR_PID"
wait_status cuo_lowrank_r10_cub_seed1 "$CUB_PID"
