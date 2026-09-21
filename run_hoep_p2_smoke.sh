#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

run_pair() {
    local gpu_ids="$1"
    local config="$2"
    local log="$3"
    local name
    name="$(basename "$config" .json)"
    echo "===== $(date '+%F %T') START $name GPU=$gpu_ids =====" | tee -a "$log"
    CUDA_VISIBLE_DEVICES="$gpu_ids" torchrun --standalone --nproc_per_node=2 \
        main.py --config="$config" >> "$log" 2>&1
    local status=$?
    echo "===== $(date '+%F %T') END $name status=$status =====" | tee -a "$log"
    return "$status"
}

run_pair "0,1" "./exps/hoep_p2_c100_seed1993_t2_e2.json" \
    "./hoep_p2_c100_seed1993_t2_e2.log"
c100_status=$?

run_pair "0,1" "./exps/hoep_p2_cub_seed1_t2_e2.json" \
    "./hoep_p2_cub_seed1_t2_e2.log"
cub_status=$?

if (( c100_status != 0 || cub_status != 0 )); then
    exit 1
fi
