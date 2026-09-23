#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHRUN="/home/hongzhijun/miniconda3/envs/sdlora/bin/torchrun"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

cd "${ROOT}"

wait_for_pair() {
    local pair="$1" first second
    IFS=, read -r first second <<< "${pair}"
    while true; do
        local memory=()
        mapfile -t memory < <(
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
        )
        if [[ "${memory[first]:-999999}" =~ ^[0-9]+$ \
            && "${memory[second]:-999999}" =~ ^[0-9]+$ \
            && "${memory[first]}" -lt 512 \
            && "${memory[second]}" -lt 512 ]]; then
            return
        fi
        printf '[%s] Waiting for free GPU pair %s\n' "$(date '+%F %T')" "${pair}"
        sleep 60
    done
}

run_job() {
    local dataset="$1" pair="$2" base_config="$3" output_dir="$4"
    local name="sbgc_attribution_${dataset}_t2_20260923"
    local config="exps/${name}.json"
    wait_for_pair "${pair}"
    if [[ -e "${output_dir}" ]]; then
        printf '[%s] Refusing to overwrite %s\n' "$(date '+%F %T')" "${output_dir}"
        return 2
    fi
    jq --arg name "${name}" --arg output "./${output_dir}/" \
        '.prefix = $name | .filepath = $output | .max_tasks = 2 | .sa_g_boundary_attribution = true' \
        "${base_config}" > "${config}"
    printf '[%s] START %s GPUs=%s\n' "$(date '+%F %T')" "${name}" "${pair}"
    local status
    if nohup env CUDA_VISIBLE_DEVICES="${pair}" \
        HF_ENDPOINT="${HF_ENDPOINT}" CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG}" \
        "${TORCHRUN}" --standalone --nproc_per_node=2 \
        main.py --config="./${config}" > "${name}.log" 2>&1; then
        status=0
    else
        status=$?
    fi
    printf '[%s] END %s status=%s\n' "$(date '+%F %T')" "${name}" "${status}"
    return "${status}"
}

run_job c100 "0,1" exps/sbgc_p2_fisher_c100_seed1993_t10.json \
    SBGC_ATTRIBUTION_C100_SEED1993_T2_20260923 > sbgc_attribution_c100_t2_queue.log 2>&1 &
c100_pid=$!
run_job inr "4,5" exps/sbgc_p2_fisher_inr_seed1995_t10.json \
    SBGC_ATTRIBUTION_INR_SEED1995_T2_20260923 > sbgc_attribution_inr_t2_queue.log 2>&1 &
inr_pid=$!
run_job cub "6,7" exps/sbgc_p2_fisher_cub_seed1_t10.json \
    SBGC_ATTRIBUTION_CUB_SEED1_T2_20260923 > sbgc_attribution_cub_t2_queue.log 2>&1 &
cub_pid=$!

status=0
wait "${c100_pid}" || status=1
wait "${inr_pid}" || status=1
wait "${cub_pid}" || status=1
exit "${status}"
