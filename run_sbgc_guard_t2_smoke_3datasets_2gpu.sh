#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHRUN="/home/hongzhijun/miniconda3/envs/sdlora/bin/torchrun"
JQ="/home/hongzhijun/miniconda3/bin/jq"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
cd "${ROOT}"

run_job() {
    local dataset="$1" pair="$2" source="$3"
    local name="sbgc_guard_smoke_${dataset}_20260923"
    local output="SBGC_GUARD_SMOKE_${dataset^^}_20260923"
    local config="exps/${name}.json"
    if [[ -e "${output}" || -e "${config}" ]]; then
        printf '[%s] Refusing to overwrite %s or %s\n' "$(date '+%F %T')" "${output}" "${config}"
        return 2
    fi
    "${JQ}" --arg name "${name}" --arg output "./${output}/" \
        '.prefix=$name | .filepath=$output | .sa_g_plasticity_guard=true |
         .sa_g_holdout_fraction=0.1 | .sa_g_guard_ce_tolerance=0.01' \
        "${source}" > "${config}"
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

run_job c100 "0,1" exps/sbgc_p0_c100_seed1993_t2_e2.json \
    > sbgc_guard_smoke_c100_queue.log 2>&1 &
c100_pid=$!
run_job inr "4,5" exps/sbgc_p0_inr_seed1995_t2_e2.json \
    > sbgc_guard_smoke_inr_queue.log 2>&1 &
inr_pid=$!
run_job cub "6,7" exps/sbgc_p0_cub_seed1_t2_e2.json \
    > sbgc_guard_smoke_cub_queue.log 2>&1 &
cub_pid=$!

status=0
wait "${c100_pid}" || status=1
wait "${inr_pid}" || status=1
wait "${cub_pid}" || status=1
exit "${status}"
