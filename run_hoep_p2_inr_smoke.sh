#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

log="./hoep_p2_inr_seed1995_t2_e2.log"
name="hoep_p2_inr_seed1995_t2_e2"
echo "===== $(date '+%F %T') START $name GPU=2,3 =====" | tee -a "$log"
CUDA_VISIBLE_DEVICES="2,3" torchrun --standalone --nproc_per_node=2 \
    main.py --config="./exps/hoep_p2_inr_seed1995_t2_e2.json" >> "$log" 2>&1
status=$?
echo "===== $(date '+%F %T') END $name status=$status =====" | tee -a "$log"
exit "$status"
