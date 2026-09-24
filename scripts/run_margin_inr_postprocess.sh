#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_dir=FROZEN_MARGIN_INR_T10_SEED1995_20260924_R3
unit=frozen-margin-inr-t10-20260924-r3.service
export PATH=/home/hongzhijun/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_ENDPOINT=https://hf-mirror.com

while systemctl --user is-active --quiet "$unit"; do
    sleep 60
done

for task in {0..9}; do
    printf -v snapshot '%s/task_snapshots/task_%03d/complete.json' "$run_dir" "$task"
    if [[ ! -s "$snapshot" ]]; then
        printf 'INCOMPLETE: missing %s; postprocessing skipped\n' "$snapshot" >&2
        exit 1
    fi
done

python scripts/counterfactual_g_head_swap.py \
    --run-dir "$run_dir" --anchor-task 0 --target-task 9 \
    --device cuda:0 --batch-size 64 \
    --output FROZEN_BRANCH_COUNTERFACTUAL_20260924/inr_t0_to_t9.json

python scripts/counterfactual_g_head_swap.py \
    --run-dir "$run_dir" --anchor-task 8 --target-task 9 \
    --device cuda:0 --batch-size 64 \
    --output FROZEN_BRANCH_COUNTERFACTUAL_20260924/inr_t8_to_t9.json

python scripts/evaluate_margin_calibration.py \
    --run-dir "$run_dir" --target-task 9 --device cuda:0 --batch-size 64 \
    --output MARGIN_CALIBRATION_20260924/inr_t10.json

for task in 1 6 9; do
    python scripts/evaluate_margin_g_proxy.py \
        --run-dir "$run_dir" --task "$task" \
        --rho-state MARGIN_CALIBRATION_20260924/inr_t10.pt \
        --device cuda:0 --batch-size 64 \
        --output "MARGIN_G_PROXY_20260924/inr_task${task}.json"
done

python scripts/summarize_margin_study.py \
    --calibration MARGIN_CALIBRATION_20260924/c100_t10.json \
    --calibration MARGIN_CALIBRATION_20260924/cub_t10.json \
    --calibration MARGIN_CALIBRATION_20260924/inr_t10.json \
    --proxy MARGIN_G_PROXY_20260924/c100_task1.json \
    --proxy MARGIN_G_PROXY_20260924/c100_task6.json \
    --proxy MARGIN_G_PROXY_20260924/c100_task7.json \
    --proxy MARGIN_G_PROXY_20260924/c100_task9.json \
    --proxy MARGIN_G_PROXY_20260924/cub_task1.json \
    --proxy MARGIN_G_PROXY_20260924/cub_task6.json \
    --proxy MARGIN_G_PROXY_20260924/cub_task8.json \
    --proxy MARGIN_G_PROXY_20260924/cub_task9.json \
    --proxy MARGIN_G_PROXY_20260924/inr_task1.json \
    --proxy MARGIN_G_PROXY_20260924/inr_task6.json \
    --proxy MARGIN_G_PROXY_20260924/inr_task9.json \
    --output MARGIN_G_PROXY_20260924/stage_gate_all.json
