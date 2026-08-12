#!/usr/bin/env bash
# P6 efficiency measurements for the frozen protocol (run on one GPU).
#   1. training-step peak memory curves: full vs SD-LoRA at T=5/10/20
#   2. inference FLOPs / throughput of the merged Live-A Dual-B backbone
#   3. minimal artifact export + exact logits/FC/prototype verification
set -uo pipefail

ROOT="/home/zhaoyang/SD-Lora-CL"
PY="/home/zhaoyang/miniconda3/envs/sdlora/bin/python"

cd "$ROOT" || exit 1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

echo "===== $(date +%F_%T) P6 measurements start ====="

echo "--- training-step peak memory (full method) ---"
"$PY" scripts/measure_train_peak_memory.py \
  --config exps/p5_full_t5_inr_nccl.json \
  --artifact INR_P5_FULL_SEED1995_T5_NCCL \
  --device cuda:0 --steps 3
"$PY" scripts/measure_train_peak_memory.py \
  --config exps/inr_p1_livea_dual_b_seed1995_nccl.json \
  --artifact INR_P1_LIVEA_DUALB_SEED1995_NCCL \
  --device cuda:0 --steps 3
"$PY" scripts/measure_train_peak_memory.py \
  --config exps/p5_full_t20_inr_nccl.json \
  --artifact INR_P5_FULL_SEED1995_T20_NCCL \
  --device cuda:0 --steps 3

echo "--- training-step peak memory (SD-LoRA baseline) ---"
"$PY" scripts/measure_train_peak_memory.py \
  --config exps/p5_sdlora_t5_inr_nccl.json \
  --artifact INR_P5_SDLORA_SEED1995_T5_NCCL \
  --device cuda:0 --steps 3
"$PY" scripts/measure_train_peak_memory.py \
  --config exps/inr_p1_sdlora_seed1995_nccl.json \
  --artifact INR_P1_SDLORA_SEED1995_NCCL \
  --device cuda:0 --steps 3
"$PY" scripts/measure_train_peak_memory.py \
  --config exps/p5_sdlora_t20_inr_nccl.json \
  --artifact INR_P5_SDLORA_SEED1995_T20_NCCL \
  --device cuda:0 --steps 3

echo "--- inference FLOPs / throughput (merged Live-A Dual-B, INR T10) ---"
"$PY" scripts/measure_sa_flops.py \
  --config exps/inr_p1_livea_dual_b_seed1995_nccl.json \
  --artifact INR_P1_LIVEA_DUALB_SEED1995_NCCL \
  --device cuda:0 --batch-size 32 --iters 20 --warmup 3

echo "--- minimal artifact export + verify (CIFAR-100 seed1 T10) ---"
"$PY" scripts/export_minimal_artifact.py \
  --config exps/p3_c100_livea_dual_b_seed1_nccl.json \
  --artifact C100_P3_LIVEA_DUALB_SEED1_NCCL \
  --out C100_P3_LIVEA_DUALB_SEED1_NCCL_MINIMAL
"$PY" scripts/verify_minimal_artifact.py \
  --config exps/p3_c100_livea_dual_b_seed1_nccl.json \
  --artifact C100_P3_LIVEA_DUALB_SEED1_NCCL \
  --minimal C100_P3_LIVEA_DUALB_SEED1_NCCL_MINIMAL \
  --log p3_c100_livea_dual_b_seed1_nccl.log \
  --device cuda:0 --samples 8

echo "--- minimal artifact export + verify (CUB seed1 T10) ---"
"$PY" scripts/export_minimal_artifact.py \
  --config exps/p5_cub_livea_dual_b_seed1_nccl.json \
  --artifact CUB_P5_LIVEA_DUALB_SEED1_NCCL \
  --out CUB_P5_LIVEA_DUALB_SEED1_NCCL_MINIMAL
"$PY" scripts/verify_minimal_artifact.py \
  --config exps/p5_cub_livea_dual_b_seed1_nccl.json \
  --artifact CUB_P5_LIVEA_DUALB_SEED1_NCCL \
  --minimal CUB_P5_LIVEA_DUALB_SEED1_NCCL_MINIMAL \
  --log p5_cub_livea_dual_b_seed1_nccl.log \
  --device cuda:0 --samples 8

echo "===== $(date +%F_%T) P6 measurements end ====="
