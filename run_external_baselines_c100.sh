#!/usr/bin/env bash
# Local reproduction of three external baselines on CIFAR-100 (T=10, seed 1)
# using their official implementations and default hyper-parameters.
# Each run uses one GPU (0,1,2).  Launch whole script with:
#   setsid nohup bash -lc 'bash run_external_baselines_c100.sh' \
#     > external_baselines_c100_queue.log 2>&1 < /dev/null &
set -uo pipefail

ROOT="/home/zhaoyang/SD-Lora-CL"
PY="/home/zhaoyang/miniconda3/envs/bestformer/bin/python"

echo "===== $(date +%F_%T) external-baseline C100 queue start ====="

# Install tracked configs into the official clones and ensure data links.
cp "$ROOT/external_baselines_configs/infolora/cifar100_inflora_sdlocal_seed1.json" \
  "$ROOT/external_baselines/infolora/configs/"
cp "$ROOT/external_baselines_configs/cllora/cifar_t10_seed1.json" \
  "$ROOT/external_baselines/cllora/exps/"
cp "$ROOT/external_baselines_configs/lora_drs/cifar100_t10_seed1.json" \
  "$ROOT/external_baselines/lora_drs/configs/"
for repo in infolora cllora lora_drs; do
  if [ ! -e "$ROOT/external_baselines/$repo/data" ]; then
    ln -s "$ROOT/data" "$ROOT/external_baselines/$repo/data"
  fi
done

# InfLoRA on GPU 0
(
  cd "$ROOT/external_baselines/infolora" || exit 1
  exec "$PY" main.py --config configs/cifar100_inflora_sdlocal_seed1.json --device 0 \
    > "$ROOT/baseline_infolora_c100_seed1.log" 2>&1
) &
pid_inf=$!

# CL-LoRA on GPU 1
(
  cd "$ROOT/external_baselines/cllora" || exit 1
  exec "$PY" main.py exps/cifar_t10_seed1.json \
    > "$ROOT/baseline_cllora_c100_seed1.log" 2>&1
) &
pid_cl=$!

# LoRA-Sub-DRS on GPU 2
(
  cd "$ROOT/external_baselines/lora_drs" || exit 1
  exec "$PY" main.py --config configs/cifar100_t10_seed1.json --device 2 \
    > "$ROOT/baseline_lora_drs_c100_seed1.log" 2>&1
) &
pid_drs=$!

status_inf=0
status_cl=0
status_drs=0
wait "$pid_inf" || status_inf=$?
wait "$pid_cl" || status_cl=$?
wait "$pid_drs" || status_drs=$?

echo "===== $(date +%F_%T) external-baseline C100 queue end"
echo "inf=$status_inf cl=$status_cl drs=$status_drs"
if [ "$status_inf" -ne 0 ] || [ "$status_cl" -ne 0 ] || [ "$status_drs" -ne 0 ]; then
  exit 1
fi
