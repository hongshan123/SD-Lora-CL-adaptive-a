#!/usr/bin/env bash
# Lightweight monitor for the P5 CUB multi-seed queue.
set -uo pipefail

ROOT="/home/zhaoyang/SD-Lora-CL"
cd "$ROOT" || exit 1

while :; do
  echo "===== MONITOR $(date '+%F %T') ====="
  if ! pgrep -f 'run_p5_cub_multi_queue' > /dev/null 2>&1; then
    echo "queue process not running"
    tail -n 8 p5_cub_multi_queue.log
    break
  fi
  echo "procs=$(pgrep -fc 'torchrun.*p5_cub')"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  grep -E '^(===== .*START|===== .*END|P5-CUB-MULTI)' p5_cub_multi_queue.log | tail -n 3
  sleep 300
done
