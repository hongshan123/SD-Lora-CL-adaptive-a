#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL-coordinate"
QUEUE_PID="${1:?queue pid required}"

cd "$PROJECT_ROOT" || exit 1

check_once() {
  echo "===== MONITOR $(date +%F_%T) queue_pid=$QUEUE_PID ====="
  if kill -0 "$QUEUE_PID" 2>/dev/null; then
    echo "queue_alive=yes"
  else
    echo "queue_alive=no"
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader
  tail -n 12 coordinate_stable_single_seed_queue.log 2>/dev/null || true
  active=$(find . -maxdepth 1 -type f -name '*coordinate_stable*nccl.log' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
  if [ -n "$active" ]; then
    echo "active_log=$active"
    tail -n 15 "$active"
    grep -Eni \
      'traceback|cuda out of memory|nccl.*error|runtimeerror|exception|killed' \
      "$active" || true
  fi
  df -h /home | tail -n 1
}

while kill -0 "$QUEUE_PID" 2>/dev/null; do
  check_once
  sleep 1800
done
check_once
echo "coordinate-stable monitor exiting"
