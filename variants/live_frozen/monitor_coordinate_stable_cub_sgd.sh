#!/usr/bin/env bash
set -euo pipefail

QUEUE_PID="${1:?queue pid required}"
PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL-coordinate"

cd "$PROJECT_ROOT"

while kill -0 "$QUEUE_PID" 2>/dev/null; do
  echo "===== MONITOR $(date '+%F %T') queue_pid=$QUEUE_PID ====="
  ps -o pid,ppid,stat,etime,cmd -p "$QUEUE_PID" || true
  pgrep -af "torchrun|main.py --config=/tmp/cub_coordinate_stable_sgd" || true
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  current_log="$(ls -t cub_coordinate_stable_sgd_seed*.log 2>/dev/null | head -n 1 || true)"
  if [ -n "$current_log" ]; then
    echo "active_log=$current_log"
    tail -n 12 "$current_log"
  fi
  sleep 1800
done

echo "===== MONITOR $(date '+%F %T') queue finished ====="
tail -n 50 coordinate_stable_cub_sgd_queue.log || true
