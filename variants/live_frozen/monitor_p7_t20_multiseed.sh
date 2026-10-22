#!/usr/bin/env bash
# Record an immediate health check and then one check every 30 minutes.
set -uo pipefail

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL"
QUEUE_PID="${1:?queue PID is required}"
INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-1800}"

cd "$PROJECT_ROOT" || exit 1

check_once() {
  echo "===== MONITOR $(date +%F_%T) queue_pid=$QUEUE_PID ====="
  if kill -0 "$QUEUE_PID" 2>/dev/null; then
    echo "queue_alive=yes"
  else
    echo "queue_alive=no"
  fi
  echo "[processes]"
  pgrep -af 'torchrun.*p7_t20_|main.py.*p7_t20_' || true
  echo "[gpu]"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader || true
  echo "[queue_tail]"
  tail -n 12 p7_t20_multiseed_queue.log 2>/dev/null || true
  active_log=$(find . -maxdepth 1 -type f -name 'p7_t20_*_nccl.log' \
    -printf '%T@ %f\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)
  if [ -n "$active_log" ]; then
    echo "[active_log] $active_log"
    tr '\r' '\n' < "$active_log" | tail -n 12
    echo "[active_errors]"
    grep -Ein 'traceback|runtimeerror|out of memory|killed|exception|nccl.*error' \
      "$active_log" | tail -n 20 || true
  fi
  echo "[disk]"
  df -h /home/zhaoyang | tail -n 1
  echo
}

while kill -0 "$QUEUE_PID" 2>/dev/null; do
  check_once
  sleep "$INTERVAL_SECONDS"
done

check_once
echo "===== MONITOR EXIT $(date +%F_%T) ====="
