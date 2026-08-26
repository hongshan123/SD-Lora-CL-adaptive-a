#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL-coordinate"
PID_FILE="$PROJECT_ROOT/coordinate_stable_normcap_multiseed_queue.pid"
MONITOR_LOG="$PROJECT_ROOT/coordinate_stable_normcap_multiseed_monitor.log"

while true; do
  timestamp="$(date '+%F %T')"
  queue_pid=""
  if [ -f "$PID_FILE" ]; then
    queue_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  fi
  {
    echo "===== $timestamp ====="
    if [ -n "$queue_pid" ] && kill -0 "$queue_pid" 2>/dev/null; then
      echo "queue_pid=$queue_pid status=running"
    else
      echo "queue_pid=${queue_pid:-missing} status=stopped"
    fi
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
      --format=csv,noheader 2>/dev/null || true
    for log in "$PROJECT_ROOT"/*coordinate_stable_normcap_seed[123]_nccl.log; do
      [ -f "$log" ] || continue
      echo "log=$(basename "$log")"
      grep -E "Learning on|absorption task|operator alignment|CNN:|NME:|Traceback|Error" "$log" \
        | tail -n 12 || true
    done
  } >> "$MONITOR_LOG"
  if [ -z "$queue_pid" ] || ! kill -0 "$queue_pid" 2>/dev/null; then
    exit 0
  fi
  sleep 1800
done
