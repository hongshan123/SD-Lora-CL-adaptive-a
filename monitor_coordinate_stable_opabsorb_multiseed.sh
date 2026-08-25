#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL-coordinate"
PID_FILE="$PROJECT_ROOT/coordinate_stable_opabsorb_multiseed_followup.pid"
MONITOR_LOG="$PROJECT_ROOT/coordinate_stable_opabsorb_multiseed_monitor.log"

while true; do
  timestamp="$(date '+%F %T')"
  followup_pid=""
  if [ -f "$PID_FILE" ]; then
    followup_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  fi
  {
    echo "===== $timestamp ====="
    if [ -n "$followup_pid" ] && kill -0 "$followup_pid" 2>/dev/null; then
      echo "followup_pid=$followup_pid status=running-or-waiting"
    else
      echo "followup_pid=${followup_pid:-missing} status=stopped"
    fi
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
      --format=csv,noheader 2>/dev/null || true
    for log in "$PROJECT_ROOT"/*coordinate_stable_opabsorb_seed[123]_nccl.log; do
      [ -f "$log" ] || continue
      echo "log=$(basename "$log")"
      grep -E "Learning on|absorption task|operator alignment|CNN:|NME:|Traceback|Error" "$log" \
        | tail -n 12 || true
    done
  } >> "$MONITOR_LOG"
  if [ -z "$followup_pid" ] || ! kill -0 "$followup_pid" 2>/dev/null; then
    exit 0
  fi
  sleep 1800
done
