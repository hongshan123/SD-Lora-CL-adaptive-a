#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL-coordinate"
QUEUE_LOG="$PROJECT_ROOT/coordinate_stable_core_ablations_queue.log"
INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-1800}"

cd "$PROJECT_ROOT" || exit 1
while true; do
  echo "===== $(date '+%F %T') CORE ABLATION MONITOR ====="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>&1
  tail -n 14 "$QUEUE_LOG" 2>/dev/null || true
  active_log="$(ls -1t *_abl_coordinate_only_seed*_nccl.log \
    *_abl_transport_only_seed*_nccl.log 2>/dev/null | head -n 1 || true)"
  if [ -n "$active_log" ]; then
    echo "active_log=$active_log"
    tail -n 12 "$active_log" 2>/dev/null || true
  fi
  if grep -q "Core CoordinateStable ablation queue DONE" "$QUEUE_LOG" 2>/dev/null; then
    echo "Core ablation monitor DONE"
    exit 0
  fi
  if grep -q "Core ablation queue FAIL" "$QUEUE_LOG" 2>/dev/null; then
    echo "Core ablation monitor detected FAIL"
    exit 1
  fi
  sleep "$INTERVAL_SECONDS"
done
