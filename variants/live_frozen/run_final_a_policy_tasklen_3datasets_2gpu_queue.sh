#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-/home/hongzhijun/miniconda3/envs/sdlora/bin/python}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')_final_a_policy}"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.runtime_final_a_policy_tasklen_$RUN_TAG}"
QUEUE_LOG="${QUEUE_LOG:-$PROJECT_ROOT/final_a_policy_tasklen_${RUN_TAG}.log}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORLD_SIZE="${WORLD_SIZE:-2}"
GPU_PAIRS="${GPU_PAIRS:-0,1;4,5;6,7}"
PREDECESSOR_QUEUE_LOG="${PREDECESSOR_QUEUE_LOG:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

cd "$PROJECT_ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

if [[ -n "$PREDECESSOR_QUEUE_LOG" ]]; then
    "$PYTHON_BIN" scripts/wait_for_queue_success.py \
        "$PREDECESSOR_QUEUE_LOG" --poll-seconds "$WAIT_POLL_SECONDS"
fi

"$PYTHON_BIN" scripts/generate_final_a_policy_tasklen_configs.py \
    "$PROJECT_ROOT" "$RUNTIME_DIR" "$RUN_TAG" "$BATCH_SIZE" "$WORLD_SIZE"

if [[ "$PREPARE_ONLY" == "1" ]]; then
    echo "PREPARE_ONLY complete: $RUNTIME_DIR/manifest.json"
    exit 0
fi

"$PYTHON_BIN" - "$RUNTIME_DIR/manifest.json" "$PROJECT_ROOT" \
    "$GPU_PAIRS" "$WORLD_SIZE" "$QUEUE_LOG" <<'PY'
import json
import sys
from pathlib import Path

from scripts.tasklen_fla_queue import (
    active_jobs,
    dispatch_jobs,
    pending_jobs,
)

manifest_path = Path(sys.argv[1])
project_root = Path(sys.argv[2]).resolve()
gpu_pairs = tuple(value for value in sys.argv[3].split(";") if value)
world_size = int(sys.argv[4])
queue_log = Path(sys.argv[5])
manifest = json.loads(manifest_path.read_text())

if len(manifest) != 36:
    raise SystemExit("expected 36 jobs, got {}".format(len(manifest)))
if any(item["dataset"] not in {"c100", "inr", "cub"} for item in manifest):
    raise SystemExit("manifest contains an unexpected dataset")
if any(
    item["method"] not in {"frozen_a", "live_a", "adaptive_a"}
    for item in manifest
):
    raise SystemExit("manifest contains an unexpected method")
if any(len(pair.split(",")) != world_size for pair in gpu_pairs):
    raise SystemExit("each GPU pair must contain world-size GPU IDs")

for item in manifest:
    config = json.loads(Path(item["config"]).read_text())
    assert config["batch_size"] == 64
    assert config["epochs"] == 20 and config["init_epoch"] == 20
    assert config["optimizer"] == "sgd" and config["lora_rank"] == 10
    assert config["sa_dual_head"] is False
    assert config["sa_coordinate_stable_transport"] is True
    assert config["sa_live_a_coordinate_align"] is True
    assert config["sa_live_a_absorb_mode"] == "bounded_norm_calibrated_absorb"

pending = pending_jobs(project_root, manifest)
active_runs = active_jobs(project_root, pending)
active_names = {item["name"] for item, _gpu_ids in active_runs}
jobs = [item for item in pending if item["name"] not in active_names]

queue_log.parent.mkdir(parents=True, exist_ok=True)
with queue_log.open("a") as stream:
    stream.write(
        "QUEUE START jobs={} pending={} active={} datasets=c100,inr,cub "
        "methods=frozen_a,live_a,adaptive_a tasks=5,10,20,50 "
        "gpu_pairs={} batch_size=64 world_size={}\n".format(
            len(manifest), len(jobs), len(active_runs), ",".join(gpu_pairs), world_size
        )
    )
    for item in jobs:
        stream.write(
            "PENDING {} dataset={} tasks={} method={} GPUs=dynamic\n".format(
                item["name"], item["dataset"], item["tasks"], item["method"]
            )
        )
    stream.flush()

status = dispatch_jobs(
    project_root,
    jobs,
    gpu_pairs,
    world_size,
    initial_active=active_runs,
)
with queue_log.open("a") as stream:
    stream.write("QUEUE END status={} jobs={}\n".format(status, len(manifest)))
raise SystemExit(status)
PY
