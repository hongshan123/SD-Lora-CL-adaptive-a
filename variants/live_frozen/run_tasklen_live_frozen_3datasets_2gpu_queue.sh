#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-/home/hongzhijun/miniconda3/envs/sdlora/bin/python}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')_live_frozen}"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.runtime_tasklen_live_frozen_3datasets_$RUN_TAG}"
QUEUE_LOG="${QUEUE_LOG:-$PROJECT_ROOT/tasklen_live_frozen_3datasets_${RUN_TAG}.log}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORLD_SIZE="${WORLD_SIZE:-2}"
GPU_PAIRS="${GPU_PAIRS:-0,1;4,5;6,7}"

cd "$PROJECT_ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

"$PYTHON_BIN" scripts/generate_live_frozen_tasklen_configs.py \
    "$PROJECT_ROOT" "$RUNTIME_DIR" "$RUN_TAG" "$BATCH_SIZE" "$WORLD_SIZE"

"$PYTHON_BIN" - "$RUNTIME_DIR/manifest.json" "$PROJECT_ROOT" "$GPU_PAIRS" "$WORLD_SIZE" "$QUEUE_LOG" <<'PY'
import json
import sys
from pathlib import Path

from scripts.tasklen_fla_queue import dispatch_jobs

manifest_path = Path(sys.argv[1])
project_root = Path(sys.argv[2]).resolve()
gpu_pairs = tuple(value for value in sys.argv[3].split(";") if value)
world_size = int(sys.argv[4])
queue_log = Path(sys.argv[5])
manifest = json.loads(manifest_path.read_text())

if len(manifest) != 24:
    raise SystemExit(f"expected 24 jobs, got {len(manifest)}")
if any(item["dataset"] not in {"c100", "inr", "cub"} for item in manifest):
    raise SystemExit("manifest contains an unexpected dataset")
if any(item["method"] not in {"live_a", "frozen_a"} for item in manifest):
    raise SystemExit("manifest contains an unexpected method")
if any(len(pair.split(",")) != world_size for pair in gpu_pairs):
    raise SystemExit("each GPU pair must contain world-size GPU IDs")

for item in manifest:
    config = json.loads(Path(item["config"]).read_text())
    assert config["batch_size"] == 64
    assert config["epochs"] == 20 and config["init_epoch"] == 20
    assert config["optimizer"] == "sgd" and config["lora_rank"] == 10
    assert config["sa_adaptive_a_enabled"] is False
    assert config["sa_live_a_coordinate_align"] is True
    assert config["sa_live_a_absorb_mode"] == "operator_preserving_absorb"
    assert config["sa_dual_head"] is False
    expected_live = item["method"] == "live_a"
    assert config["sa_train_a_all_tasks"] is expected_live
    log_path = project_root / (item["name"] + ".log")
    output_path = Path(config["filepath"])
    if log_path.exists() or output_path.exists():
        raise SystemExit(f"refusing to overwrite {item['name']}")

queue_log.parent.mkdir(parents=True, exist_ok=True)
with queue_log.open("w") as stream:
    stream.write(
        "QUEUE START jobs={} datasets=c100,inr,cub methods=live_a,frozen_a "
        "tasks=5,10,20,50 gpu_pairs={} batch_size={} world_size={}\n".format(
            len(manifest), ",".join(gpu_pairs), 64, world_size
        )
    )
    for item in manifest:
        stream.write(
            "PENDING {} dataset={} tasks={} method={} GPUs=dynamic\n".format(
                item["name"], item["dataset"], item["tasks"], item["method"]
            )
        )

status = dispatch_jobs(
    project_root,
    manifest,
    gpu_pairs,
    world_size,
)
with queue_log.open("a") as stream:
    stream.write("QUEUE END status={} jobs={}\n".format(status, len(manifest)))
raise SystemExit(status)
PY
