#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONDA_HOME=${CONDA_HOME:-/home/hongzhijun/miniconda3}

if [[ ${1:-} == __worker ]]; then
  shift
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate sdlora
  export HF_ENDPOINT=https://hf-mirror.com
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  exec nohup "$CONDA_PREFIX/bin/python" -u "$SCRIPT_DIR/scripts/imageneta_live_frozen.py" "$@"
fi

RUNTIME=${1:?Usage: bash run_imageneta_live_frozen_queue.sh RUNTIME smoke\|formal\|monitor [GPU_PAIRS]}
PHASE=${2:-formal}
GPU_PAIRS=${3:-}
case "$PHASE" in
  smoke|formal|monitor) ;;
  *) printf 'Unknown phase: %s\n' "$PHASE" >&2; exit 2 ;;
esac
RUNTIME=$(realpath "$RUNTIME")
SOURCE=$("$CONDA_HOME/envs/sdlora/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$RUNTIME/manifest.json")
if [[ -z "$GPU_PAIRS" ]]; then
  GPU_PAIRS=$("$CONDA_HOME/envs/sdlora/bin/python" -c 'import json,sys; print(";".join(json.load(open(sys.argv[1])).get("gpu_pairs", ["0,1", "4,5", "6,7"])))' "$RUNTIME/manifest.json")
fi
TAG=$(basename "$RUNTIME")
UNIT="ina-${PHASE}-${TAG#.runtime_}"
ACTION=queue
[[ "$PHASE" == monitor ]] && ACTION=monitor
mkdir -p "$RUNTIME/logs"

systemd-run --user --collect --unit="$UNIT" \
  --property="WorkingDirectory=$SOURCE" \
  --property="StandardOutput=append:$RUNTIME/${PHASE}_queue.log" \
  --property="StandardError=append:$RUNTIME/${PHASE}_queue.log" \
  --property=KillMode=control-group \
  --property=TimeoutStopSec=45 \
  --setenv="CONDA_HOME=$CONDA_HOME" \
  nohup bash "$SOURCE/run_imageneta_live_frozen_queue.sh" __worker \
    "$ACTION" --runtime "$RUNTIME" --phase "${PHASE/monitor/formal}" --gpu-pairs "$GPU_PAIRS"
printf 'Unit: %s.service\nLog: %s/%s_queue.log\n' "$UNIT" "$RUNTIME" "$PHASE"
