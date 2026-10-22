#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
if [[ "${1:-}" == "__worker" ]]; then
    shift
    source "$CONDA_HOME/etc/profile.d/conda.sh"
    conda activate sdlora
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec nohup "$CONDA_PREFIX/bin/python" -u "$ROOT/scripts/historical_only_multiseed.py" "$@"
fi

RUNTIME="${RUNTIME:-$ROOT/.runtime_historical_only_seed4_13_20260930}"
PHASE="${1:-formal}"
[[ "$PHASE" == "smoke" || "$PHASE" == "formal" || "$PHASE" == "monitor" ]] || exit 2
[[ -f "$RUNTIME/manifest.json" ]] || { printf 'Missing manifest: %s\n' "$RUNTIME" >&2; exit 2; }
UNIT="historical-only-${PHASE}-$(basename "$RUNTIME" | tr '_' '-')"
ARGS=(queue --runtime "$RUNTIME" --phase "$PHASE")
if [[ "$PHASE" == "monitor" ]]; then
    ARGS=(monitor --runtime "$RUNTIME")
fi
exec systemd-run --user --collect --unit="$UNIT" \
    --property="WorkingDirectory=$ROOT" \
    --property="StandardOutput=append:$RUNTIME/${PHASE}_queue.log" \
    --property="StandardError=append:$RUNTIME/${PHASE}_queue.log" \
    --property="KillMode=control-group" --property="TimeoutStopSec=45" \
    --setenv="CONDA_HOME=$CONDA_HOME" \
    nohup bash "$ROOT/run_historical_only_seed4_13_queue.sh" __worker "${ARGS[@]}"
