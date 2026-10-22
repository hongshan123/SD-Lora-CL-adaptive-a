#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_functional_signal_threeway_single_seed"
QUEUE_LOG="$PROJECT_ROOT/functional_signal_threeway_single_seed_queue.log"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

if [ ! -f "$CONDA_SH" ]; then
    echo "Conda initialization script not found: $CONDA_SH" >&2
    exit 1
fi

source "$CONDA_SH"
conda activate sdlora
cd "$PROJECT_ROOT"
mkdir -p "$RUNTIME_DIR"

if ! python - "$PROJECT_ROOT" "$RUNTIME_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime = Path(sys.argv[2])
datasets = (
    ("c100", "c100_coordinate_stable_adaptive_a_seed1993_nccl.json", 1993),
    ("inr", "inr_coordinate_stable_adaptive_a_seed1995_nccl.json", 1995),
    ("cub", "cub_coordinate_stable_adaptive_a_seed1_nccl.json", 1),
)
variants = (
    ("full_logit", "function_safe_pareto", "full"),
    ("historical_logit", "function_safe_pareto", "historical"),
    ("hbd", "functional_halfspace", "full"),
)

for dataset, source_name, seed in datasets:
    source = json.loads((root / "exps" / source_name).read_text())
    for variant, strategy, scope in variants:
        name = (
            f"{dataset}_functional_signal_{variant}_seed{seed}"
            "_t10_i4_bs128_2gpu"
        )
        config = dict(source)
        config.update(
            {
                "prefix": name,
                "filepath": f"./{name.upper()}/",
                "device": ["0", "1"],
                "seed": [seed],
                "batch_size": 64,
                "max_tasks": 10,
                "sa_dual_head": False,
                "sa_hbd_enabled": False,
                "sa_adaptive_a_enabled": True,
                "sa_adaptive_a_strategy": strategy,
                "sa_functional_student_scope": scope,
                "sa_adaptive_a_crossfit_interval": 4,
                "sa_functional_diagnostics_interval": 0,
                "sa_functional_conflict_cosine": 0.05,
                "sa_functional_temperature": 2.0,
                "sa_resume": False,
            }
        )
        config.pop("sa_dual_head_schedule", None)
        (runtime / f"{name}.json").write_text(
            json.dumps(config, indent=4) + "\n"
        )
PY
then
    echo "Failed to generate runtime configs" >&2
    exit 2
fi

datasets=("c100|1993" "inr|1995" "cub|1")
variants=(
    "full_logit|function_safe_pareto|full|0,1"
    "historical_logit|function_safe_pareto|historical|2,3"
    "hbd|functional_halfspace|full|4,5"
)

experiment_name() {
    printf '%s_functional_signal_%s_seed%s_t10_i4_bs128_2gpu' "$1" "$2" "$3"
}

preflight_one() {
    local dataset="$1" seed="$2" variant="$3" strategy="$4" scope="$5" gpu_ids="$6"
    local name config log output_dir gpu_count
    name="$(experiment_name "$dataset" "$variant" "$seed")"
    config="$RUNTIME_DIR/${name}.json"
    log="$PROJECT_ROOT/${name}.log"
    output_dir="$PROJECT_ROOT/${name^^}"
    IFS=',' read -r -a gpu_list <<< "$gpu_ids"
    gpu_count="${#gpu_list[@]}"
    if [ "$gpu_count" -ne 2 ]; then
        echo "$name requires exactly two GPUs; got $gpu_ids" >&2
        return 2
    fi
    if [ -e "$log" ] || [ -e "$output_dir" ]; then
        echo "Refusing to overwrite existing output for $name" >&2
        return 2
    fi
    if pgrep -f "[t]orchrun.*${name}.json" >/dev/null; then
        echo "Duplicate experiment process exists for $name" >&2
        return 2
    fi
    if ! python - "$config" "$seed" "$strategy" "$scope" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
seed = int(sys.argv[2])
strategy = sys.argv[3]
scope = sys.argv[4]
assert config["seed"] == [seed]
assert config["batch_size"] == 64
assert config["max_tasks"] == 10
assert config["epochs"] == 20
assert config["optimizer"] == "sgd"
assert config["sa_dual_head"] is False
assert config["sa_hbd_enabled"] is False
assert config["sa_adaptive_a_enabled"] is True
assert config["sa_adaptive_a_strategy"] == strategy
assert config["sa_functional_student_scope"] == scope
assert config["sa_adaptive_a_crossfit_interval"] == 4
assert config["sa_functional_diagnostics_interval"] == 0
assert config["sa_functional_conflict_cosine"] == 0.05
assert config["sa_functional_temperature"] == 2.0
assert config["sa_train_a_all_tasks"] is True
assert config["sa_cumulative_state"] is True
assert config["sa_cumulative_merge"] == "live_a_aggregate_b"
assert config["sa_live_a_coordinate_align"] is True
assert config["sa_live_a_absorb_mode"] == "bounded_norm_calibrated_absorb"
assert config["sa_coordinate_stable_transport"] is True
PY
    then
        echo "Protocol validation failed for $name" >&2
        return 2
    fi
    echo "preflight PASS name=$name GPUs=$gpu_ids config=$config"
}

for dataset_spec in "${datasets[@]}"; do
    IFS='|' read -r dataset seed <<< "$dataset_spec"
    for variant_spec in "${variants[@]}"; do
        IFS='|' read -r variant strategy scope gpu_ids <<< "$variant_spec"
        preflight_one "$dataset" "$seed" "$variant" "$strategy" "$scope" "$gpu_ids" || exit $?
    done
done

if [ "$PREPARE_ONLY" = "1" ]; then
    echo "PREPARE_ONLY complete"
    exit 0
fi
if [ -e "$QUEUE_LOG" ]; then
    echo "Refusing to overwrite queue log: $QUEUE_LOG" >&2
    exit 2
fi

export HF_ENDPOINT PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1

run_one() {
    local dataset="$1" seed="$2" variant="$3" gpu_ids="$4"
    local name config log status
    name="$(experiment_name "$dataset" "$variant" "$seed")"
    config="$RUNTIME_DIR/${name}.json"
    log="$PROJECT_ROOT/${name}.log"
    {
        echo "===== $(date '+%F %T') START $name GPUs=$gpu_ids commit=$(git rev-parse HEAD) ====="
        echo "config_sha=$(sha256sum "$config" | awk '{print $1}')"
        CUDA_VISIBLE_DEVICES="$gpu_ids" torchrun --standalone \
            --nproc_per_node=2 main.py --config="$config"
        status=$?
        echo "===== $(date '+%F %T') END $name GPUs=$gpu_ids status=$status ====="
        return "$status"
    } > "$log" 2>&1
}

echo "===== $(date '+%F %T') QUEUE START commit=$(git rev-parse HEAD) =====" > "$QUEUE_LOG"
overall_status=0
for dataset_spec in "${datasets[@]}"; do
    IFS='|' read -r dataset seed <<< "$dataset_spec"
    echo "===== $(date '+%F %T') WAVE START dataset=$dataset =====" >> "$QUEUE_LOG"
    pids=()
    for variant_spec in "${variants[@]}"; do
        IFS='|' read -r variant strategy scope gpu_ids <<< "$variant_spec"
        run_one "$dataset" "$seed" "$variant" "$gpu_ids" &
        pids+=("$!")
        echo "launched dataset=$dataset variant=$variant GPUs=$gpu_ids pid=${pids[-1]}" >> "$QUEUE_LOG"
    done
    wave_status=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            wave_status=1
        fi
    done
    echo "===== $(date '+%F %T') WAVE END dataset=$dataset status=$wave_status =====" >> "$QUEUE_LOG"
    if [ "$wave_status" -ne 0 ]; then
        overall_status=1
        break
    fi
done
echo "===== $(date '+%F %T') QUEUE END status=$overall_status =====" >> "$QUEUE_LOG"
exit "$overall_status"
