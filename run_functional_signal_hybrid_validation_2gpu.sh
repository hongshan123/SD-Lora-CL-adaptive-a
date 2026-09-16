#!/usr/bin/env bash
# Single-seed, three-dataset validation of Historical-only versus two HBD hybrids.
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_functional_signal_hybrid_validation"
QUEUE_LOG="$PROJECT_ROOT/functional_signal_hybrid_validation_queue.log"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

source "$CONDA_SH"
conda activate sdlora
cd "$PROJECT_ROOT"
mkdir -p "$RUNTIME_DIR"

python - "$PROJECT_ROOT" "$RUNTIME_DIR" <<'PY'
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
arms = (
    ("historical", "historical", 0.5),
    ("fixed_hybrid", "fixed_hybrid", 0.5),
    ("reliability_hybrid", "reliability_hybrid", 0.5),
)

for dataset, source_name, seed in datasets:
    source = json.loads((root / "exps" / source_name).read_text())
    for arm, signal, weight in arms:
        name = f"{dataset}_fs_{arm}_seed{seed}_t10_i4_bs128_2gpu"
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
                "sa_adaptive_a_strategy": "function_safe_pareto",
                "sa_functional_student_scope": "historical",
                "sa_functional_stability_signal": signal,
                "sa_functional_hybrid_weight": weight,
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

datasets=("c100" "inr" "cub")
arms=("historical" "fixed_hybrid" "reliability_hybrid")
seeds=("1993" "1995" "1")
gpu_pairs=("0,1" "2,3" "4,5" "6,7")
jobs=()

for dataset_index in "${!datasets[@]}"; do
    dataset="${datasets[$dataset_index]}"
    seed="${seeds[$dataset_index]}"
    for arm in "${arms[@]}"; do
        jobs+=("$dataset|$arm|$seed")
    done
done

experiment_name() {
    printf '%s_fs_%s_seed%s_t10_i4_bs128_2gpu' "$1" "$2" "$3"
}

preflight_one() {
    local dataset="$1" arm="$2" seed="$3"
    local name config log output_dir
    name="$(experiment_name "$dataset" "$arm" "$seed")"
    config="$RUNTIME_DIR/${name}.json"
    log="$PROJECT_ROOT/${name}.log"
    output_dir="$PROJECT_ROOT/${name^^}"
    if [ -e "$log" ] || [ -e "$output_dir" ]; then
        echo "Refusing to overwrite existing output for $name" >&2
        return 2
    fi
    python - "$config" "$arm" "$seed" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
arm, seed = sys.argv[2], int(sys.argv[3])
assert config["seed"] == [seed]
assert config["batch_size"] == 64
assert config["max_tasks"] == 10
assert config["epochs"] == 20
assert config["optimizer"] == "sgd"
assert config["sa_dual_head"] is False
assert config["sa_hbd_enabled"] is False
assert config["sa_adaptive_a_strategy"] == "function_safe_pareto"
assert config["sa_functional_student_scope"] == "historical"
assert config["sa_functional_stability_signal"] == arm
assert config["sa_train_a_all_tasks"] is True
assert config["sa_cumulative_merge"] == "live_a_aggregate_b"
assert config["sa_live_a_coordinate_align"] is True
assert config["sa_live_a_absorb_mode"] == "bounded_norm_calibrated_absorb"
assert config["sa_coordinate_stable_transport"] is True
PY
    echo "preflight PASS name=$name"
}

for job in "${jobs[@]}"; do
    IFS='|' read -r dataset arm seed <<< "$job"
    preflight_one "$dataset" "$arm" "$seed" || exit $?
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
    local dataset="$1" arm="$2" seed="$3" gpu_ids="$4"
    local name config log status
    name="$(experiment_name "$dataset" "$arm" "$seed")"
    config="$RUNTIME_DIR/${name}.json"
    log="$PROJECT_ROOT/${name}.log"
    {
        echo "===== $(date '+%F %T') START $name GPUs=$gpu_ids commit=$(git rev-parse HEAD) ====="
        echo "config_sha=$(sha256sum "$config" | awk '{print $1}')"
        CUDA_VISIBLE_DEVICES="$gpu_ids" torchrun --standalone --nproc_per_node=2 \
            main.py --config="$config"
        status=$?
        echo "===== $(date '+%F %T') END $name GPUs=$gpu_ids status=$status ====="
        return "$status"
    } > "$log" 2>&1
}

echo "===== $(date '+%F %T') QUEUE START commit=$(git rev-parse HEAD) =====" > "$QUEUE_LOG"
overall_status=0
next=0
while [ "$next" -lt "${#jobs[@]}" ]; do
    pids=()
    for slot in "${!gpu_pairs[@]}"; do
        [ "$next" -lt "${#jobs[@]}" ] || break
        IFS='|' read -r dataset arm seed <<< "${jobs[$next]}"
        gpu_ids="${gpu_pairs[$slot]}"
        run_one "$dataset" "$arm" "$seed" "$gpu_ids" &
        pids+=("$!")
        echo "launched dataset=$dataset arm=$arm seed=$seed GPUs=$gpu_ids pid=${pids[-1]}" >> "$QUEUE_LOG"
        next=$((next + 1))
    done
    wave_status=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            wave_status=1
        fi
    done
    echo "===== $(date '+%F %T') WAVE END status=$wave_status =====" >> "$QUEUE_LOG"
    if [ "$wave_status" -ne 0 ]; then
        overall_status=1
        break
    fi
done
echo "===== $(date '+%F %T') QUEUE END status=$overall_status =====" >> "$QUEUE_LOG"
exit "$overall_status"
