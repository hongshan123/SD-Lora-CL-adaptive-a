#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
USER_HOME_DIR="${USER_HOME_DIR:-/home/$(id -un)}"
CONDA_SH="${CONDA_SH:-$USER_HOME_DIR/miniconda3/etc/profile.d/conda.sh}"
GPU_IDS="${GPU_IDS:-0,1}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"
RUNTIME_DIR="$PROJECT_ROOT/.runtime_nodual_core_then_threeway_bs128"

IFS=',' read -r -a gpu_id_list <<< "$GPU_IDS"
if [ "${#gpu_id_list[@]}" -ne 2 ]; then
  echo "This queue requires exactly two independent GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi
for gpu_id in "${gpu_id_list[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "GPU IDs must be numeric; got: $GPU_IDS" >&2
    exit 2
  fi
done
if [ ! -f "$CONDA_SH" ]; then
  echo "Conda initialization script not found: $CONDA_SH" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate sdlora
cd "$PROJECT_ROOT"
mkdir -p "$RUNTIME_DIR"

python - "$PROJECT_ROOT" "$RUNTIME_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime_dir = Path(sys.argv[2])
datasets = {
    "c100": (1993, 1994, 1995),
    "inr": (1995, 1996, 1997),
    "cub": (1, 2, 3),
}
core_variants = {
    "adaptive_a": dict(train_a=True, adaptive=True, align=True, transport=True,
                       absorb="bounded_norm_calibrated_absorb"),
    "adaptive_no_transport": dict(train_a=True, adaptive=True, align=True,
                                  transport=False,
                                  absorb="bounded_norm_calibrated_absorb"),
    "adaptive_no_normcap": dict(train_a=True, adaptive=True, align=True,
                                transport=True,
                                absorb="operator_preserving_absorb"),
    "live_a": dict(train_a=True, adaptive=False, align=True, transport=True,
                   absorb="bounded_norm_calibrated_absorb"),
    "live_coordinate_only": dict(train_a=True, adaptive=False, align=True,
                                 transport=False,
                                 absorb="bounded_norm_calibrated_absorb"),
    "live_transport_only": dict(train_a=True, adaptive=False, align=False,
                                transport=True,
                                absorb="bounded_norm_calibrated_absorb"),
    "live_none": dict(train_a=True, adaptive=False, align=False,
                      transport=False,
                      absorb="bounded_norm_calibrated_absorb"),
}
paired_variants = {
    "live_a": core_variants["live_a"],
    "frozen_a": dict(train_a=False, adaptive=False, align=True, transport=True,
                     absorb="bounded_norm_calibrated_absorb"),
    "adaptive_a": core_variants["adaptive_a"],
}


def source_config(dataset, seed):
    return root / ".runtime_adaptive_a_multiseed_single_gpu_bs128" / (
        f"{dataset}_coordinate_stable_adaptive_a_multiseed_seed{seed}_single_bs128.json"
    )


def build(dataset, seed, variant, settings):
    source = source_config(dataset, seed)
    if not source.is_file():
        raise SystemExit(f"missing source config: {source}")
    config = json.loads(source.read_text())
    required = {
        "batch_size": 128,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_use_prototype_classifier": True,
        "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise SystemExit(f"unexpected source protocol {source}: {mismatches}")
    if any("aob" in key.lower() and config[key] for key in config):
        raise SystemExit(f"AOB field enabled in source protocol: {source}")

    name = f"nodual_{dataset}_{variant}_seed{seed}_single_bs128"
    config.update(
        {
            "prefix": name,
            "filepath": f"./{name.upper()}/",
            "device": ["0"],
            "batch_size": 128,
            "sa_dual_head": False,
            "sa_use_prototype_classifier": True,
            "sa_train_a_all_tasks": settings["train_a"],
            "sa_adaptive_a_enabled": settings["adaptive"],
            "sa_live_a_coordinate_align": settings["align"],
            "sa_coordinate_stable_transport": settings["transport"],
            "sa_live_a_absorb_mode": settings["absorb"],
            "sa_resume": False,
        }
    )
    config.pop("sa_dual_head_schedule", None)
    output = runtime_dir / f"{name}.json"
    output.write_text(json.dumps(config, indent=4) + "\n")
    return output


phase1 = []
for dataset, seeds in datasets.items():
    for variant, settings in core_variants.items():
        phase1.append(build(dataset, seeds[0], variant, settings))

phase2 = []
for dataset, seeds in datasets.items():
    for seed in seeds:
        for variant, settings in paired_variants.items():
            phase2.append(build(dataset, seed, variant, settings))

(runtime_dir / "phase1_core_single_seed.txt").write_text(
    "\n".join(map(str, phase1)) + "\n"
)
(runtime_dir / "phase2_threeway_paired.txt").write_text(
    "\n".join(map(str, phase2)) + "\n"
)
print(f"generated phase1={len(phase1)} phase2={len(phase2)}")
PY

mapfile -t phase1_configs < "$RUNTIME_DIR/phase1_core_single_seed.txt"
mapfile -t phase2_configs < "$RUNTIME_DIR/phase2_threeway_paired.txt"

is_complete() {
  local name="$1"
  local output_dir="$PROJECT_ROOT/${name^^}"
  local log_file="$PROJECT_ROOT/${name}.log"
  [ -f "$output_dir/run_manifest.json" ] \
    && [ -f "$log_file" ] \
    && grep -q "Forgetting (CNN):" "$log_file"
}

preflight_one() {
  local config="$1"
  local name output_dir log_file
  name="$(basename "$config" .json)"
  output_dir="$PROJECT_ROOT/${name^^}"
  log_file="$PROJECT_ROOT/${name}.log"
  if is_complete "$name"; then
    return 0
  fi
  if [ -e "$output_dir" ] || [ -e "$log_file" ]; then
    echo "Preflight failed: incomplete output or log exists for $name" >&2
    return 1
  fi
  if pgrep -f "[t]orchrun.*${config}" >/dev/null; then
    echo "Preflight failed: duplicate process exists for $name" >&2
    return 1
  fi
}

for config in "${phase1_configs[@]}" "${phase2_configs[@]}"; do
  preflight_one "$config"
done

if [ "$GENERATE_ONLY" = "1" ]; then
  echo "Generation and preflight completed; GENERATE_ONLY=1"
  exit 0
fi

export HF_ENDPOINT PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1
code_sha="$(sha256sum backbone/sa_lora.py models/sa_sdlora.py | sha256sum | awk '{print $1}')"

run_one() {
  local config="$1"
  local gpu_id="$2"
  local name log_file status
  name="$(basename "$config" .json)"
  log_file="$PROJECT_ROOT/${name}.log"
  if is_complete "$name"; then
    echo "===== $(date '+%F %T') SKIP complete $name ====="
    return 0
  fi
  echo "===== $(date '+%F %T') START $name GPU=$gpu_id batch_size=128 ====="
  {
    echo "===== $(date '+%F %T') START $name GPU=$gpu_id batch_size=128 ====="
    echo "code_sha=$code_sha config_sha=$(sha256sum "$config" | awk '{print $1}')"
    if CUDA_VISIBLE_DEVICES="$gpu_id" torchrun --standalone --nproc_per_node=1 \
      main.py --config="$config"; then
      status=0
    else
      status=$?
    fi
    echo "===== $(date '+%F %T') END $name status=$status ====="
  } > "$log_file" 2>&1
  echo "===== $(date '+%F %T') END $name status=$status ====="
  return "$status"
}

run_worker() {
  local gpu_id="$1"
  local parity="$2"
  shift 2
  local index=0 config
  for config in "$@"; do
    if [ $((index % 2)) -eq "$parity" ]; then
      run_one "$config" "$gpu_id"
    fi
    index=$((index + 1))
  done
}

run_phase() {
  local phase_name="$1"
  shift
  local worker0 worker1 status0=0 status1=0
  echo "===== $(date '+%F %T') PHASE START $phase_name count=$# ====="
  run_worker "${gpu_id_list[0]}" 0 "$@" &
  worker0=$!
  run_worker "${gpu_id_list[1]}" 1 "$@" &
  worker1=$!
  wait "$worker0" || status0=$?
  wait "$worker1" || status1=$?
  if [ "$status0" -ne 0 ] || [ "$status1" -ne 0 ]; then
    echo "Phase failed: $phase_name worker0=$status0 worker1=$status1" >&2
    return 1
  fi
  echo "===== $(date '+%F %T') PHASE END $phase_name status=0 ====="
}

run_phase core_single_seed "${phase1_configs[@]}"
run_phase threeway_paired "${phase2_configs[@]}"
echo "===== $(date '+%F %T') NODUAL CORE AND THREEWAY QUEUE DONE ====="
