#!/usr/bin/env python3
"""Generate matched first-task-anchor and cumulative-G screening configs."""

import json
import sys
from pathlib import Path


DATASETS = {
    "c100": {
        "source": "c100_coordinate_stable_adaptive_a_seed1993_nccl.json",
        "seed": 1993,
    },
    "inr": {
        "source": "inr_coordinate_stable_adaptive_a_seed1995_nccl.json",
        "seed": 1995,
    },
    "cub": {
        "source": "cub_coordinate_stable_adaptive_a_seed1_nccl.json",
        "seed": 1,
    },
}

METHODS = {
    "sa_lora_bank": {
        "sa_cumulative_state": False,
        "sa_cumulative_merge": "gauge",
        "sa_freeze_old_scales": False,
        "sa_normalize_current_branch": True,
        "sa_live_a_absorb_mode": "normalized_absorb",
    },
    "frozen_b_bank": {
        "sa_cumulative_state": False,
        "sa_cumulative_merge": "gauge",
        "sa_freeze_old_scales": True,
        "sa_normalize_current_branch": True,
        "sa_live_a_absorb_mode": "normalized_absorb",
    },
    "exact_g": {
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_freeze_old_scales": True,
        "sa_normalize_current_branch": True,
        "sa_live_a_absorb_mode": "normalized_absorb",
    },
    "bounded_g": {
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_freeze_old_scales": True,
        "sa_normalize_current_branch": False,
        "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
    },
}


def build_config(
    source, method, dataset_key, runtime_root, run_tag, batch_size, world_size
):
    spec = DATASETS[dataset_key]
    name = "anchor_{}_{}_seed{}_bs{}_{}gpu_{}".format(
        method, dataset_key, spec["seed"], batch_size, world_size, run_tag
    )
    config = dict(source)
    for key in list(config):
        if key.startswith("sa_adaptive_a_"):
            config.pop(key)
    config.pop("sa_dual_head_schedule", None)
    config.update(
        {
            "prefix": name,
            "device": [str(index) for index in range(world_size)],
            "seed": [spec["seed"]],
            "filepath": str((runtime_root / "results" / name).resolve()) + "/",
            "batch_size": int(batch_size),
            "lora_rank": 10,
            "model_name": "sa_sdlora",
            "sa_shared_a_orthogonal": True,
            "sa_train_a_all_tasks": False,
            "sa_delete_per_task_files": bool(METHODS[method]["sa_cumulative_state"]),
            "sa_use_prototype_classifier": True,
            "sa_live_a_history_groups": 1,
            "sa_live_a_coordinate_align": False,
            "sa_coordinate_stable_transport": False,
            "sa_hbd_enabled": False,
            "sa_dual_head": False,
            "sa_adaptive_a_enabled": False,
            "sa_deterministic_training": True,
            "sa_resume": False,
        }
    )
    config.update(METHODS[method])
    return name, config


def generate_configs(root, runtime_root, run_tag, batch_size=128, world_size=1):
    root = Path(root).resolve()
    runtime_root = Path(runtime_root).resolve()
    config_root = runtime_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "results").mkdir(parents=True, exist_ok=True)
    manifest = []
    for method in METHODS:
        for dataset_key, spec in DATASETS.items():
            source = json.loads((root / "exps" / spec["source"]).read_text())
            name, config = build_config(
                source,
                method,
                dataset_key,
                runtime_root,
                run_tag,
                batch_size,
                world_size,
            )
            config_path = config_root / (name + ".json")
            config_path.write_text(json.dumps(config, indent=4) + "\n")
            manifest.append(
                {
                    "name": name,
                    "canonical_name": name,
                    "dataset": dataset_key,
                    "method": method,
                    "seed": spec["seed"],
                    "config": str(config_path),
                    "output": config["filepath"],
                }
            )
    (runtime_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main(argv):
    if len(argv) not in (4, 5, 6):
        raise SystemExit(
            "usage: generate_anchor_subspace_configs.py ROOT RUNTIME_DIR "
            "RUN_TAG [BATCH_SIZE] [WORLD_SIZE]"
        )
    batch_size = int(argv[4]) if len(argv) == 5 else 128
    if len(argv) == 6:
        batch_size = int(argv[4])
    world_size = int(argv[5]) if len(argv) == 6 else 1
    manifest = generate_configs(
        argv[1], argv[2], argv[3], batch_size, world_size
    )
    print("generated {} configs".format(len(manifest)))


if __name__ == "__main__":
    main(sys.argv)
