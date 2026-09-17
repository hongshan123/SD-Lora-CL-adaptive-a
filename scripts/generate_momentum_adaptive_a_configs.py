#!/usr/bin/env python
"""Generate matched T=10 configs for momentum-aware Adaptive-A."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.domainnet import balanced_task_increments


DATASETS = {
    "c100": {
        "source": "p3_c100_livea_dual_b_seed1_nccl.json",
        "dataset": "cifar224",
        "seed": 1993,
        "num_classes": 100,
        "gpu_ids": "0,1",
    },
    "inr": {
        "source": "p3_inr_livea_dual_b_seed1_nccl.json",
        "dataset": "imagenetr",
        "seed": 1995,
        "num_classes": 200,
        "gpu_ids": "4,5",
    },
    "cub": {
        "source": "p5_cub_livea_dual_b_seed1_nccl.json",
        "dataset": "cub",
        "seed": 1,
        "num_classes": 200,
        "gpu_ids": "6,7",
    },
}


def build_config(source, dataset_key, runtime_root, run_tag):
    """Build one theoretical-main config without mutating its source."""
    spec = DATASETS[dataset_key]
    increments = balanced_task_increments(spec["num_classes"], 10)
    name = "momentum_adaptive_a_{}_t10_seed{}_bs128_2gpu_{}".format(
        dataset_key, spec["seed"], run_tag
    )
    config = dict(source)
    for key in list(config):
        if key.startswith("sa_adaptive_a_"):
            config.pop(key)
    config.pop("sa_dual_head_schedule", None)
    config.update(
        {
            "prefix": name,
            "dataset": spec["dataset"],
            "device": ["0", "1"],
            "seed": [spec["seed"]],
            "filepath": str((runtime_root / "results" / name).resolve()) + "/",
            "init_cls": increments[0],
            "increment": increments[0],
            "task_increments": increments,
            "max_tasks": 10,
            "batch_size": 64,
            "lora_rank": 10,
            "model_name": "sa_sdlora",
            "sa_shared_a_orthogonal": True,
            "sa_train_a_all_tasks": True,
            "sa_delete_per_task_files": False,
            "sa_use_prototype_classifier": True,
            "sa_cumulative_state": True,
            "sa_cumulative_merge": "live_a_aggregate_b",
            "sa_live_a_history_groups": 1,
            "sa_live_a_coordinate_align": True,
            "sa_live_a_absorb_mode": "operator_preserving_absorb",
            "sa_coordinate_stable_transport": False,
            "sa_hbd_enabled": False,
            "sa_dual_head": False,
            "sa_adaptive_a_enabled": True,
            "sa_adaptive_a_strategy": "impact_ratio",
            "sa_adaptive_a_stability_weight": 1.0,
            "sa_adaptive_a_gate_formula": "ratio",
            "sa_adaptive_a_gate_floor": 0.0,
            "sa_adaptive_a_gate_momentum": 0.0,
            "sa_adaptive_a_eps": 1e-8,
            "sa_deterministic_training": True,
            "sa_resume": False,
        }
    )
    return name, config


def generate_configs(root, runtime_root, run_tag):
    root = Path(root).resolve()
    runtime_root = Path(runtime_root).resolve()
    config_root = runtime_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "results").mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset_key, spec in DATASETS.items():
        source = json.loads((root / "exps" / spec["source"]).read_text())
        name, config = build_config(source, dataset_key, runtime_root, run_tag)
        config_path = config_root / (name + ".json")
        config_path.write_text(json.dumps(config, indent=4) + "\n")
        manifest.append(
            {
                "name": name,
                "dataset": dataset_key,
                "seed": spec["seed"],
                "gpu_ids": spec["gpu_ids"],
                "config": str(config_path),
                "output": config["filepath"],
            }
        )
    (runtime_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main(argv):
    if len(argv) != 4:
        raise SystemExit(
            "usage: generate_momentum_adaptive_a_configs.py "
            "ROOT RUNTIME_DIR RUN_TAG"
        )
    manifest = generate_configs(argv[1], argv[2], argv[3])
    print("generated {} configs".format(len(manifest)))


if __name__ == "__main__":
    main(sys.argv)
