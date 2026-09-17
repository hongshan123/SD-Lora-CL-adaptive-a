#!/usr/bin/env python
"""Generate task-length configs for Frozen-A, Live-A, and Adaptive-A."""

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
    },
    "inr": {
        "source": "p3_inr_livea_dual_b_seed1_nccl.json",
        "dataset": "imagenetr",
        "seed": 1995,
        "num_classes": 200,
    },
    "cub": {
        "source": "p5_cub_livea_dual_b_seed1_nccl.json",
        "dataset": "cub",
        "seed": 1,
        "num_classes": 200,
    },
    "domainnet": {
        "source": "p5_cub_livea_dual_b_seed1_nccl.json",
        "dataset": "domainnet",
        "seed": 1993,
        "num_classes": 200,
    },
}
TASK_COUNTS = (5, 10, 20, 50)
METHODS = ("frozen_a", "live_a", "adaptive_a")


def canonical_name(dataset_key, task_count, method, batch_size=64, world_size=2):
    """Return the name used by the previous two-GPU task-length queue."""
    spec = DATASETS[dataset_key]
    return "pair_{}_t{}_{}_seed{}_bs{}_{}gpu".format(
        dataset_key,
        task_count,
        method,
        spec["seed"],
        batch_size,
        world_size,
    )


def build_config(
    source,
    dataset_key,
    task_count,
    method,
    runtime_root,
    run_tag,
    batch_size=64,
    world_size=2,
):
    """Build one isolated config without changing the source config."""
    if method not in METHODS:
        raise ValueError("unknown method: {}".format(method))
    spec = DATASETS[dataset_key]
    increments = balanced_task_increments(spec["num_classes"], task_count)
    name = "fla_{}_t{}_{}_seed{}_bs{}_{}gpu_{}".format(
        dataset_key,
        task_count,
        method,
        spec["seed"],
        batch_size,
        world_size,
        run_tag,
    )
    config = dict(source)
    config.update(
        {
            "prefix": name,
            "dataset": spec["dataset"],
            "device": [str(index) for index in range(world_size)],
            "seed": [spec["seed"]],
            "filepath": str((runtime_root / "results" / name).resolve()) + "/",
            "init_cls": increments[0],
            "increment": increments[0],
            "task_increments": increments,
            "batch_size": batch_size,
            "lora_rank": 10,
            "sa_deterministic_training": True,
            "sa_resume": False,
        }
    )
    if dataset_key == "domainnet":
        config.update(
            {
                "domainnet_root": "/data/dataset/DomainNet",
                "domainnet_top_classes": 200,
                "domainnet_domains": [
                    "clipart",
                    "infograph",
                    "painting",
                    "quickdraw",
                    "real",
                    "sketch",
                ],
            }
        )

    config.update(
        {
            "model_name": "sa_sdlora",
            "sa_shared_a_orthogonal": True,
            "sa_train_a_all_tasks": method != "frozen_a",
            "sa_delete_per_task_files": False,
            "sa_use_prototype_classifier": True,
            "sa_cumulative_state": True,
            "sa_cumulative_merge": "live_a_aggregate_b",
            "sa_live_a_history_groups": 1,
            "sa_live_a_coordinate_align": method == "adaptive_a",
            "sa_dual_head": False,
        }
    )
    config.pop("sa_dual_head_schedule", None)
    if method == "adaptive_a":
        config.update(
            {
                "sa_adaptive_a_enabled": True,
                "sa_adaptive_a_strategy": "impact_ratio",
                "sa_adaptive_a_stability_weight": 1.0,
                "sa_adaptive_a_gate_floor": 0.05,
                "sa_adaptive_a_gate_momentum": 0.9,
                "sa_adaptive_a_eps": 1e-8,
            }
        )
    else:
        for key in list(config):
            if key.startswith("sa_adaptive_a_"):
                config.pop(key)
        config["sa_adaptive_a_enabled"] = False
    return name, config


def generate_configs(root, runtime_root, run_tag, batch_size=64, world_size=2):
    root = Path(root).resolve()
    runtime_root = Path(runtime_root).resolve()
    config_root = runtime_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "results").mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset_key, spec in DATASETS.items():
        source = json.loads((root / "exps" / spec["source"]).read_text())
        for task_count in TASK_COUNTS:
            for method in METHODS:
                name, config = build_config(
                    source,
                    dataset_key,
                    task_count,
                    method,
                    runtime_root,
                    run_tag,
                    batch_size=batch_size,
                    world_size=world_size,
                )
                path = config_root / (name + ".json")
                path.write_text(json.dumps(config, indent=4) + "\n")
                manifest.append(
                    {
                        "name": name,
                        "canonical_name": canonical_name(
                            dataset_key,
                            task_count,
                            method,
                            batch_size=batch_size,
                            world_size=world_size,
                        ),
                        "dataset": dataset_key,
                        "tasks": task_count,
                        "method": method,
                        "seed": spec["seed"],
                        "config": str(path),
                    }
                )
    (runtime_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main(argv):
    if len(argv) not in (4, 6):
        raise SystemExit(
            "usage: generate_tasklen_fla_configs.py ROOT RUNTIME_DIR RUN_TAG "
            "[BATCH_SIZE WORLD_SIZE]"
        )
    root = Path(argv[1])
    runtime_root = Path(argv[2])
    run_tag = argv[3]
    batch_size = int(argv[4]) if len(argv) == 6 else 64
    world_size = int(argv[5]) if len(argv) == 6 else 2
    if not run_tag or batch_size <= 0 or world_size <= 0:
        raise SystemExit("RUN_TAG, BATCH_SIZE, and WORLD_SIZE must be valid")
    manifest = generate_configs(
        root, runtime_root, run_tag, batch_size=batch_size, world_size=world_size
    )
    print("generated {} configs under {}".format(
        len(manifest), runtime_root / "configs"
    ))


if __name__ == "__main__":
    main(sys.argv)
