#!/usr/bin/env python
"""Generate reproducible single-seed task-length comparison configs."""

import json
import sys
from pathlib import Path

# The script is invoked as ``python scripts/...`` from the repository root,
# so make the repository packages importable without requiring installation.
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
TASK_COUNTS = (5, 10, 20, 40)
METHODS = ("live_a", "frozen_a", "sdlora")


def _strip_sa_options(config):
    for key in list(config):
        if key.startswith("sa_") and key != "sa_deterministic_training":
            config.pop(key)


def build_config(
    source,
    dataset_key,
    task_count,
    method,
    runtime_root,
    batch_size=32,
    world_size=4,
):
    spec = DATASETS[dataset_key]
    task_increments = balanced_task_increments(spec["num_classes"], task_count)
    name = "pair_{}_t{}_{}_seed{}_bs{}_{}gpu".format(
        dataset_key, task_count, method, spec["seed"], batch_size, world_size
    )
    config = dict(source)
    config.update(
        {
            "prefix": name,
            "dataset": spec["dataset"],
            "device": [str(index) for index in range(world_size)],
            "seed": [spec["seed"]],
            "filepath": str((runtime_root / "results" / name).resolve()) + "/",
            "init_cls": task_increments[0],
            "increment": task_increments[0],
            "task_increments": task_increments,
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

    if method == "sdlora":
        _strip_sa_options(config)
        config["model_name"] = "sdlora"
    else:
        config.update(
            {
                "model_name": "sa_sdlora",
                "sa_shared_a_orthogonal": True,
                "sa_train_a_all_tasks": method == "live_a",
                "sa_delete_per_task_files": False,
                "sa_use_prototype_classifier": True,
                "sa_cumulative_state": True,
                "sa_cumulative_merge": "live_a_aggregate_b",
                "sa_live_a_history_groups": 1,
                "sa_live_a_coordinate_align": False,
                "sa_dual_head": False,
            }
        )
        config.pop("sa_dual_head_schedule", None)
    return name, config


def main(argv):
    if len(argv) not in (3, 5):
        raise SystemExit(
            "usage: generate_tasklen_pair_configs.py ROOT RUNTIME_DIR "
            "[BATCH_SIZE WORLD_SIZE]"
        )
    root = Path(argv[1]).resolve()
    runtime_root = Path(argv[2]).resolve()
    batch_size = int(argv[3]) if len(argv) == 5 else 32
    world_size = int(argv[4]) if len(argv) == 5 else 4
    if batch_size <= 0 or world_size <= 0:
        raise SystemExit("BATCH_SIZE and WORLD_SIZE must be positive")
    config_root = runtime_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "results").mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset_key in DATASETS:
        source_path = root / "exps" / DATASETS[dataset_key]["source"]
        source = json.loads(source_path.read_text())
        for task_count in TASK_COUNTS:
            for method in METHODS:
                name, config = build_config(
                    source,
                    dataset_key,
                    task_count,
                    method,
                    runtime_root,
                    batch_size=batch_size,
                    world_size=world_size,
                )
                path = config_root / (name + ".json")
                path.write_text(json.dumps(config, indent=4) + "\n")
                manifest.append(
                    {
                        "name": name,
                        "dataset": dataset_key,
                        "tasks": task_count,
                        "method": method,
                        "config": str(path),
                    }
                )
    (runtime_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("generated {} configs under {}".format(len(manifest), config_root))


if __name__ == "__main__":
    main(sys.argv)
