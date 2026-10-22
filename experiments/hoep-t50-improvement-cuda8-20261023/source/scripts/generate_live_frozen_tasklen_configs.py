#!/usr/bin/env python3
"""Generate the matched Live-A/Frozen-A T=5/10/20/50 configs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_momentum_adaptive_a_configs import DATASETS, build_config


TASK_COUNTS = (5, 10, 20, 50)
METHODS = ("live_a", "frozen_a")
DATASET_KEYS = ("c100", "inr", "cub")


def generate_configs(root, runtime_root, run_tag, batch_size=64, world_size=2):
    root = Path(root).resolve()
    runtime_root = Path(runtime_root).resolve()
    config_root = runtime_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "results").mkdir(parents=True, exist_ok=True)
    manifest = []

    for dataset_key in DATASET_KEYS:
        spec = DATASETS[dataset_key]
        source = json.loads((root / "exps" / spec["source"]).read_text())
        for task_count in TASK_COUNTS:
            for method in METHODS:
                _, config = build_config(
                    source,
                    dataset_key,
                    runtime_root,
                    run_tag,
                    task_count=task_count,
                )
                name = "tasklen_{}_{}_t{}_seed{}_bs{}_{}gpu_{}".format(
                    method,
                    dataset_key,
                    task_count,
                    spec["seed"],
                    batch_size,
                    world_size,
                    run_tag,
                )
                config.update(
                    {
                        "prefix": name,
                        "filepath": str((runtime_root / "results" / name).resolve()) + "/",
                        "batch_size": batch_size,
                        "device": [str(index) for index in range(world_size)],
                        "sa_train_a_all_tasks": method == "live_a",
                        "sa_adaptive_a_enabled": False,
                        "sa_live_a_coordinate_align": True,
                        "sa_live_a_absorb_mode": "operator_preserving_absorb",
                        "sa_coordinate_stable_transport": False,
                        "sa_hbd_enabled": False,
                        "sa_dual_head": False,
                    }
                )
                for key in list(config):
                    if key.startswith("sa_adaptive_a_") and key != "sa_adaptive_a_enabled":
                        config.pop(key)
                config_path = config_root / (name + ".json")
                config_path.write_text(json.dumps(config, indent=4) + "\n")
                manifest.append(
                    {
                        "name": name,
                        "canonical_name": name,
                        "dataset": dataset_key,
                        "tasks": task_count,
                        "method": method,
                        "seed": spec["seed"],
                        "config": str(config_path),
                    }
                )

    (runtime_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main(argv):
    if len(argv) not in (4, 6):
        raise SystemExit(
            "usage: generate_live_frozen_tasklen_configs.py ROOT RUNTIME_DIR "
            "RUN_TAG [BATCH_SIZE WORLD_SIZE]"
        )
    root = Path(argv[1])
    runtime_root = Path(argv[2])
    run_tag = argv[3]
    batch_size = int(argv[4]) if len(argv) == 6 else 64
    world_size = int(argv[5]) if len(argv) == 6 else 2
    manifest = generate_configs(
        root, runtime_root, run_tag, batch_size=batch_size, world_size=world_size
    )
    print("generated {} configs under {}".format(len(manifest), runtime_root))


if __name__ == "__main__":
    main(sys.argv)
