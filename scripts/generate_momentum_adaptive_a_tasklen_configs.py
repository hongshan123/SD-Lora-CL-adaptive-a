#!/usr/bin/env python
"""Generate T=5/10/20/50 momentum-aware Adaptive-A configs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_momentum_adaptive_a_configs import DATASETS, build_config


TASK_COUNTS = (5, 10, 20, 50)


def generate_configs(root, runtime_root, run_tag):
    root = Path(root).resolve()
    runtime_root = Path(runtime_root).resolve()
    config_root = runtime_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "results").mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset_key, spec in DATASETS.items():
        source = json.loads((root / "exps" / spec["source"]).read_text())
        for task_count in TASK_COUNTS:
            name, config = build_config(
                source,
                dataset_key,
                runtime_root,
                run_tag,
                task_count=task_count,
            )
            config_path = config_root / (name + ".json")
            config_path.write_text(json.dumps(config, indent=4) + "\n")
            manifest.append(
                {
                    "name": name,
                    "dataset": dataset_key,
                    "tasks": task_count,
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
            "usage: generate_momentum_adaptive_a_tasklen_configs.py "
            "ROOT RUNTIME_DIR RUN_TAG"
        )
    manifest = generate_configs(argv[1], argv[2], argv[3])
    print("generated {} configs".format(len(manifest)))


if __name__ == "__main__":
    main(sys.argv)
