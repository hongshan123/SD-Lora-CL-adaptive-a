#!/usr/bin/env python3
"""Generate the final no-Dual-B Frozen/Live/Adaptive task-length matrix."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_momentum_adaptive_a_configs import DATASETS, build_config


TASK_COUNTS = (5, 10, 20, 50)
METHODS = ("frozen_a", "live_a", "adaptive_a")
DATASET_KEYS = ("c100", "inr", "cub")


def _set_a_policy(config, method):
    config["sa_train_a_all_tasks"] = method != "frozen_a"
    if method == "adaptive_a":
        config.update(
            {
                "sa_adaptive_a_enabled": True,
                "sa_adaptive_a_strategy": "impact_ratio",
                "sa_adaptive_a_stability_weight": 1.0,
                "sa_adaptive_a_gate_formula": "ratio",
                "sa_adaptive_a_gate_floor": 0.0,
                "sa_adaptive_a_gate_momentum": 0.0,
                "sa_adaptive_a_eps": 1e-8,
            }
        )
        return

    for key in list(config):
        if key.startswith("sa_adaptive_a_"):
            config.pop(key)
    config["sa_adaptive_a_enabled"] = False


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
                name = "final_a_policy_{}_{}_t{}_seed{}_bs{}_{}gpu_{}".format(
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
                        "filepath": str((runtime_root / "results" / name).resolve())
                        + "/",
                        "batch_size": int(batch_size),
                        "device": [str(index) for index in range(world_size)],
                        "sa_use_prototype_classifier": True,
                        "sa_cumulative_state": True,
                        "sa_cumulative_merge": "live_a_aggregate_b",
                        "sa_live_a_coordinate_align": True,
                        "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
                        "sa_coordinate_stable_transport": True,
                        "sa_coordinate_transport_rank": 10,
                        "sa_coordinate_transport_reg": 1e-4,
                        "sa_coordinate_transport_min_gain": 0.0,
                        "sa_hbd_enabled": False,
                        "sa_dual_head": False,
                        "sa_deterministic_training": True,
                        "sa_resume": False,
                    }
                )
                config.pop("sa_dual_head_schedule", None)
                _set_a_policy(config, method)
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
            "usage: generate_final_a_policy_tasklen_configs.py ROOT RUNTIME_DIR "
            "RUN_TAG [BATCH_SIZE WORLD_SIZE]"
        )
    batch_size = int(argv[4]) if len(argv) == 6 else 64
    world_size = int(argv[5]) if len(argv) == 6 else 2
    manifest = generate_configs(
        argv[1], argv[2], argv[3], batch_size=batch_size, world_size=world_size
    )
    print("generated {} configs under {}".format(len(manifest), argv[2]))


if __name__ == "__main__":
    main(sys.argv)
