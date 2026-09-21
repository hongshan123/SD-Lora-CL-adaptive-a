#!/usr/bin/env python3
"""Generate the preregistered HOEP-A P3 single-seed comparison matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path


DATASETS = {
    "c100": {
        "source": "hoep_c100_seed1993_t10.json",
        "seed": 1993,
    },
    "inr": {
        "source": "hoep_inr_seed1995_t10.json",
        "seed": 1995,
    },
    "cub": {
        "source": "hoep_cub_seed1_t10.json",
        "seed": 1,
    },
}

# This order fills three two-GPU slots with Frozen for all datasets first.
# HOEP, Live, and prior scalar-ratio controls are dispatched as slots free up.
METHOD_ORDER = ("frozen", "hoep", "live", "ratio")


def _method_settings(method):
    if method == "frozen":
        return {
            "sa_train_a_all_tasks": False,
            "sa_adaptive_a_enabled": False,
        }
    if method == "live":
        return {
            "sa_train_a_all_tasks": True,
            "sa_adaptive_a_enabled": False,
        }
    if method == "ratio":
        return {
            "sa_train_a_all_tasks": True,
            "sa_adaptive_a_enabled": True,
            "sa_adaptive_a_strategy": "impact_ratio",
            "sa_adaptive_a_stability_weight": 1.0,
            "sa_adaptive_a_gate_formula": "ratio",
            "sa_adaptive_a_gate_floor": 0.05,
            "sa_adaptive_a_gate_momentum": 0.9,
            "sa_adaptive_a_eps": 1e-8,
        }
    if method == "hoep":
        return {
            "sa_train_a_all_tasks": True,
            "sa_adaptive_a_enabled": True,
            "sa_adaptive_a_strategy": "operator_energy_partition",
            "sa_hoep_energy_budget": 0.05,
            "sa_hoep_eigenvalue_rtol": 1e-6,
            "sa_adaptive_a_eps": 1e-8,
        }
    raise ValueError("unknown P3 method: {}".format(method))


def build_config(source, dataset_key, method, runtime_root, run_tag):
    spec = DATASETS[dataset_key]
    config = dict(source)
    for key in list(config):
        if key.startswith("sa_adaptive_a_") or key.startswith("sa_hoep_"):
            config.pop(key)
    config.pop("dist_backend", None)
    name = "hoep_p3_{}_{}_seed{}_t10_e20_bs128_2gpu_{}".format(
        dataset_key, method, spec["seed"], run_tag
    )
    config.update(
        {
            "prefix": name,
            "device": ["0", "1"],
            "seed": [spec["seed"]],
            "filepath": str((runtime_root / "results" / name).resolve()) + "/",
            "max_tasks": 10,
            "init_epoch": 20,
            "epochs": 20,
            "batch_size": 64,
            "dist_backend": "nccl",
            "lora_rank": 10,
            "sa_shared_a_orthogonal": True,
            "sa_delete_per_task_files": False,
            "sa_use_prototype_classifier": True,
            "sa_cumulative_state": True,
            "sa_cumulative_merge": "live_a_aggregate_b",
            "sa_live_a_history_groups": 1,
            "sa_live_a_coordinate_align": True,
            "sa_live_a_absorb_mode": "operator_preserving_absorb",
            "sa_normalize_current_branch": False,
            "sa_coordinate_stable_transport": False,
            "sa_dual_head": False,
            "sa_hbd_enabled": False,
            "sa_deterministic_training": True,
            "sa_resume": False,
        }
    )
    config.update(_method_settings(method))
    return name, config


def generate_configs(project_root, runtime_root, run_tag):
    project_root = Path(project_root).resolve()
    runtime_root = Path(runtime_root).resolve()
    config_root = runtime_root / "configs"
    result_root = runtime_root / "results"
    config_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for method in METHOD_ORDER:
        for dataset_key, spec in DATASETS.items():
            source = json.loads(
                (project_root / "exps" / spec["source"]).read_text()
            )
            name, config = build_config(
                source, dataset_key, method, runtime_root, run_tag
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
                    "config": str(config_path.resolve()),
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
            "usage: generate_hoep_p3_configs.py PROJECT_ROOT RUNTIME_DIR RUN_TAG"
        )
    manifest = generate_configs(argv[1], argv[2], argv[3])
    print("generated {} configs".format(len(manifest)))


if __name__ == "__main__":
    main(sys.argv)
