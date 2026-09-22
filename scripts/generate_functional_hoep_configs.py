#!/usr/bin/env python3
"""Generate preregistered shadow-diagnostic Functional-HOEP configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "c100": ("cifar224", 1993, 10, 10),
    "inr": ("imagenetr", 1995, 20, 20),
    "cub": ("cub", 1, 20, 20),
}


def _source_name(dataset, phase):
    if phase == "smoke":
        return "hoep_p2_{}_seed{}_t2_e2.json".format(
            dataset, DATASETS[dataset][1]
        )
    return "hoep_{}_seed{}_t10.json".format(dataset, DATASETS[dataset][1])


def generate_configs(project_root, runtime_root, run_tag, phase):
    if phase not in ("smoke", "phase_a"):
        raise ValueError("phase must be smoke or phase_a")
    project_root = Path(project_root).resolve()
    runtime_root = Path(runtime_root).resolve()
    config_root = runtime_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "results").mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset, (_, seed, _, _) in DATASETS.items():
        source_path = project_root / "exps" / _source_name(dataset, phase)
        source = json.loads(source_path.read_text())
        tasks = 2 if phase == "smoke" else 10
        epochs = 2 if phase == "smoke" else 20
        name = "functional_hoep_{}_{}_seed{}_t{}_e{}_bs128_2gpu_{}".format(
            phase, dataset, seed, tasks, epochs, run_tag
        )
        source.update(
            {
                "prefix": name,
                "device": ["0", "1"],
                "seed": [seed],
                "filepath": str(runtime_root / "results" / name) + "/",
                "max_tasks": tasks,
                "init_epoch": epochs,
                "epochs": epochs,
                "batch_size": 64,
                "dist_backend": "nccl",
                "lora_rank": 10,
                "sa_train_a_all_tasks": True,
                "sa_adaptive_a_enabled": True,
                "sa_adaptive_a_strategy": "operator_energy_partition",
                "sa_hoep_energy_budget": 0.05,
                "sa_hoep_eigenvalue_rtol": 1e-6,
                "sa_hoep_energy_metric": "operator",
                "sa_hoep_functional_diagnostics": True,
                "sa_hoep_activation_calibration_batch_size": 64,
                "sa_coordinate_stable_transport": False,
                "sa_dual_head": False,
                "sa_hbd_enabled": False,
                "sa_live_a_absorb_mode": "operator_preserving_absorb",
                "sa_normalize_current_branch": False,
                "sa_deterministic_training": True,
                "sa_resume": False,
            }
        )
        config_path = config_root / (name + ".json")
        config_path.write_text(json.dumps(source, indent=4) + "\n")
        manifest.append(
            {
                "name": name,
                "canonical_name": name,
                "dataset": dataset,
                "method": "operator_shadow_functional",
                "seed": seed,
                "config": str(config_path),
                "output": source["filepath"],
            }
        )
    (runtime_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--phase", choices=("smoke", "phase_a"), required=True)
    args = parser.parse_args()
    manifest = generate_configs(
        args.project_root, args.runtime_dir, args.run_tag, args.phase
    )
    print("generated {} {} configs".format(len(manifest), args.phase))


if __name__ == "__main__":
    main()
