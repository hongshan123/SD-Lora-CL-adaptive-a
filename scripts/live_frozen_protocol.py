"""Generate immutable, paired no-Dual-B Live-A/Frozen-A experiment configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = {"inr": (1995, 1996, 1997), "c100": (1993, 1994, 1995),
         "cub": (1, 2, 3)}
DATASETS = {"inr": ("imagenetr", 20), "c100": ("cifar224", 10),
            "cub": ("cub", 20)}


def make_config(source, dataset, seed, policy, snapshots, output_dir):
    expected_dataset, classes = DATASETS[dataset]
    required = {
        "dataset": expected_dataset,
        "init_cls": classes,
        "increment": classes,
        "max_tasks": 10,
        "epochs": 20,
        "lora_rank": 10,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
        "sa_use_prototype_classifier": True,
        "sa_coordinate_stable_transport": True,
        "sa_live_a_coordinate_align": True,
    }
    mismatch = {key: (source.get(key), value) for key, value in required.items()
                if source.get(key, 10 if key == "max_tasks" else None) != value}
    if mismatch:
        raise ValueError(f"source does not match no-Dual-B protocol: {mismatch}")
    if policy not in ("live", "frozen"):
        raise ValueError("policy must be live or frozen")
    phase = "snapshot_on" if snapshots else "snapshot_off"
    name = f"lfdiag_{dataset}_{policy}_seed{seed}_{phase}"
    config = dict(source)
    config.update({
        "prefix": name,
        "filepath": str(Path(output_dir).resolve() / "results" / name),
        "device": ["0"],
        "seed": [seed],
        "batch_size": 128,
        "max_tasks": 10,
        "sa_dual_head": False,
        "sa_train_a_all_tasks": policy == "live",
        "sa_adaptive_a_enabled": False,
        "sa_live_a_coordinate_align": True,
        "sa_coordinate_stable_transport": True,
        "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
        "sa_deterministic_training": True,
        "sa_save_task_snapshots": snapshots,
        "sa_resume": False,
    })
    config.pop("sa_dual_head_schedule", None)
    return config


def generate_configs(source_dir, output_dir):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"smoke": [], "formal": []}
    for dataset, seeds in SEEDS.items():
        source_path = source_dir / f"{dataset}.json"
        source = json.loads(source_path.read_text())
        for seed in seeds:
            for policy in ("live", "frozen"):
                phases = (False, True) if dataset == "inr" and seed == 1995 else (True,)
                for snapshots in phases:
                    config = make_config(source, dataset, seed, policy, snapshots, output_dir)
                    config_path = config_dir / f"{config['prefix']}.json"
                    if config_path.exists():
                        raise FileExistsError(config_path)
                    config_path.write_text(json.dumps(config, indent=2) + "\n")
                    entry = {
                        "dataset": dataset, "seed": seed, "policy": policy,
                        "snapshots": snapshots, "config": str(config_path.resolve()),
                        "source": str(source_path.resolve()),
                    }
                    manifest["smoke" if dataset == "inr" and seed == 1995 else "formal"].append(entry)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate_configs(args.source_dir, args.output_dir)
    print(json.dumps({key: len(value) for key, value in manifest.items()}))


if __name__ == "__main__":
    main()
