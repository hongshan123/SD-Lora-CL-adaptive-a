#!/usr/bin/env python3
"""Export the minimal recoverable artifact for the frozen Live-A Dual-B method.

Only these files are copied:
  * sa_state.pt        -- O(1) v4 Live-A Aggregate-B LoRA state
  * sa_merged_lora.pt  -- merged LoRA for inference-only backbone
  * sa_prototypes.pt   -- per-class prototypes
  * sa_dual_head.pt    -- Dual-B schedule/lambda/temperatures
  * CLs_weight{N}.pt / CLs_bias{N}.pt -- final FC head (N = num_tasks - 1)
  * minimal_manifest.json -- provenance

Usage:
  python scripts/export_minimal_artifact.py \
    --config exps/p5_cub_livea_dual_b_seed1_nccl.json \
    --artifact CUB_P5_LIVEA_DUALB_SEED1_NCCL \
    --out CUB_P5_LIVEA_DUALB_SEED1_NCCL_MINIMAL
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    cli = parse_args()
    with open(cli.config) as handle:
        config = json.load(handle)

    artifact = Path(cli.artifact).resolve()
    out = Path(cli.out).resolve()
    if not artifact.is_dir():
        raise FileNotFoundError(artifact)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    num_tasks = int(config.get("total_sessions", 0)) or None
    if num_tasks is None:
        # derive from config increments: 100 / increment for C100, 200 / increment for INR/CUB
        dataset = config["dataset"]
        total = 200 if dataset in ("imagenetr", "inr", "cub", "ImageNet_R") else 100
        num_tasks = (total - config["init_cls"]) // config["increment"] + 1
    task_id = num_tasks - 1

    names = [
        "sa_state.pt",
        "sa_merged_lora.pt",
        "sa_prototypes.pt",
        "sa_dual_head.pt",
        "CLs_weight{}.pt".format(task_id),
        "CLs_bias{}.pt".format(task_id),
    ]
    for name in names:
        src = artifact / name
        if not src.is_file():
            raise FileNotFoundError("missing {}".format(src))
        shutil.copy2(src, out / name)

    sizes = {name: (out / name).stat().st_size for name in names}
    manifest = {
        "source_artifact": str(artifact),
        "config": cli.config,
        "git_commit": subprocess.check_output(
            ["git", "-C", str(artifact.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "num_tasks": num_tasks,
        "files": sizes,
        "total_bytes": sum(sizes.values()),
    }
    (out / "minimal_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    total = sum(sizes.values())
    print("minimal artifact: {}".format(out))
    for name in names:
        print("  {:<22} {:>12,} bytes".format(name, sizes[name]))
    print("  {:<22} {:>12,} bytes".format("total", total))
    print("num_tasks={} final_task_id={}".format(num_tasks, task_id))


if __name__ == "__main__":
    main()
