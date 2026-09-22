#!/usr/bin/env python3
"""Dynamically dispatch Functional-HOEP smoke or Phase-A jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_functional_hoep_configs import generate_configs
from scripts.tasklen_fla_queue import dispatch_jobs, pending_jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--phase", choices=("smoke", "phase_a"), required=True)
    parser.add_argument("--gpu-pairs", default="0,1;4,5;6,7")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    runtime_root = Path(args.runtime_dir).resolve()
    gpu_pairs = tuple(
        pair.strip() for pair in args.gpu_pairs.split(";") if pair.strip()
    )
    if not gpu_pairs or any(len(pair.split(",")) != 2 for pair in gpu_pairs):
        raise SystemExit("Functional-HOEP requires two GPUs per slot")
    if any(set(pair.split(",")) & {"2", "3"} for pair in gpu_pairs):
        raise SystemExit("GPU 2/3 are excluded from Functional-HOEP experiments")

    manifest = generate_configs(
        project_root, runtime_root, args.run_tag, args.phase
    )
    jobs = pending_jobs(project_root, manifest)
    queue_log = runtime_root / "queue.log"
    with queue_log.open("a") as stream:
        stream.write(
            "QUEUE START phase={} jobs={} gpu_pairs={}\n".format(
                args.phase, len(jobs), ";".join(gpu_pairs)
            )
        )
    status = dispatch_jobs(
        project_root,
        jobs,
        gpu_pairs,
        world_size=2,
        stop_on_failure=True,
    )
    with queue_log.open("a") as stream:
        stream.write("QUEUE END status={}\n".format(status))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
