#!/usr/bin/env python3
"""Dynamically dispatch the HOEP-A P3 matrix to independent GPU slots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_hoep_p3_configs import generate_configs
from scripts.tasklen_fla_queue import active_jobs, dispatch_jobs, pending_jobs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--gpu-pairs", default="0,1;2,3;4,5")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    runtime_root = Path(args.runtime_dir).resolve()
    gpu_pairs = tuple(
        value.strip() for value in args.gpu_pairs.split(";") if value.strip()
    )
    if not gpu_pairs or any(len(pair.split(",")) != 2 for pair in gpu_pairs):
        raise SystemExit("P3 requires two physical GPU IDs per slot")

    manifest = generate_configs(
        project_root, runtime_root, args.run_tag
    )
    pending = pending_jobs(project_root, manifest)
    recovered = active_jobs(project_root, pending) if args.resume else []
    recovered_names = {job["name"] for job, _gpu in recovered}
    jobs = [job for job in pending if job["name"] not in recovered_names]

    queue_log = runtime_root / "queue.log"
    mode = "a" if args.resume or queue_log.exists() else "w"
    with queue_log.open(mode) as stream:
        stream.write(
            "QUEUE {} total={} pending={} active={} gpu_pairs={}\n".format(
                "RESUME" if args.resume else "START",
                len(manifest),
                len(jobs),
                len(recovered),
                ";".join(gpu_pairs),
            )
        )
        for job, gpu_id in recovered:
            stream.write(
                "ACTIVE {} dataset={} method={} GPU={}\n".format(
                    job["name"], job["dataset"], job["method"], gpu_id
                )
            )
        for job in jobs:
            stream.write(
                "PENDING {} dataset={} method={}\n".format(
                    job["name"], job["dataset"], job["method"]
                )
            )

    status = dispatch_jobs(
        project_root,
        jobs,
        gpu_pairs,
        world_size=2,
        initial_active=recovered,
        stop_on_failure=True,
    )
    with queue_log.open("a") as stream:
        stream.write("QUEUE END status={}\n".format(status))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
