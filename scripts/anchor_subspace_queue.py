#!/usr/bin/env python3
"""Dynamically dispatch anchor-subspace screening jobs to single-GPU slots."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_anchor_subspace_configs import generate_configs
from scripts.tasklen_fla_queue import active_jobs, dispatch_jobs, pending_jobs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-ids", default="0,1,4,5")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    gpu_slots = tuple(value.strip() for value in args.gpu_ids.split(",") if value.strip())
    if not gpu_slots:
        raise SystemExit("at least one GPU ID is required")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate_configs(
        project_root, runtime_dir, args.run_tag, batch_size=args.batch_size
    )
    pending = pending_jobs(project_root, manifest)
    active_runs = active_jobs(project_root, pending)
    active_names = {item["name"] for item, _gpu_id in active_runs}
    jobs = [item for item in pending if item["name"] not in active_names]
    queue_log = runtime_dir / "queue.log"
    mode = "a" if args.resume or queue_log.exists() else "w"
    with queue_log.open(mode) as stream:
        stream.write(
            "QUEUE {} jobs={} pending={} active={} gpu_slots={} batch_size={}\n".format(
                "RESUME" if args.resume else "START",
                len(manifest),
                len(jobs),
                len(active_runs),
                ",".join(gpu_slots),
                args.batch_size,
            )
        )
        for item in jobs:
            stream.write(
                "PENDING {} dataset={} method={}\n".format(
                    item["name"], item["dataset"], item["method"]
                )
            )
        stream.flush()
    status = dispatch_jobs(
        project_root,
        jobs,
        gpu_slots,
        1,
        initial_active=active_runs,
    )
    with queue_log.open("a") as stream:
        stream.write("QUEUE END status={} completed_pending={}\n".format(status, len(jobs)))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
