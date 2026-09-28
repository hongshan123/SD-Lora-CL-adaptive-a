"""Wait for an unused pair, then run the approved dataset queues serially."""

import argparse
import csv
import datetime
import io
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def gpu_pair_idle(gpu_text, process_text, gpu_ids):
    try:
        cards = {
            int(row[0]): (row[1].strip(), int(row[2]), int(row[3]))
            for row in csv.reader(io.StringIO(gpu_text)) if row
        }
        processes = {
            row[0].strip() for row in csv.reader(io.StringIO(process_text))
            if row and int(row[1]) > 0
        }
        return all(
            cards[index][0] not in processes
            and cards[index][1] <= 64 and cards[index][2] <= 5
            for index in gpu_ids
        )
    except (IndexError, KeyError, ValueError):
        return False


def log(message):
    print(datetime.datetime.now().astimezone().isoformat(), message, flush=True)


def wait_for_pair(gpu_ids, poll_seconds, idle_checks):
    streak = 0
    while streak < idle_checks:
        try:
            cards = subprocess.run([
                "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, check=True, timeout=20).stdout
            processes = subprocess.run([
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, check=True, timeout=20).stdout
            idle = gpu_pair_idle(cards, processes, gpu_ids)
        except (OSError, subprocess.SubprocessError) as error:
            log("GPU_QUERY_FAILED " + repr(error))
            idle = False
        streak = streak + 1 if idle else 0
        log(f"WAIT_FOR_GPUS ids={gpu_ids} idle={idle} streak={streak}/{idle_checks}")
        if streak < idle_checks:
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-checks", type=int, default=2)
    args = parser.parse_args()
    try:
        gpu_ids = tuple(int(value) for value in args.gpu_ids.split(","))
    except ValueError:
        parser.error("GPU IDs must be integers")
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or any(
        index < 0 or index in (2, 3) for index in gpu_ids
    ):
        parser.error("use two distinct GPUs, excluding 2 and 3")
    if args.poll_seconds <= 0 or args.idle_checks < 2:
        parser.error("require a positive interval and at least two idle observations")
    environment = dict(os.environ, GPU_IDS=args.gpu_ids)
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    for dataset in ("cub", "inr"):
        wait_for_pair(gpu_ids, args.poll_seconds, args.idle_checks)
        log(f"DATASET_START dataset={dataset} GPUs={args.gpu_ids}")
        queue_log = ROOT / f"early_a_{dataset}_formal_queue_20260928.log"
        with queue_log.open("a") as handle:
            result = subprocess.run([
                "/usr/bin/bash", str(ROOT / "run_early_a_freeze_queue.sh"),
                dataset, "formal",
            ], cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        log(f"DATASET_END dataset={dataset} status={result.returncode}")
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
