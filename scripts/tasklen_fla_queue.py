#!/usr/bin/env python
"""Run task-length Frozen/Live/Adaptive experiments with dynamic GPU slots."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_tasklen_fla_configs import generate_configs


START_RE = re.compile(r"^.*START .* GPUs=(?P<gpu_ids>\S+)(?:\s|$)", re.MULTILINE)
END_RE = re.compile(r"^.*END .* status=(?P<status>-?\d+)\s*=*\s*$", re.MULTILINE)


def _log_status(log_path):
    """Return the last explicit queue status, or None for a nonterminal log."""
    path = Path(log_path)
    if not path.is_file():
        return None
    matches = list(END_RE.finditer(path.read_text(errors="replace")))
    return int(matches[-1].group("status")) if matches else None


def _current_run_succeeded(project_root, job_name):
    return _log_status(Path(project_root) / (job_name + ".log")) == 0


def torchrun_command(world_size, config_path):
    """Run torch.distributed with the interpreter hosting this scheduler."""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node={}".format(world_size),
        "main.py",
        "--config={}".format(config_path),
    ]


def _archive_stale_result_dir(config_path):
    """Move a prior failed result aside before starting a fresh attempt."""
    with open(config_path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    filepath = config.get("filepath")
    if not filepath:
        return None
    target = Path(filepath).expanduser()
    if not target.exists():
        return None
    if not target.is_dir():
        raise RuntimeError("result filepath exists but is not a directory: {}".format(target))

    suffix = ".retry_{}_{}".format(time.strftime("%Y%m%d_%H%M%S"), os.getpid())
    archived = target.with_name(target.name + suffix)
    target.rename(archived)
    return archived


def old_run_succeeded(project_root, canonical_name):
    """Reuse only an explicitly successful previous two-GPU run."""
    log_path = Path(project_root) / (canonical_name + ".log")
    return _log_status(log_path) == 0


def pending_jobs(project_root, manifest):
    return [
        item
        for item in manifest
        if not old_run_succeeded(project_root, item["canonical_name"])
        and not (
            item.get("name")
            and _current_run_succeeded(project_root, item["name"])
        )
    ]


def active_jobs(project_root, manifest):
    """Find generated jobs started before a scheduler restart.

    A detached torchrun can outlive its Python scheduler.  Its per-job log has
    a START marker but no END marker, so the scheduler can reclaim that slot
    by waiting for the existing process instead of launching a duplicate.
    """
    project_root = Path(project_root)
    recovered = []
    for item in manifest:
        log_path = project_root / (item["name"] + ".log")
        if not log_path.is_file() or _log_status(log_path) is not None:
            continue
        matches = list(START_RE.finditer(log_path.read_text(errors="replace")))
        if matches:
            recovered.append((item, matches[-1].group("gpu_ids")))
    return recovered


def _process_is_alive(job):
    """Check for torchrun/main processes belonging to a generated config."""
    target = str(job["config"])
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        # A failed process listing should not cause a duplicate launch.
        return True
    own_pid = str(os.getpid())
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and fields[0] != own_pid and target in fields[1]:
            return True
    return False


def _training_log_is_complete(log_path):
    text = Path(log_path).read_text(errors="replace")
    return "Accuracy Matrix (CNN):" in text and "Forgetting (CNN):" in text


def wait_for_existing(project_root, job, gpu_ids, _world_size):
    """Wait for an orphaned torchrun and reconstruct its queue status."""
    log_path = Path(project_root) / (job["name"] + ".log")
    missing_since = None
    while True:
        status = _log_status(log_path)
        if status is not None:
            return status
        if _process_is_alive(job):
            missing_since = None
        else:
            if missing_since is None:
                missing_since = time.monotonic()
            elif time.monotonic() - missing_since >= 15:
                status = 0 if _training_log_is_complete(log_path) else 1
                with log_path.open("a") as stream:
                    stream.write(
                        "===== END {} GPUs={} status={} =====\n".format(
                            job["name"], gpu_ids, status
                        )
                    )
                return status
        time.sleep(10)


def run_one(project_root, job, gpu_ids, world_size):
    """Execute one torchrun and write an independent, auditable log."""
    project_root = Path(project_root)
    log_path = project_root / (job["name"] + ".log")
    archived_result_dir = _archive_stale_result_dir(job["config"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["PYTHONUNBUFFERED"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["TORCH_DETERMINISTIC"] = "1"
    command = torchrun_command(world_size, job["config"])
    with log_path.open("w") as stream:
        stream.write(
            "===== START {} GPUs={} canonical={} =====\n".format(
                job["name"], gpu_ids, job["canonical_name"]
            )
        )
        stream.write("command={}\n".format(" ".join(command)))
        if archived_result_dir is not None:
            stream.write("archived_stale_result_dir={}\n".format(archived_result_dir))
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stream.write("===== END {} GPUs={} status={} =====\n".format(
            job["name"], gpu_ids, completed.returncode
        ))
    return completed.returncode


def dispatch_jobs(
    project_root,
    jobs,
    gpu_pairs,
    world_size,
    runner=run_one,
    initial_active=(),
    existing_runner=wait_for_existing,
):
    """Keep every slot full and submit the next job on each completion."""
    if not gpu_pairs:
        return 0
    statuses = []
    next_index = 0
    active = {}
    active_by_gpu = {gpu_ids: job for job, gpu_ids in initial_active}
    with ThreadPoolExecutor(max_workers=len(gpu_pairs)) as executor:
        for slot, gpu_ids in enumerate(gpu_pairs):
            if gpu_ids in active_by_gpu:
                future = executor.submit(
                    existing_runner,
                    project_root,
                    active_by_gpu[gpu_ids],
                    gpu_ids,
                    world_size,
                )
                active[future] = (slot, gpu_ids, active_by_gpu[gpu_ids])
                continue
            if next_index >= len(jobs):
                break
            future = executor.submit(
                runner, project_root, jobs[next_index], gpu_ids, world_size
            )
            active[future] = (slot, gpu_ids, jobs[next_index])
            next_index += 1
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                slot, gpu_ids, _job = active.pop(future)
                statuses.append(future.result())
                if next_index < len(jobs):
                    next_job = jobs[next_index]
                    replacement = executor.submit(
                        runner, project_root, next_job, gpu_ids, world_size
                    )
                    active[replacement] = (slot, gpu_ids, next_job)
                    next_index += 1
    return 0 if all(status == 0 for status in statuses) else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument(
        "--gpu-pairs", default="0,1;4,5;6,7",
        help="semicolon-separated physical GPU pairs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append queue state and reclaim started jobs after scheduler restart",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    gpu_pairs = tuple(
        value.strip() for value in args.gpu_pairs.split(";") if value.strip()
    )
    if any(len(pair.split(",")) != args.world_size for pair in gpu_pairs):
        raise SystemExit("each GPU pair must contain world-size GPU IDs")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate_configs(
        project_root,
        runtime_dir,
        args.run_tag,
        batch_size=args.batch_size,
        world_size=args.world_size,
    )
    pending = pending_jobs(project_root, manifest)
    active_runs = active_jobs(project_root, pending)
    active_names = {item["name"] for item, _gpu_ids in active_runs}
    jobs = [item for item in pending if item["name"] not in active_names]
    queue_log = runtime_dir / "queue.log"
    mode = "a" if args.resume or queue_log.exists() else "w"
    with queue_log.open(mode) as stream:
        stream.write(
            "QUEUE {} jobs={} pending={} active={} gpu_pairs={} batch_size={} world_size={}\n".format(
                "RESUME" if args.resume else "START",
                len(manifest),
                len(jobs),
                len(active_runs),
                ",".join(gpu_pairs),
                args.batch_size,
                args.world_size,
            )
        )
        for item in manifest:
            if item not in pending:
                stream.write("SKIP existing_success {}\n".format(item["canonical_name"]))
        for item, gpu_ids in active_runs:
            stream.write(
                "ACTIVE {} dataset={} tasks={} method={} GPUs={}\n".format(
                    item["name"], item["dataset"], item["tasks"], item["method"], gpu_ids
                )
            )
        for item in jobs:
            stream.write(
                "PENDING {} dataset={} tasks={} method={}\n".format(
                    item["name"], item["dataset"], item["tasks"], item["method"]
                )
            )
        stream.flush()
    status = dispatch_jobs(
        project_root,
        jobs,
        gpu_pairs,
        args.world_size,
        initial_active=active_runs,
    )
    with queue_log.open("a") as stream:
        stream.write("QUEUE END status={} completed_pending={}\n".format(
            status, len(jobs)
        ))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
