#!/usr/bin/env python3
"""Four two-GPU lanes, with dependency-aware handover of an existing run."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hoep_cifar_ablation import (
    atomic_json, completed_run_metrics, read_metrics, verify_sources,
)

PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
PRIORITY = {"hoep": 0, "frozen": 1, "live": 2,
            "canonical_live": 3, "random": 4, "high": 5}


def process_info(pid):
    """Read Linux process identity and the exit status of an unreaped child."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw.rsplit(")", 1)[1].split()  # fields start at Linux field3
        return {"pid": int(pid), "state": fields[0], "parent_pid": int(fields[1]),
                "start_ticks": int(fields[19]),
                "exit_code": os.waitstatus_to_exitcode(int(fields[49])) if fields[0] == "Z" else None,
                "command": Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")}
    except (FileNotFoundError, ProcessLookupError):
        return None


def pause_parent(pid, start_ticks):
    info = process_info(pid)
    if info is None or info["start_ticks"] != start_ticks:
        raise RuntimeError("scheduler process identity changed before handover")
    os.kill(pid, signal.SIGSTOP)  # deliberately NOT a process-group signal
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        info = process_info(pid)
        if info and info["state"] == "T" and info["start_ticks"] == start_ticks:
            return
        time.sleep(.01)
    info = process_info(pid)
    if info and info["start_ticks"] == start_ticks:
        os.kill(pid, signal.SIGCONT)
    raise RuntimeError("serial scheduler did not pause")


def worker_progress(running, proof_path):
    """Return a terminal proof, or check a live worker including after recovery."""
    proof_path = Path(proof_path)
    proof = json.loads(proof_path.read_text()) if proof_path.exists() else None
    if proof and proof["phase"] in ("complete", "failed"):
        return proof
    info = process_info(running["pid"])
    if info and info["state"] != "Z" and info["start_ticks"] == running["start_ticks"]:
        return None
    # The worker writes its terminal proof just before exiting. Reread after
    # observing its disappearance so a normal completion cannot race this check.
    proof = json.loads(proof_path.read_text()) if proof_path.exists() else None
    if proof and proof["phase"] in ("complete", "failed"):
        return proof
    raise RuntimeError(f"worker exited without a terminal result: {running['name']}")


def ready_jobs(jobs, completed, active):
    providers = {job["seed"]: job["name"] for job in jobs if job["method"] == "hoep"}
    ready = []
    for job in jobs:
        if job["name"] in completed or job["name"] in active:
            continue
        if not set(job.get("depends_on", [])).issubset(completed):
            continue
        if job["method"] in ("random", "high") and job["seed"] != 1993:
            if providers.get(job["seed"]) not in completed:
                continue
        ready.append(job)
    return sorted(ready, key=lambda job: (job.get("dispatch_priority", PRIORITY[job["method"]]), job["seed"]))


def validate_reference(path, seed, expected_tasks=10):
    data = json.loads(Path(path).read_text())
    if data.get("version") != 1 or data.get("seed") != seed:
        raise ValueError("HOEP reference version/seed mismatch")
    for task in range(1, expected_tasks):
        counts = data.get("tasks", {}).get(str(task))
        if not isinstance(counts, list) or len(counts) != 24:
            raise ValueError(f"HOEP reference missing complete task {task}")
        if any(isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= 10 for k in counts):
            raise ValueError(f"HOEP reference has invalid counts at task {task}")
    return data


def validated_result(entry, proof):
    if proof.get("phase") != "complete" or proof.get("returncode") != 0:
        raise ValueError("training did not have a verified zero exit status")
    config_path = Path(entry["config"])
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    expected_tasks = config.get("max_tasks", entry.get("tasks", 10))
    if isinstance(expected_tasks, bool) or not isinstance(expected_tasks, int) or expected_tasks <= 0:
        raise ValueError("expected task count must be a positive integer")
    metrics = read_metrics(entry["log"], expected_tasks)
    if "within_bound=False" in Path(entry["log"]).read_text(errors="replace"):
        raise ValueError("operator residual bound failed")
    if entry["method"] == "hoep":
        validate_reference(Path(entry["output"]) / "hoep_partition_schedule.json", entry["seed"], expected_tasks)
    if entry["method"] in ("random", "high"):
        config = json.loads(Path(entry["config"]).read_text())
        reference = validate_reference(config["sa_hoep_reference_counts_path"], entry["seed"], expected_tasks)
        actual = validate_reference(Path(entry["output"]) / "hoep_partition_schedule.json", entry["seed"], expected_tasks)
        if any(actual["tasks"][str(t)] != reference["tasks"][str(t)] for t in range(1, expected_tasks)):
            raise ValueError("actual direction counts differ from the paired HOEP")
    if config.get("sa_coordinate_stable_transport", False):
        path = Path(entry["output"]) / "sa_coordinate_diagnostics.json"
        if not path.is_file():
            raise ValueError("prototype transport diagnostics are missing")
        records = json.loads(path.read_text())
        if [record.get("task_id") for record in records] != list(range(1, expected_tasks)):
            raise ValueError("prototype transport boundary coverage is incomplete")
        increments = config.get("task_increments") or [config["init_cls"]] + [config["increment"]] * (expected_tasks-1)
        for record in records:
            if record.get("num_prototypes") != sum(increments[:record["task_id"]]):
                raise ValueError("transport prototype count differs from old class set")
    return {**entry, **metrics, "gpu_pair": proof.get("gpu_pair"),
            "completion_proof": proof.get("proof_path"),
            "log_sha256": hashlib.sha256(Path(entry["log"]).read_bytes()).hexdigest()}


def prepare(project_root, runtime_root):
    project_root, runtime_root = Path(project_root).resolve(), Path(runtime_root).resolve()
    verify_sources(project_root, runtime_root)
    manifest = json.loads((runtime_root / "manifest.json").read_text())
    if len(manifest["jobs"]) != 15 or len(manifest["reused"]) != 3:
        raise RuntimeError("unexpected experiment matrix")
    events = [json.loads(line) for line in (runtime_root / "events.jsonl").read_text().splitlines()]
    if not any(event.get("kind") == "preflight_passed" for event in events):
        raise RuntimeError("original preflights have not passed")
    validate_reference(runtime_root / "references/hoep_seed1993_counts.json", 1993)
    work = runtime_root / "parallel"
    (work / "job_results").mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    target = work / "dispatcher_source.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if target.exists():
        if json.loads(target.read_text())["sha256"] != digest:
            raise RuntimeError("refusing to replace an existing dispatcher source pin")
    else:
        shutil.copy2(source, work / "dispatcher_snapshot.py")
        atomic_json(target, {"source": str(source), "sha256": digest,
                             "gpu_pairs": PAIRS, "original_runtime": str(runtime_root),
                             "training_revision": json.loads((runtime_root / "frozen_sources.json").read_text())["git_revision"]})
    return manifest


def verify_dispatcher(work):
    pin = json.loads((Path(work) / "dispatcher_source.json").read_text())
    if hashlib.sha256(Path(pin["source"]).read_bytes()).hexdigest() != pin["sha256"]:
        raise RuntimeError("parallel dispatcher source changed")


def worker(project_root, runtime_root, name, pair):
    project_root, runtime_root = Path(project_root).resolve(), Path(runtime_root).resolve()
    work = runtime_root / "parallel"
    proof_path = work / "job_results" / (name + ".json")
    proof = {"name": name, "phase": "starting", "gpu_pair": list(pair),
             "worker_pid": os.getpid(), "worker_start_ticks": process_info(os.getpid())["start_ticks"],
             "proof_path": str(proof_path)}
    try:
        verify_sources(project_root, runtime_root)
        verify_dispatcher(work)
        jobs = json.loads((runtime_root / "manifest.json").read_text())["jobs"]
        entry = next(job for job in jobs if job["name"] == name)
        if Path(entry["log"]).exists() or Path(entry["output"]).exists():
            raise RuntimeError(f"refusing to overwrite {name}")
        if entry["method"] in ("random", "high"):
            config = json.loads(Path(entry["config"]).read_text())
            validate_reference(config["sa_hoep_reference_counts_path"], entry["seed"])
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, pair)), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   "main.py", "--config=" + entry["config"]]
        with Path(entry["log"]).open("x") as stream:
            process = subprocess.Popen(command, cwd=project_root, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            proof.update(phase="running", training_pid=process.pid, command=command)
            atomic_json(proof_path, proof)
            code = process.wait()
        proof.update(phase="complete" if code == 0 else "failed", returncode=code)
        result = validated_result(entry, proof)
        proof["result"] = result
        atomic_json(proof_path, proof)
        return 0
    except Exception as error:
        proof.update(phase="failed", error=str(error))
        atomic_json(proof_path, proof)
        print(str(error), file=sys.stderr, flush=True)
        return 1


def handover(runtime_root, legacy_service):
    work = Path(runtime_root) / "parallel"
    target = work / "legacy_handoff.json"
    if target.exists():
        return json.loads(target.read_text())
    pid = int(subprocess.run(["systemctl", "--user", "show", legacy_service, "-p", "MainPID", "--value"],
                             capture_output=True, text=True, check=True).stdout.strip())
    parent = process_info(pid)
    if not parent or "scripts/hoep_cifar_ablation.py" not in parent["command"] or str(runtime_root) not in parent["command"]:
        raise RuntimeError("serial scheduler identity does not match this runtime")
    paused = False
    try:
        pause_parent(pid, parent["start_ticks"])
        paused = True
        state = json.loads((Path(runtime_root) / "status.json").read_text())
        if state.get("phase") != "training" or "current" not in state:
            raise RuntimeError("serial scheduler is not at a handover-compatible training boundary")
        child = process_info(state["training_pid"])
        if not child or child["parent_pid"] != pid:
            raise RuntimeError("adopted torchrun has no verified scheduler parent")
        if child["state"] != "Z" and state["current"]["config"] not in child["command"]:
            raise RuntimeError("adopted torchrun config does not match recorded current job")
        result = {"serial_parent": parent, "adopted_process": child,
                  "entry": state["current"], "gpu_pair": [0, 1],
                  "serial_service": legacy_service, "original_status": state}
        atomic_json(target, result)
        return result
    except Exception:
        if paused:
            os.kill(pid, signal.SIGCONT)
        raise


def gpu_measurements():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            timeout=20, check=True)
    return {int(parts[0]): (int(parts[1]), int(parts[2]))
            for line in result.stdout.splitlines() if (parts := line.split(","))}


def run(project_root, runtime_root, legacy_service):
    project_root, runtime_root = Path(project_root).resolve(), Path(runtime_root).resolve()
    work = runtime_root / "parallel"
    verify_sources(project_root, runtime_root)
    verify_dispatcher(work)
    manifest = json.loads((runtime_root / "manifest.json").read_text())
    jobs = manifest["jobs"]
    lock = (work / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active, completed, failures = {}, {}, []
    processes = {}
    idle_streak = {pair: 0 for pair in PAIRS}

    def event(kind, **detail):
        record = {"at": datetime.now(timezone.utc).isoformat(), "kind": kind, **detail}
        with (work / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def save_status(phase="training"):
        pending = [job["name"] for job in jobs if job["name"] not in completed and job["name"] not in active]
        state = {"phase": phase, "scheduler": "four_pair_parallel", "formal_total": 15, "reused": 3,
                 "gpu_pairs": PAIRS, "completed": list(completed), "active": list(active.values()),
                 "pending": pending, "failures": failures, "updated_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(work / "status.json", state)
        atomic_json(runtime_root / "status.json", state)
        results = json.loads((runtime_root / "reuse_audit.json").read_text()) + list(completed.values())
        atomic_json(work / "results_summary.json", results)
        atomic_json(runtime_root / "results_summary.json", results)

    def retire_serial_parent(handoff):
        parent = handoff["serial_parent"]
        info = process_info(parent["pid"])
        if info:
            if info["start_ticks"] != parent["start_ticks"]:
                raise RuntimeError("serial scheduler identity changed during handover")
            if info["state"] == "Z":
                return
            if info["state"] != "T":
                raise RuntimeError("serial scheduler identity/state changed during handover")
            os.kill(parent["pid"], signal.SIGKILL)  # only after adopted torchrun exit is verified
            event("serial_scheduler_retired", pid=parent["pid"], adopted_training_preserved=True)

    try:
        for job in jobs:
            previous = completed_run_metrics(job, runtime_root / "events.jsonl")
            if previous is not None:
                completed[job["name"]] = {**job, **previous}
            proof_path = work / "job_results" / (job["name"] + ".json")
            if proof_path.exists():
                proof = json.loads(proof_path.read_text())
                if proof["phase"] == "complete":
                    completed[job["name"]] = validated_result(job, proof)
                elif proof["phase"] == "failed":
                    failures.append({"name": job["name"], "error": proof.get("error", "worker failed")})
                else:
                    recovered = {"entry": job, "name": job["name"], "kind": "worker",
                                 "pid": proof["worker_pid"], "start_ticks": proof["worker_start_ticks"],
                                 "gpu_pair": proof["gpu_pair"]}
                    terminal = worker_progress(recovered, proof_path)
                    if terminal is None:
                        active[job["name"]] = recovered
                    elif terminal["phase"] == "complete":
                        completed[job["name"]] = validated_result(job, terminal)
                    else:
                        failures.append({"name": job["name"], "error": terminal.get("error", "worker failed")})
        handoff = handover(runtime_root, legacy_service)
        adopted = handoff["entry"]
        if adopted["name"] not in completed and not any(f.get("name") == adopted["name"] for f in failures):
            active[adopted["name"]] = {"entry": adopted, "name": adopted["name"], "kind": "adopted",
                                       "pid": handoff["adopted_process"]["pid"],
                                       "start_ticks": handoff["adopted_process"]["start_ticks"], "gpu_pair": [0, 1]}
        else:
            retire_serial_parent(handoff)
        event("parallel_queue_start", completed=list(completed), gpu_pairs=PAIRS,
              adopted=adopted["name"], adopted_pid=handoff["adopted_process"]["pid"])
        save_status()
        while active or len(completed) + len(failures) < len(jobs):
            for name, running in list(active.items()):
                proof_path = work / "job_results" / (name + ".json")
                if running["kind"] == "adopted":
                    info = process_info(running["pid"])
                    if info is None or info["start_ticks"] != running["start_ticks"]:
                        raise RuntimeError("adopted torchrun disappeared without a verified exit")
                    if info["state"] != "Z":
                        continue
                    proof = {"name": name, "phase": "complete" if info["exit_code"] == 0 else "failed",
                             "returncode": info["exit_code"], "gpu_pair": [0, 1],
                             "proof_path": str(proof_path), "exit_proof": "unreaped Linux process exit status"}
                else:
                    proof = worker_progress(running, proof_path)
                    if proof is None:
                        continue
                try:
                    result = validated_result(running["entry"], proof)
                    completed[name] = result
                    proof["result"] = result
                    event("training_complete", name=name, gpu_pair=running["gpu_pair"],
                          final=result["final"], aaa=result["aaa"])
                except Exception as error:
                    detail = proof.get("error") or str(error)
                    proof.update(phase="failed", error=detail)
                    failures.append({"name": name, "error": detail})
                    event("training_failed", name=name, error=detail)
                atomic_json(proof_path, proof)
                if running["kind"] == "adopted":
                    retire_serial_parent(handoff)
                elif name in processes:
                    processes.pop(name).wait()
                del active[name]
            if not failures:
                measurements = gpu_measurements()
                occupied = {tuple(item["gpu_pair"]) for item in active.values()}
                ready = ready_jobs(jobs, set(completed), set(active))
                for pair in PAIRS:
                    if pair in occupied:
                        idle_streak[pair] = 0
                        continue
                    idle = all(memory <= 64 and utilization <= 5 for memory, utilization in
                               (measurements[gpu] for gpu in pair))
                    idle_streak[pair] = idle_streak[pair] + 1 if idle else 0
                    if idle_streak[pair] < 2 or not ready:
                        continue
                    entry = ready.pop(0)
                    verify_sources(project_root, runtime_root)
                    verify_dispatcher(work)
                    if Path(entry["log"]).exists() or Path(entry["output"]).exists():
                        raise RuntimeError(f"unclaimed existing output refuses overwrite: {entry['name']}")
                    if entry["method"] in ("random", "high"):
                        config = json.loads(Path(entry["config"]).read_text())
                        validate_reference(config["sa_hoep_reference_counts_path"], entry["seed"])
                    command = [sys.executable, str(Path(__file__).resolve()), "--project-root", str(project_root),
                               "--runtime-dir", str(runtime_root), "--worker", entry["name"],
                               "--gpu-pair", ",".join(map(str, pair))]
                    process = subprocess.Popen(command, cwd=project_root)
                    processes[entry["name"]] = process
                    active[entry["name"]] = {"entry": entry, "name": entry["name"], "kind": "worker",
                                             "pid": process.pid, "start_ticks": process_info(process.pid)["start_ticks"],
                                             "gpu_pair": list(pair)}
                    event("training_start", name=entry["name"], method=entry["method"], seed=entry["seed"],
                          gpu_pair=pair, worker_pid=process.pid)
            save_status("draining_failed" if failures else "training")
            if failures and not active:
                break
            if not active and len(completed) < len(jobs) and not ready_jobs(jobs, set(completed), set()):
                raise RuntimeError("reference dependency deadlock")
            time.sleep(3)
        save_status("failed" if failures else "complete")
        event("parallel_queue_end", completed=len(completed), failures=failures)
        return 1 if failures else 0
    except Exception as error:
        failures.append({"error": str(error)})
        event("parallel_scheduler_failed", error=str(error))
        save_status("failed")
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--legacy-service", default="codex-hoep-cifar15-cuda7-20261008-r2.service")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--gpu-pair")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.project_root, args.runtime_dir)
        print("Four GPU pairs prepared; training sources/configs unchanged.")
        return
    if args.worker:
        pair = tuple(int(value) for value in args.gpu_pair.split(","))
        if pair not in PAIRS:
            parser.error("worker GPU pair is outside the authorized four pairs")
        raise SystemExit(worker(args.project_root, args.runtime_dir, args.worker, pair))
    raise SystemExit(run(args.project_root, args.runtime_dir, args.legacy_service))


if __name__ == "__main__":
    main()
