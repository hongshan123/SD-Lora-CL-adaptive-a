#!/usr/bin/env python3
"""Matched CIFAR direction ablations: three reused cells, fifteen new runs."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_hoep_p3_configs import build_config

SEEDS = (1993, 1994, 1995)
POLICIES = ("frozen", "live", "hoep", "canonical_live", "random", "high")
OLD_TAG = "20260921_164138"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_metrics(log, expected_tasks=10):
    body = Path(log).read_text(errors="replace")
    matches = re.findall(r"CNN top1 curve:\s*(\[[^\n]+\])", body)
    encoded = matches[-1] if matches else "[]"
    # NumPy2 renders list elements as np.float64(...). Strip only numeric
    # wrappers; keep literal_eval so arbitrary calls in a log never execute.
    encoded = re.sub(r"np\.float(?:32|64)\(([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)",
                     r"\1", encoded)
    curve = ast.literal_eval(encoded)
    if len(curve) != expected_tasks or "Forgetting (CNN):" not in body:
        raise ValueError(f"log must contain {expected_tasks} completed tasks: {log}")
    return {"final": float(curve[-1]), "aaa": sum(map(float, curve)) / len(curve),
            "curve": curve}


def freeze_sources(project_root, runtime_root, manifest):
    """Snapshot production sources and pin configs before any GPU run."""
    project_root, runtime_root = Path(project_root), Path(runtime_root)
    paths = [project_root / name for name in ("main.py", "trainer.py")
             if (project_root / name).is_file()]
    for directory in ("backbone", "models", "utils", "scripts"):
        paths.extend(sorted((project_root / directory).rglob("*.py")))
    source_hashes = {}
    for path in paths:
        relative = path.relative_to(project_root)
        source_hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot = runtime_root / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
    configs = [Path(job["config"]) for job in manifest["jobs"] + manifest["preflight"]]
    configs.extend(runtime_root / name for name in
                   ("manifest.json", "reuse_audit.json", "references/hoep_seed1993_counts.json")
                   if (runtime_root / name).is_file())
    config_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in configs}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                              capture_output=True, text=True).stdout.strip()
    patch = subprocess.run(["git", "diff", "--binary"], cwd=project_root,
                           capture_output=True, text=True).stdout
    (runtime_root / "experiment_patch.diff").write_text(patch)
    atomic_json(runtime_root / "frozen_sources.json",
                {"git_revision": revision, "source_hashes": source_hashes,
                 "config_hashes": config_hashes, "snapshot": str(runtime_root / "source_snapshot")})


def verify_sources(project_root, runtime_root):
    provenance = json.loads((Path(runtime_root) / "frozen_sources.json").read_text())
    for relative, expected in provenance["source_hashes"].items():
        path = Path(project_root) / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"experiment source changed: {path}")
    for name, expected in provenance["config_hashes"].items():
        path = Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"experiment config changed: {path}")


def completed_run_metrics(entry, events_path, expected_tasks=10):
    """Recover only runs whose subprocess exit and complete log were recorded."""
    path = Path(events_path)
    if not path.is_file():
        return None
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(event.get("name") == entry["name"] and event.get("kind") in
           ("training_complete", "preflight_complete") for event in events):
        return read_metrics(entry["log"], expected_tasks)
    return None


def generate_manifest(project_root, runtime_root, previous_root, run_tag):
    project_root, runtime_root, previous_root = map(
        lambda p: Path(p).resolve(), (project_root, runtime_root, previous_root))
    source = json.loads((project_root / "exps/hoep_c100_seed1993_t10.json").read_text())
    config_root = runtime_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    old_runtime = previous_root / f".runtime_hoep_p3_{OLD_TAG}_no23"
    reused = []
    for method in ("frozen", "live", "hoep"):
        name = f"hoep_p3_c100_{method}_seed1993_t10_e20_bs128_2gpu_{OLD_TAG}"
        reused.append({"name": name, "method": method, "seed": 1993,
                       "log": str(previous_root / (name + ".log")),
                       "config": str(old_runtime / "configs" / (name + ".json")),
                       "output": str(old_runtime / "results" / name)})
    order = [(method, seed) for seed in (1994, 1995) for method in ("frozen", "live", "hoep")]
    order += [(method, seed) for seed in SEEDS for method in ("canonical_live", "random", "high")]
    jobs = []
    for method, seed in order:
        base_method = method if method in ("frozen", "live", "hoep") else "hoep"
        _, config = build_config(source, "c100", base_method, runtime_root, run_tag)
        name = f"hoep_cifar_abl_{method}_seed{seed}_t10_e20_bs128_2gpu_{run_tag}"
        output = runtime_root / "results" / name
        config.update(prefix=name, seed=[seed], filepath=str(output) + "/")
        if base_method == "hoep":
            config["sa_hoep_partition_policy"] = {
                "hoep": "low", "canonical_live": "all", "random": "random", "high": "high"
            }[method]
            config["sa_hoep_partition_seed"] = seed
        if method in ("random", "high"):
            if seed == 1993:
                reference = runtime_root / "references/hoep_seed1993_counts.json"
            else:
                reference_job = next(j for j in jobs if j["method"] == "hoep" and j["seed"] == seed)
                reference = Path(reference_job["output"]) / "hoep_partition_schedule.json"
            config["sa_hoep_reference_counts_path"] = str(reference)
        config_path = config_root / (name + ".json")
        atomic_json(config_path, config)
        jobs.append({"name": name, "method": method, "seed": seed,
                     "config": str(config_path), "output": str(output),
                     "log": str(runtime_root / "logs" / (name + ".log"))})
    smokes = []
    for method in ("hoep", "random", "high", "canonical_live"):
        _, config = build_config(source, "c100", "hoep", runtime_root, run_tag)
        name = f"hoep_cifar_preflight_{method}_seed1993_{run_tag}"
        config.update(prefix=name, filepath=str(runtime_root / "preflight" / name) + "/",
                      init_epoch=1, epochs=1, max_tasks=2,
                      sa_hoep_partition_seed=1993,
                      sa_hoep_partition_policy={"hoep": "low", "canonical_live": "all", "random": "random", "high": "high"}[method])
        if method in ("random", "high"):
            config["sa_hoep_reference_counts_path"] = str(
                Path(smokes[0]["output"]) / "hoep_partition_schedule.json")
        path = config_root / (name + ".json")
        atomic_json(path, config)
        smokes.append({"name": name, "method": method, "seed": 1993,
                       "config": str(path), "output": config["filepath"],
                       "log": str(runtime_root / "logs" / (name + ".log"))})
    manifest = {"version": 1, "run_tag": run_tag, "base_revision": "6380f77",
                "host": "cuda7", "gpu_ids": [0, 1], "reused": reused,
                "jobs": jobs, "preflight": smokes}
    atomic_json(runtime_root / "manifest.json", manifest)
    return manifest


def audit_reused(manifest, previous_root, runtime_root):
    import torch
    previous_root, runtime_root = Path(previous_root), Path(runtime_root)
    audit = []
    for entry in manifest["reused"]:
        body = Path(entry["log"]).read_text()
        if not re.search(r"END .*status=0", body):
            raise ValueError(f"previous run lacks success marker: {entry['name']}")
        config = json.loads(Path(entry["config"]).read_text())
        if config["batch_size"] != 64 or config["seed"] != [1993] or config["device"] != ["0", "1"]:
            raise ValueError("previous run protocol mismatch")
        if any(config[k] for k in ("sa_normalize_current_branch", "sa_coordinate_stable_transport", "sa_dual_head", "sa_hbd_enabled")):
            raise ValueError("previous run has extra modules enabled")
        audit.append({**entry, **read_metrics(entry["log"]),
                      "log_sha256": hashlib.sha256(Path(entry["log"]).read_bytes()).hexdigest()})
    original = Path(next(j["output"] for j in manifest["reused"] if j["method"] == "hoep"))
    shadow = previous_root / "FUNCTIONAL_HOEP_PHASE_A_20260922/results/functional_hoep_phase_a_c100_seed1993_t10_e20_bs128_2gpu_20260922"
    a = torch.load(original / "sa_state.pt", map_location="cpu", weights_only=False)
    b = torch.load(shadow / "sa_state.pt", map_location="cpu", weights_only=False)
    for key in ("shared_a", "aggregate_up"):
        if len(a[key]) != len(b[key]) or not all(torch.equal(x, y) for x, y in zip(a[key], b[key])):
            raise ValueError("shadow reference does not reproduce original A/G")
    a = torch.load(original / "sa_prototypes.pt", map_location="cpu", weights_only=False)
    b = torch.load(shadow / "sa_prototypes.pt", map_location="cpu", weights_only=False)
    if set(a) != set(b) or not all(torch.equal(a[k], b[k]) for k in a):
        raise ValueError("shadow reference does not reproduce original prototypes")
    shadow_log = previous_root / "functional_hoep_phase_a_c100_seed1993_t10_e20_bs128_2gpu_20260922.log"
    if read_metrics(shadow_log)["curve"] != read_metrics(next(j["log"] for j in manifest["reused"] if j["method"] == "hoep"))["curve"]:
        raise ValueError("shadow reference has a different ten-task curve")
    source = shadow / "hoep_functional_diagnostics.pt"
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    counts = {str(t["task_id"]): [int(branch["operator_selected_directions"]) for branch in t["branches"]]
              for t in artifact["tasks"]}
    if set(counts) != {str(t) for t in range(1, 10)} or any(len(v) != 24 for v in counts.values()):
        raise ValueError("incomplete seed1993 reference counts")
    atomic_json(runtime_root / "references/hoep_seed1993_counts.json",
                {"version": 1, "seed": 1993, "tasks": counts,
                 "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                 "verification": "exact final shared A/G and prototypes; identical ten-task curve"})
    atomic_json(runtime_root / "reuse_audit.json", audit)
    return audit


def run_queue(project_root, runtime_root):
    project_root, runtime_root = Path(project_root).resolve(), Path(runtime_root).resolve()
    manifest = json.loads((runtime_root / "manifest.json").read_text())
    lock = (runtime_root / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"phase": "starting", "completed": [], "formal_total": 15, "reused": 3}

    def event(kind, **details):
        record = {"at": datetime.now(timezone.utc).isoformat(), "kind": kind, **details}
        with (runtime_root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        state.update(last_event=record)
        atomic_json(runtime_root / "status.json", state)
        print(json.dumps(record), flush=True)

    def wait_idle():
        streak = 0
        while streak < 2:
            result = subprocess.run(["nvidia-smi", "-i", "0,1", "--query-gpu=memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=True)
            measurements = [[int(v.strip()) for v in line.split(",")] for line in result.stdout.strip().splitlines()]
            idle = len(measurements) == 2 and all(memory <= 64 and utilization <= 5 for memory, utilization in measurements)
            streak = streak + 1 if idle else 0
            if not idle:
                event("waiting_for_gpu", measurements=measurements)
            time.sleep(3 if idle else 30)

    def train(entry, preflight=False):
        verify_sources(project_root, runtime_root)
        recovered = completed_run_metrics(entry, runtime_root / "events.jsonl", 2 if preflight else 10)
        if recovered is not None:
            event("completed_run_reused", name=entry["name"])
            return {**entry, **recovered}
        if Path(entry["output"]).exists() or Path(entry["log"]).exists():
            raise RuntimeError(f"refusing to overwrite run {entry['name']}")
        wait_idle()
        verify_sources(project_root, runtime_root)
        Path(entry["log"]).parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="0,1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   "main.py", "--config=" + entry["config"]]
        event("preflight_start" if preflight else "training_start", name=entry["name"], method=entry["method"], seed=entry["seed"])
        with Path(entry["log"]).open("x") as stream:
            process = subprocess.Popen(command, cwd=project_root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            state.update(current=entry, training_pid=process.pid)
            atomic_json(runtime_root / "status.json", state)
            code = process.wait()
        if code:
            raise RuntimeError(f"{entry['name']} exited with status {code}; see {entry['log']}")
        metrics = read_metrics(entry["log"], 2 if preflight else 10)
        if "within_bound=False" in Path(entry["log"]).read_text():
            raise RuntimeError(f"operator residual bound failed: {entry['name']}")
        if entry["method"] in ("random", "high"):
            config = json.loads(Path(entry["config"]).read_text())
            reference = json.loads(Path(config["sa_hoep_reference_counts_path"]).read_text())
            actual = json.loads((Path(entry["output"]) / "hoep_partition_schedule.json").read_text())
            for task in range(1, 2 if preflight else 10):
                if actual["tasks"][str(task)] != reference["tasks"][str(task)]:
                    raise RuntimeError("direction counts differ from matched HOEP reference")
        event("preflight_complete" if preflight else "training_complete", name=entry["name"], **metrics)
        return {**entry, **metrics}

    try:
        verify_sources(project_root, runtime_root)
        event("queue_start", formal_total=15, reused=3, gpu_ids=[0, 1])
        state["phase"] = "preflight"
        for entry in manifest["preflight"]:
            train(entry, preflight=True)
        task_zero_hashes = [json.loads((Path(entry["output"]) / "p0_hashes.json").read_text())["0"]
                           for entry in manifest["preflight"]]
        if any(h != task_zero_hashes[0] for h in task_zero_hashes[1:]):
            raise RuntimeError("preflight policies changed Task0 model or RNG")
        event("preflight_passed", task_zero_weights_rng_equal=True)
        state["phase"] = "training"
        results = json.loads((runtime_root / "reuse_audit.json").read_text())
        for entry in manifest["jobs"]:
            result = train(entry)
            state["completed"].append(entry["name"])
            results.append(result)
            atomic_json(runtime_root / "results_summary.json", results)
            atomic_json(runtime_root / "status.json", state)
        state["phase"] = "complete"
        event("queue_complete", formal_completed=15)
    except Exception as error:
        state["phase"] = "failed"
        event("queue_failed", error=str(error))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--previous-root")
    parser.add_argument("--run-tag", default="20261008")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if not args.previous_root:
            parser.error("--prepare requires --previous-root")
        manifest = generate_manifest(args.project_root, args.runtime_dir, args.previous_root, args.run_tag)
        audit_reused(manifest, args.previous_root, args.runtime_dir)
        freeze_sources(args.project_root, args.runtime_dir, manifest)
        print("Prepared 15 new full runs, 3 audited reused cells, and four short preflights.")
    else:
        run_queue(args.project_root, args.runtime_dir)


if __name__ == "__main__":
    main()
