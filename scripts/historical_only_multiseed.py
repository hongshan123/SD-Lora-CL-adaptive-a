#!/usr/bin/env python3
"""Extend the sealed Historical-only protocol with seeds 4--13."""

import argparse
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.imageneta_live_frozen import (
    _literal, archive_attempt, check_idle, parse_training_log, sha256_file,
    validate_gpu_pairs, write_json,
)
from scripts.tasklen_fla_queue import dispatch_jobs, torchrun_command


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("c100", "inr", "cub")
SEEDS = tuple(range(4, 14))
GPU_PAIRS = ("0,1", "2,5", "6,7")
TRAIN_COMMIT = "c68e12a557bef0b197841a0a8ecb9a308c4b83cf"


def make_jobs(templates, runtime, phase):
    runtime = Path(runtime).resolve()
    jobs = []
    for seed in (SEEDS if phase == "formal" else (4,)):
        for dataset in DATASETS:
            name = f"{dataset}_historical_only_{phase}_seed{seed}_t10_i4_bs128"
            config = dict(templates[dataset])
            config.update(prefix=name, seed=[seed],
                          filepath=str(runtime / "results" / name) + "/")
            if phase == "smoke":
                config.update(max_tasks=2, init_epoch=1, epochs=1)
                if "tuned_epoch" in config:
                    config["tuned_epoch"] = 1
            path = runtime / "configs" / (name + ".json")
            write_json(path, config, exclusive=True)
            jobs.append(dict(name=name, dataset=dataset, seed=seed,
                             config=str(path), config_sha256=sha256_file(path),
                             result=config["filepath"], tasks=config["max_tasks"],
                             log=str(runtime / "logs" / (name + ".log"))))
    return jobs


def corrected_forgetting(groups):
    if not groups:
        return None
    keys = sorted((key for key in groups[-1] if re.fullmatch(r"\d+-\d+", key)),
                  key=lambda key: int(key.split("-")[0]))
    return statistics.mean(
        max(float(group[key]) for group in groups if key in group) - float(groups[-1][key])
        for key in keys[:-1]
    ) if len(keys) > 1 else 0.0


def parse_log(text, tasks):
    normalized = re.sub(r"^===== .*?END (\S+).*?status=(-?\d+) =====$",
                        r"===== END \1 status=\2 =====", text, flags=re.M)
    result = parse_training_log(normalized, tasks)
    groups = [_literal(value) for value in re.findall(r"CNN:\s*(\{[^\n]*\})", text)]
    result["forgetting_logged"] = result.get("forgetting")
    result["forgetting"] = corrected_forgetting(groups)
    result["latest_progress"] = text.replace("\r", "\n").splitlines()[-1][-400:] if text else "pending"
    return result


def inspect_job(job, verify_artifacts=True):
    path = Path(job["log"])
    text = path.read_text(errors="replace") if path.is_file() else ""
    record = dict(name=job["name"], dataset=job["dataset"], seed=job["seed"],
                  **parse_log(text, job["tasks"]))
    if verify_artifacts and record["complete"]:
        import torch
        try:
            root = Path(job["result"])
            provenance = json.loads((root / "run_manifest.json").read_text())
            if provenance["git_commit"] != TRAIN_COMMIT or provenance["config_sha256"] != job["config_sha256"]:
                raise ValueError("training commit/config provenance mismatch")
            state = torch.load(root / "sa_state.pt", map_location="cpu", weights_only=False)
            merged = torch.load(root / "sa_merged_lora.pt", map_location="cpu", weights_only=False)
            if state["task_id"] != job["tasks"] or merged["task_id"] != job["tasks"] - 1:
                raise ValueError("incomplete task count")
            for artifact, key, shape in ((state, "shared_a", (10, 768)),
                                          (state, "aggregate_up", (768, 10)),
                                          (merged, "shared_a", (10, 768)),
                                          (merged, "merged_b", (768, 10))):
                values = artifact[key]
                if len(values) != 24 or any(tuple(v.shape) != shape or not torch.isfinite(v).all() for v in values):
                    raise ValueError("missing/nonfinite Q/V branch state")
            if state["absorb_mode"] != "bounded_norm_calibrated_absorb" or not state["coordinate_aligned"]:
                raise ValueError("incorrect alignment/absorption mode")
            record["artifact_verified"] = True
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            record.update(complete=False, artifact_error=str(error))
    return record


def summarize(records):
    summary = {}
    for dataset in DATASETS:
        selected = [item for item in records if item["dataset"] == dataset]
        valid = [item for item in selected if item["complete"]]
        if len(selected) != 13 or len(valid) != 13 or sorted(item["seed"] for item in valid) != list(range(1, 14)):
            summary[dataset] = None
            continue
        summary[dataset] = {"n": 13}
        for metric in ("final", "aaa", "forgetting"):
            values = [item[metric] for item in valid]
            summary[dataset][metric] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))
    return summary


def validate_source(manifest):
    source = Path(manifest["source"])
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True).strip()
    if actual != TRAIN_COMMIT or dirty:
        raise RuntimeError("sealed training source changed")
    if subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=source, capture_output=True).returncode != 1:
        raise RuntimeError("training clone must stay detached")


def prepare(runtime):
    runtime.mkdir(parents=True, exist_ok=False)
    templates, prior = {}, []
    for dataset in DATASETS:
        for seed in range(1, 4):
            name = f"{dataset}_historical_signal_seed{seed}_t10_i4_bs128_2gpu"
            config_path = ROOT / ".runtime_historical_signal_multiseed" / (name + ".json")
            config = json.loads(config_path.read_text())
            if seed == 1:
                templates[dataset] = config
            job = dict(name=name, dataset=dataset, seed=seed, tasks=10,
                       config=str(config_path), config_sha256=sha256_file(config_path),
                       log=str(ROOT / (name + ".log")), result=str(ROOT / name.upper()))
            audited = inspect_job(job)
            if not audited["complete"]:
                raise RuntimeError(f"existing seed cannot be reused: {name}: {audited}")
            prior.append(job)
    source = runtime / "source"
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(ROOT), str(source)], check=True)
    subprocess.run(["git", "checkout", "--detach", TRAIN_COMMIT], cwd=source, check=True)
    (source / "data").symlink_to(ROOT / "data", target_is_directory=True)
    for directory in ("data/cifar-100-python", "data/imagenet-r/train", "data/imagenet-r/test", "data/cub/train", "data/cub/test"):
        if not (source / directory).is_dir():
            raise FileNotFoundError(directory)
    from huggingface_hub import hf_hub_download
    weights = Path(hf_hub_download("timm/vit_base_patch16_224.augreg2_in21k_ft_in1k", "model.safetensors", local_files_only=True))
    import torch
    import timm
    manifest = dict(created_at=datetime.now().isoformat(), source=str(source),
                    git_commit=TRAIN_COMMIT, runner_commit=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    seeds=list(SEEDS), reused_seeds=[1, 2, 3], gpu_pairs=list(GPU_PAIRS),
                    excluded_gpus=[3, 4], world_size=2, effective_batch_size=128,
                    templates=templates, prior=prior,
                    base_weights=str(weights), base_weights_sha256=sha256_file(weights),
                    environment=dict(python=sys.version, torch=torch.__version__, timm=timm.__version__),
                    smoke=make_jobs(templates, runtime, "smoke"),
                    formal=make_jobs(templates, runtime, "formal"))
    validate_source(manifest)
    write_json(runtime / "manifest.json", manifest, exclusive=True)
    write_json(ROOT / "historical_only_seed4_13_registration.json", manifest, exclusive=True)
    print(f"PREPARED formal=30 prior=9 source={source}", flush=True)


def load_manifest(runtime):
    return json.loads((Path(runtime) / "manifest.json").read_text())


def run_one(source, job, gpu_ids, world_size):
    manifest = load_manifest(Path(source).parent)
    validate_source(manifest)
    if sha256_file(job["config"]) != job["config_sha256"]:
        raise RuntimeError(f"config changed: {job['name']}")
    if inspect_job(job)["complete"]:
        return 0
    if Path(job["log"]).exists() or Path(job["result"]).exists():
        listing = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
        if any(job["config"] in line for line in listing.splitlines()):
            raise RuntimeError(f"training already active: {job['name']}")
        archive_attempt(job, Path(source).parent / "archive")
    Path(job["log"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["result"]).parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu_ids, PYTHONUNBUFFERED="1", HF_ENDPOINT="https://hf-mirror.com",
               HF_HUB_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", TORCH_DETERMINISTIC="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = ["nohup"] + torchrun_command(world_size, job["config"])
    print(f"START {job['name']} GPUs={gpu_ids} time={datetime.now().isoformat()}", flush=True)
    with Path(job["log"]).open("x") as stream:
        stream.write(f"===== START {job['name']} GPUs={gpu_ids} =====\ncommand={json.dumps(command)}\n")
        stream.flush()
        process = subprocess.run(command, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"===== END {job['name']} status={process.returncode} =====\n")
    audited = inspect_job(job)
    status = 0 if audited["complete"] else (process.returncode or 1)
    print(f"END {job['name']} status={status} time={datetime.now().isoformat()} audit={json.dumps(audited)}", flush=True)
    return status


def report(runtime, phase="formal"):
    manifest = load_manifest(runtime)
    jobs = manifest[phase] + (manifest["prior"] if phase == "formal" else [])
    records = [inspect_job(job) for job in jobs]
    result = dict(phase=phase, records=records, summary=summarize(records) if phase == "formal" else None)
    write_json(Path(runtime) / (phase + "_results.json"), result)
    return result


def export_csv(runtime):
    result = report(runtime)
    with (Path(runtime) / "formal_results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "seed", "complete", "final", "aaa", "forgetting_corrected", "forgetting_logged", "curve"])
        for item in result["records"]:
            writer.writerow([item["dataset"], item["seed"], item["complete"], item.get("final"),
                             item.get("aaa"), item.get("forgetting"), item.get("forgetting_logged"),
                             json.dumps(item.get("curve", []))])


def queue(runtime, phase):
    with (runtime / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = load_manifest(runtime)
        validate_source(manifest)
        pairs = validate_gpu_pairs(";".join(manifest["gpu_pairs"]), excluded_gpus=manifest["excluded_gpus"])
        if sha256_file(manifest["base_weights"]) != manifest["base_weights_sha256"]:
            raise RuntimeError("pretrained weights changed")
        for job in manifest[phase]:
            if sha256_file(job["config"]) != job["config_sha256"]:
                raise RuntimeError("registered config changed")
        if phase == "formal" and not all(item["complete"] for item in report(runtime, "smoke")["records"]):
            raise RuntimeError("three dataset Task0/1 smoke checks must pass first")
        check_idle(pairs)
        jobs = [job for job in manifest[phase] if not inspect_job(job)["complete"]]
        write_json(runtime / (phase + "_queue_status.json"), dict(state="running", jobs=len(jobs)))
        print(f"QUEUE START phase={phase} jobs={len(jobs)} commit={TRAIN_COMMIT}", flush=True)
        try:
            status = dispatch_jobs(manifest["source"], jobs, pairs, 2, runner=run_one, stop_on_failure=True)
            report(runtime, phase)
            if phase == "formal":
                export_csv(runtime)
        except Exception as error:
            write_json(runtime / (phase + "_queue_status.json"), dict(state="failed", error=str(error)))
            raise
        write_json(runtime / (phase + "_queue_status.json"), dict(state="complete" if status == 0 else "failed", status=status))
        print(f"QUEUE END phase={phase} status={status}", flush=True)
        return status


def monitor(runtime):
    while True:
        result = report(runtime)
        export_csv(runtime)
        print(json.dumps(dict(time=datetime.now().isoformat(), records=result["records"], summary=result["summary"])), flush=True)
        print(subprocess.check_output(["nvidia-smi"], text=True), flush=True)
        processes = subprocess.check_output(["ps", "-eo", "pid,etimes,args"], text=True)
        print("\n".join(line for line in processes.splitlines() if str(runtime) in line), flush=True)
        state_path = runtime / "formal_queue_status.json"
        if state_path.exists() and json.loads(state_path.read_text())["state"] in {"complete", "failed"}:
            print(f"MONITOR END: {state_path.read_text()}", flush=True)
            return
        time.sleep(1800)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "queue", "monitor", "report"))
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--phase", choices=("smoke", "formal"), default="formal")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if args.action == "prepare":
        prepare(runtime)
    elif args.action == "queue":
        return queue(runtime, args.phase)
    elif args.action == "monitor":
        monitor(runtime)
    else:
        result = report(runtime, args.phase)
        if args.phase == "formal":
            export_csv(runtime)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
