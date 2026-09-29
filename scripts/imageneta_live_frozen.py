#!/usr/bin/env python3
"""Audited, paired ImageNet-A experiments; no changes to training algorithms."""

import argparse
import ast
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tasklen_fla_queue import dispatch_jobs, torchrun_command


SEEDS = (1995, 1996, 1997)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}
ERROR_RE = re.compile(r"Traceback \(most recent call last\)|CUDA out of memory|NCCL.*(?:timeout|error)|ChildFailedError", re.I)
END_RE = re.compile(r"^===== END .* status=(-?\d+) =====$", re.M)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(serialized)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(serialized)
        os.replace(temporary, path)


def audit_dataset(root, expected_classes=200, decode=True):
    """Fingerprint ImageFolder paths/content and reject cross-split leakage."""
    from PIL import Image

    root = Path(root).resolve()
    mappings = {}
    records = []
    for split in ("train", "test"):
        directory = root / split
        classes = sorted(path.name for path in directory.iterdir() if path.is_dir())
        if len(classes) != expected_classes:
            raise ValueError(f"{split}: expected {expected_classes} classes, found {len(classes)}")
        mappings[split] = {label: index for index, label in enumerate(classes)}
        for label in classes:
            images = sorted(path for path in (directory / label).rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file())
            if not images:
                raise ValueError(f"empty class: {split}/{label}")
            for path in images:
                if decode:
                    try:
                        with Image.open(path) as image:
                            image.convert("RGB").load()
                    except Exception as error:
                        raise ValueError(f"image decode failed: {path}") from error
                records.append({"split": split, "path": path.relative_to(directory).as_posix(), "class": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if mappings["train"] != mappings["test"]:
        raise ValueError("train/test class mapping mismatch")
    hashes = {split: {item["sha256"] for item in records if item["split"] == split} for split in mappings}
    overlap = hashes["train"] & hashes["test"]
    if overlap:
        raise ValueError(f"train/test content hash overlap: {len(overlap)}")
    paths = {split: {item["path"] for item in records if item["split"] == split} for split in mappings}
    payload = json.dumps({"mapping": mappings["train"], "files": records}, sort_keys=True, separators=(",", ":")).encode()
    return {
        "root": str(root), "class_to_idx": mappings["train"], "files": records,
        "counts": {split: sum(item["split"] == split for item in records) for split in mappings},
        "data_sha256": hashlib.sha256(payload).hexdigest(),
        "relative_path_overlap": len(paths["train"] & paths["test"]),
        "content_hash_overlap": 0, "decode_checked": decode,
    }


def prepare_data_link(root, dataset):
    link = Path(root) / "data/imagenet-a"
    dataset = Path(dataset).resolve(strict=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve() != dataset:
            raise FileExistsError(f"refusing to replace {link}")
    else:
        link.symlink_to(dataset, target_is_directory=True)
    return link


def build_jobs(root, config_root, runtime, tag, smoke=False):
    if not re.fullmatch(r"[A-Za-z0-9_]+", tag):
        raise ValueError("run tag must contain only letters, digits and underscores")
    source = json.loads((Path(root) / "exps/frozen_margin_inr_t10_seed1995.json").read_text())
    runtime = Path(runtime).resolve()
    jobs = []
    for method in ("frozen", "live"):
        for seed in ((1995,) if smoke else SEEDS):
            name = f"ina_{'smoke' if smoke else 'formal'}_{method}_seed{seed}_{tag}"
            config = dict(source)
            config.update(dataset="imageneta", prefix=name, filepath=str(runtime / "results" / name) + "/", seed=[seed], sa_train_a_all_tasks=method == "live")
            if smoke:
                config.update(max_tasks=2, init_epoch=2, epochs=2, tuned_epoch=2)
            path = Path(config_root).resolve() / (name + ".json")
            write_json(path, config, exclusive=True)
            jobs.append(dict(name=name, method=method, seed=seed, config=str(path), config_sha256=sha256_file(path), result=config["filepath"], log=str(runtime / "logs" / (name + ".log")), tasks=config["max_tasks"]))
    return jobs


def validate_gpu_pairs(value):
    pairs = tuple(pair.strip() for pair in value.split(";") if pair.strip())
    used = []
    for pair in pairs:
        ids = pair.split(",")
        if len(ids) != 2 or any(not item.isdigit() for item in ids):
            raise ValueError("every GPU slot must contain exactly two GPU IDs")
        used.extend(int(item) for item in ids)
    if not pairs or len(set(used)) != len(used) or any(item not in {0, 1, 4, 5, 6, 7} for item in used):
        raise ValueError("GPU pairs overlap or contain an excluded/nonexistent GPU")
    return pairs


def _literal(value):
    # NumPy scalar repr is present in this repo's metric dictionaries.
    class ScalarRepr(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "np" and node.func.attr in {"float64", "float32"} and len(node.args) == 1 and not node.keywords:
                return self.visit(node.args[0])
            raise ValueError("unsupported metric expression")
    return ast.literal_eval(ScalarRepr().visit(ast.parse(value, mode="eval")))


def parse_training_log(text, tasks=10):
    statuses = END_RE.findall(text)
    curves = re.findall(r"CNN top1 curve:\s*(\[[^\n]*\])", text)
    grouped = re.findall(r"CNN:\s*(\{[^\n]*\})", text)
    forgetting = re.findall(r"Forgetting \(CNN\):\s*([-+\d.eE]+)", text)
    result = {"complete": False, "status": int(statuses[-1]) if statuses else None, "errors": ERROR_RE.findall(text)}
    try:
        curve = [float(value) for value in _literal(curves[-1])] if curves else []
        groups = [_literal(value) for value in grouped]
        if any(not math.isfinite(value) or not 0 <= value <= 100 for value in curve):
            raise ValueError("invalid accuracy")
        result.update(curve=curve, old_curve=[float(group["old"]) for group in groups], new_curve=[float(group["new"]) for group in groups])
        if forgetting and math.isfinite(float(forgetting[-1])):
            result["forgetting"] = float(forgetting[-1])
        if curve:
            result.update(final=curve[-1], aaa=statistics.mean(curve))
        result["complete"] = result["status"] == 0 and not result["errors"] and len(curve) == len(groups) == tasks and "Accuracy Matrix (CNN):" in text and "forgetting" in result
    except (ValueError, SyntaxError, KeyError, TypeError):
        result["parse_error"] = True
    return result


def paired_summary(records):
    selected = {(item["method"], item["seed"]): item for item in records if item["complete"]}
    if len(selected) != 6 or any((method, seed) not in selected for method in ("frozen", "live") for seed in SEEDS):
        return None
    summary = {}
    for method in ("frozen", "live"):
        summary[method] = {}
        for metric in ("final", "aaa", "forgetting"):
            values = [selected[method, seed][metric] for seed in SEEDS]
            summary[method][metric] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))
    summary["paired_live_minus_frozen"] = {}
    for metric in ("final", "aaa", "forgetting"):
        values = [selected["live", seed][metric] - selected["frozen", seed][metric] for seed in SEEDS]
        summary["paired_live_minus_frozen"][metric] = dict(values=values, mean=statistics.mean(values), sample_std=statistics.stdev(values))
    return summary


def archive_attempt(job, archive_root):
    archive = Path(archive_root) / (job.get("name", "attempt") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    archive.mkdir(parents=True, exist_ok=False)
    for key in ("log", "result"):
        path = Path(job[key])
        if path.exists():
            shutil.move(str(path), str(archive / path.name))
    write_json(archive / "job.json", job)
    return archive


def inspect_job(job, verify_artifacts=False):
    log = Path(job["log"])
    text = log.read_text(errors="replace") if log.is_file() else ""
    result = dict(name=job["name"], method=job["method"], seed=job["seed"], **parse_training_log(text, job["tasks"]))
    result["latest_progress"] = text.splitlines()[-1][-400:] if text else "pending"
    if verify_artifacts and result["complete"]:
        from utils.sa_task_snapshots import audit_snapshot
        try:
            directories = sorted((Path(job["result"]) / "task_snapshots").glob("task_*"))
            if len(directories) != job["tasks"]:
                raise ValueError("missing task snapshots")
            result["snapshots"] = [audit_snapshot(path) for path in directories]
            if not all(item["verified"] for item in result["snapshots"]):
                raise ValueError("unfinalized task snapshot")
        except (ValueError, FileNotFoundError) as error:
            result.update(complete=False, artifact_error=str(error))
    return result


def report(runtime, phase):
    runtime = Path(runtime)
    manifest = json.loads((runtime / "manifest.json").read_text())
    records = [inspect_job(job, verify_artifacts=True) for job in manifest[phase]]
    result = dict(phase=phase, records=records, summary=paired_summary(records) if phase == "formal" else None)
    write_json(runtime / (phase + "_results.json"), result)
    with (runtime / (phase + "_results.csv")).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "seed", "complete", "final", "aaa", "forgetting", "task", "top1_global", "old_global", "new_global"])
        for item in records:
            for task, top1 in enumerate(item.get("curve", [])):
                writer.writerow([item["method"], item["seed"], item["complete"], item.get("final"), item.get("aaa"), item.get("forgetting"), task, top1, item["old_curve"][task] if task < len(item["old_curve"]) else "", item["new_curve"][task] if task < len(item["new_curve"]) else ""])
    return result


def git_commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def validate_seal(runtime, manifest):
    source = Path(manifest["source"])
    if git_commit(source) != manifest["git_commit"]:
        raise RuntimeError("training source commit changed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise RuntimeError("training source is not clean")
    for job in manifest["smoke"] + manifest["formal"]:
        if sha256_file(job["config"]) != job["config_sha256"]:
            raise RuntimeError(f"configuration changed: {job['name']}")
    audit = json.loads((runtime / "data_audit.json").read_text())
    current = audit_dataset(audit["root"], decode=False)
    if current["data_sha256"] != audit["data_sha256"]:
        raise RuntimeError("dataset changed after audit")


def run_one(source, job, gpu_ids, world_size):
    result = Path(job["result"])
    log = Path(job["log"])
    if result.exists() or log.exists():
        if inspect_job(job, verify_artifacts=True)["complete"]:
            print(f"REUSE {job['name']}", flush=True)
            return 0
        # systemd owns the process tree; a restarted queue must not duplicate it.
        listing = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
        if any(job["config"] in line and str(os.getpid()) != line.strip().split()[0] for line in listing.splitlines()):
            raise RuntimeError(f"training still active: {job['name']}")
        archive = archive_attempt(job, log.parent.parent / "archive")
        print(f"ARCHIVE {job['name']} {archive}", flush=True)
    if sha256_file(job["config"]) != job["config_sha256"]:
        raise RuntimeError("configuration checksum changed")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu_ids, PYTHONUNBUFFERED="1", HF_ENDPOINT="https://hf-mirror.com", CUBLAS_WORKSPACE_CONFIG=":4096:8", TORCH_DETERMINISTIC="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = ["nohup"] + torchrun_command(world_size, job["config"])
    print(f"START {job['name']} GPUs={gpu_ids} time={datetime.now().isoformat()}", flush=True)
    with log.open("x") as stream:
        stream.write(f"===== START {job['name']} GPUs={gpu_ids} =====\ncommand={json.dumps(command)}\n")
        stream.flush()
        process = subprocess.run(command, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"===== END {job['name']} status={process.returncode} =====\n")
    audited = inspect_job(job, verify_artifacts=True)
    status = 0 if audited["complete"] else (process.returncode or 1)
    print(f"END {job['name']} status={status} time={datetime.now().isoformat()}", flush=True)
    return status


def check_idle(pairs):
    used = {int(item) for pair in pairs for item in pair.split(",")}
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    for row in rows.splitlines():
        index, memory, utilization = [int(value.strip()) for value in row.split(",")]
        if index in used and (memory > 256 or utilization > 10):
            raise RuntimeError(f"GPU {index} is occupied; refusing to overlap training")


def queue(runtime, phase, pairs):
    runtime = Path(runtime).resolve()
    with (runtime / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((runtime / "manifest.json").read_text())
        validate_seal(runtime, manifest)
        if phase == "formal":
            gate = json.loads((runtime / "smoke_verification.json").read_text())
            if gate.get("passed") is not True or gate.get("git_commit") != manifest["git_commit"]:
                raise RuntimeError("formal training requires verified smoke for this commit")
        check_idle(pairs)
        jobs = [job for job in manifest[phase] if not inspect_job(job, verify_artifacts=True)["complete"]]
        print(f"QUEUE START phase={phase} jobs={len(jobs)} commit={manifest['git_commit']}", flush=True)
        status = dispatch_jobs(manifest["source"], jobs, pairs, 2, runner=run_one, stop_on_failure=True)
        report(runtime, phase)
        print(f"QUEUE END phase={phase} status={status}", flush=True)
        return status


def monitor(runtime, interval=1800):
    runtime = Path(runtime)
    while True:
        result = report(runtime, "formal")
        print(json.dumps({"time": datetime.now().isoformat(), "jobs": result["records"]}, allow_nan=False), flush=True)
        for command in (["nvidia-smi"], ["ps", "-eo", "pid,etimes,args"]):
            output = subprocess.check_output(command, text=True)
            if command[0] == "ps":
                output = "\n".join(line for line in output.splitlines() if "imageneta_live_frozen" in line or "ina_formal_" in line)
            print(output, flush=True)
        if all(item["complete"] for item in result["records"]):
            print("MONITOR END: all six formal results verified", flush=True)
            return
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "seal", "queue", "report", "monitor"))
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--dataset", type=Path, default=Path("/data/dataset/Imagenet-A"))
    parser.add_argument("--phase", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--gpu-pairs", default="0,1;4,5;6,7")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if args.action == "prepare":
        audit = audit_dataset(args.dataset)
        if audit["counts"] != {"train": 5981, "test": 1519}:
            raise ValueError("unexpected ImageNet-A split sizes")
        prepare_data_link(args.root, args.dataset)
        configs = args.root / "exps" / ("imageneta_live_frozen_" + args.tag)
        manifest = dict(version=1, tag=args.tag, data_sha256=audit["data_sha256"], smoke=build_jobs(args.root, configs, runtime, args.tag, smoke=True), formal=build_jobs(args.root, configs, runtime, args.tag))
        write_json(runtime / "data_audit.json", audit, exclusive=True)
        write_json(runtime / "manifest.json", manifest, exclusive=True)
        print(json.dumps({key: audit[key] for key in ("counts", "data_sha256", "relative_path_overlap", "content_hash_overlap")}))
    elif args.action == "seal":
        manifest = json.loads((runtime / "manifest.json").read_text())
        if "git_commit" in manifest:
            raise ValueError("run is already sealed")
        prepare_data_link(args.source, args.dataset)
        manifest.update(source=str(args.source.resolve()), git_commit=git_commit(args.source))
        validate_seal(runtime, manifest)
        write_json(runtime / "manifest.json", manifest)
        print(f"SEALED {manifest['git_commit']}")
    elif args.action == "queue":
        sys.exit(queue(runtime, args.phase, validate_gpu_pairs(args.gpu_pairs)))
    elif args.action == "report":
        print(json.dumps(report(runtime, args.phase), indent=2))
    else:
        monitor(runtime)


if __name__ == "__main__":
    main()
