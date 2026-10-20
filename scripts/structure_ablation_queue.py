"""CIFAR structural ablations, isolated attempts and GPU 0--3 leases."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, "/home/hongzhijun/hongshan/cuda9-baselines-multiseed-20261020")
from gpu_leases import GpuLease

MODES = ("independent_bank", "frozen_bank", "frozen_fold", "hoep_fold")
VARIANTS = [(m, True) for m in MODES] + [("hoep_fold", False)]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def read(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(pid=pid, start=int(fields[19]),
                    boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def make_config(root, mode, transport, tasks, smoke, micro):
    c = read(SOURCE/"experiments/hoep_t50_diagnosis_20261020/formal.json")
    name = f"{mode}_transport{int(transport)}_T{tasks}_seed1993"
    c.update(prefix=name, init_cls=100//tasks, increment=100//tasks,
             sa_structure_mode=mode, sa_adaptive_a_enabled=mode == "hoep_fold",
             sa_train_a_all_tasks=mode in ("hoep_fold", "independent_bank"),
             sa_live_a_coordinate_align=mode == "hoep_fold",
             sa_coordinate_stable_transport=transport, sa_train_microbatch=micro,
             sa_boundary_snapshots=False, sa_boundary_diagnose=False,
             sa_resume=False, max_tasks=2 if smoke else tasks,
             epochs=1 if smoke else 20, init_epoch=1 if smoke else 20)
    return c


def parse_curve(path, count):
    text = Path(path).read_text(errors="replace")
    rows = re.findall(r"CNN top1 curve:\s*(\[[^\n]+\])", text)
    if not rows or "Forgetting (CNN):" not in text:
        raise ValueError("Missing final evaluation/forgetting")
    encoded = re.sub(r"np\.float(?:32|64)\(([-+0-9.eE]+)\)", r"\1", rows[-1])
    curve = ast.literal_eval(encoded)
    if len(curve) != count:
        raise ValueError(f"Expected {count} tasks, got {len(curve)}")
    return dict(curve=curve, final_accuracy=curve[-1],
                average_incremental_accuracy=sum(curve)/len(curve))


def worker(jobfile):
    job = read(jobfile)
    attempt = Path(job["attempt_dir"])
    write(attempt/"started.json", dict(identity=identity(os.getpid()), config_sha256=sha(jobfile)))
    code = 1
    try:
        frozen = read(Path(job["runtime"])/"source_hashes.json")
        for path, digest in frozen.items():
            if sha(SOURCE/path) != digest:
                raise RuntimeError("Source changed: "+path)
        if job["kind"] == "resource":
            command = [sys.executable, str(SOURCE/"scripts/structure_resource_gate.py"),
                       "--mode", job["mode"], "--output", str(attempt/"output")]
        else:
            cfg = job["config"]
            cfg.update(filepath=str(attempt/"output")+"/", sa_p0_run_id=job["name"]+f"_attempt{job['attempt']}")
            write(attempt/"config.json", cfg)
            command = [sys.executable, str(SOURCE/"main.py"), "--config", str(attempt/"config.json")]
        with (attempt/"train.log").open("w") as log:
            subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)
        if job["kind"] == "resource":
            metrics = read(attempt/"output/gate.json")
            if not metrics or not metrics.get("passed"):
                raise RuntimeError("Resource gate failed")
        else:
            cfg = read(attempt/"config.json")
            metrics = parse_curve(attempt/"train.log", cfg["max_tasks"])
            for file in ("sa_state.pt", "sa_merged_lora.pt", "sa_prototypes.pt"):
                if not (attempt/"output"/file).is_file():
                    raise RuntimeError("Missing artifact: "+file)
            metrics.update(tasks=cfg["max_tasks"], smoke=job["kind"] == "smoke",
                           mode=job["mode"], transport=cfg["sa_coordinate_stable_transport"],
                           seed=1993, microbatch=cfg["sa_train_microbatch"])
        write(attempt/"metrics.json", metrics)
        code = 0
    except BaseException:
        traceback.print_exc()
    finally:
        write(attempt/"exit.json", dict(exit_code=code, finished=time.time(),
                                       config_sha256=sha(jobfile)))
    return code


def idle():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                         "--format=csv,noheader"], text=True).splitlines()
    available = []
    for line in out.splitlines():
        index, uuid, memory, util = [x.strip() for x in line.split(",")]
        if int(index) in (0, 1, 2, 3) and int(memory) < 300 and int(util) < 5 and uuid not in processes:
            available.append(int(index))
    return available


def run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/"queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read(root/"manifest.json")
    if manifest is None:
        jobs = [dict(name="resource_"+m, mode=m, kind="resource") for m in MODES]
        for m, transport in VARIANTS:
            jobs.append(dict(name=f"smoke_{m}_transport{int(transport)}", mode=m,
                             transport=transport, tasks=50, kind="smoke"))
        for tasks in (50, 10):
            for m, transport in VARIANTS:
                jobs.append(dict(name=f"{m}_transport{int(transport)}_T{tasks}_seed1993",
                                 mode=m, transport=transport, tasks=tasks, kind="formal"))
        manifest = dict(host="cuda8", gpus=[0, 1, 2, 3], seed=1993, formal_count=10, jobs=jobs)
        write(root/"manifest.json", manifest)
        paths = [SOURCE/"main.py", SOURCE/"trainer.py",
                 SOURCE/"experiments/hoep_t50_diagnosis_20261020/formal.json"]
        for folder in ("backbone", "models", "utils", "scripts"):
            paths.extend((SOURCE/folder).glob("*.py"))
        write(root/"source_hashes.json", {str(p.relative_to(SOURCE)): sha(p) for p in paths})
    status = read(root/"status.json") or {"jobs": {j["name"]: dict(state="pending", attempt=0) for j in manifest["jobs"]}}
    children, leases = {}, {}
    while True:
        for name, job in status["jobs"].items():
            if job["state"] not in ("running", "launching"):
                continue
            attempt = Path(job["attempt_dir"])
            proof, end = read(attempt/"started.json"), read(attempt/"exit.json")
            ident = (proof or {}).get("identity", job.get("identity"))
            if ident and identity(ident["pid"]) == ident:
                continue
            if end:
                valid = end.get("config_sha256") == sha(attempt/"job.json")
                job.update(state="complete" if end["exit_code"] == 0 and valid else "failed",
                           ended=end["finished"], metrics=read(attempt/"metrics.json"))
            elif job["state"] == "launching" and time.time()-job["launched"] < 30:
                continue
            else:
                job.update(state="pending", interrupted=time.time())
            if name in children:
                children.pop(name).wait()
            if name in leases:
                leases.pop(name).close()
        status["updated"] = time.time()
        write(root/"status.json", status)
        formal = [dict(j, **status["jobs"][j["name"]]) for j in manifest["jobs"] if j["kind"] == "formal"]
        write(root/"results_summary.json", dict(updated=time.time(),records=formal))
        pre = [j for j in manifest["jobs"] if j["kind"] != "formal"]
        if any(status["jobs"][j["name"]]["state"] == "failed" for j in pre):
            raise RuntimeError("Preflight failed; inspect preserved attempt logs")
        gates_done = all(status["jobs"]["resource_"+m]["state"] == "complete" for m in MODES)
        smokes_done = all(status["jobs"][j["name"]]["state"] == "complete" for j in pre)
        micro = min(status["jobs"]["resource_"+m]["metrics"]["microbatch"] for m in MODES) if gates_done else None
        if gates_done:
            write(root/"memory_plan.json", dict(effective_batch=128, microbatch=micro,
                                               accumulation_steps=128//micro, shared_by_all_cells=True))
        if all(j["state"] in ("complete", "failed") for j in status["jobs"].values()):
            return
        for gpu in idle():
            cfg = next((j for j in manifest["jobs"] if status["jobs"][j["name"]]["state"] == "pending"
                        and (j["kind"] == "resource" or gates_done and j["kind"] == "smoke"
                             or smokes_done and j["kind"] == "formal")), None)
            if cfg is None:
                break
            lease = GpuLease.acquire([gpu])
            if lease is None:
                continue
            if gpu not in idle():
                lease.close()
                continue
            job = status["jobs"][cfg["name"]]
            existing = [int(p.name.split("-")[-1]) for p in (root/"results"/cfg["name"]).glob("attempt-*")]
            number = max([job["attempt"]]+existing)+1
            attempt = root/"results"/cfg["name"]/f"attempt-{number:03d}"
            attempt.mkdir(parents=True, exist_ok=False)
            job.update(state="launching", attempt=number, gpu=gpu, attempt_dir=str(attempt), launched=time.time())
            spec = dict(cfg, attempt=number, attempt_dir=str(attempt), runtime=str(root))
            if cfg["kind"] != "resource":
                spec["config"] = make_config(root, cfg["mode"], cfg["transport"], cfg["tasks"],
                                            cfg["kind"] == "smoke", micro)
            write(attempt/"job.json", spec)
            write(root/"status.json", status)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
            with (attempt/"worker.log").open("w") as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(attempt/"job.json")],
                                         cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=lease.fds)
            children[cfg["name"]] = child
            leases[cfg["name"]] = lease
            job.update(state="running", identity=identity(child.pid))
            write(root/"status.json", status)
            print("Launched", cfg["name"], "GPU", gpu, "PID", child.pid, flush=True)
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(SOURCE/".runtime_structure_ablation_cifar_20261020"))
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        sys.exit(worker(args.worker))
    run(args.root)
