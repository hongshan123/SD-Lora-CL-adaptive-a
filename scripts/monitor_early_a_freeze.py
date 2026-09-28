"""Read-only 30-minute monitoring for the two early-freeze queues."""

import argparse
import datetime
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNITS = (
    "codex-early-a-cub-formal-20260928.service",
    "codex-early-a-inr-formal-20260928.service",
)


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr


def main(controller_unit=None):
    terminal = True
    print(datetime.datetime.now().astimezone().isoformat(), flush=True)
    for unit in ((controller_unit,) if controller_unit else UNITS):
        status = command([
            "systemctl", "--user", "show", unit, "--property=LoadState",
            "--property=ActiveState", "--property=SubState",
            "--property=MainPID", "--property=ExecMainStatus",
        ])
        print(unit, status, flush=True)
        if "ActiveState=active\n" in status or "ActiveState=activating\n" in status:
            terminal = False
    print(command([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
    ]), flush=True)
    needles = (
        "Task ", "CNN top1 curve", "Forgetting (CNN)", "SnapshotResume",
        "SharedAFreeze", "Traceback", "NCCL", "Error", "QUEUE_EXIT",
        "START ", "END ", "FORK_VERIFIED",
        "WAIT_FOR_GPUS", "DATASET_START", "DATASET_END", "GPU_QUERY_FAILED",
    )
    for path in sorted(ROOT.glob("early_a_*20260928*.log")):
        if "smoke" in path.name:
            continue
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 100000))
            tail = handle.read().decode(errors="replace").splitlines()
        print(path.name, flush=True)
        print("\n".join(line for line in tail if any(word in line for word in needles))[-6000:], flush=True)
    if terminal:
        print("All observed queues are terminal; stop only this monitor timer.", flush=True)
        print(command([
            "systemctl", "--user", "stop", "codex-early-a-monitor-20260928.timer",
        ]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-unit")
    main(parser.parse_args().controller_unit)
