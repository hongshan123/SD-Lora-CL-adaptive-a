#!/usr/bin/env python3
"""Collect P5 task-length (T5/T20) results from the frozen-protocol queue.

Reads every ``p5_*_nccl.log`` and prints Final / AAA / Forgetting with the
run's dataset, method and task count.  Methods: full (Live-A Dual-B),
sdlora, exp009, rank1 (SD-LoRA rank-1 + Dual-B).

Usage:
  python scripts/collect_p5_tasklen.py
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.parse_log_metrics import parse


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "p5_*_t*_*_nccl.log"))):
        name = os.path.basename(path).replace("_nccl.log", "")
        match = re.match(r"p5_(\w+)_t(\d+)_(c100|inr)", name)
        if match is None:
            continue
        method, tasks, dataset = match.groups()
        info = parse(path)
        curves = [c for c in info["curves"] if len(c) == int(tasks)]
        final = curves[-1][-1] if curves else float("nan")
        avg = info["avg"][-1] if info["avg"] else float("nan")
        forgetting = info["forgetting"][-1] if info["forgetting"] else float("nan")
        rows.append((dataset, method, tasks, final, avg, forgetting, path))
    if not rows:
        print("no P5 logs found")
        return 1
    print("{:<6} {:<8} {:>3} {:>8} {:>8} {:>8}  {}".format(
        "dataset", "method", "T", "Final", "AAA", "Forget", "log"
    ))
    for dataset, method, tasks, final, avg, forgetting, path in rows:
        print("{:<6} {:<8} {:>3} {:>8.2f} {:>8.2f} {:>8.2f}  {}".format(
            dataset, method, tasks, final, avg, forgetting,
            os.path.basename(path),
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
