#!/usr/bin/env python3
"""P1.1 Dual-B head decomposition from frozen P3 logs (no retraining).

For every confirmation seed (1-5) and both datasets this extracts, per task:
  * FC head top1/top5;
  * prototype head top1/top5;
  * fused head top1/top5;
and per-stage differences relative to SD-LoRA / EXP-009.

This quantifies the Dual-B contribution; it never selects a per-task best
head from test results (forbidden by the experiment guide).

Usage:
  python scripts/diagnose_dual_b_head.py
"""

import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.parse_log_metrics import parse


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [1, 2, 3, 4, 5]


def parse_dual_head(log_path):
    """Return {task: {mode: (top1, top5)}} for fc/proto/fused."""
    text = open(log_path, errors="ignore").read()
    out = {}
    pattern = re.compile(
        r"\[DualHead\] task (\d+) mode=(fc|proto|fused) "
        r"top1=([\d.]+) top5=([\d.]+)"
    )
    for match in pattern.finditer(text):
        task = int(match.group(1))
        mode = match.group(2)
        top1 = float(match.group(3))
        top5 = float(match.group(4))
        out.setdefault(task, {})[mode] = (top1, top5)
    return out


def final_curve(log_path):
    info = parse(log_path)
    curves = [c for c in info["curves"] if len(c) == 10]
    if not curves:
        raise ValueError("no full 10-task curve in {}".format(log_path))
    return curves[-1]


def avg(values):
    return statistics.mean(values)


def std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_group(dataset, method):
    """Return {seed: {task: value}} for per-task top1."""
    out = {}
    for seed in SEEDS:
        if method == "dual":
            path = os.path.join(
                ROOT, "p3_{}_{}_seed{}_nccl.log".format(
                    dataset, "livea_dual_b", seed
                )
            )
            heads = parse_dual_head(path)
            out[seed] = {
                task: heads[task] for task in sorted(heads)
            }
        else:
            path = os.path.join(
                ROOT, "p3_{}_{}_seed{}_nccl.log".format(
                    dataset, "sdlora" if method == "sdlora" else "exp009", seed
                )
            )
            out[seed] = {
                task: final_curve(path)[task]
                for task in range(10)
            }
    return out


def main():
    out_path = os.path.join(ROOT, "p1_dual_b_head_decomposition_output.txt")
    lines = []

    def emit(text=""):
        lines.append(text)

    def format_row(label, values):
        emit("  {:<14}".format(label) + " ".join(
            "{:7.2f}".format(v) for v in values
        ))

    emit("P1.1 Dual-B head decomposition (frozen P3, confirmation seeds 1-5)")
    emit("Per-task top1 (means over seeds) and AAA; top5 in parentheses where available")
    emit("")

    for dataset in ("inr", "c100"):
        dual = load_group(dataset, "dual")
        sdlora = load_group(dataset, "sdlora")
        exp009 = load_group(dataset, "exp009")

        modes = ("fc", "proto", "fused")
        per_task = {mode: {seed: [dual[seed][t][mode][0] for t in range(10)]
                           for seed in SEEDS} for mode in modes}
        per_task_top5 = {mode: {seed: [dual[seed][t][mode][1] for t in range(10)]
                                for seed in SEEDS} for mode in modes}
        sdlora_per_task = {seed: [sdlora[seed][t] for t in range(10)]
                           for seed in SEEDS}
        exp009_per_task = {seed: [exp009[seed][t] for t in range(10)]
                           for seed in SEEDS}

        emit("Dataset: {}".format("ImageNet-R" if dataset == "inr" else "CIFAR-100"))
        emit("")
        for mode in modes:
            means = [avg([per_task[mode][s][t] for s in SEEDS]) for t in range(10)]
            stds = [std([per_task[mode][s][t] for s in SEEDS]) for t in range(10)]
            means5 = [avg([per_task_top5[mode][s][t] for s in SEEDS]) for t in range(10)]
            emit("[{}] per-task top1 mean:".format(mode))
            format_row("task", range(10))
            format_row("top1", means)
            format_row("std", stds)
            format_row("top5", means5)
            aaa = [avg([per_task[mode][s][t] for s in SEEDS]) for t in range(10)]
            emit("  AAA (mean over tasks) = {:.3f}".format(avg(aaa)))
            emit("")

        emit("[fused - sdlora] per-stage top1 differences (mean over seeds):")
        diffs = [avg([per_task["fused"][s][t] - sdlora_per_task[s][t]
                      for s in SEEDS]) for t in range(10)]
        format_row("task", range(10))
        format_row("diff", diffs)
        emit("  AAA diff = {:+.3f}".format(avg(diffs)))
        emit("")
        emit("[fused - exp009] per-stage top1 differences (mean over seeds):")
        diffs = [avg([per_task["fused"][s][t] - exp009_per_task[s][t]
                      for s in SEEDS]) for t in range(10)]
        format_row("task", range(10))
        format_row("diff", diffs)
        emit("  AAA diff = {:+.3f}".format(avg(diffs)))
        emit("")
        emit("[proto - sdlora] per-stage top1 differences (mean over seeds):")
        diffs = [avg([per_task["proto"][s][t] - sdlora_per_task[s][t]
                      for s in SEEDS]) for t in range(10)]
        format_row("task", range(10))
        format_row("diff", diffs)
        emit("  AAA diff = {:+.3f}".format(avg(diffs)))
        emit("")
        emit("[fc - sdlora] per-stage top1 differences (mean over seeds):")
        diffs = [avg([per_task["fc"][s][t] - sdlora_per_task[s][t]
                      for s in SEEDS]) for t in range(10)]
        format_row("task", range(10))
        format_row("diff", diffs)
        emit("  AAA diff = {:+.3f}".format(avg(diffs)))
        emit("")
        emit("Per-seed AAA by head mode:")
        for seed in SEEDS:
            emit("  seed {}: fc={:.3f} proto={:.3f} fused={:.3f} "
                 "sdlora={:.3f} exp009={:.3f}".format(
                     seed,
                     avg([per_task["fc"][seed][t] for t in range(10)]),
                     avg([per_task["proto"][seed][t] for t in range(10)]),
                     avg([per_task["fused"][seed][t] for t in range(10)]),
                     avg(sdlora_per_task[seed]),
                     avg(exp009_per_task[seed]),
                 ))
        emit("")

    with open(out_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwritten to {}".format(out_path))


if __name__ == "__main__":
    main()
