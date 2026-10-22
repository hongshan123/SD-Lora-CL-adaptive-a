#!/usr/bin/env python3
"""Collect local external-baseline results from their official log formats.

Handles:
  * InfLoRA:  `CNN top1 curve: [..]` (+ compute AAA; no forgetting print)
  * CL-LoRA:  `CNN top1 curve: [..]`, `Average Accuracy (CNN):`, `Forgetting (CNN):`
  * LoRA-DRS: `ACC top1 curve: [..]`, `Average Accuracy:`

Usage:
  python scripts/collect_external_baselines.py
"""

import glob
import os
import re


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_curve(text, patterns):
    for pattern in patterns:
        match = re.findall(pattern, text)
        if match:
            raw = match[-1]
            values = [float(x) for x in re.findall(r"np\.float64\(([\d.]+)\)", raw)]
            if not values:
                values = [float(x.strip()) for x in raw.split(",")]
            return values
    return []


def main():
    files = {
        "infolora": os.path.join(ROOT, "baseline_infolora_c100_seed1.log"),
        "cllora": os.path.join(ROOT, "baseline_cllora_c100_seed1.log"),
        "lora_drs": os.path.join(ROOT, "baseline_lora_drs_c100_seed1.log"),
    }
    print("{:<10} {:>8} {:>8} {:>8} {:>10}".format(
        "method", "Final", "AAA", "Forget", "tasks"
    ))
    for method, path in files.items():
        if not os.path.exists(path):
            print("{:<10} log missing".format(method))
            continue
        text = open(path, errors="ignore").read()
        if method == "infolora":
            curve = parse_curve(text, [r"CNN top1 curve: \[(.*?)\]"])
            avg = sum(curve) / len(curve) if curve else float("nan")
            forget = float("nan")
        elif method == "cllora":
            curve = parse_curve(text, [r"CNN top1 curve: \[(.*?)\]"])
            avgs = [float(x) for x in re.findall(
                r"Average Accuracy \(CNN\): ([\d.]+)", text
            )]
            forgets = [float(x) for x in re.findall(
                r"Forgetting \(CNN\): ([\d.]+)", text
            )]
            avg = avgs[-1] if avgs else float("nan")
            forget = forgets[-1] if forgets else float("nan")
        else:
            curve = parse_curve(text, [r"ACC top1 curve: \[(.*?)\]"])
            avgs = [float(x) for x in re.findall(
                r"Average Accuracy: ([\d.]+)", text
            )]
            avg = avgs[-1] if avgs else float("nan")
            forget = float("nan")
        final = curve[-1] if curve else float("nan")
        print("{:<10} {:>8.2f} {:>8.2f} {:>8.2f} {:>10}".format(
            method, final, avg, forget, len(curve)
        ))


if __name__ == "__main__":
    main()
