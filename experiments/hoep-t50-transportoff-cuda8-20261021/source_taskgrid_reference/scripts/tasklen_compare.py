#!/usr/bin/env python3
"""Print the fair task-length comparison table (main vs EXP-009 vs SD-LoRA).

All entries use seed1995 (ImageNet-R) / seed1993 (CIFAR-100), same class
order, 20 epochs/task, batch 32.  Pending runs are printed as 'n/a'.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.parse_log_metrics import parse


ROOT = Path(__file__).resolve().parents[1]


def metric(path):
    path = ROOT / path
    if not path.exists():
        return None
    info = parse(str(path))
    if not info["curves"] or not info["avg"] or not info["forgetting"]:
        return None
    return {
        "final": float(info["curves"][-1][-1]),
        "avg": float(info["avg"][-1]),
        "forgetting": float(info["forgetting"][-1]),
    }


ENTRIES = [
    # (dataset, T, method, log)
    ("ImageNet-R", 5, "cumulative+gauge", "sa_cumulative_inr_seed1995_gauge_t5.log"),
    ("ImageNet-R", 10, "cumulative+gauge", "sa_cumulative_inr_seed1995_gauge.log"),
    ("ImageNet-R", 20, "cumulative+gauge", "sa_cumulative_inr_seed1995_gauge_t20.log"),
    ("ImageNet-R", 40, "cumulative+gauge", "sa_cumulative_inr_seed1995_gauge_t40.log"),
    ("ImageNet-R", 5, "EXP-009", "sa_sdlora_proto_inr_seed1995_t5.log"),
    ("ImageNet-R", 10, "EXP-009", "sa_sdlora_proto_inr_seed1995.log"),
    ("ImageNet-R", 20, "EXP-009", "sa_sdlora_proto_inr_seed1995_t20.log"),
    ("ImageNet-R", 40, "EXP-009", "sa_sdlora_proto_inr_seed1995_t40.log"),
    ("ImageNet-R", 10, "SD-LoRA", "sdlora_inr_seed1995_proto_baseline.log"),
    ("ImageNet-R", 20, "SD-LoRA", "sdlora_inr_seed1995_proto_baseline_t20.log"),
    ("ImageNet-R", 40, "SD-LoRA", "sdlora_inr_seed1995_proto_baseline_t40.log"),
    ("CIFAR-100", 5, "cumulative+gauge", "sa_cumulative_c100_seed1993_gauge_t5.log"),
    ("CIFAR-100", 10, "cumulative+gauge", "sa_cumulative_c100_seed1993_gauge.log"),
    ("CIFAR-100", 20, "cumulative+gauge", "sa_cumulative_c100_seed1993_gauge_t20.log"),
    ("CIFAR-100", 5, "EXP-009", "sa_sdlora_proto_c100_seed1993_t5.log"),
    ("CIFAR-100", 10, "EXP-009", "sa_sdlora_proto_c100_seed1993.log"),
    ("CIFAR-100", 20, "EXP-009", "sa_sdlora_proto_c100_seed1993_t20.log"),
]


def main():
    print(
        "{:<12} {:>4} {:<18} {:>8} {:>8} {:>8}".format(
            "dataset", "T", "method", "final", "avgacc", "forget"
        )
    )
    for dataset, t, method, log in ENTRIES:
        values = metric(log)
        if values is None:
            print(
                "{:<12} {:>4} {:<18} {:>8} {:>8} {:>8}".format(
                    dataset, t, method, "n/a", "n/a", "n/a"
                )
            )
            continue
        print(
            "{:<12} {:>4} {:<18} {:>8.2f} {:>8.2f} {:>8.2f}".format(
                dataset,
                t,
                method,
                values["final"],
                values["avg"],
                values["forgetting"],
            )
        )


if __name__ == "__main__":
    main()
