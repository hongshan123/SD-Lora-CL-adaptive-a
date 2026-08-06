#!/usr/bin/env python3
"""Report mean +/- std and paired tests for multi-seed log groups.

Example:
  python scripts/multiseed_stats.py \
    --group main_inr 'sa_cumulative_inr_seed*_gauge.log' 'sa_cumulative_inr_seed1995_gauge.log' \
    --group exp009_inr 'sa_sdlora_proto_inr_seed*.log' 'sa_sdlora_proto_inr_seed1995.log' \
    --metric final --paired main_inr exp009_inr
"""

import argparse
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.parse_log_metrics import parse


def collect(pattern, metric):
    values = []
    for path in sorted(glob.glob(pattern)):
        info = parse(path)
        if not info["curves"]:
            continue
        if metric == "final":
            value = info["curves"][-1][-1]
        elif metric == "avg":
            value = info["avg"][-1]
        elif metric == "forgetting":
            value = info["forgetting"][-1]
        else:
            raise ValueError(metric)
        values.append((path, float(value)))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", nargs=2, action="append", required=True,
                        metavar=("NAME", "LOG_GLOB"))
    parser.add_argument("--metric", default="final",
                        choices=["final", "avg", "forgetting"])
    parser.add_argument("--paired", nargs=2, metavar=("A", "B"),
                        help="paired test between two group names")
    args = parser.parse_args()

    groups = {}
    for name, pattern in args.group:
        groups[name] = collect(pattern, args.metric)
        values = [v for _, v in groups[name]]
        print(
            "{:<16} n={} mean={:8.2f} std={:8.2f}".format(
                name,
                len(values),
                statistics.mean(values) if values else float("nan"),
                statistics.stdev(values) if len(values) > 1 else float("nan"),
            )
        )
        for path, value in groups[name]:
            print("    {:<75} {:8.2f}".format(path, value))

    if args.paired:
        a_name, b_name = args.paired
        a = [v for _, v in groups[a_name]]
        b = [v for _, v in groups[b_name]]
        if len(a) != len(b):
            print("paired test skipped: unequal counts ({} vs {})".format(len(a), len(b)))
            return
        if len(a) < 2:
            print("paired test skipped: need >= 2 pairs")
            return
        try:
            from scipy.stats import ttest_rel

            stat, p = ttest_rel(a, b)
            diff = statistics.mean(x - y for x, y in zip(a, b))
            print(
                "paired {} - {}: mean_diff={:+.2f} t={:+.3f} p={:.4f} (n={})".format(
                    a_name, b_name, diff, stat, p, len(a)
                )
            )
        except Exception as exc:
            print("paired test failed: {}".format(exc))


if __name__ == "__main__":
    main()
