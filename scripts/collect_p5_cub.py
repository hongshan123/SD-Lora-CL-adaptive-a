#!/usr/bin/env python3
"""Collect P5 CUB-200 results (frozen protocol) into a compact table.

Reads ``p5_cub_{full|exp009|sdlora}_seed{1,2,3}_nccl.log`` and prints
per-seed and mean Final / AAA / Forgetting.

Usage:
  python scripts/collect_p5_cub.py
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
    for path in sorted(glob.glob(os.path.join(ROOT, "p5_cub_*_seed*_nccl.log"))):
        name = os.path.basename(path).replace("_nccl.log", "")
        match = re.match(r"p5_cub_(\w+)_seed(\d+)", name)
        if match is None:
            continue
        method, seed = match.groups()
        info = parse(path)
        curves = [c for c in info["curves"] if len(c) == 10]
        if not curves:
            continue
        final = curves[-1][-1]
        avg = info["avg"][-1] if info["avg"] else float("nan")
        forgetting = info["forgetting"][-1] if info["forgetting"] else float("nan")
        rows.append((method, int(seed), final, avg, forgetting, path))

    if not rows:
        print("no p5_cub logs found")
        return 1

    print("{:<8} {:>4} {:>8} {:>8} {:>8}  {}".format(
        "method", "seed", "Final", "AAA", "Forget", "log"
    ))
    for method, seed, final, avg, forgetting, path in rows:
        print("{:<8} {:>4} {:>8.2f} {:>8.2f} {:>8.2f}  {}".format(
            method, seed, final, avg, forgetting, os.path.basename(path)
        ))
    print("")
    methods = sorted({r[0] for r in rows})
    for method in methods:
        vals = [r for r in rows if r[0] == method]
        if len(vals) < 2:
            continue
        finals = [r[2] for r in vals]
        avgs = [r[3] for r in vals]
        forgets = [r[4] for r in vals]
        print("{:<8} mean Final {:.2f}±{:.2f}  AAA {:.2f}±{:.2f}  "
              "Forgetting {:.2f}±{:.2f}  (n={})".format(
                  method,
                  sum(finals) / len(finals),
                  (sum((x - sum(finals) / len(finals)) ** 2 for x in finals)
                   / max(len(finals) - 1, 1)) ** 0.5,
                  sum(avgs) / len(avgs),
                  (sum((x - sum(avgs) / len(avgs)) ** 2 for x in avgs)
                   / max(len(avgs) - 1, 1)) ** 0.5,
                  sum(forgets) / len(forgets),
                  (sum((x - sum(forgets) / len(forgets)) ** 2 for x in forgets)
                   / max(len(forgets) - 1, 1)) ** 0.5,
                  len(vals),
              ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
