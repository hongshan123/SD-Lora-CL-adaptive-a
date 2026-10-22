#!/usr/bin/env python3
"""Parse trainer/proto-router logs into compact metric summaries."""

import argparse
import glob
import re
import sys


def parse(path):
    text = open(path, errors="ignore").read()
    curves = re.findall(r"CNN top1 curve: \[(.*?)\]", text)
    avgs = re.findall(r"Average Accuracy \(CNN\): ([\d.]+)", text)
    forgets = re.findall(r"Forgetting \(CNN\): ([\d.]+)", text)
    routing = re.findall(
        r"\[ProtoRouter\]\[(block4|block6|full|oracle)\] RoutingAcc=([\d.]+)",
        text,
    )
    rows = []
    for curve in curves:
        try:
            values = [float(x) for x in re.findall(r"np\.float64\(([\d.]+)\)", curve)]
            if not values:
                values = [float(x.strip()) for x in curve.split(",")]
            rows.append(values)
        except ValueError:
            continue
    return {
        "curves": rows,
        "avg": [float(x) for x in avgs],
        "forgetting": [float(x) for x in forgets],
        "routing": routing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()
    for pattern in args.logs:
        for path in sorted(glob.glob(pattern)):
            info = parse(path)
            final = info["curves"][-1][-1] if info["curves"] else float("nan")
            avg = info["avg"][-1] if info["avg"] else float("nan")
            fgt = info["forgetting"][-1] if info["forgetting"] else float("nan")
            print(
                "{:<70} final_top1={:8.2f} avgacc={:8.2f} forgetting={:6.2f} tasks={}".format(
                    path, final, avg, fgt, len(info["curves"])
                )
            )
            if info["routing"]:
                last = [r for r in info["routing"] if r[0] != "oracle"]
                if last:
                    print(
                        "    final routing: "
                        + "  ".join(
                            "{}={}".format(k, v) for k, v in last[-4:]
                        )
                    )


if __name__ == "__main__":
    main()
