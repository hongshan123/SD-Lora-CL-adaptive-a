#!/usr/bin/env python3
"""Summarize cumulative Shared-A runs from logs and artifacts.

For each log matching the patterns, print final Top1 / AvgAcc / Forgetting,
task count, gauge diagnostics (mean projection residual / rotation /
preservation), and artifact LoRA parameters.  Requires the corresponding
artifact directory (config's filepath) to exist for parameter reporting.
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.measure_sa_artifact import _count_tensors


def _parse_log(path):
    text = open(path, errors="ignore").read()
    curve = re.findall(r"CNN top1 curve: \[(.*?)\]", text)
    if not curve:
        return None
    values = [float(x) for x in re.findall(r"np\.float64\(([\d.]+)\)", curve[-1])]
    if not values:
        values = [float(x.strip()) for x in curve[-1].split(",")]
    avgs = [float(x) for x in re.findall(r"Average Accuracy \(CNN\): ([\d.]+)", text)]
    forgets = [float(x) for x in re.findall(r"Forgetting \(CNN\): ([\d.]+)", text)]
    tasks = len(re.findall(r"\[sdlora\.py\] => Learning on", text))
    residuals = [
        float(x)
        for x in re.findall(
            r"relative_projection_residual=([\d.eE+-]+)", text
        )
    ]
    rotations = [
        float(x)
        for x in re.findall(r"basis_rotation_fro=([\d.eE+-]+)", text)
    ]
    preservations = [
        float(x)
        for x in re.findall(r"operator_preservation=([\d.eE+-]+)", text)
    ]
    truncations = [
        float(x)
        for x in re.findall(
            r"max_relative_truncation_error=([\d.eE+-]+)", text
        )
    ]
    return {
        "final": values[-1] if values else float("nan"),
        "avg": avgs[-1] if avgs else float("nan"),
        "forgetting": forgets[-1] if forgets else float("nan"),
        "tasks": tasks,
        "residual": sum(residuals) / len(residuals) if residuals else float("nan"),
        "rotation": sum(rotations) / len(rotations) if rotations else float("nan"),
        "preservation": (
            sum(preservations) / len(preservations) if preservations else float("nan")
        ),
        "truncation": (
            sum(truncations) / len(truncations) if truncations else float("nan")
        ),
    }


def _artifact_params(config_path, artifact_dir):
    if not os.path.exists(config_path) or not os.path.isdir(artifact_dir):
        return None
    with open(config_path) as handle:
        config = json.load(handle)
    state_path = os.path.join(artifact_dir, "sa_state.pt")
    if not os.path.exists(state_path):
        return None
    state = __import__("torch").load(
        state_path, map_location="cpu", weights_only=True
    )
    if int(state.get("version", -1)) in (2, 3):
        total = _count_tensors(state.get("canonical_down", [])) + _count_tensors(
            state.get("cumulative_up", [])
        ) + _count_tensors(state.get("triangular_r", []))
    else:
        total = _count_tensors(state.get("shared_a", [])) + _count_tensors(
            list(state.get("scales", {}).values())
        )
        for task_id in range(len(state.get("scales", {}))):
            path = os.path.join(
                artifact_dir, "sa_lora_w_b_{}.pt".format(task_id)
            )
            if os.path.exists(path):
                total += _count_tensors(
                    __import__("torch").load(
                        path, map_location="cpu", weights_only=True
                    )
                )
    proto_path = os.path.join(artifact_dir, "sa_prototypes.pt")
    prototypes = 0
    if os.path.exists(proto_path):
        prototypes = _count_tensors(
            __import__("torch").load(
                proto_path, map_location="cpu", weights_only=True
            )
        )
    return total, prototypes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns",
        nargs="+",
        help="log glob patterns, e.g. 'sa_cumulative_*gauge.log'",
    )
    args = parser.parse_args()
    print(
        "{:<62} {:>8} {:>8} {:>8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "log",
            "final",
            "avgacc",
            "forget",
            "tasks",
            "residual",
            "rotation",
            "preserve",
            "trunc_err",
            "lora_params",
        )
    )
    for pattern in args.patterns:
        for path in sorted(glob.glob(pattern)):
            info = _parse_log(path)
            if info is None:
                print("{:<62} no-result".format(path))
                continue
            artifact_dir = None
            config_path = None
            stem = os.path.basename(path).replace(".log", "")
            config_path = "exps/{}.json".format(stem)
            if os.path.exists(config_path):
                with open(config_path) as handle:
                    artifact_dir = json.load(handle)["filepath"]
            params = None
            if artifact_dir:
                params = _artifact_params(config_path, artifact_dir)
            print(
                "{:<62} {:8.2f} {:8.2f} {:8.2f} {:6d} {:10.2e} {:10.2e} "
                "{:10.2e} {:10.2e} {:>10}".format(
                    path,
                    info["final"],
                    info["avg"],
                    info["forgetting"],
                    info["tasks"],
                    info["residual"],
                    info["rotation"],
                    info["preservation"],
                    info["truncation"],
                    (
                        "{:,}".format(params[0])
                        if params
                        else "n/a"
                    ),
                )
            )


if __name__ == "__main__":
    main()
