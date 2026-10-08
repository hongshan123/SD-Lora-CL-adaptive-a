#!/usr/bin/env python3
"""Compare snapshot-off/on ImageNet-R trajectories before formal runs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def parse_curve(log_text):
    matches = re.findall(r"CNN top1 curve:\s*(\[[^\n]*\])", log_text)
    if not matches:
        raise ValueError("completed CNN top1 curve is absent")
    literal = matches[-1].replace("np.float64(", "(")
    curve = ast.literal_eval(literal)
    if not isinstance(curve, list):
        raise ValueError("CNN top1 curve must be a list")
    return [float(value) for value in curve]


def audit_pair(off_dir, on_dir, off_log, on_log):
    off = json.loads((Path(off_dir) / "p0_hashes.json").read_text())
    on = json.loads((Path(on_dir) / "p0_hashes.json").read_text())
    first = None
    if set(off) != set(map(str, range(10))) or set(on) != set(map(str, range(10))):
        first = "incomplete task hashes"
    else:
        for task in range(10):
            for key in ("post_train_model_hash", "post_train_rng_hash",
                        "post_eval_model_hash", "post_eval_rng_hash"):
                if (not isinstance(off[str(task)].get(key), str)
                        or not off[str(task)][key]
                        or not isinstance(on[str(task)].get(key), str)
                        or not on[str(task)][key]):
                    first = f"task {task} {key} missing"
                    break
                if off[str(task)][key] != on[str(task)][key]:
                    first = f"task {task} {key}"
                    break
            if first is not None:
                break
    off_curve = parse_curve(off_log)
    on_curve = parse_curve(on_log)
    if len(off_curve) != 10 or len(on_curve) != 10:
        first = first or "incomplete accuracy curve"
    elif any(abs(a - b) > 1e-4 for a, b in zip(off_curve, on_curve)):
        first = first or "per-task accuracy curve"
    return {"passed": first is None, "first_mismatch": first,
            "off_curve": off_curve, "on_curve": on_curve,
            "task0_model_rng_equal": off.get("0") == on.get("0")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir
    results = {}
    for policy in ("live", "frozen"):
        prefix = f"lfdiag_inr_{policy}_seed1995"
        off = root / "results" / (prefix + "_snapshot_off")
        on = root / "results" / (prefix + "_snapshot_on")
        off_log = (root / "logs" / (prefix + "_snapshot_off.log")).read_text()
        on_log = (root / "logs" / (prefix + "_snapshot_on.log")).read_text()
        results[policy] = audit_pair(off, on, off_log, on_log)
    path = root / "smoke_audit.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(path)
    if not all(result["passed"] for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
