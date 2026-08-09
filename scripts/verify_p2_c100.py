"""Verify one P2 CIFAR-100 paired run (Dual-B vs SD-LoRA vs EXP-009).

P1 proved Dual-B and Live-A control share the training trajectory, so the
Dual-B prototype curve is the same-trajectory control for CIFAR-100.
"""

import argparse
import json
import re
from pathlib import Path


DUAL_MODE_RE = re.compile(
    r"\[DualHead\] task (\d+) mode=(\w+) top1=([0-9.]+) top5=([0-9.]+)"
)
FUSED_DIFF_RE = re.compile(
    r"\[DualHead\] task (\d+) fused_proto_max_diff=([0-9.eE+-]+) PASS"
)
RANK_RE = re.compile(
    r"\[DualHead\] rank (\d+) task (\d+) "
    r"lambda=([0-9.]+) tau_fc=([0-9.]+) tau_proto=([0-9.]+)"
)
CNN_TOTAL_RE = re.compile(r"CNN: \{'total': np\.float64\(([0-9.]+)\)")
FORGETTING_RE = re.compile(r"Forgetting \(CNN\): ([0-9.]+)")
AVG_RE = re.compile(r"Average Accuracy \(CNN\): ([0-9.]+)")


def last_metric(path, regex):
    values = [
        float(match.group(1))
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        for match in [regex.search(line)]
        if match
    ]
    return values[-1] if values else None


def parse_dual_modes(path):
    modes = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        match = DUAL_MODE_RE.search(line)
        if match:
            task = int(match.group(1))
            modes.setdefault(task, {})[match.group(2)] = float(match.group(3))
    return modes


def parse_rank_values(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        match = RANK_RE.search(line)
        if match:
            task = int(match.group(2))
            values.setdefault(task, set()).add(
                (match.group(3), match.group(4), match.group(5))
            )
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-log", required=True)
    parser.add_argument("--sdlora-log", required=True)
    parser.add_argument("--exp009-log", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    dual_modes = parse_dual_modes(args.dual_log)
    rank_values = parse_rank_values(args.dual_log)
    fused_diffs = [
        (int(match.group(1)), float(match.group(2)))
        for line in Path(args.dual_log).read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
        for match in [FUSED_DIFF_RE.search(line)]
        if match
    ]
    dual_final = dual_modes.get(9, {}).get("fused")
    proto_final = dual_modes.get(9, {}).get("proto")
    dual_aaa = last_metric(args.dual_log, AVG_RE)
    dual_forgetting = last_metric(args.dual_log, FORGETTING_RE)
    sdlora_final = last_metric(args.sdlora_log, CNN_TOTAL_RE)
    sdlora_aaa = last_metric(args.sdlora_log, AVG_RE)
    exp009_final = last_metric(args.exp009_log, CNN_TOTAL_RE)
    exp009_aaa = last_metric(args.exp009_log, AVG_RE)

    final_diff = (
        None if dual_final is None or sdlora_final is None else dual_final - sdlora_final
    )
    aaa_diff = (
        None if dual_aaa is None or sdlora_aaa is None else dual_aaa - sdlora_aaa
    )
    max_fused_diff = dict(fused_diffs).get(9)

    checks = {
        "rank_sync": all(len(values) == 1 for values in rank_values.values()),
        "fused_final_equals_proto_final": (
            dual_final is not None
            and proto_final is not None
            and abs(dual_final - proto_final) < 1e-9
        ),
        "fused_proto_logit_identity": (
            max_fused_diff is not None and max_fused_diff <= 1e-6
        ),
        "final_vs_sdlora_gate": final_diff is not None and final_diff >= 0.0,
        "aaa_vs_sdlora_gate": aaa_diff is not None and aaa_diff >= -0.3,
    }

    summary = {
        "checks": checks,
        "dual_b": {
            "final_fused": dual_final,
            "final_proto": proto_final,
            "aaa_fused": dual_aaa,
            "forgetting": dual_forgetting,
        },
        "sdlora": {"final": sdlora_final, "aaa": sdlora_aaa},
        "exp009": {"final": exp009_final, "aaa": exp009_aaa},
        "gates": {
            "dual_final_minus_sdlora": final_diff,
            "dual_aaa_minus_sdlora": aaa_diff,
        },
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        print("P2 VERIFY FAIL: {}".format(failed))
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)
    print("P2 VERIFY PASS")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
