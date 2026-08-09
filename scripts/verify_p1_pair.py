"""Verify one P1 paired run (control vs Dual-B vs SD-LoRA vs EXP-009).

Checks trajectory equality, eval RNG invariance, four-rank dual sync, final
fused/prototype identity, and the P1 acceptance gates.  All numbers are parsed
from logs; no manual task/seed selection.
"""

import argparse
import json
import re
from pathlib import Path


POST_TRAIN_RE = re.compile(
    r"\[PostTrainHash\] task (\d+) hash=([0-9a-f]+) rng=([0-9a-f]+)"
)
EVAL_RNG_RE = re.compile(
    r"\[RNGHash\] task (\d+) eval before=([0-9a-f]+) after=([0-9a-f]+) PASS"
)
DUAL_MODE_RE = re.compile(
    r"\[DualHead\] task (\d+) mode=(\w+) top1=([0-9.]+) top5=([0-9.]+)"
)
FUSED_DIFF_RE = re.compile(
    r"\[DualHead\] task (\d+) fused_proto_max_diff=([0-9.eE+-]+) PASS"
)
RANK_RE = re.compile(
    r"\[DualHead\] rank (\d+) task (\d+) "
    r"lambda=([0-9]+\.[0-9]{6}) tau_fc=([0-9]+\.[0-9]{6}) "
    r"tau_proto=([0-9]+\.[0-9]{6})"
)
CNN_TOTAL_RE = re.compile(r"CNN: \{'total': np\.float64\(([0-9.]+)\)")
FORGETTING_RE = re.compile(r"Forgetting \(CNN\): ([0-9.]+)")
AVG_RE = re.compile(r"Average Accuracy \(CNN\): ([0-9.]+)")


def parse_post_train(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        match = POST_TRAIN_RE.search(line)
        if match:
            result[int(match.group(1))] = (
                match.group(2),
                match.group(3),
            )
    return result


def parse_eval_rng(path):
    return [
        (int(match.group(1)), match.group(2), match.group(3))
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        for match in [EVAL_RNG_RE.search(line)]
        if match
    ]


def parse_cnn_totals(path):
    return [
        float(match.group(1))
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        for match in [CNN_TOTAL_RE.search(line)]
        if match
    ]


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
    ranks = set()
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        match = RANK_RE.search(line)
        if match:
            ranks.add(int(match.group(1)))
            task = int(match.group(2))
            values.setdefault(task, set()).add(
                (match.group(3), match.group(4), match.group(5))
            )
    return values, ranks


def last_metric(path, regex):
    values = [
        float(match.group(1))
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        for match in [regex.search(line)]
        if match
    ]
    return values[-1] if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", required=True)
    parser.add_argument("--dual-log", required=True)
    parser.add_argument("--sdlora-log", required=True)
    parser.add_argument("--exp009-log", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    control_post = parse_post_train(args.control_log)
    dual_post = parse_post_train(args.dual_log)
    control_rng = parse_eval_rng(args.control_log)
    dual_rng = parse_eval_rng(args.dual_log)
    control_totals = parse_cnn_totals(args.control_log)
    dual_modes = parse_dual_modes(args.dual_log)
    rank_values, ranks_seen = parse_rank_values(args.dual_log)
    fused_diffs = [
        (int(match.group(1)), float(match.group(2)))
        for line in Path(args.dual_log).read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
        for match in [FUSED_DIFF_RE.search(line)]
        if match
    ]

    checks = {}
    checks["post_train_equal"] = control_post == dual_post
    checks["eval_rng_equal"] = control_rng == dual_rng
    checks["prototype_curve_equal"] = (
        len(control_totals) == len(dual_modes)
        and all(
            abs(control_totals[i] - dual_modes[i]["proto"]) < 1e-9
            for i in range(len(control_totals))
        )
    )
    checks["rank_sync"] = (
        ranks_seen == {0, 1, 2, 3}
        and all(len(values) == 1 for values in rank_values.values())
    )
    final_fused_diff = dict(fused_diffs)
    checks["fused_proto_final_identity"] = bool(
        final_fused_diff and 9 in final_fused_diff
    ) and final_fused_diff.get(9, 1.0) <= 1e-6

    control_final = control_totals[-1] if control_totals else None
    control_aaa = last_metric(args.control_log, AVG_RE)
    control_forgetting = last_metric(args.control_log, FORGETTING_RE)
    dual_final = dual_modes.get(9, {}).get("fused")
    dual_aaa = last_metric(args.dual_log, AVG_RE)
    dual_forgetting = last_metric(args.dual_log, FORGETTING_RE)
    sdlora_final = last_metric(args.sdlora_log, CNN_TOTAL_RE)
    sdlora_aaa = last_metric(args.sdlora_log, AVG_RE)
    exp009_final = last_metric(args.exp009_log, CNN_TOTAL_RE)
    exp009_aaa = last_metric(args.exp009_log, AVG_RE)

    aaa_gain = (
        None
        if dual_aaa is None or control_aaa is None
        else dual_aaa - control_aaa
    )
    final_vs_sdlora = (
        None if dual_final is None or sdlora_final is None else dual_final - sdlora_final
    )
    aaa_vs_sdlora = (
        None if dual_aaa is None or sdlora_aaa is None else dual_aaa - sdlora_aaa
    )
    checks["aaa_gain_gate"] = aaa_gain is not None and aaa_gain >= 0.7
    checks["final_vs_sdlora_gate"] = (
        final_vs_sdlora is not None and final_vs_sdlora >= 0.0
    )
    checks["aaa_vs_sdlora_gate"] = (
        aaa_vs_sdlora is not None and aaa_vs_sdlora >= -0.3
    )

    summary = {
        "checks": checks,
        "control": {
            "final": control_final,
            "aaa": control_aaa,
            "forgetting": control_forgetting,
        },
        "dual_b": {
            "final_fused": dual_final,
            "aaa_fused": dual_aaa,
            "forgetting": dual_forgetting,
            "aaa_gain_vs_control": aaa_gain,
        },
        "sdlora": {"final": sdlora_final, "aaa": sdlora_aaa},
        "exp009": {"final": exp009_final, "aaa": exp009_aaa},
        "gates": {
            "dual_final_minus_sdlora": final_vs_sdlora,
            "dual_aaa_minus_sdlora": aaa_vs_sdlora,
        },
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        print("P1 VERIFY FAIL: {}".format(failed))
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)
    print("P1 VERIFY PASS")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
