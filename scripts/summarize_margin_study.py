#!/usr/bin/env python3
"""Apply preregistered gates to read-only margin and G-proxy diagnostics."""

import argparse
import json
import math
from pathlib import Path


def summarize_calibration(record, tolerance_pp=0.30):
    final_delta = record["bias_final"] - record["baseline_final"]
    aaa_delta = record["bias_aaa"] - record["baseline_aaa"]
    return {
        "dataset": record["dataset"],
        "final_delta_pp": final_delta,
        "aaa_delta_pp": aaa_delta,
        "bias_only_prescreen": (
            "pass" if min(final_delta, aaa_delta) >= -tolerance_pp else "fail"
        ),
        "note": "Bias-only is not the joint G-plus-head acceptance test.",
    }


def _midranks(values):
    ranks = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for position in order[start:end]:
            ranks[position] = (start + end - 1) / 2
        start = end
    return ranks


def _rank_correlation(left, right):
    x, y = _midranks(left), _midranks(right)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    covariance = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = sum((a - x_mean) ** 2 for a in x)
    y_norm = sum((b - y_mean) ** 2 for b in y)
    if x_norm == 0 or y_norm == 0:
        return None
    return covariance / math.sqrt(x_norm * y_norm)


def summarize_proxy(records, min_old_change_pp=0.20, min_informative_tasks=2,
                    required_agreement=0.65):
    if not records:
        return {
            "status": "insufficient_evidence", "informative_tasks": 0,
            "pair_count": 0, "agreeing_pairs": 0, "agreement": None,
            "mean_inverse_kl_spearman": None,
        }
    dataset = records[0]["dataset"]
    task_ids = set()
    informative_tasks = set()
    correlations = []
    agreeing = 0
    pairs = 0
    for record in records:
        if record["dataset"] != dataset or record["task"] in task_ids:
            raise ValueError("proxy records mix datasets or duplicate a task")
        task_ids.add(record["task"])
        rows = record["candidate_rows"]
        if len(rows) < 2:
            raise ValueError("each task requires at least two G candidates")
        informative = False
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                accuracy_difference = (
                    left["diagnostic_old_top1"] - right["diagnostic_old_top1"]
                )
                if abs(accuracy_difference) < min_old_change_pp:
                    continue
                proxy_difference = (
                    left["proxy_current_old_kl"] - right["proxy_current_old_kl"]
                )
                informative = True
                pairs += 1
                if proxy_difference * accuracy_difference < 0:
                    agreeing += 1
        if informative:
            informative_tasks.add(record["task"])
            correlation = _rank_correlation(
                [-row["proxy_current_old_kl"] for row in rows],
                [row["diagnostic_old_top1"] for row in rows],
            )
            if correlation is not None:
                correlations.append(correlation)
    fraction = agreeing / pairs if pairs else None
    mean_correlation = sum(correlations) / len(correlations) if correlations else None
    enough = len(informative_tasks) >= min_informative_tasks
    return {
        "status": (
            "pass" if enough and fraction >= required_agreement
            and mean_correlation is not None and mean_correlation > 0 else
            "fail" if enough else "insufficient_evidence"
        ),
        "informative_tasks": len(informative_tasks),
        "pair_count": pairs,
        "agreeing_pairs": agreeing,
        "agreement": fraction,
        "mean_inverse_kl_spearman": mean_correlation,
        "criterion": (
            f"at least {min_informative_tasks} informative tasks and "
            f"{required_agreement:.0%} inverse KL/old-Top1 pair agreement "
            "and positive mean inverse-KL Spearman"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, action="append", default=[])
    parser.add_argument("--proxy", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibrations = [json.loads(path.read_text()) for path in args.calibration]
    proxies = [json.loads(path.read_text()) for path in args.proxy]
    datasets = sorted({item["dataset"] for item in calibrations + proxies})
    report = {
        "version": 1,
        "use_of_test_labels": "diagnostic gate only; never used to fit bias or select a G candidate",
        "datasets": {},
    }
    for dataset in datasets:
        matched_calibration = [item for item in calibrations if item["dataset"] == dataset]
        if len(matched_calibration) > 1:
            raise ValueError(f"duplicate calibration for {dataset}")
        matched_proxy = [item for item in proxies if item["dataset"] == dataset]
        report["datasets"][dataset] = {
            "calibration": (
                summarize_calibration(matched_calibration[0])
                if matched_calibration else None
            ),
            "g_proxy": summarize_proxy(matched_proxy),
        }
    passed = sum(
        item["g_proxy"]["status"] == "pass"
        for item in report["datasets"].values()
    )
    report["g_proxy_two_dataset_gate"] = (
        "pass" if passed >= 2 else "not_met"
    )
    report["deployment_decision"] = "shadow_only"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
