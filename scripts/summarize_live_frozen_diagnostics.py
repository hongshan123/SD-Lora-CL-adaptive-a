#!/usr/bin/env python3
"""Aggregate all preregistered Live-A/Frozen-A diagnostic boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_live_frozen_smoke import parse_curve


STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "H_LS",
          "formal_new_prototypes")


def task_row(result, policy):
    row = {"dataset": result["dataset"], "seed": result["seed"],
           "task": result["task"], "policy": policy,
           "r_rec": result["aggregate"]["r_rec"],
           "r_weighted": result["aggregate"]["r_weighted"]}
    for stage in STAGES:
        row[f"{stage}_old"] = result["metrics"][stage]["old"]["top1"]
        row[f"{stage}_global"] = result["metrics"][stage]["global_top1"]
    row["old_drop_hls"] = row["S0_old"] - row["H_LS_old"]
    row["old_drop_deploy"] = row["S0_old"] - row["formal_new_prototypes_old"]
    for before, after in zip(STAGES[:7], STAGES[1:7]):
        row[f"old_delta_{before}_{after}"] = row[f"{after}_old"] - row[f"{before}_old"]
    row["old_delta_S6_formal"] = row["formal_new_prototypes_old"] - row["S6_old"]
    return row


def paired_rows(rows):
    cells = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["seed"], row["task"])
        if row["policy"] in cells[key]:
            raise ValueError(f"duplicate policy for {key}")
        cells[key][row["policy"]] = row
    pairs = []
    for (dataset, seed, task), policies in sorted(cells.items()):
        if set(policies) != {"live", "frozen"}:
            raise ValueError(f"incomplete policy pair for {(dataset, seed, task)}")
        live, frozen = policies["live"], policies["frozen"]
        pair = {"dataset": dataset, "seed": seed, "task": task}
        for key in live:
            if key in ("dataset", "seed", "task", "policy"):
                continue
            pair[f"delta_{key}"] = live[key] - frozen[key]
        pairs.append(pair)
    return pairs


def _correlations(rows):
    output = []
    live = [row for row in rows if row["policy"] == "live"]
    for dataset in sorted({row["dataset"] for row in live}):
        subset = [row for row in live if row["dataset"] == dataset]
        seeds = sorted({row["seed"] for row in subset})
        for seed in (None, *seeds):
            current = subset if seed is None else [row for row in subset if row["seed"] == seed]
            for residual in ("r_rec", "r_weighted"):
                for outcome in ("old_drop_hls", "old_drop_deploy"):
                    x = [row[residual] for row in current]
                    y = [row[outcome] for row in current]
                    coefficient = float(spearmanr(x, y).statistic)
                    output.append({"dataset": dataset, "seed": seed,
                                   "n_boundaries": len(current),
                                   "residual": residual, "outcome": outcome,
                                   "spearman": coefficient})
    return output


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(experiment_dir):
    root = Path(experiment_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = [entry for entry in manifest["smoke"] + manifest["formal"]
               if entry["snapshots"]]
    if len(entries) != 18:
        raise ValueError("expected exactly 18 prespecified paired runs")
    rows, final = [], []
    for entry in entries:
        name = Path(entry["config"]).stem
        directory = root / "diagnostics" / name
        for task in range(1, 10):
            result = json.loads((directory / f"task_{task:03d}.json").read_text())
            if (result["seed"], result["task"]) != (entry["seed"], task):
                raise ValueError(f"diagnostic metadata mismatch: {name} task {task}")
            rows.append(task_row(result, entry["policy"]))
        log = (root / "logs" / f"{name}.log").read_text(errors="replace")
        curve = parse_curve(log)
        match = re.search(r"Forgetting \(CNN\):\s*([-+0-9.eE]+)", log)
        if len(curve) != 10 or match is None:
            raise ValueError(f"incomplete final metrics for {name}")
        final.append({"dataset": entry["dataset"], "seed": entry["seed"],
                      "policy": entry["policy"], "final": curve[-1],
                      "aaa": mean(curve), "forgetting": float(match.group(1))})
    pairs = paired_rows(rows)
    if len(pairs) != 81:
        raise ValueError("expected 81 paired task boundaries")
    correlations = _correlations(rows)
    report_dir = root / "reports"
    report_dir.mkdir(exist_ok=True)
    _write_csv(report_dir / "boundary_rows.csv", rows)
    _write_csv(report_dir / "paired_boundaries.csv", pairs)
    _write_csv(report_dir / "final_per_seed.csv", final)
    (report_dir / "correlations.json").write_text(json.dumps(correlations, indent=2) + "\n")
    lines = ["# Live-A / Frozen-A 边界诊断结果", "",
             "全部预定 seed；相关性为描述性分析，不作因果或显著性声明。", "",
             "| 数据集 | 方法 | Final | AAA | Forgetting |", "|---|---|---:|---:|---:|"]
    for dataset in ("inr", "c100", "cub"):
        for policy in ("live", "frozen"):
            group = [row for row in final if row["dataset"] == dataset and row["policy"] == policy]
            values = []
            for metric in ("final", "aaa", "forgetting"):
                sample = [row[metric] for row in group]
                values.append(f"{mean(sample):.2f}±{pstdev(sample):.2f}")
            lines.append(f"| {dataset} | {policy} | {' | '.join(values)} |")
    lines.extend(["", "## 可恢复性与旧类变化", "",
                  "每项为 Live-A 的 seed×任务边界 Spearman 相关；逐层值不作为独立样本。", "",
                  "| 数据集 | Seed | 残差 | 旧类损失 | ρ |",
                  "|---|---:|---|---|---:|"])
    for item in correlations:
        seed = "汇总" if item["seed"] is None else item["seed"]
        lines.append(f"| {item['dataset']} | {seed} | {item['residual']} | "
                     f"{item['outcome']} | {item['spearman']:.3f} |")
    lines.extend(["", "逐任务阶段指标见 `boundary_rows.csv`；Live−Frozen 配对差值见 `paired_boundaries.csv`。", ""])
    path = report_dir / "summary.md"
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.experiment_dir))


if __name__ == "__main__":
    main()
