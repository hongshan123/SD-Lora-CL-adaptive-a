#!/usr/bin/env python3
"""Export completed continual-learning log records into a deduplicated CSV ledger."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


CONFIG_RE = re.compile(r"=> config:\s*(.+)")
DATASET_RE = re.compile(r"=> dataset:\s*(\S+)")
CURVE_RE = re.compile(r"=> CNN top1 curve:\s*\[(.*)\]")
TOP5_RE = re.compile(r"=> CNN top5 curve:\s*\[(.*)\]")
AAA_RE = re.compile(r"Average Accuracy \(CNN\):\s*([0-9.]+)")
FORGET_RE = re.compile(r"Forgetting \(CNN\):\s*([0-9.eE+-]+)")
NUMBER_RE = re.compile(r"(?:np\.float64\()?([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


def values(raw: str) -> list[float]:
    return [float(x) for x in NUMBER_RE.findall(raw)]


def last_match(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None


def parse_log(path: Path, source: str, root: Path) -> dict[str, str] | None:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None

    config = last_match(CONFIG_RE, text)
    dataset = last_match(DATASET_RE, text)
    curve_raw = last_match(CURVE_RE, text)
    aaa = last_match(AAA_RE, text)
    forgetting = last_match(FORGET_RE, text)
    if not (config and dataset and curve_raw and aaa and forgetting):
        return None

    curve = values(curve_raw)
    if not curve:
        return None
    top5_raw = last_match(TOP5_RE, text)
    top5 = values(top5_raw) if top5_raw else []
    rel_path = path.relative_to(root)
    stat = path.stat()
    return {
        "source": source,
        "repo": rel_path.parts[0] if len(rel_path.parts) > 1 else "",
        "log_path": str(rel_path),
        "config_path": config,
        "dataset": dataset,
        "method": Path(config).stem,
        "num_tasks": str(len(curve)),
        "final_top1": f"{curve[-1]:.6f}",
        "aaa": f"{float(aaa):.6f}",
        "forgetting": f"{float(forgetting):.6f}",
        "top1_curve": ";".join(f"{x:.6f}" for x in curve),
        "top5_curve": ";".join(f"{x:.6f}" for x in top5),
        "modified_epoch": str(int(stat.st_mtime)),
    }


def signature(row: dict[str, str]) -> tuple[str, ...]:
    # Config basename is intentionally included: equal numbers from different methods stay distinct.
    return tuple(row[key] for key in ("dataset", "method", "num_tasks", "final_top1", "aaa", "forgetting", "top1_curve"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for path in sorted(args.root.rglob("*.log")):
        row = parse_log(path, args.source, args.root)
        if row:
            rows.append(row)

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[signature(row)].append(row)

    fieldnames = [
        "source", "repo", "log_path", "config_path", "dataset", "method", "num_tasks",
        "final_top1", "aaa", "forgetting", "top1_curve", "top5_curve", "modified_epoch",
        "duplicate_count", "all_log_paths",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entries in sorted(grouped.values(), key=lambda group: (group[0]["source"], group[0]["repo"], group[0]["log_path"])):
            canonical = min(entries, key=lambda item: item["log_path"]).copy()
            canonical["duplicate_count"] = str(len(entries))
            canonical["all_log_paths"] = " | ".join(item["log_path"] for item in entries)
            writer.writerow(canonical)
    print(f"direct_completed_logs={len(rows)} deduplicated_records={len(grouped)} output={args.output}")


if __name__ == "__main__":
    main()
