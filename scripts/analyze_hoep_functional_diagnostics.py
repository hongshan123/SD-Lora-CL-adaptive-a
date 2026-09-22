#!/usr/bin/env python3
"""Evaluate preregistered Functional-HOEP shadow-diagnostic Go criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def summarize(path):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    tasks = [item for item in artifact.get("tasks", []) if int(item["task_id"]) > 0]
    if not tasks:
        raise RuntimeError("{} contains no Task 1+ diagnostics".format(path))
    transition_hits = [
        item["global"]["operator_mask_functional_energy_ratio"] > 0.075
        or item["global"]["jaccard"] < 0.8
        for item in tasks
    ]
    branches = [branch for item in tasks for branch in item["branches"]]
    mixed = [
        0 < int(branch["functional_mask"].sum()) < branch["functional_mask"].numel()
        for branch in branches
    ]
    return {
        "path": str(path),
        "transitions": len(tasks),
        "transition_hit_fraction": sum(transition_hits) / len(transition_hits),
        "branches": len(branches),
        "functional_mixed_branch_fraction": sum(mixed) / len(mixed),
        "mean_jaccard": sum(item["global"]["jaccard"] for item in tasks)
        / len(tasks),
        "mean_operator_mask_functional_energy_ratio": sum(
            item["global"]["operator_mask_functional_energy_ratio"]
            for item in tasks
        )
        / len(tasks),
        "finite": all(item["global"]["finite"] for item in tasks),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+")
    args = parser.parse_args()
    summaries = [summarize(Path(path)) for path in args.artifacts]
    dataset_passes = sum(
        item["transition_hit_fraction"] >= 0.30 and item["finite"]
        for item in summaries
    )
    mixed_pass = (
        sum(item["branches"] * item["functional_mixed_branch_fraction"] for item in summaries)
        / sum(item["branches"] for item in summaries)
        >= 0.30
    )
    output = {
        "datasets": summaries,
        "dataset_transition_passes": dataset_passes,
        "mixed_branch_condition": mixed_pass,
        "go": dataset_passes >= 2 and mixed_pass,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
