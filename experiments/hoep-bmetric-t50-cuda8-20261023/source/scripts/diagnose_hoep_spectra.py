#!/usr/bin/env python3
"""Inspect HOEP-A plastic dimensions in existing live-A checkpoints."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.operator_energy_partition import (
    historical_energy_coordinates,
    select_global_low_energy_partition,
)
from backbone.sa_lora import SA_STATE_VERSION_LIVE_A


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("states", nargs="+", type=Path)
    parser.add_argument(
        "--budgets", nargs="+", type=float, default=(0.01, 0.05, 0.10)
    )
    parser.add_argument("--tie-rtol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def inspect_state(path, budgets, tie_rtol):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if int(state.get("version", -1)) != SA_STATE_VERSION_LIVE_A:
        raise ValueError(f"{path}: expected live-A v4 state")
    shared_a = state.get("shared_a", [])
    aggregate_up = state.get("aggregate_up", [])
    if not shared_a or len(shared_a) != len(aggregate_up):
        raise ValueError(f"{path}: inconsistent shared_a/aggregate_up")
    spectra = [
        historical_energy_coordinates(g.float(), a.float())[0]
        for a, g in zip(shared_a, aggregate_up)
    ]
    branches = []
    for index, values in enumerate(spectra):
        total = float(values.sum())
        branches.append(
            {
                "index": index,
                "block": index // 2,
                "projection": "q" if index % 2 == 0 else "v",
                "energy": total,
                "normalized_spectrum": (
                    (values / total).tolist() if total > 0 else values.tolist()
                ),
            }
        )
    selections = {}
    for budget in budgets:
        partition = select_global_low_energy_partition(
            spectra, budget, tie_rtol
        )
        counts = [int(mask.sum()) for mask in partition.plastic_masks]
        selections[str(budget)] = {
            "selected_directions": partition.selected_directions,
            "total_directions": partition.total_directions,
            "selected_energy": partition.selected_energy,
            "total_energy": partition.total_energy,
            "selected_energy_ratio": (
                partition.selected_energy / partition.total_energy
                if partition.total_energy > 0
                else 0.0
            ),
            "plastic_dimensions_per_branch": counts,
            "mixed_branch_fraction": sum(
                0 < count < len(spectra[index])
                for index, count in enumerate(counts)
            )
            / len(counts),
            "frozen_branch_fraction": sum(count == 0 for count in counts)
            / len(counts),
            "live_branch_fraction": sum(
                count == len(spectra[index])
                for index, count in enumerate(counts)
            )
            / len(counts),
        }
    return {
        "path": str(path.resolve()),
        "task_id": int(state.get("task_id", -1)),
        "rank": int(state.get("rank", shared_a[0].shape[0])),
        "branches": branches,
        "selections": selections,
    }


def main():
    args = parse_args()
    results = [
        inspect_state(path, args.budgets, args.tie_rtol)
        for path in args.states
    ]
    payload = {"states": results}
    serialized = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
