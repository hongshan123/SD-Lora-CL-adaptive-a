#!/usr/bin/env python3
"""Measure Shared-A SD-LoRA artifact parameters (with/without prototypes).

Reports the parameters that count toward the research acceptance budget.
Legacy v1 counts shared A + per-task B + task scales; cumulative v2 counts
canonical down + cumulative up + triangular factors.  Stored prototypes are
added separately. LRPT transport U/V are temporary and discarded, so they are
reported separately as "tmp". Baseline SD-LoRA is 3,686,400; the round-2
budget is 2,211,840.
"""

import argparse
import glob
import os

import torch


def _count_tensors(values):
    if isinstance(values, dict):
        values = list(values.values())
    total = 0
    for value in values:
        if torch.is_tensor(value):
            total += value.numel()
        elif isinstance(value, (list, tuple)):
            total += _count_tensors(value)
    return total


def measure_artifact(directory):
    """Return persisted-state counts without double-counting deployment files."""
    directory = os.fspath(directory).rstrip("/")
    shared_a = 0
    b_files = 0
    scales = 0
    canonical_down = 0
    cumulative_up = 0
    triangular_r = 0
    aggregate_up = 0
    cuo_projection_down = 0
    cuo_unified_up = 0
    cuo_gram_scalars = 0
    prototypes = 0
    merged_b = 0

    state_path = os.path.join(directory, "sa_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        version = int(state.get("version", -1))
        if version in (2, 3):
            canonical_down = _count_tensors(state.get("canonical_down", []))
            cumulative_up = _count_tensors(state.get("cumulative_up", []))
            triangular_r = _count_tensors(state.get("triangular_r", []))
        elif version == 4:
            shared_a = _count_tensors(state.get("shared_a", []))
            aggregate_up = _count_tensors(state.get("aggregate_up", []))
        elif version == 5 and state.get("merge_mode") == "cuo_lowrank":
            cuo_projection_down = _count_tensors(
                state.get("projection_down", [])
            )
            cuo_unified_up = _count_tensors(state.get("unified_up", []))
            cuo_gram_scalars = _count_tensors(state.get("projected_gram", []))
        else:
            shared_a = _count_tensors(state.get("shared_a", []))
            scales = _count_tensors(list(state.get("scales", {}).values()))

    for path in sorted(glob.glob(os.path.join(directory, "sa_lora_w_b_*.pt"))):
        b_files += _count_tensors(
            torch.load(path, map_location="cpu", weights_only=True)
        )

    proto_path = os.path.join(directory, "sa_prototypes.pt")
    if os.path.exists(proto_path):
        prototypes = _count_tensors(
            torch.load(proto_path, map_location="cpu", weights_only=True)
        )

    merged_path = os.path.join(directory, "sa_merged_lora.pt")
    if os.path.exists(merged_path):
        merged = torch.load(merged_path, map_location="cpu", weights_only=True)
        merged_b = _count_tensors(merged.get("merged_b", []))

    lora_total = (
        shared_a
        + b_files
        + scales
        + canonical_down
        + cumulative_up
        + triangular_r
        + aggregate_up
        + cuo_projection_down
        + cuo_unified_up
    )
    persistent_scalar_total = lora_total + cuo_gram_scalars
    with_prototypes = persistent_scalar_total + prototypes
    baseline = 3_686_400
    budget = 2_211_840

    return {
        "directory": directory,
        "shared_a": shared_a,
        "per_task_b": b_files,
        "task_scales": scales,
        "canonical_down": canonical_down,
        "cumulative_up": cumulative_up,
        "triangular_r": triangular_r,
        "aggregate_up": aggregate_up,
        "lora_factor_scalars": lora_total,
        "cuo_projection_down": cuo_projection_down,
        "cuo_unified_up": cuo_unified_up,
        "cuo_gram_scalars": cuo_gram_scalars,
        "persistent_scalar_total": persistent_scalar_total,
        "prototypes": prototypes,
        "with_prototypes": with_prototypes,
        "merged_b": merged_b,
        "baseline": baseline,
        "budget": budget,
    }


def _print_summary(summary):
    print("artifact: {}".format(summary["directory"]))
    print("  shared_a            = {:>10,}".format(summary["shared_a"]))
    print("  per_task_b          = {:>10,}".format(summary["per_task_b"]))
    print("  task_scales         = {:>10,}".format(summary["task_scales"]))
    print("  canonical_down (v2) = {:>10,}".format(summary["canonical_down"]))
    print("  cumulative_up (v2)  = {:>10,}".format(summary["cumulative_up"]))
    print("  triangular_r (v2)   = {:>10,}".format(summary["triangular_r"]))
    print("  aggregate_up (v4)   = {:>10,}".format(summary["aggregate_up"]))
    print("  CUO projection (v5) = {:>10,}".format(summary["cuo_projection_down"]))
    print("  CUO unified_up (v5) = {:>10,}".format(summary["cuo_unified_up"]))
    print("  LoRA factors        = {:>10,}  ({:.2%} of baseline)".format(
        summary["lora_factor_scalars"],
        summary["lora_factor_scalars"] / summary["baseline"],
    ))
    print("  CUO Gram stats (v5) = {:>10,}".format(summary["cuo_gram_scalars"]))
    print("  persistent total    = {:>10,}".format(summary["persistent_scalar_total"]))
    print("  prototypes          = {:>10,}".format(summary["prototypes"]))
    print("  with_prototypes     = {:>10,}  ({:.2%} of baseline)".format(
        summary["with_prototypes"],
        summary["with_prototypes"] / summary["baseline"],
    ))
    print("  budget 2,211,840    = {:>10,}  ({} within budget)".format(
        summary["with_prototypes"],
        "YES" if summary["with_prototypes"] <= summary["budget"] else "NO",
    ))
    print("  merged_b (storage)  = {:>10,}".format(summary["merged_b"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?")
    parser.add_argument("--artifact")
    args = parser.parse_args()
    directory = args.artifact or args.directory
    if directory is None:
        parser.error("provide DIRECTORY or --artifact DIRECTORY")
    _print_summary(measure_artifact(directory))


if __name__ == "__main__":
    main()
