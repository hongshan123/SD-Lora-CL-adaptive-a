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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()

    directory = args.directory.rstrip("/")
    shared_a = 0
    b_files = 0
    scales = 0
    canonical_down = 0
    cumulative_up = 0
    triangular_r = 0
    aggregate_up = 0
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
    )
    with_prototypes = lora_total + prototypes
    baseline = 3_686_400
    budget = 2_211_840

    print("artifact: {}".format(directory))
    print("  shared_a            = {:>10,}".format(shared_a))
    print("  per_task_b          = {:>10,}".format(b_files))
    print("  task_scales         = {:>10,}".format(scales))
    print("  canonical_down (v2) = {:>10,}".format(canonical_down))
    print("  cumulative_up (v2)  = {:>10,}".format(cumulative_up))
    print("  triangular_r (v2)   = {:>10,}".format(triangular_r))
    print("  aggregate_up (v4)   = {:>10,}".format(aggregate_up))
    print("  lora_total          = {:>10,}  ({:.2%} of baseline)".format(
        lora_total, lora_total / baseline
    ))
    print("  prototypes          = {:>10,}".format(prototypes))
    print("  with_prototypes     = {:>10,}  ({:.2%} of baseline)".format(
        with_prototypes, with_prototypes / baseline
    ))
    print("  budget 2,211,840    = {:>10,}  ({} within budget)".format(
        with_prototypes, "YES" if with_prototypes <= budget else "NO"
    ))
    print("  merged_b (storage)  = {:>10,}".format(merged_b))


if __name__ == "__main__":
    main()
