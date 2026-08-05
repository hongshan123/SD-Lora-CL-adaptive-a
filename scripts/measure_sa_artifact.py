#!/usr/bin/env python3
"""Measure Shared-A SD-LoRA artifact parameters (with/without prototypes).

Reports the parameters that count toward the research acceptance budget:
shared A + per-task B + task scales + stored prototypes. LRPT transport U/V
are temporary and discarded, so they are reported separately as "tmp".
Baseline SD-LoRA is 3,686,400; the round-2 budget is 2,211,840.
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
    prototypes = 0
    merged_b = 0

    state_path = os.path.join(directory, "sa_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=True)
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

    lora_total = shared_a + b_files + scales
    with_prototypes = lora_total + prototypes
    baseline = 3_686_400
    budget = 2_211_840

    print("artifact: {}".format(directory))
    print("  shared_a            = {:>10,}".format(shared_a))
    print("  per_task_b          = {:>10,}".format(b_files))
    print("  task_scales         = {:>10,}".format(scales))
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
