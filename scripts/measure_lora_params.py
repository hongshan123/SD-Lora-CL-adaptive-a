#!/usr/bin/env python3
"""Estimate total stored LoRA parameters for an experiment artifact directory.

SD-LoRA baseline stores per-task lora_w_a/b_{t}.pt lists; K-CMS additionally
stores k_cms_state.pt clusters and optional shared LoRA. The classifier heads
are excluded because they are identical across methods.
"""

import argparse
import glob
import os
import sys

import torch


def tensor_count(t):
    if torch.is_tensor(t):
        return t.numel()
    if hasattr(t, "weight") and torch.is_tensor(t.weight):
        return t.weight.numel()
    return 0


def count_files(directory):
    total = 0
    for pattern in ("lora_w_a_*.pt", "lora_w_b_*.pt"):
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            value = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(value, (list, tuple)):
                total += sum(tensor_count(v) for v in value)
            else:
                total += tensor_count(value)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+")
    args = parser.parse_args()
    for directory in args.directories:
        per_task = count_files(directory)
        cluster_count = 0
        state_path = os.path.join(directory, "k_cms_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            for cluster in state.get("clusters", []):
                for a, b in zip(cluster.get("low_A", []), cluster.get("low_B", [])):
                    cluster_count += tensor_count(a) + tensor_count(b)
        shared_count = 0
        shared_path = os.path.join(directory, "k_cms_shared_lora.pt")
        if os.path.exists(shared_path):
            shared = torch.load(shared_path, map_location="cpu", weights_only=True)
            shared_count = sum(
                tensor_count(v)
                for v in list(shared.get("w_A", [])) + list(shared.get("w_B", []))
            )
        total = per_task + cluster_count + shared_count
        print(
            "{:<70} per_task={:>10} clusters={:>10} shared={:>8} total={:>12}".format(
                directory, per_task, cluster_count, shared_count, total
            )
        )


if __name__ == "__main__":
    main()
