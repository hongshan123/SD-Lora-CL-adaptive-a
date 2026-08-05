#!/usr/bin/env python3
"""Measure inference FLOPs, throughput, and peak GPU memory of the merged
Shared-A SD-LoRA backbone loaded from disk (paper-level audit).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import timm
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_sa_sdlora import build_merged_backbone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    cli = parse_args()
    with open(cli.config) as handle:
        config = json.load(handle)
    torch.manual_seed(cli.seed)
    device = torch.device(cli.device)

    base_model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0
    ).to(device)
    merged_path = "{}/sa_merged_lora.pt".format(cli.artifact)
    merged_state = torch.load(merged_path, map_location=device, weights_only=True)
    backbone = build_merged_backbone(base_model, merged_state).to(device).eval()

    x = torch.randn(cli.batch_size, 3, 224, 224, device=device)

    # FLOPs on one forward.
    with torch.no_grad():
        for _ in range(cli.warmup):
            backbone(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        with FlopCounterMode(display=False) as flop_counter:
            backbone(x)
    flops = flop_counter.get_total_flops()

    # Throughput.
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(cli.iters):
            backbone(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    images_per_sec = cli.batch_size * cli.iters / max(elapsed, 1e-9)
    peak_mib = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            backbone(x)
        peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    print("batch_size      : {}".format(cli.batch_size))
    print("FLOPs (forward) : {:.3e}".format(flops))
    print("throughput      : {:.1f} images/sec".format(images_per_sec))
    print("peak mem (MiB)  : {}".format(
        "{:.1f}".format(peak_mib) if peak_mib is not None else "n/a (cpu)"
    ))


if __name__ == "__main__":
    main()
