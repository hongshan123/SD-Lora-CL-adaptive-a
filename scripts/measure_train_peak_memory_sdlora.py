#!/usr/bin/env python3
"""Measure training-step peak GPU memory for the original SD-LoRA (v1 bank).

Loads per-task ``lora_w_a_{t}.pt`` / ``lora_w_b_{t}.pt`` files into a
``LoRA_ViT_timm`` backbone at the final task and runs synthetic steps.

Usage:
  python scripts/measure_train_peak_memory_sdlora.py \
    --config exps/p5_sdlora_t5_inr_nccl.json \
    --artifact INR_P5_SDLORA_SEED1995_T5_NCCL \
    --device cuda:3 --steps 3
"""

import argparse
import json
import sys
from pathlib import Path

import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.lora import LoRA_ViT_timm
from utils.data_manager import DataManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    cli = parse_args()
    with open(cli.config) as handle:
        config = json.load(handle)
    torch.manual_seed(cli.seed)
    device = torch.device(cli.device)

    seed = int(config["seed"][0] if isinstance(config["seed"], list) else config["seed"])
    data_manager = DataManager(
        config["dataset"],
        config["shuffle"],
        seed,
        config["init_cls"],
        config["increment"],
        config,
    )
    num_tasks = data_manager.nb_tasks

    base_model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0
    ).to(device)
    backbone = LoRA_ViT_timm(
        vit_model=base_model.eval(),
        r=int(config.get("lora_rank", 10)),
        num_classes=0,
        increment=config["increment"],
        filepath=cli.artifact.rstrip("/") + "/",
        cur_task_index=num_tasks,
    ).to(device)
    backbone.train()
    head = torch.nn.Linear(768, 100, device=device)
    opt = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()), lr=1e-4
    )
    x = torch.randn(cli.batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 100, (cli.batch_size,), device=device)

    torch.cuda.reset_peak_memory_stats(device)
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(head(backbone(x)), y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(cli.steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(backbone(x)), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    print("config          : {}".format(cli.config))
    print("artifact        : {}".format(cli.artifact))
    print("tasks           : {}".format(num_tasks))
    print("batch_size      : {}".format(cli.batch_size))
    print("train-step peak : {:.1f} MiB".format(peak_mib))


if __name__ == "__main__":
    main()
