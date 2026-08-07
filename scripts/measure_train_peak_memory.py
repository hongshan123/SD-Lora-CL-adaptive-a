#!/usr/bin/env python3
"""Measure training-step peak GPU memory of a Shared-A artifact.

Builds the deployed training-time backbone at the final task (all historical
branches for v1, the single cumulative state for v2/v3), then runs a few
forward/backward steps on synthetic inputs.  Reports peak CUDA memory
allocated during the step (optimizer/data-loader memory excluded).

Usage:
  python scripts/measure_train_peak_memory.py \
    --config exps/sa_sdlora_proto_inr_seed1995.json \
    --artifact ImageNetR_SA_SDLORA_PROTO_SEED1995 \
    --device cuda:0 --steps 3
"""

import argparse
import json
import sys
from pathlib import Path

import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import SharedALoRA_ViT_timm
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
    backbone = SharedALoRA_ViT_timm(
        vit_model=base_model.eval(),
        r=int(config.get("lora_rank", 10)),
        increment=config["increment"],
        filepath=cli.artifact,
        cur_task_index=num_tasks,
        shared_a_orthogonal=config.get("sa_shared_a_orthogonal", True),
        train_a_all_tasks=config.get("sa_train_a_all_tasks", False),
        delete_per_task_files=False,
        cumulative_state=config.get("sa_cumulative_state", False),
        cumulative_gauge=config.get("sa_cumulative_gauge", True),
        cumulative_merge=config.get("sa_cumulative_merge", "gauge"),
        cumulative_rank=config.get("sa_cumulative_rank", None),
    ).to(device)
    backbone.train()
    head = torch.nn.Linear(backbone.out_dim, 100, device=device)
    opt = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()), lr=1e-4
    )
    x = torch.randn(cli.batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 100, (cli.batch_size,), device=device)

    # Warmup once to trigger any lazy allocations.
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
