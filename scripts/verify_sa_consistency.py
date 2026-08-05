#!/usr/bin/env python3
"""Verify that the per-task Shared-A bank and the on-disk merged LoRA produce
identical features/logits for fixed inputs, and that the stored prototype head
agrees across both backbones.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.linears import PrototypeCosineHead
from backbone.sa_lora import SharedALoRA_ViT_timm
from scripts.evaluate_sa_sdlora import build_merged_backbone
from utils.data_manager import DataManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    cli = parse_args()
    with open(cli.config) as handle:
        config = json.load(handle)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
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
    all_classes = np.arange(data_manager.nb_classes)
    loader = torch.utils.data.DataLoader(
        data_manager.get_dataset(all_classes, source="test", mode="test"),
        batch_size=cli.samples,
        shuffle=False,
        num_workers=0,
    )
    _, inputs, _ = next(iter(loader))
    inputs = inputs[: cli.samples].to(device)

    per_task = SharedALoRA_ViT_timm(
        vit_model=timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        ).eval(),
        r=int(config.get("lora_rank", 10)),
        increment=config["increment"],
        filepath=cli.artifact,
        cur_task_index=num_tasks,
        shared_a_orthogonal=config.get("sa_shared_a_orthogonal", True),
        delete_per_task_files=False,
    ).to(device)

    merged_path = "{}/sa_merged_lora.pt".format(cli.artifact)
    merged_state = torch.load(merged_path, map_location=device, weights_only=True)
    merged = build_merged_backbone(
        timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        ).eval(),
        merged_state,
    ).to(device)

    per_task.eval()
    merged.eval()
    with torch.no_grad():
        feats_a = per_task(inputs)
        feats_b = merged(inputs)
    diff = (feats_a - feats_b).abs().max().item()
    print("feature max abs diff: {:.3e}".format(diff))
    ok = diff < 1e-4

    proto_path = "{}/sa_prototypes.pt".format(cli.artifact)
    if Path(proto_path).exists():
        prototypes = torch.load(proto_path, map_location="cpu", weights_only=True)
        head = PrototypeCosineHead(prototypes).to(device)
        with torch.no_grad():
            logits_a = head(feats_a)["logits"]
            logits_b = head(feats_b)["logits"]
        logit_diff = (logits_a - logits_b).abs().max().item()
        print("prototype logit max abs diff: {:.3e}".format(logit_diff))
        ok = ok and logit_diff < 1e-4

    print("CONSISTENCY: {}".format("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
