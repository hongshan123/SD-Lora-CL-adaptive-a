#!/usr/bin/env python3
"""Verify a minimal artifact reproduces the original artifact exactly.

For a fixed batch of test inputs it builds the Live-A backbone from both the
original and the minimal artifact, applies the final FC head, prototype head
and Dual-B fusion, and requires bit-identical logits (max abs diff == 0).
It also reports Final top-1 on the full test set from the minimal artifact and
compares it with the value recorded in the training log.

Usage:
  python scripts/verify_minimal_artifact.py \
    --config exps/p5_cub_livea_dual_b_seed1_nccl.json \
    --artifact CUB_P5_LIVEA_DUALB_SEED1_NCCL \
    --minimal CUB_P5_LIVEA_DUALB_SEED1_NCCL_MINIMAL \
    --log p5_cub_livea_dual_b_seed1_nccl.log
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.linears import PrototypeCosineHead
from backbone.sa_lora import SharedALoRA_ViT_timm
from utils.data_manager import DataManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--minimal", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=8)
    return parser.parse_args()


def build_network(config, artifact, num_tasks, device):
    base_model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0
    ).to(device)
    return SharedALoRA_ViT_timm(
        vit_model=base_model.eval(),
        r=int(config.get("lora_rank", 10)),
        increment=config["increment"],
        filepath=artifact,
        cur_task_index=num_tasks,
        shared_a_orthogonal=config.get("sa_shared_a_orthogonal", True),
        train_a_all_tasks=config.get("sa_train_a_all_tasks", False),
        delete_per_task_files=False,
        cumulative_state=config.get("sa_cumulative_state", False),
        cumulative_gauge=config.get("sa_cumulative_gauge", True),
        cumulative_merge=config.get("sa_cumulative_merge", "gauge"),
        cumulative_rank=config.get("sa_cumulative_rank", None),
        live_a_history_groups=config.get("sa_live_a_history_groups", 1),
    ).to(device)


def load_final_fc(artifact, num_tasks, device):
    task_id = num_tasks - 1
    weight = torch.load(
        "{}/CLs_weight{}.pt".format(artifact, task_id),
        map_location=device,
        weights_only=True,
    ).float()
    bias = torch.load(
        "{}/CLs_bias{}.pt".format(artifact, task_id),
        map_location=device,
        weights_only=True,
    ).float()
    return weight, bias


def load_dual_head(artifact):
    return torch.load(
        "{}/sa_dual_head.pt".format(artifact),
        map_location="cpu",
        weights_only=True,
    )


def fused_logits(artifact, features, weight, bias, dual, device):
    fc_logits = F.linear(features, weight, bias) / dual["tau_fc"]
    prototypes = torch.load(
        "{}/sa_prototypes.pt".format(artifact),
        map_location=device,
        weights_only=True,
    )
    head = PrototypeCosineHead(prototypes).to(device)
    proto_logits = head(features)["logits"] / dual["tau_proto"]
    fused = (
        (1.0 - dual["lambda"]) * fc_logits + dual["lambda"] * proto_logits
    )
    return fc_logits, proto_logits, fused


def main():
    cli = parse_args()
    with open(cli.config) as handle:
        config = json.load(handle)
    device = torch.device(cli.device)
    seed = int(config["seed"][0] if isinstance(config["seed"], list) else config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    loader = DataLoader(
        data_manager.get_dataset(all_classes, source="test", mode="test"),
        batch_size=cli.samples,
        shuffle=False,
        num_workers=0,
    )
    _, inputs, targets = next(iter(loader))
    inputs = inputs[: cli.samples].to(device)
    targets = targets[: cli.samples]

    net_a = build_network(config, cli.artifact, num_tasks, device)
    net_b = build_network(config, cli.minimal, num_tasks, device)
    weight_a, bias_a = load_final_fc(cli.artifact, num_tasks, device)
    weight_b, bias_b = load_final_fc(cli.minimal, num_tasks, device)
    dual_a = load_dual_head(cli.artifact)
    dual_b = load_dual_head(cli.minimal)

    with torch.no_grad():
        feats_a = net_a(inputs)
        feats_b = net_b(inputs)
        fc_a, proto_a, fused_a = fused_logits(
            cli.artifact, feats_a, weight_a, bias_a, dual_a, device
        )
        fc_b, proto_b, fused_b = fused_logits(
            cli.minimal, feats_b, weight_b, bias_b, dual_b, device
        )

    checks = {
        "features": (feats_a - feats_b).abs().max().item(),
        "fc_logits": (fc_a - fc_b).abs().max().item(),
        "proto_logits": (proto_a - proto_b).abs().max().item(),
        "fused_logits": (fused_a - fused_b).abs().max().item(),
        "fc_weight": (weight_a - weight_b).abs().max().item(),
        "fc_bias": (bias_a - bias_b).abs().max().item(),
    }
    ok = all(v == 0.0 for v in checks.values()) and dual_a == dual_b
    print("num_tasks={} samples={}".format(num_tasks, cli.samples))
    for name, value in checks.items():
        print("  max_abs_diff {:<14} = {:.3e}".format(name, value))
    print("  dual_head identical = {}".format(dual_a == dual_b))

    # Full test-set Final top-1 from the minimal artifact.
    net_b.eval()
    correct, total = 0, 0
    prototypes = torch.load(
        "{}/sa_prototypes.pt".format(cli.minimal),
        map_location=device,
        weights_only=True,
    )
    head = PrototypeCosineHead(prototypes).to(device)
    with torch.no_grad():
        for _, inputs_batch, targets_batch in loader:
            inputs_batch = inputs_batch.to(device)
            feats = net_b(inputs_batch)
            fc_logits = F.linear(feats, weight_b, bias_b) / dual_b["tau_fc"]
            proto_logits = head(feats)["logits"] / dual_b["tau_proto"]
            fused = (
                (1.0 - dual_b["lambda"]) * fc_logits
                + dual_b["lambda"] * proto_logits
            )
            preds = fused.max(dim=1).indices.cpu()
            correct += (preds == targets_batch).sum().item()
            total += targets_batch.shape[0]
    final = 100.0 * correct / max(total, 1)
    print("  minimal Final top1 = {:.2f} ({}/{})".format(final, correct, total))

    log_text = open(cli.log, errors="ignore").read()
    curves = re.findall(r"CNN top1 curve: \[(.*?)\]", log_text)
    if curves:
        values = [
            float(x)
            for x in re.findall(r"np\.float64\(([\d.]+)\)", curves[-1])
        ]
        if not values:
            values = [float(x.strip()) for x in curves[-1].split(",")]
        logged_final = values[-1] if values else float("nan")
        print("  log Final top1      = {:.2f}".format(logged_final))
        print("  diff vs log         = {:.3f}".format(final - logged_final))

    if ok:
        print("MINIMAL ARTIFACT VERIFY PASS")
        return 0
    print("MINIMAL ARTIFACT VERIFY FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
