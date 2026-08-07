#!/usr/bin/env python3
"""Offline diagnostic of prototype-coordinate expiry vs backbone interference.

DIAGNOSTIC ONLY: this script intentionally re-reads old training data to
recompute prototypes in different feature spaces.  It is not part of the
rehearsal-free method and must never be used as an acceptance result.

Checks implemented:
  1. current cumulative backbone + stored (training-time) prototypes;
  2. current cumulative backbone + offline recomputed prototypes;
  3. frozen base ViT + base-space prototypes;
  4. per-class cosine similarity between base-space and final-space
     prototypes (feature drift magnitude).

Decision rule (from next_improvement_sd.md):
  - if (2) >> (1): main problem is prototype coordinate expiry;
  - if (2) ~ (1) and both << (3): main problem is backbone representation
    interference / capacity, not prototype coordinates.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.linears import PrototypeCosineHead
from scripts.evaluate_sa_sdlora import _MergedQKV, build_merged_backbone
from utils.data_manager import DataManager


num_workers = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def extract_prototypes(backbone, loader, device, normalize=True):
    sums = {}
    counts = {}
    backbone.eval()
    with torch.no_grad():
        for _, inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            feats = backbone(inputs)
            if normalize:
                feats = F.normalize(feats, p=2, dim=1)
            for f, target in zip(feats.cpu(), targets):
                c = int(target.item())
                sums[c] = sums.get(c, 0.0) + f
                counts[c] = counts.get(c, 0) + 1
    return {
        c: F.normalize(sums[c] / counts[c], p=2, dim=0)
        for c in sums
    }


def eval_prototypes(backbone, prototypes, loader, device):
    head = PrototypeCosineHead(prototypes).to(device)
    backbone.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = head(backbone(inputs))["logits"]
            _, preds = logits.max(dim=1)
            correct += (preds.cpu() == targets).sum().item()
            total += targets.shape[0]
    return 100.0 * correct / max(total, 1)


def main():
    cli = parse_args()
    with open(cli.config) as handle:
        config = json.load(handle)
    seed = int(config["seed"][0] if isinstance(config["seed"], list) else config["seed"])
    device = torch.device(cli.device)
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
    all_classes = np.arange(data_manager.nb_classes)
    train_loader = DataLoader(
        data_manager.get_dataset(all_classes, source="train", mode="test"),
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        data_manager.get_dataset(all_classes, source="test", mode="test"),
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    base_model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0
    ).to(device)
    base_model.eval()
    # build_merged_backbone mutates its input in place; use a separate copy so
    # the "frozen base" backbone is a true unmodified ViT (P0 diagnostic fix).
    merged_base = copy.deepcopy(base_model).to(device)
    assert merged_base is not base_model

    merged_path = Path(cli.artifact) / "sa_merged_lora.pt"
    merged_state = torch.load(merged_path, map_location=device, weights_only=True)
    backbone = build_merged_backbone(merged_base, merged_state).to(device)
    for blk in base_model.blocks:
        assert not isinstance(blk.attn.qkv, _MergedQKV), (
            "base_model must remain a true frozen base (no merged LoRA)"
        )
    for blk in backbone.blocks:
        assert isinstance(blk.attn.qkv, _MergedQKV), (
            "merged backbone must contain _MergedQKV wrappers"
        )

    stored_path = Path(cli.artifact) / "sa_prototypes.pt"
    stored = torch.load(stored_path, map_location="cpu", weights_only=True)
    stored_acc = eval_prototypes(backbone, stored, test_loader, device)
    print("[DIAGNOSTIC ONLY] stored training-time prototypes: {:.2f}".format(stored_acc))

    final_protos = extract_prototypes(backbone, train_loader, device)
    recomputed_acc = eval_prototypes(backbone, final_protos, test_loader, device)
    print("[DIAGNOSTIC ONLY] recomputed final-space prototypes: {:.2f}".format(recomputed_acc))

    base_protos = extract_prototypes(base_model, train_loader, device)
    base_acc = eval_prototypes(base_model, base_protos, test_loader, device)
    print("[DIAGNOSTIC ONLY] frozen base ViT + base prototypes: {:.2f}".format(base_acc))

    common = sorted(set(final_protos) & set(base_protos))
    cosines = [
        float((final_protos[c] * base_protos[c]).sum())
        for c in common
    ]
    print(
        "[DIAGNOSTIC ONLY] base-vs-final prototype cosine: "
        "mean={:.4f} min={:.4f} max={:.4f} (n={})".format(
            float(np.mean(cosines)),
            float(np.min(cosines)),
            float(np.max(cosines)),
            len(cosines),
        )
    )

    if recomputed_acc > stored_acc + 0.5:
        print("[DIAGNOSTIC] prototype coordinate expiry dominates: recomputing "
              "prototypes in the final space recovers accuracy.")
    elif base_acc > recomputed_acc + 0.5:
        print("[DIAGNOSTIC] backbone representation interference/capacity "
              "dominates: even recomputed prototypes cannot match the frozen "
              "base classifier.")
    else:
        print("[DIAGNOSTIC] neither effect dominates clearly; inspect per-class "
              "drift and margins next.")


if __name__ == "__main__":
    main()
