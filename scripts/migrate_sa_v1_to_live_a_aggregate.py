#!/usr/bin/env python3
"""Offline v1 -> Live-A Aggregate-B conversion and equivalence check.

Live-A Aggregate-B stores, per Q/V branch:
    G = sum_i s_i * B_i / ||B_i||
and uses the live shared A at training/inference time:
    y_old = G A x / ||A||

For a fixed final A this is numerically identical to the EXP-009 bank
``sum_i s_i B_i A x / (||B_i|| ||A||)``.  This script converts an existing
EXP-009 (v1) artifact into ``sa_live_a_aggregate.pt`` and verifies that the
aggregate backbone reproduces the stored merged backbone's features and
prototype logits (max abs diff <= 1e-5).

Usage:
  python scripts/migrate_sa_v1_to_live_a_aggregate.py \
    --artifact ImageNetR_SA_SDLORA_PROTO_SEED1995 \
    --config exps/sa_sdlora_proto_inr_seed1995.json \
    --device cuda:0 [--eval]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.linears import PrototypeCosineHead
from scripts.evaluate_sa_sdlora import _MergedQKV, build_merged_backbone
from utils.data_manager import DataManager


LIVE_A_AGGREGATE_FILENAME = "sa_live_a_aggregate.pt"


def build_aggregate_backbone(base_model, aggregate_up, shared_a):
    """Wrap QKV with the Live-A aggregate branch ``G A x / ||A||``."""
    from scripts.evaluate_sa_sdlora import _MergedQKV  # reuse pattern

    for layer_index, blk in enumerate(base_model.blocks):
        qkv = blk.attn.qkv
        offset = layer_index * 2
        a_q = shared_a[offset]
        a_v = shared_a[offset + 1]
        g_q = aggregate_up[offset]
        g_v = aggregate_up[offset + 1]
        blk.attn.qkv = _MergedQKV(
            qkv,
            a_q,
            a_v,
            g_q / torch.linalg.vector_norm(a_q),
            g_v / torch.linalg.vector_norm(a_v),
        )
    base_model.head = torch.nn.Identity()
    return base_model


def load_v1_aggregate(artifact):
    state_path = Path(artifact) / "sa_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if int(state.get("version", -1)) != 1:
        raise ValueError("artifact is not v1 EXP-009 (version={})".format(
            state.get("version", -1)
        ))
    shared_a = [w.detach().cpu().float() for w in state["shared_a"]]
    scales = {
        int(task_id): torch.as_tensor(v).reshape(())
        for task_id, v in state["scales"].items()
    }
    task_ids = sorted(scales)
    branches = len(shared_a)
    aggregate_up = [torch.zeros_like(b) for b in
                    torch.load(
                        Path(artifact) / "sa_lora_w_b_{}.pt".format(task_ids[0]),
                        map_location="cpu",
                        weights_only=True,
                    )]
    for task_id in task_ids:
        b_list = torch.load(
            Path(artifact) / "sa_lora_w_b_{}.pt".format(task_id),
            map_location="cpu",
            weights_only=True,
        )
        if len(b_list) != branches:
            raise ValueError("branch count mismatch in task {}".format(task_id))
        s = scales[task_id]
        for idx, b in enumerate(b_list):
            aggregate_up[idx] = aggregate_up[idx] + s * b.float() / (
                torch.linalg.vector_norm(b.float()) + 1e-8
            )
    return {
        "version": 1,
        "task_id": len(task_ids),
        "rank": shared_a[0].shape[0],
        "aggregate_up": aggregate_up,
        "shared_a": shared_a,
        "scales": scales,
        "folded_scales": True,
    }


def verify_equivalence(base_model, aggregate_state, artifact, device, samples=8):
    merged_path = Path(artifact) / "sa_merged_lora.pt"
    merged_state = torch.load(merged_path, map_location=device, weights_only=True)
    merged_backbone = build_merged_backbone(
        timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        ).to(device).eval(),
        merged_state,
    ).to(device).eval()
    aggregate_backbone = build_aggregate_backbone(
        base_model,
        aggregate_state["aggregate_up"],
        aggregate_state["shared_a"],
    ).to(device).eval()

    x = torch.randn(samples, 3, 224, 224, device=device)
    with torch.no_grad():
        feat_a = aggregate_backbone(x)
        feat_m = merged_backbone(x)
    feature_diff = (feat_a - feat_m).abs().max().item()

    proto_path = Path(artifact) / "sa_prototypes.pt"
    logit_diff = None
    if proto_path.exists():
        prototypes = torch.load(proto_path, map_location="cpu", weights_only=True)
        head = PrototypeCosineHead(prototypes).to(device)
        with torch.no_grad():
            logit_a = head(feat_a)["logits"]
            logit_m = head(feat_m)["logits"]
        logit_diff = (logit_a - logit_m).abs().max().item()

    ok = feature_diff <= 1e-5 and (logit_diff is None or logit_diff <= 1e-5)
    print("feature max abs diff  : {:.3e}".format(feature_diff))
    if logit_diff is not None:
        print("prototype logit diff  : {:.3e}".format(logit_diff))
    print("EQUIVALENCE: {}".format("PASS" if ok else "FAIL"))
    return ok


def evaluate_top1(backbone, prototypes, data_manager, device, batch_size=64):
    from torch.utils.data import DataLoader

    head = PrototypeCosineHead(prototypes).to(device)
    loader = DataLoader(
        data_manager.get_dataset(
            np.arange(data_manager.nb_classes), source="test", mode="test"
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    correct = 0
    total = 0
    backbone.eval()
    with torch.no_grad():
        for _, inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = head(backbone(inputs))["logits"]
            _, preds = logits.max(dim=1)
            correct += (preds.cpu() == targets).sum().item()
            total += targets.shape[0]
    return 100.0 * correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    aggregate_state = load_v1_aggregate(args.artifact)
    out_path = Path(args.artifact) / LIVE_A_AGGREGATE_FILENAME
    torch.save(aggregate_state, out_path)
    print("wrote {}".format(out_path))

    base_model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0
    ).to(device).eval()
    ok = verify_equivalence(
        base_model, aggregate_state, args.artifact, device, samples=args.samples
    )
    if not ok:
        raise SystemExit(1)

    if args.eval:
        with open(args.config) as handle:
            config = json.load(handle)
        seed = int(config["seed"][0] if isinstance(config["seed"], list) else config["seed"])
        data_manager = DataManager(
            config["dataset"],
            config["shuffle"],
            seed,
            config["init_cls"],
            config["increment"],
            config,
        )
        prototypes = torch.load(
            Path(args.artifact) / "sa_prototypes.pt",
            map_location="cpu",
            weights_only=True,
        )
        top1 = evaluate_top1(base_model, prototypes, data_manager, device)
        print("aggregate final Top1: {:.2f}".format(top1))


if __name__ == "__main__":
    main()
