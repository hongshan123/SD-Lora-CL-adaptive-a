#!/usr/bin/env python3
"""Evaluation-only Stage B2 dual-head diagnostic for Live-A artifacts.

For each task end, evaluate the stored per-task FC head and the prototype head
built from the stored prototype subset on the classes seen so far. Fixed
pre-registered fusion schedules A/B are evaluated with per-head temperature
calibration fit only on the current task's training classes.

NOTE: this first-pass diagnostic uses the *final* merged backbone because Live-A
artifacts do not persist per-task backbone snapshots. It is a final-space
diagnostic (upper/lower-bound insight), not the exact per-task-end evaluation;
the formal Stage B2 verification requires the retrained dual-head pipeline.
"""

import argparse
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
from utils.data_manager import DataManager


SCHEDULE_A = [0.00, 0.25, 0.50, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
SCHEDULE_B = [0.00, 0.00, 0.25, 0.50, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00]


class _MergedQKV(torch.nn.Module):
    def __init__(self, qkv, a_q, a_v, b_q, b_v):
        super().__init__()
        self.qkv = qkv
        self.a_q = torch.nn.Parameter(a_q, requires_grad=False)
        self.a_v = torch.nn.Parameter(a_v, requires_grad=False)
        self.b_q = torch.nn.Parameter(b_q, requires_grad=False)
        self.b_v = torch.nn.Parameter(b_v, requires_grad=False)
        self.dim = qkv.in_features

    def forward(self, x):
        new_q = F.linear(F.linear(x, self.a_q), self.b_q)
        new_v = F.linear(F.linear(x, self.a_v), self.b_v)
        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


def build_merged_backbone(base_model, merged_state):
    for layer_index, blk in enumerate(base_model.blocks):
        qkv = blk.attn.qkv
        offset = layer_index * 2
        blk.attn.qkv = _MergedQKV(
            qkv,
            merged_state["shared_a"][offset],
            merged_state["shared_a"][offset + 1],
            merged_state["merged_b"][offset],
            merged_state["merged_b"][offset + 1],
        )
    base_model.head = torch.nn.Identity()
    return base_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def extract_features(backbone, loader, device):
    features, targets = [], []
    backbone.eval()
    with torch.no_grad():
        for _, inputs, batch_targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            features.append(backbone(inputs).detach().cpu())
            targets.append(batch_targets)
    return torch.cat(features), torch.cat(targets)


def fit_temperature(logits, targets, lo=0.05, hi=5.0, steps=60):
    best_tau, best_loss = 1.0, float("inf")
    for tau in torch.linspace(lo, hi, steps).tolist():
        loss = F.cross_entropy(logits / tau, targets).item()
        if loss < best_loss:
            best_loss, best_tau = loss, tau
    return best_tau


def evaluate_logits(logits, targets, topk=1):
    preds = logits.topk(topk, dim=1).indices.cpu()
    hits = preds.eq(targets.unsqueeze(1)).any(dim=1)
    return 100.0 * hits.float().mean().item()


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
    num_tasks = data_manager.nb_tasks
    if num_tasks != len(SCHEDULE_A):
        raise ValueError(
            "diagnostic schedules are defined for T=10; got T={}".format(num_tasks)
        )

    base_model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0
    ).to(device)
    merged_state = torch.load(
        "{}/sa_merged_lora.pt".format(cli.artifact), map_location=device, weights_only=True
    )
    backbone = build_merged_backbone(base_model, merged_state).to(device)
    backbone.eval()

    stored_prototypes = torch.load(
        "{}/sa_prototypes.pt".format(cli.artifact), map_location="cpu", weights_only=True
    )

    rows = []
    for task_id in range(num_tasks):
        seen_classes = np.arange((task_id + 1) * config["increment"])
        current_classes = np.arange(
            task_id * config["increment"], (task_id + 1) * config["increment"]
        )

        weight = torch.load(
            "{}/CLs_weight{}.pt".format(cli.artifact, task_id),
            map_location="cpu",
            weights_only=True,
        ).float()
        bias = torch.load(
            "{}/CLs_bias{}.pt".format(cli.artifact, task_id),
            map_location="cpu",
            weights_only=True,
        ).float()
        proto_subset = {int(c): stored_prototypes[int(c)] for c in seen_classes}
        proto_head = PrototypeCosineHead(proto_subset).to(device)

        cal_loader = DataLoader(
            data_manager.get_dataset(current_classes, source="train", mode="test"),
            batch_size=cli.batch_size,
            shuffle=False,
            num_workers=cli.num_workers,
        )
        test_loader = DataLoader(
            data_manager.get_dataset(seen_classes, source="test", mode="test"),
            batch_size=cli.batch_size,
            shuffle=False,
            num_workers=cli.num_workers,
        )
        cal_feats, cal_targets = extract_features(backbone, cal_loader, device)
        test_feats, test_targets = extract_features(backbone, test_loader, device)

        with torch.no_grad():
            fc_cal = F.linear(cal_feats.to(device), weight.to(device), bias.to(device))
            proto_cal = proto_head(cal_feats.to(device))["logits"]
            fc_test = F.linear(test_feats.to(device), weight.to(device), bias.to(device))
            proto_test = proto_head(test_feats.to(device))["logits"]

        tau_fc = fit_temperature(fc_cal, cal_targets.to(device))
        tau_proto = fit_temperature(proto_cal, cal_targets.to(device))

        lambdas = (SCHEDULE_A[task_id], SCHEDULE_B[task_id])
        fc_top1 = evaluate_logits(fc_test, test_targets)
        proto_top1 = evaluate_logits(proto_test, test_targets)
        fc_top5 = evaluate_logits(fc_test, test_targets, topk=5)
        proto_top5 = evaluate_logits(proto_test, test_targets, topk=5)
        fused_rows = []
        for name, lam in zip(("A", "B"), lambdas):
            fused = (1.0 - lam) * (fc_test / tau_fc) + lam * (proto_test / tau_proto)
            fused_rows.append((name, evaluate_logits(fused, test_targets), evaluate_logits(fused, test_targets, topk=5)))

        row = {
            "task": task_id,
            "classes": len(seen_classes),
            "fc_top1": fc_top1,
            "proto_top1": proto_top1,
            "fc_top5": fc_top5,
            "proto_top5": proto_top5,
            "tau_fc": tau_fc,
            "tau_proto": tau_proto,
            "fusedA_top1": fused_rows[0][1],
            "fusedA_top5": fused_rows[0][2],
            "fusedB_top1": fused_rows[1][1],
            "fusedB_top5": fused_rows[1][2],
        }
        rows.append(row)
        print(
            "task={:02d} fc={:7.2f}/{:.2f} proto={:7.2f}/{:.2f} "
            "A={:7.2f}/{:.2f} B={:7.2f}/{:.2f} tau=({:.3f},{:.3f})".format(
                task_id,
                fc_top1,
                fc_top5,
                proto_top1,
                proto_top5,
                fused_rows[0][1],
                fused_rows[0][2],
                fused_rows[1][1],
                fused_rows[1][2],
                tau_fc,
                tau_proto,
            )
        )

    def avg(key):
        return sum(r[key] for r in rows) / len(rows)

    print("\nAvgAcc final-space diagnostic:")
    for key in ("fc_top1", "proto_top1", "fusedA_top1", "fusedB_top1"):
        print("  {:<12} = {:.3f}".format(key, avg(key)))
    print("  theoretical per-task max(fc,proto) = {:.3f}".format(
        sum(max(r["fc_top1"], r["proto_top1"]) for r in rows) / len(rows)
    ))
    print("  task0-3 fc/proto/A/B:")
    for r in rows[:4]:
        print(
            "    t{} fc={:.2f} proto={:.2f} A={:.2f} B={:.2f}".format(
                r["task"], r["fc_top1"], r["proto_top1"], r["fusedA_top1"], r["fusedB_top1"]
            )
        )


if __name__ == "__main__":
    main()
