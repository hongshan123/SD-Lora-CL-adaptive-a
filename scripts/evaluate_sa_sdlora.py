#!/usr/bin/env python3
"""Offline final-head calibration evaluation for Shared-A SD-LoRA artifacts."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import SharedALoRA_ViT_timm
from utils.data_manager import DataManager


num_workers = 8


class _MergedQKV(torch.nn.Module):
    """QKV wrapper for the exact merged Shared-A LoRA (B* A x)."""

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
    parser.add_argument("--tune-epochs", type=int, default=5)
    parser.add_argument("--tune-lr", type=float, default=1e-3)
    parser.add_argument("--align", action="store_true", default=True)
    parser.add_argument("--cosine", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def load_final_head(artifact, num_tasks, device):
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


def extract_features(backbone, loader, device, batch_size):
    features, targets = [], []
    torch.cuda.empty_cache()
    backbone.eval()
    with torch.no_grad():
        for _, inputs, batch_targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            features.append(backbone(inputs).detach().cpu())
            targets.append(batch_targets)
    return torch.cat(features), torch.cat(targets)


def evaluate(backbone, weight, bias, loader, device):
    backbone.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            features = backbone(inputs)
            logits = F.linear(features, weight, bias)
            _, preds = logits.max(dim=1)
            correct += (preds.cpu() == targets).sum().item()
            total += targets.shape[0]
    return 100.0 * correct / max(total, 1)


def evaluate_cosine(backbone, weight, loader, device):
    """Data-free cosine classifier evaluation (L2-normalized features/weights)."""
    backbone.eval()
    weight_norm = F.normalize(weight, p=2, dim=1)
    correct = 0
    total = 0
    with torch.no_grad():
        for _, inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            features = backbone(inputs)
            logits = F.linear(F.normalize(features, p=2, dim=1), weight_norm)
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
    num_tasks = data_manager.nb_tasks
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
    merged_path = "{}/sa_merged_lora.pt".format(cli.artifact)
    has_merged = Path(merged_path).exists()
    has_per_task = Path(
        "{}/sa_lora_w_b_0.pt".format(cli.artifact)
    ).exists()
    if has_per_task:
        backbone = SharedALoRA_ViT_timm(
            vit_model=base_model.eval(),
            r=int(config.get("lora_rank", 10)),
            increment=config["increment"],
            filepath=cli.artifact,
            cur_task_index=num_tasks,
            shared_a_orthogonal=config.get("sa_shared_a_orthogonal", True),
            delete_per_task_files=False,
        ).to(device)
    elif has_merged:
        merged_state = torch.load(
            merged_path, map_location=device, weights_only=True
        )
        backbone = build_merged_backbone(base_model, merged_state).to(device)
    else:
        raise FileNotFoundError(
            "artifact has neither per-task B files nor merged LoRA"
        )

    weight, bias = load_final_head(cli.artifact, num_tasks, device)
    before = evaluate(backbone, weight, bias, test_loader, device)
    print("final top1 before calibration: {:.2f}".format(before))

    if cli.align:
        increment = int(config["increment"])
        old_norm = torch.norm(weight[:-increment], p=2, dim=1).mean()
        new_norm = torch.norm(weight[-increment:], p=2, dim=1).mean()
        gamma = old_norm / (new_norm + 1e-8)
        weight[-increment:] *= gamma
        after_align = evaluate(backbone, weight, bias, test_loader, device)
        print("final top1 after weight align: {:.2f}".format(after_align))

    if cli.cosine:
        after_cosine = evaluate_cosine(backbone, weight, test_loader, device)
        print("final top1 with cosine norm: {:.2f}".format(after_cosine))

    if cli.tune_epochs > 0:
        feature_matrix, target_vector = extract_features(
            backbone, train_loader, device, cli.batch_size
        )
        weight = torch.nn.Parameter(weight.detach().clone())
        bias = torch.nn.Parameter(bias.detach().clone())
        optimizer = optim.SGD([weight, bias], lr=cli.tune_lr, momentum=0.9)
        for epoch in range(cli.tune_epochs):
            permutation = torch.randperm(feature_matrix.shape[0])
            total_loss = 0.0
            steps = 0
            for start in range(0, feature_matrix.shape[0], cli.batch_size):
                indices = permutation[start : start + cli.batch_size]
                logits = F.linear(
                    feature_matrix[indices].to(device), weight, bias
                )
                loss = F.cross_entropy(
                    logits, target_vector[indices].to(device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                steps += 1
            print(
                "head tune epoch {}/{} loss={:.4f}".format(
                    epoch + 1, cli.tune_epochs, total_loss / max(steps, 1)
                )
            )
        after_tune = evaluate(
            backbone, weight.detach(), bias.detach(), test_loader, device
        )
        print("final top1 after head tune: {:.2f}".format(after_tune))


if __name__ == "__main__":
    main()
