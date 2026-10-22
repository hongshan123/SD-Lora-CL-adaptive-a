#!/usr/bin/env python3
"""Evaluate the class-prototype router using completed v1 expert artifacts."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.class_proto_routed_lora import (
    CLASS_ROUTER_STATE_VERSION,
    ClassPrototypeRoutedLoRAViT,
    joint_topk_task_class_logits,
    task_head_logits,
    task_ids_from_ranges,
)
from utils.data_manager import DataManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Matching v1 experiment config")
    parser.add_argument("--artifact", required=True, help="Directory with all LoRA/FC files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--router-temperature", type=float, default=0.07)
    parser.add_argument("--classifier-temperature", type=float, default=1.0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_ranges(data_manager):
    ranges, low = [], 0
    for task_id in range(data_manager.nb_tasks):
        high = low + data_manager.get_task_size(task_id)
        ranges.append([low, high])
        low = high
    return ranges


def build_class_router_state(base_vit, loader, ranges, device):
    num_classes = ranges[-1][1]
    feature_dim = int(base_vit.embed_dim)
    sums = torch.zeros(num_classes, feature_dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    base_vit.eval()
    with torch.no_grad():
        for batch_index, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device)
            tokens = base_vit.forward_features(inputs)
            features = base_vit.forward_head(tokens, pre_logits=True).float()
            sums.index_add_(0, targets, features)
            counts.index_add_(0, targets, torch.ones_like(targets, dtype=torch.float32))
            if batch_index % 100 == 0:
                print("prototype batches:", batch_index, flush=True)
    if torch.any(counts <= 0):
        raise RuntimeError("class prototype extraction missed at least one class")
    mapping = torch.empty(num_classes, dtype=torch.long)
    for task_id, (low, high) in enumerate(ranges):
        mapping[low:high] = task_id
    return {
        "version": CLASS_ROUTER_STATE_VERSION,
        "task_ids": list(range(len(ranges))),
        "class_ranges": ranges,
        "class_ids": torch.arange(num_classes),
        "class_to_task": mapping,
        "class_feature_sums": sums.cpu(),
        "class_counts": counts.cpu(),
        "global_feature_sum": sums.sum(dim=0).cpu(),
        "global_feature_count": counts.sum().cpu(),
    }


def load_scales(artifact, num_tasks):
    for name in ("class_router_state.pt", "task_prototypes.pt"):
        path = os.path.join(artifact, name)
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=True)
            scales = torch.as_tensor(state.get("lora_scales", []), dtype=torch.float32)
            if scales.numel() == num_tasks:
                return scales
    raise FileNotFoundError("could not recover one LoRA scale per task from artifact")


def load_final_head(artifact, num_tasks, device):
    task_id = num_tasks - 1
    weight_path = os.path.join(artifact, "CLs_weight{}.pt".format(task_id))
    bias_path = os.path.join(artifact, "CLs_bias{}.pt".format(task_id))
    if not os.path.exists(weight_path) or not os.path.exists(bias_path):
        raise FileNotFoundError("missing final global classifier files")
    weight = torch.load(weight_path, map_location=device, weights_only=True).float()
    bias = torch.load(bias_path, map_location=device, weights_only=True).float()
    return weight, bias


def topk(logits, k=5):
    return torch.topk(logits, min(k, logits.shape[1]), dim=1).indices


def metric(predictions, targets):
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    return {
        "top1": round(float((predictions[:, 0] == targets).float().mean() * 100), 4),
        "top5": round(
            float((predictions == targets[:, None]).any(dim=1).float().mean() * 100),
            4,
        ),
    }


def selected_features(model, inputs, task_ids):
    model.prepare_routing("selected", task_ids)
    return model(inputs)


def evaluate(model, loader, state, weight, bias, temperatures, seed, device):
    ranges = state["class_ranges"]
    num_tasks = len(ranges)
    names = (
        "class_top1_global",
        "class_top1_task_head",
        "class_top2_task_head_joint",
        "oracle_global",
        "oracle_task_head",
        "base_only_global",
        "random_global",
        "all_global",
    )
    predictions = {name: [] for name in names}
    targets_all, top1_all, top2_all = [], [], []
    random_generator = torch.Generator().manual_seed(seed + 991)
    model.eval()
    with torch.no_grad():
        for batch_index, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets_device = targets.to(device)
            true_tasks = task_ids_from_ranges(targets_device, ranges)
            router_features = model.extract_router_features(inputs)
            candidates, task_scores, _ = model.route_from_features(router_features, 2)

            first_features = selected_features(model, inputs, candidates[:, 0])
            first_logits = F.linear(first_features, weight, bias)
            predictions["class_top1_global"].append(topk(first_logits).cpu())
            local_logits = task_head_logits(
                first_features, weight, bias, candidates[:, 0], ranges
            )
            predictions["class_top1_task_head"].append(topk(local_logits).cpu())

            second_features = selected_features(model, inputs, candidates[:, 1])
            joint = joint_topk_task_class_logits(
                task_scores,
                candidates,
                [first_features, second_features],
                weight,
                bias,
                ranges,
                temperatures[0],
                temperatures[1],
            )
            predictions["class_top2_task_head_joint"].append(topk(joint).cpu())

            oracle_features = selected_features(model, inputs, true_tasks)
            oracle_logits = F.linear(oracle_features, weight, bias)
            predictions["oracle_global"].append(topk(oracle_logits).cpu())
            oracle_local = task_head_logits(
                oracle_features, weight, bias, true_tasks, ranges
            )
            predictions["oracle_task_head"].append(topk(oracle_local).cpu())

            base_logits = F.linear(router_features, weight, bias)
            predictions["base_only_global"].append(topk(base_logits).cpu())

            random_tasks = torch.randint(
                num_tasks,
                (inputs.shape[0],),
                generator=random_generator,
            ).to(device)
            random_features = selected_features(model, inputs, random_tasks)
            predictions["random_global"].append(
                topk(F.linear(random_features, weight, bias)).cpu()
            )

            model.prepare_routing("all")
            all_features = model(inputs)
            predictions["all_global"].append(
                topk(F.linear(all_features, weight, bias)).cpu()
            )
            targets_all.append(targets.cpu())
            top1_all.append(candidates[:, 0].cpu())
            top2_all.append(candidates.cpu())
            if batch_index % 50 == 0:
                print("evaluation batches:", batch_index, flush=True)

    targets = torch.cat(targets_all)
    true_tasks = task_ids_from_ranges(targets, ranges)
    selected_top1 = torch.cat(top1_all)
    selected_top2 = torch.cat(top2_all)
    results = {name: metric(predictions[name], targets_all) for name in names}
    results["routing"] = {
        "top1": round(float((selected_top1 == true_tasks).float().mean() * 100), 4),
        "top2": round(
            float(
                (selected_top2 == true_tasks[:, None]).any(dim=1).float().mean() * 100
            ),
            4,
        ),
        "utilization": torch.bincount(selected_top1, minlength=num_tasks).tolist(),
        "confusion": torch.bincount(
            true_tasks * num_tasks + selected_top1,
            minlength=num_tasks * num_tasks,
        )
        .view(num_tasks, num_tasks)
        .tolist(),
    }
    return results


def main():
    cli = parse_args()
    with open(cli.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    seed_value = config["seed"]
    seed = int(seed_value[0] if isinstance(seed_value, list) else seed_value)
    seed_everything(seed)
    device = torch.device(cli.device)
    config["device"] = [0]
    data_manager = DataManager(
        config["dataset"],
        config["shuffle"],
        seed,
        config["init_cls"],
        config["increment"],
        config,
    )
    ranges = class_ranges(data_manager)
    all_classes = np.arange(data_manager.nb_classes)
    train_loader = DataLoader(
        data_manager.get_dataset(all_classes, source="train", mode="test"),
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        data_manager.get_dataset(all_classes, source="test", mode="test"),
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=True,
    )

    base_model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0
    ).to(device)
    state = build_class_router_state(base_model, train_loader, ranges, device)
    state["lora_scales"] = load_scales(cli.artifact, len(ranges))
    model = ClassPrototypeRoutedLoRAViT(
        vit_model=base_model.eval(),
        r=int(config.get("lora_rank", 10)),
        filepath=cli.artifact,
        increment=config["increment"],
        inference_only=True,
        router_state=state,
    ).to(device)
    weight, bias = load_final_head(cli.artifact, len(ranges), device)
    results = evaluate(
        model,
        test_loader,
        state,
        weight,
        bias,
        (cli.router_temperature, cli.classifier_temperature),
        seed,
        device,
    )

    os.makedirs(cli.output_dir, exist_ok=True)
    torch.save(state, os.path.join(cli.output_dir, "class_router_state.pt"))
    output_path = os.path.join(cli.output_dir, "class_router_metrics.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    print("saved:", output_path, flush=True)


if __name__ == "__main__":
    main()
