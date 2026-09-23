#!/usr/bin/env python3
"""Read-only old/new class counterfactual from a saved pre-merge task state."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.causal_subspace_intervention import TaskModel, prototypes_from_current
from utils.data_manager import DataManager


def build_model(snapshot, device):
    if snapshot["normalize_current_branch"]:
        raise ValueError("normalized current branches are not supported by this diagnostic")
    branches = snapshot["branches"]
    if len(branches) != 24:
        raise ValueError("expected 24 Q/V branches for ViT-B/16")
    state = {
        "projection_down": [item["down"] for item in branches],
        "unified_up": [
            item["historical_up"] / (item["down"].norm() + 1e-8)
            for item in branches
        ],
    }
    model = TaskModel(
        state,
        snapshot["fc_weight"][:snapshot["known_classes"]],
        snapshot["fc_bias"][:snapshot["known_classes"]],
        snapshot["total_classes"] - snapshot["known_classes"],
        live=False,
    )
    with torch.no_grad():
        model.scale.copy_(snapshot["current_scale"])
        model.fc.weight.copy_(snapshot["fc_weight"])
        model.fc.bias.copy_(snapshot["fc_bias"])
        for layer, wrapper in enumerate(model.wrappers):
            wrapper.b_q.copy_(branches[2 * layer]["current_up"])
            wrapper.b_v.copy_(branches[2 * layer + 1]["current_up"])
    return model.to(device).eval()


def summarize_intrusion(labels, history_logits, full_logits, old_count):
    labels = labels.long().cpu()
    history_pred = history_logits.argmax(dim=1).cpu()
    full_pred = full_logits.argmax(dim=1).cpu()
    old = labels < old_count
    new = ~old
    result = {}
    for name, mask, lower, upper in (
        ("old", old, 0, old_count),
        ("new", new, old_count, full_logits.shape[1]),
    ):
        count = int(mask.sum())
        if not count:
            result[name] = {"count": 0}
            continue
        restricted_labels = labels[mask] - lower
        result[name] = {
            "count": count,
            "history_top1": 100 * float((history_pred[mask] == labels[mask]).float().mean()),
            "full_top1": 100 * float((full_pred[mask] == labels[mask]).float().mean()),
            "history_restricted_top1": 100 * float(
                (history_logits[mask, lower:upper].argmax(dim=1).cpu() == restricted_labels).float().mean()
            ),
            "full_restricted_top1": 100 * float(
                (full_logits[mask, lower:upper].argmax(dim=1).cpu() == restricted_labels).float().mean()
            ),
        }
    if bool(old.any()):
        result["old"].update({
            "history_correct_to_full_new_error_rate": 100 * float((
                (history_pred[old] == labels[old]) & (full_pred[old] >= old_count)
            ).float().mean()),
            "full_predicted_new_rate": 100 * float((full_pred[old] >= old_count).float().mean()),
            "history_predicted_new_rate": 100 * float((history_pred[old] >= old_count).float().mean()),
            "old_class_untouched_prediction_rate": 100 * float((
                full_pred[old] == history_pred[old]
            ).float().mean()),
        })
    return result


def collect_logits(model, loader, weights, device):
    labels, full_logits, history_logits = [], [], []
    with torch.no_grad():
        for _, images, targets in loader:
            images = images.to(device)
            model.wrappers[0].current_enabled = True
            for wrapper in model.wrappers:
                wrapper.current_enabled = True
            _, full_features = model(images)
            for wrapper in model.wrappers:
                wrapper.current_enabled = False
            _, history_features = model(images)
            labels.append(targets.cpu())
            full_logits.append((F.normalize(full_features, dim=-1) @ weights.T).cpu())
            history_logits.append((F.normalize(history_features, dim=-1) @ weights.T).cpu())
    for wrapper in model.wrappers:
        wrapper.current_enabled = True
    return torch.cat(labels), torch.cat(history_logits), torch.cat(full_logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.run_dir / "task_snapshots" / f"task_{args.task:03d}"
    snapshot = torch.load(directory / "pre_merge.pt", map_location="cpu", weights_only=False)
    config = json.loads((directory / "config.json").read_text())
    if snapshot["task_id"] != args.task or args.task == 0:
        raise ValueError("diagnostic requires a matching incremental task >= 1")
    device = torch.device(args.device)
    model = build_model(snapshot, device)
    seed = int(config["seed"][0] if isinstance(config["seed"], list) else config["seed"])
    manager = DataManager(
        config["dataset"], config["shuffle"], seed,
        config["init_cls"], config["increment"], config,
    )
    known = snapshot["known_classes"]
    total = snapshot["total_classes"]
    train_new = manager.get_dataset(np.arange(known, total), source="train", mode="test")
    old_test = manager.get_dataset(np.arange(known), source="test", mode="test")
    new_test = manager.get_dataset(np.arange(known, total), source="test", mode="test")
    def loader(dataset):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    new_prototypes = prototypes_from_current(
        model, loader(train_new), known, total - known, device
    )
    old = snapshot["old_prototypes"]
    if sorted(old) != list(range(known)):
        raise RuntimeError("old prototypes are incomplete or incorrectly ordered")
    old_prototypes = torch.stack([old[index] for index in range(known)]).to(device)
    weights = F.normalize(torch.cat((old_prototypes, new_prototypes), dim=0), dim=-1)
    labels = []
    history_logits = []
    full_logits = []
    for dataset in (old_test, new_test):
        y, historical, full = collect_logits(model, loader(dataset), weights, device)
        labels.append(y)
        history_logits.append(historical)
        full_logits.append(full)
    result = {
        "protocol": "pre-merge model; fixed old saved and new pre-merge train prototypes; test data only for evaluation",
        "dataset": config["dataset"], "seed": seed, "task": args.task,
        "known_classes": known, "total_classes": total,
        "metrics": summarize_intrusion(
            torch.cat(labels), torch.cat(history_logits), torch.cat(full_logits), known
        ),
    }
    output = args.output or directory / "branch_intrusion.json"
    predictions = {
        "labels": torch.cat(labels),
        "historical_logits": torch.cat(history_logits),
        "full_logits": torch.cat(full_logits),
        "old_prototypes": old_prototypes.cpu(),
        "new_prototypes": new_prototypes.cpu(),
        "old_count": known,
    }
    prediction_path = output.with_suffix(".pt")
    temporary = prediction_path.with_suffix(".pt.tmp")
    torch.save(predictions, temporary)
    os.replace(temporary, prediction_path)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
