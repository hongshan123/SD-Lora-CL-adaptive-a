#!/usr/bin/env python3
"""Read-only, matched-sample G/head state swaps across task snapshots."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_sa_sdlora import build_merged_backbone
from utils.data_manager import DataManager
from utils.sa_task_snapshots import audit_snapshot


def load_task(directory, task):
    snapshot = directory / "task_snapshots" / f"task_{task:03d}"
    audit_snapshot(snapshot)
    config = json.loads((snapshot / "config.json").read_text())
    merged = torch.load(snapshot / "sa_merged_lora.pt", map_location="cpu", weights_only=True)
    prototypes = torch.load(snapshot / "sa_prototypes.pt", map_location="cpu", weights_only=True)
    fc_weight = torch.load(snapshot / f"CLs_weight{task}.pt", map_location="cpu", weights_only=True)
    fc_bias = torch.load(snapshot / f"CLs_bias{task}.pt", map_location="cpu", weights_only=True)
    if merged["task_id"] != task or sorted(prototypes) != list(range(len(prototypes))):
        raise ValueError(f"incomplete or mismatched task {task} artifact")
    if fc_weight.shape[0] != len(prototypes) or fc_bias.shape != (len(prototypes),):
        raise ValueError(f"task {task} classifier shape does not match prototypes")
    if not all(torch.is_tensor(value) for value in prototypes.values()):
        raise ValueError("only one prototype per class is supported")
    return config, merged, prototypes, fc_weight, fc_bias


def graft_old_rows(anchor, target, old_count):
    if anchor.ndim != 2 or target.ndim != 2:
        raise ValueError("head weights must be rank-two tensors")
    if anchor.shape != (old_count, target.shape[1]) or target.shape[0] <= old_count:
        raise ValueError("head dimensions do not match the old/new class boundary")
    result = target.clone()
    result[:old_count] = anchor
    return result


def prototype_matrix(prototypes):
    return torch.stack([prototypes[index].float() for index in range(len(prototypes))])


def score(features, weights, bias=None):
    if bias is None:
        return F.normalize(features.float(), dim=1) @ F.normalize(weights.float(), dim=1).T
    return F.linear(features.float(), weights.float(), bias.float())


def metrics(predictions, labels, old_count):
    if not 0 < old_count <= predictions.shape[1]:
        raise ValueError("invalid old class count")
    labels = labels.long()
    pred = predictions.argmax(dim=1)
    restricted = predictions[:, :old_count].argmax(dim=1)
    return {
        "top1": 100.0 * (pred == labels).float().mean().item(),
        "old_restricted_top1": 100.0 * (restricted == labels).float().mean().item(),
        "predicted_later_rate": 100.0 * (pred >= old_count).float().mean().item(),
    }, pred


def build_backbone(state, device):
    base = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    return build_merged_backbone(base, state).to(device).eval()


def collect_features(anchor_state, target_state, loader, device):
    old_model = build_backbone(anchor_state, device)
    new_model = build_backbone(target_state, device)
    rows = {"anchor": [], "target": [], "labels": [], "indices": []}
    with torch.no_grad():
        for indices, images, labels in loader:
            images = images.to(device, non_blocking=True)
            rows["anchor"].append(old_model(images).cpu())
            rows["target"].append(new_model(images).cpu())
            rows["labels"].append(labels.cpu())
            rows["indices"].append(indices.cpu())
    return {key: torch.cat(values) for key, values in rows.items()}


def evaluate_grid(features, anchor_weight, target_weight, old_count,
                  anchor_bias=None, target_bias=None):
    weights = {
        "anchor_rows": graft_old_rows(anchor_weight, target_weight, old_count),
        "target_rows": target_weight,
    }
    biases = {"anchor_rows": None, "target_rows": None}
    if (anchor_bias is None) != (target_bias is None):
        raise ValueError("both bias tensors must be supplied together")
    if anchor_bias is not None:
        biases["anchor_rows"] = torch.cat((anchor_bias, target_bias[old_count:]))
        biases["target_rows"] = target_bias
    results, predictions = {}, {}
    for g_name in ("anchor", "target"):
        for head_name, weight in weights.items():
            key = f"{g_name}_G__{head_name}"
            logits = score(features[g_name], weight, biases[head_name])
            results[key], predictions[key] = metrics(logits, features["labels"], old_count)
    baseline = results["target_G__target_rows"]["top1"]
    for item in results.values():
        item["delta_vs_deployed_pp"] = item["top1"] - baseline
    return results, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--anchor-task", type=int, required=True)
    parser.add_argument("--target-task", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.anchor_task < 0 or args.target_task <= args.anchor_task:
        raise ValueError("require 0 <= anchor-task < target-task")

    anchor_cfg, anchor_g, anchor_proto, anchor_fc, anchor_bias = load_task(
        args.run_dir, args.anchor_task
    )
    target_cfg, target_g, target_proto, target_fc, target_bias = load_task(
        args.run_dir, args.target_task
    )
    for key in ("dataset", "seed", "init_cls", "increment", "shuffle"):
        if anchor_cfg[key] != target_cfg[key]:
            raise ValueError(f"configuration mismatch at {key}")
    if len(anchor_g["shared_a"]) != len(target_g["shared_a"]) or not all(
        torch.equal(old, new)
        for old, new in zip(anchor_g["shared_a"], target_g["shared_a"])
    ):
        raise ValueError("A changed across snapshots; this is not a G-only intervention")

    old_count = len(anchor_proto)
    seed = anchor_cfg["seed"]
    seed = int(seed[0] if isinstance(seed, list) else seed)
    manager = DataManager(
        target_cfg["dataset"], target_cfg["shuffle"], seed,
        target_cfg["init_cls"], target_cfg["increment"], target_cfg,
    )
    if old_count != sum(manager.get_task_size(t) for t in range(args.anchor_task + 1)):
        raise ValueError("anchor class count does not match the data protocol")
    if len(target_proto) != sum(manager.get_task_size(t) for t in range(args.target_task + 1)):
        raise ValueError("target class count does not match the data protocol")
    dataset = manager.get_dataset(np.arange(old_count), source="test", mode="test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    features = collect_features(anchor_g, target_g, loader, torch.device(args.device))
    labels = features["labels"].long()
    if not torch.all((labels >= 0) & (labels < old_count)):
        raise ValueError("evaluation cohort includes non-anchor classes")
    prototype_results, proto_preds = evaluate_grid(
        features, prototype_matrix(anchor_proto), prototype_matrix(target_proto), old_count
    )
    fc_results, fc_preds = evaluate_grid(
        features, anchor_fc.float(), target_fc.float(), old_count,
        anchor_bias.float(), target_bias.float()
    )
    result = {
        "protocol": "fixed old-class test samples, target class space, later-class rows fixed to target",
        "deployed_head": "prototype_cosine",
        "fc_status": "shadow_only; training linear FC with bias is not the deployed classifier",
        "dataset": target_cfg["dataset"],
        "seed": seed,
        "anchor_task": args.anchor_task,
        "target_task": args.target_task,
        "old_classes": old_count,
        "target_classes": len(target_proto),
        "sample_count": len(labels),
        "labels_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
        "prototype_head": prototype_results,
        "fc_head_shadow": fc_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "labels": labels,
        "dataset_indices": features["indices"],
        "prototype_predictions": proto_preds,
        "fc_predictions": fc_preds,
    }, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
