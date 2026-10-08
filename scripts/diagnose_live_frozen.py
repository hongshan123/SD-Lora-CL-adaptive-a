#!/usr/bin/env python3
"""Read-only S0-S6 and recoverability analysis of one saved task boundary."""

from __future__ import annotations

import argparse
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

from scripts.diagnose_joint_ab_boundary import (  # noqa: E402
    paired_prediction_changes, summarize_group_logits,
)
from scripts.evaluate_sa_sdlora import build_merged_backbone  # noqa: E402
from scripts.live_frozen_stages import build_stage_factors, effective_operator  # noqa: E402
from utils.canonical_hash import hash_named_tensors  # noqa: E402
from utils.data_manager import DataManager  # noqa: E402
from utils.sa_task_snapshots import audit_snapshot  # noqa: E402


def mean_prototypes(features, labels, class_ids):
    features = F.normalize(features.float(), p=2, dim=1)
    result = {}
    for class_id in class_ids:
        selected = features[labels == class_id]
        if len(selected) == 0:
            raise ValueError(f"missing training examples for class {class_id}")
        result[int(class_id)] = F.normalize(selected.mean(dim=0), p=2, dim=0)
    return result


def fixed_head(old_prototypes, new_prototypes, total_classes):
    if set(old_prototypes) | set(new_prototypes) != set(range(total_classes)):
        raise ValueError("prototype class IDs do not cover the evaluation space")
    if set(old_prototypes) & set(new_prototypes):
        raise ValueError("old and new prototype IDs overlap")
    vectors = [old_prototypes.get(index, new_prototypes.get(index))
               for index in range(total_classes)]
    if any(not torch.is_tensor(vector) for vector in vectors):
        raise ValueError("this protocol requires one tensor prototype per class")
    return F.normalize(torch.stack(vectors).float(), p=2, dim=1)


def set_stage(model, state):
    if len(state["shared_a"]) != 2 * len(model.blocks):
        raise ValueError("Q/V branch count differs from backbone depth")
    with torch.no_grad():
        for layer, block in enumerate(model.blocks):
            wrapper = block.attn.qkv
            for offset, suffix in ((0, "q"), (1, "v")):
                index = 2 * layer + offset
                getattr(wrapper, "a_" + suffix).copy_(
                    state["shared_a"][index].to(model.blocks[0].attn.qkv.a_q.device)
                )
                getattr(wrapper, "b_" + suffix).copy_(
                    state["merged_b"][index].to(model.blocks[0].attn.qkv.a_q.device)
                )
    model.eval()


def summarize_stage(logits, labels, old_count, previous=None):
    result = summarize_group_logits(logits, labels, old_count)
    result["global_top1"] = 100.0 * int((logits.argmax(1) == labels).sum()) / len(labels)
    if previous is not None:
        result["flips_vs_previous"] = paired_prediction_changes(
            previous, logits, labels, old_count
        )
    return result


def _loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _features(model, loader, device):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for _, images, targets in loader:
            features.append(model(images.to(device, non_blocking=True)).float().cpu())
            labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


def _logits(model, loader, head, device):
    ids, labels, logits = [], [], []
    model.eval()
    with torch.no_grad():
        for indices, images, targets in loader:
            features = model(images.to(device, non_blocking=True))
            scores = F.normalize(features.float(), p=2, dim=1) @ head.T
            ids.append(indices.cpu())
            labels.append(targets.cpu())
            logits.append(scores.cpu())
    return torch.cat(ids), torch.cat(labels), torch.cat(logits)


def _difference(left, right):
    delta = left.float() - right.float()
    return {"max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / left.float().norm().clamp_min(1e-12))}


def _old_input_weighted_energy(model, loader, previous, stages, device):
    """Stream all old training images; capture each layer's S0 input tokens."""
    branch_ops = []
    for index, (old_g, old_a, new_a) in enumerate(zip(
        previous["aggregate_up"], previous["shared_a"], stages["S1"]["shared_a"]
    )):
        old_a = old_a.double()
        new_a = new_a.double()
        old = old_g.double() @ (old_a / (old_a.norm() + 1e-8))
        projector = torch.linalg.pinv(new_a) @ new_a
        lost = old @ (torch.eye(new_a.shape[1], dtype=torch.float64) - projector)
        branch_ops.append((old.float().to(device), lost.float().to(device)))
    energy = torch.zeros(len(branch_ops), 2, dtype=torch.float64)
    handles = []
    for layer, block in enumerate(model.blocks):
        def capture(module, args, layer_index=layer):
            flat = args[0].detach().float().reshape(-1, args[0].shape[-1])
            for offset in (0, 1):
                index = 2 * layer_index + offset
                old, lost = branch_ops[index]
                energy[index, 0] += (flat @ lost.T).square().sum().double().cpu()
                energy[index, 1] += (flat @ old.T).square().sum().double().cpu()
        handles.append(block.attn.qkv.register_forward_pre_hook(capture))
    try:
        model.eval()
        with torch.no_grad():
            for _, images, _ in loader:
                model(images.to(device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()
    return energy


def run_boundary(run_dir, task, output_dir, device, batch_size=32):
    run_dir = Path(run_dir)
    if task < 1:
        raise ValueError("Task 0 has no historical boundary")
    previous_dir = run_dir / "task_snapshots" / f"task_{task - 1:03d}"
    current_dir = run_dir / "task_snapshots" / f"task_{task:03d}"
    audit_snapshot(previous_dir)
    audit_snapshot(current_dir)
    config = json.loads((current_dir / "config.json").read_text())
    for key, expected in (
        ("sa_cumulative_merge", "live_a_aggregate_b"),
        ("sa_live_a_boundary_merge", "aligned"),
        ("sa_live_a_history_forward", "shared"),
        ("sa_k_prototypes", 1),
        ("sa_raw_prototypes", False),
    ):
        actual = config.get(key, expected)
        if actual != expected:
            raise ValueError(f"unsupported diagnostic protocol {key}={actual}")
    pre = torch.load(current_dir / "pre_merge.pt", map_location="cpu", weights_only=True)
    previous = torch.load(previous_dir / "sa_state.pt", map_location="cpu", weights_only=True)
    deployed = torch.load(current_dir / "sa_merged_lora.pt", map_location="cpu", weights_only=True)
    official_proto = torch.load(current_dir / "sa_prototypes.pt", map_location="cpu", weights_only=True)
    if pre["task_id"] != task or deployed["task_id"] != task:
        raise ValueError("task ID mismatch")
    old_count, total = pre["known_classes"], pre["total_classes"]
    old_proto = pre["old_prototypes"]
    if set(old_proto) != set(range(old_count)) or set(official_proto) != set(range(total)):
        raise ValueError("snapshot prototype IDs are incomplete")
    stages, branch_diagnostics = build_stage_factors(previous, pre)
    operator_differences = [float((effective_operator(stages["S5"], index)
                                   - effective_operator(deployed, index)).abs().max())
                            for index in range(len(pre["branches"]))]
    if max(operator_differences) > 1e-4:
        raise RuntimeError(f"reconstructed deployment operator mismatch: {max(operator_differences)}")
    seed = config["seed"]
    seed = int(seed[0] if isinstance(seed, list) else seed)
    manager = DataManager(config["dataset"], config["shuffle"], seed,
                          config["init_cls"], config["increment"], config)
    train_new = _loader(manager.get_dataset(np.arange(old_count, total),
                                            source="train", mode="test"), batch_size)
    train_old = _loader(manager.get_dataset(np.arange(old_count),
                                            source="train", mode="test"), batch_size)
    test = _loader(manager.get_dataset(np.arange(total), source="test", mode="test"),
                   batch_size)
    base = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model = build_merged_backbone(base, stages["S0"]).to(device).eval()
    set_stage(model, stages["S2"])
    new_features, new_labels = _features(model, train_new, device)
    new_proto = mean_prototypes(new_features, new_labels, range(old_count, total))
    fixed = fixed_head(old_proto, new_proto, total).to(device)
    transported_old = {index: official_proto[index] for index in range(old_count)}
    transported = fixed_head(transported_old, new_proto, total).to(device)
    formal = fixed_head(transported_old,
                        {index: official_proto[index] for index in range(old_count, total)},
                        total).to(device)
    logits = {}
    metrics = {}
    identity = None
    sequence = ("S0", "S1", "S2", "S3", "S4", "S5", "H_LS")
    for name in sequence:
        set_stage(model, stages[name])
        ids, labels, scores = _logits(model, test, fixed, device)
        if identity is None:
            identity = (ids, labels)
        elif not (torch.equal(ids, identity[0]) and torch.equal(labels, identity[1])):
            raise RuntimeError("test sample order or labels changed between stages")
        logits[name] = scores
        predecessor = {"S1": "S0", "S2": "S1", "S3": "S2", "S4": "S3",
                       "S5": "S4", "H_LS": "S0"}.get(name)
        metrics[name] = summarize_stage(scores, labels, old_count,
                                        logits.get(predecessor))
    set_stage(model, deployed)
    _, _, actual_scores = _logits(model, test, fixed, device)
    reconstruction = _difference(logits["S5"], actual_scores)
    exact = _difference(logits["S3"], logits["S4"])
    for label, comparison in (("S3/S4", exact), ("S5/deployed", reconstruction)):
        if comparison["max_abs"] > 1e-4 or comparison["relative_l2"] > 1e-5:
            raise RuntimeError(f"{label} logits differ: {comparison}")
    set_stage(model, stages["S5"])
    for name, head in (("S6", transported), ("formal_new_prototypes", formal)):
        ids, labels, scores = _logits(model, test, head, device)
        if not (torch.equal(ids, identity[0]) and torch.equal(labels, identity[1])):
            raise RuntimeError("sample order changed for prototype counterfactual")
        logits[name] = scores
        predecessor = "S5" if name == "S6" else "S6"
        metrics[name] = summarize_stage(scores, labels, old_count, logits[predecessor])
    set_stage(model, stages["S0"])
    weighted = _old_input_weighted_energy(model, train_old, previous, stages, device)
    for index, record in enumerate(branch_diagnostics):
        lost, history = weighted[index].tolist()
        record["weighted_lost_energy"] = lost
        record["weighted_historical_energy"] = history
        record["r_weighted"] = lost / max(history, 1e-12)
    total_lost = sum(record["lost_energy"] for record in branch_diagnostics)
    total_history = sum(record["historical_energy"] for record in branch_diagnostics)
    weighted_lost = sum(record["weighted_lost_energy"] for record in branch_diagnostics)
    weighted_history = sum(record["weighted_historical_energy"] for record in branch_diagnostics)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"task_{task:03d}.pt"
    torch.save({"sample_ids": identity[0], "labels": identity[1], "logits": logits,
                "temporary_new_prototypes": new_proto,
                "old_prototypes_before_transport": old_proto,
                "old_prototypes_after_transport": transported_old,
                "official_new_prototypes": {index: official_proto[index]
                                            for index in range(old_count, total)}}, artifact)
    result = {
        "dataset": config["dataset"], "seed": seed, "task": task,
        "known_classes": old_count, "total_classes": total,
        "sample_count": len(identity[0]),
        "temporary_new_prototype_source": "current task train split, test preprocessing, S2 features",
        "weighted_input_source": "old class train split, test preprocessing, S0 Q/V inputs, all tokens",
        "prototype_hashes": {
            "fixed": hash_named_tensors({str(index): fixed[index].cpu()
                                         for index in range(total)}),
            "transported": hash_named_tensors({str(index): transported[index].cpu()
                                               for index in range(total)}),
            "formal": hash_named_tensors({str(index): formal[index].cpu()
                                          for index in range(total)}),
        },
        "checks": {"s3_s4_logits": exact, "s5_deployed_logits": reconstruction,
                   "s5_deployed_operator_max_abs": max(operator_differences)},
        "aggregate": {"r_rec": total_lost / max(total_history, 1e-12),
                      "r_weighted": weighted_lost / max(weighted_history, 1e-12)},
        "branches": branch_diagnostics,
        "metrics": metrics,
        "logits_file": str(artifact),
    }
    summary = output_dir / f"task_{task:03d}.json"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    path = run_boundary(args.run_dir, args.task, args.output_dir,
                        torch.device(args.device), args.batch_size)
    print(path)


if __name__ == "__main__":
    main()
