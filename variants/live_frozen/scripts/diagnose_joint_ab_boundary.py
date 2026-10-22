#!/usr/bin/env python3
"""Read-only old/new class response at a Live-A task boundary."""

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

from backbone.coordinate_stability import align_live_a_aggregate
from backbone.sa_lora import absorb_live_a_current_projection
from scripts.diagnose_main_branch_intrusion import build_model
from scripts.evaluate_sa_sdlora import build_merged_backbone
from utils.data_manager import DataManager
from utils.sa_task_snapshots import audit_snapshot


def capped_current_up(a, b, scale, mode):
    absorbed, diagnostics = absorb_live_a_current_projection(
        a, b, scale, mode=mode
    )
    return b * diagnostics["consolidation_gain"], diagnostics["consolidation_gain"]


def aligned_counterfactual(previous, pre_merge, absorb_mode):
    branches = pre_merge["branches"]
    if not (len(branches) == len(previous["shared_a"]) == len(previous["aggregate_up"])):
        raise ValueError("previous and current branch counts differ")
    shared_a, merged_b = [], []
    for index, branch in enumerate(branches):
        a = branch["down"].float()
        old_g = previous["aggregate_up"][index].float()
        old_a = previous["shared_a"][index].float()
        aligned_g, _ = align_live_a_aggregate(old_g, old_a, a)
        current_g, _ = absorb_live_a_current_projection(
            a, branch["current_up"].float(),
            pre_merge["current_scale"].float(), mode=absorb_mode,
        )
        g = aligned_g + current_g
        shared_a.append(a)
        merged_b.append(g / (a.norm() + 1e-8))
    return {
        "task_id": pre_merge["task_id"],
        "shared_a": shared_a,
        "merged_b": merged_b,
    }


def summarize_group_logits(logits, labels, old_count):
    if logits.ndim != 2 or labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("logits and labels have incompatible shapes")
    if not 0 < old_count < logits.shape[1]:
        raise ValueError("old_count must be within the class range")
    labels = labels.long()
    prediction = logits.argmax(dim=1)
    true_logit = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], float("-inf"))
    margins = true_logit - competitors.max(dim=1).values
    result = {}
    for name, mask in (("old", labels < old_count), ("new", labels >= old_count)):
        if not mask.any():
            raise ValueError(f"empty {name} evaluation group")
        result[name] = {
            "count": int(mask.sum()),
            "top1": 100.0 * float((prediction[mask] == labels[mask]).float().mean()),
            "mean_true_margin": float(margins[mask].mean()),
            "predicted_new_rate": 100.0 * float(
                (prediction[mask] >= old_count).float().mean()
            ),
        }
    return result


def paired_prediction_changes(before, after, labels, old_count):
    if before.shape != after.shape or len(labels) != len(before):
        raise ValueError("paired logits and labels must have matching shapes")
    before_pred = before.argmax(dim=1)
    after_pred = after.argmax(dim=1)
    result = {}
    for name, mask in (("old", labels < old_count), ("new", labels >= old_count)):
        result[name] = {
            "count": int(mask.sum()),
            "prediction_changed": int(((before_pred != after_pred) & mask).sum()),
            "correct_to_wrong": int(
                ((before_pred == labels) & (after_pred != labels) & mask).sum()
            ),
            "wrong_to_correct": int(
                ((before_pred != labels) & (after_pred == labels) & mask).sum()
            ),
        }
    return result


def collect_logits(backbone, loader, weights, device):
    labels, scores = [], []
    backbone.eval()
    with torch.no_grad():
        for _, images, targets in loader:
            features = backbone(images.to(device, non_blocking=True))
            logits = F.normalize(features, dim=-1) @ weights.T
            labels.append(targets.cpu())
            scores.append(logits.cpu())
    return torch.cat(labels), torch.cat(scores)


def evaluate_merged(state, loader, weights, device):
    base = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model = build_merged_backbone(base, state).to(device)
    result = collect_logits(model, loader, weights, device)
    del model
    return result


def evaluate_pre_merge(snapshot, loader, weights, device, absorb_mode):
    model = build_model(snapshot, device)
    labels, unbounded = collect_logits(model.vit, loader, weights, device)
    gains = []
    for layer, wrapper in enumerate(model.wrappers):
        for offset, suffix in ((0, "q"), (1, "v")):
            branch = snapshot["branches"][2 * layer + offset]
            capped, gain = capped_current_up(
                branch["down"], branch["current_up"],
                snapshot["current_scale"], absorb_mode,
            )
            with torch.no_grad():
                getattr(wrapper, "b_" + suffix).copy_(capped.to(device))
            gains.append(float(gain))
    _, capped = collect_logits(model.vit, loader, weights, device)
    del model
    return labels, unbounded, capped, gains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.task < 1:
        raise ValueError("task must be at least 1")

    root = args.run_dir / "task_snapshots"
    previous_dir = root / f"task_{args.task - 1:03d}"
    current_dir = root / f"task_{args.task:03d}"
    audit_snapshot(previous_dir)
    audit_snapshot(current_dir)
    config = json.loads((current_dir / "config.json").read_text())
    pre = torch.load(current_dir / "pre_merge.pt", map_location="cpu", weights_only=False)
    if config.get("sa_live_a_history_forward", "shared") == "anchored" and pre.get(
        "history_forward"
    ) != "anchored":
        raise ValueError("legacy anchored snapshot lacks explicit historical factors")
    old = torch.load(previous_dir / "sa_state.pt", map_location="cpu", weights_only=True)
    deployed = torch.load(current_dir / "sa_merged_lora.pt", map_location="cpu", weights_only=True)
    prototypes = torch.load(current_dir / "sa_prototypes.pt", map_location="cpu", weights_only=True)
    if deployed.get("boundary_merge") != "joint_svd":
        raise ValueError("current task is not a joint-SVD deployment")
    if pre["task_id"] != args.task or deployed["task_id"] != args.task:
        raise ValueError("task ID mismatch")
    if sorted(prototypes) != list(range(pre["total_classes"])):
        raise ValueError("incomplete deployed prototype head")
    device = torch.device(args.device)
    seed = config["seed"]
    seed = int(seed[0] if isinstance(seed, list) else seed)
    manager = DataManager(
        config["dataset"], config["shuffle"], seed,
        config["init_cls"], config["increment"], config,
    )
    dataset = manager.get_dataset(
        np.arange(pre["total_classes"]), source="test", mode="test"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    weights = F.normalize(
        torch.stack([prototypes[index] for index in range(pre["total_classes"])])
        .to(device).float(), dim=-1,
    )
    labels, pre_logits, capped_logits, gains = evaluate_pre_merge(
        pre, loader, weights, device, deployed["absorb_mode"]
    )
    aligned = aligned_counterfactual(old, pre, deployed["absorb_mode"])
    aligned_labels, aligned_logits = evaluate_merged(aligned, loader, weights, device)
    deployed_labels, deployed_logits = evaluate_merged(deployed, loader, weights, device)
    if not (torch.equal(labels, aligned_labels) and torch.equal(labels, deployed_labels)):
        raise RuntimeError("evaluation sample order changed between models")
    logits = {
        "pre_train": pre_logits,
        "pre_capped": capped_logits,
        "aligned": aligned_logits,
        "joint_svd": deployed_logits,
    }
    metrics = {
        name: summarize_group_logits(value, labels, pre["known_classes"])
        for name, value in logits.items()
    }
    for name in metrics:
        for group in ("old", "new"):
            metrics[name][group]["delta_vs_pre_capped_pp"] = (
                metrics[name][group]["top1"] - metrics["pre_capped"][group]["top1"]
            )
    result = {
        "protocol": "identical test samples and deployed prototype head; no training or state changes",
        "dataset": config["dataset"],
        "seed": seed,
        "task": args.task,
        "known_classes": pre["known_classes"],
        "total_classes": pre["total_classes"],
        "mean_normcap_gain": sum(gains) / len(gains),
        "absorb_mode": deployed["absorb_mode"],
        "metrics": metrics,
        "aligned_to_joint_flips": paired_prediction_changes(
            aligned_logits, deployed_logits, labels, pre["known_classes"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"labels": labels, "logits": logits}, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
