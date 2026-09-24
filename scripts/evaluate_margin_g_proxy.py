#!/usr/bin/env python3
"""Shadow-test current-data proxies for candidate cumulative-G writes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone.coordinate_stability import (
    apply_residual_orthogonal_transport,
    fit_residual_orthogonal_transport,
)
from backbone.margin_calibration import synthetic_old_features
from scripts.counterfactual_g_head_swap import build_backbone, load_task, prototype_matrix, score
from scripts.evaluate_margin_calibration import collect_features, summarize
from utils.data_manager import DataManager


def interpolate_g(previous, deployed, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("G fraction must lie in [0, 1]")
    if len(previous["shared_a"]) != len(deployed["shared_a"]):
        raise ValueError("LoRA branch count mismatch")
    if not all(torch.equal(a, b) for a, b in zip(
        previous["shared_a"], deployed["shared_a"]
    )):
        raise ValueError("shared A changed: cannot isolate G")
    return {
        "shared_a": previous["shared_a"],
        "merged_b": [
            old + fraction * (new - old)
            for old, new in zip(previous["merged_b"], deployed["merged_b"])
        ],
    }


def candidate_prototypes(previous, old_prototypes, model, current_dataset,
                         previous_features, config, known, total, batch_size, device):
    current_features, labels = collect_features(
        model, current_dataset, batch_size, device
    )
    if not torch.equal(labels, previous_features["labels"]):
        raise ValueError("paired feature targets differ")
    transport = fit_residual_orthogonal_transport(
        previous_features["features"], current_features,
        rank=int(config["sa_coordinate_transport_rank"]),
        identity_reg=float(config["sa_coordinate_transport_reg"]),
        min_validation_gain=float(config["sa_coordinate_transport_min_gain"]),
    )
    updated_old = apply_residual_orthogonal_transport(old_prototypes, transport)
    new_prototypes = {}
    for class_id in range(known, total):
        selected = current_features[labels == class_id]
        if len(selected) == 0:
            raise ValueError("missing current class in train set")
        new_prototypes[class_id] = F.normalize(selected.mean(dim=0), dim=0)
    all_prototypes = {**updated_old, **new_prototypes}
    return prototype_matrix(all_prototypes), current_features, labels, transport


def current_data_old_kl(teacher_features, candidate_features,
                        teacher_head, candidate_head, temperature=0.1):
    teacher = score(teacher_features, teacher_head) / temperature
    student = score(candidate_features, candidate_head) / temperature
    reference = F.softmax(teacher, dim=1)
    per_sample = (reference * (
        F.log_softmax(teacher, dim=1) - F.log_softmax(student, dim=1)
    )).sum(dim=1)
    confidence = score(teacher_features, teacher_head).max(dim=1).values
    high = confidence >= torch.quantile(confidence, 0.75)
    return float(per_sample.mean()), float(per_sample[high].mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--rho-state", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.task < 1:
        raise ValueError("candidate G study requires an incremental task")
    config, previous, old_prototypes, _, _ = load_task(args.run_dir, args.task - 1)
    target_config, target, deployed_prototypes, _, _ = load_task(args.run_dir, args.task)
    if any(config[key] != target_config[key] for key in ("dataset", "seed", "shuffle")):
        raise ValueError("snapshot protocol mismatch")
    rho_state = torch.load(args.rho_state, map_location="cpu", weights_only=True)
    old_count = len(old_prototypes)
    total = len(deployed_prototypes)
    concentration = rho_state["concentration"][:old_count]
    if len(concentration) != old_count or not torch.isfinite(concentration).all():
        raise ValueError("acquisition-only concentration is incomplete")
    seed = config["seed"]
    seed = int(seed[0] if isinstance(seed, list) else seed)
    manager = DataManager(
        config["dataset"], config["shuffle"], seed,
        config["init_cls"], config["increment"], config,
    )
    train = manager.get_dataset(np.arange(old_count, total), source="train", mode="test")
    old_test = manager.get_dataset(np.arange(old_count), source="test", mode="test")
    new_test = manager.get_dataset(np.arange(old_count, total), source="test", mode="test")
    device = torch.device(args.device)
    previous_model = build_backbone(previous, device)
    previous_current, train_labels = collect_features(
        previous_model, train, args.batch_size, device
    )
    previous_features = {"features": previous_current, "labels": train_labels}
    teacher_head = prototype_matrix(old_prototypes)
    del previous_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []
    for fraction in (0.5, 0.75, 1.0):
        state = interpolate_g(previous, target, fraction)
        model = build_backbone(state, device)
        head, current, labels, transport = candidate_prototypes(
            previous, old_prototypes, model, train, previous_features,
            target_config, old_count, total, args.batch_size, device
        )
        if fraction == 1.0:
            saved = prototype_matrix(deployed_prototypes)
            max_diff = float((head - saved).abs().max())
            if max_diff > 1e-4:
                raise RuntimeError(
                    f"additive candidate prototype reconstruction failed: {max_diff}"
                )
        plain_kl, high_kl = current_data_old_kl(
            previous_current, current, teacher_head, head[:old_count]
        )
        pseudo, pseudo_labels = synthetic_old_features(
            head[:old_count], concentration, seed=seed + args.task * 101
        )
        pseudo_logits = score(pseudo, head)
        pseudo_old_to_new = 100 * float(
            (pseudo_logits.argmax(dim=1) >= old_count).float().mean()
        )
        old_features, old_labels = collect_features(
            model, old_test, args.batch_size, device
        )
        new_features, new_labels = collect_features(
            model, new_test, args.batch_size, device
        )
        old_actual, _ = summarize(score(old_features, head), old_labels, old_count)
        new_actual, _ = summarize(score(new_features, head), new_labels, old_count)
        rows.append({
            "alpha": fraction,
            "proxy_current_old_kl": plain_kl,
            "proxy_high_confidence_old_kl": high_kl,
            "proxy_synthetic_old_predicted_new_rate": pseudo_old_to_new,
            "diagnostic_old_top1": old_actual["top1"],
            "diagnostic_new_top1": new_actual["top1"],
            "transport_enabled": transport["enabled"],
            "transport_validation_gain": transport["validation_gain"],
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"task={args.task} alpha={fraction} "
            f"old={old_actual['top1']:.2f} new={new_actual['top1']:.2f} "
            f"kl={plain_kl:.5f}",
            flush=True,
        )
    output = {
        "version": 1,
        "protocol": "current train proxy; old/new test labels diagnostic only",
        "dataset": config["dataset"],
        "seed": seed,
        "task": args.task,
        "old_classes": old_count,
        "total_classes": total,
        "candidate_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
