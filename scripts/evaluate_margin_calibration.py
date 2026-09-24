#!/usr/bin/env python3
"""Offline sequential prototype-head calibration from immutable task snapshots."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone.margin_calibration import (
    class_concentration,
    fit_old_group_bias,
    stratified_two_folds,
    synthetic_old_features,
)
from scripts.counterfactual_g_head_swap import build_backbone, load_task, prototype_matrix, score
from utils.data_manager import DataManager


def collect_features(model, dataset, batch_size, device):
    features, labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for _, images, targets in loader:
            features.append(F.normalize(model(images.to(device)).float().cpu(), dim=1))
            labels.append(targets.cpu().long())
    return torch.cat(features), torch.cat(labels)


def crossfit_bias(features, labels, target_prototypes, concentration, previous_bias,
                  known, seed):
    classes = list(range(known, len(target_prototypes)))
    folds = stratified_two_folds(labels, classes, seed)
    old_synthetic, old_labels = synthetic_old_features(
        target_prototypes[:known], concentration[:known], seed=seed + 1
    )
    logits_all, labels_all = [], []
    for fold in (0, 1):
        fit = folds != fold
        heldout = ~fit
        new_prototypes = []
        for class_id in classes:
            new_prototypes.append(F.normalize(
                features[fit & (labels == class_id)].mean(dim=0), dim=0
            ))
        head = torch.cat((target_prototypes[:known], torch.stack(new_prototypes)))
        old_scores = score(old_synthetic, head)
        new_scores = score(features[heldout], head)
        old_scores[:, :known] += previous_bias[:known]
        new_scores[:, :known] += previous_bias[:known]
        logits_all.extend((old_scores, new_scores))
        labels_all.extend((old_labels, labels[heldout]))
    delta = fit_old_group_bias(
        torch.cat(logits_all), torch.cat(labels_all), known
    )
    return delta, {
        "folds": 2,
        "old_synthetic_per_class": 32,
        "current_holdout_count": int(len(labels)),
        "calibration_labels_source": "current-task train and synthetic old only",
    }


def summarize(logits, labels, known):
    predictions = logits.argmax(dim=1)
    correct = predictions == labels
    record = {"top1": 100 * correct.float().mean().item()}
    if known:
        old = labels < known
        new = ~old
        record.update({
            "old_top1": 100 * correct[old].float().mean().item(),
            "new_top1": 100 * correct[new].float().mean().item(),
            "old_predicted_new_rate": 100 * (
                predictions[old] >= known
            ).float().mean().item(),
            "new_predicted_old_rate": 100 * (
                predictions[new] < known
            ).float().mean().item(),
        })
    return record, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-task", type=int, default=9)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.target_task < 0:
        raise ValueError("target-task must be nonnegative")
    config, _, _, _, _ = load_task(args.run_dir, 0)
    seed = config["seed"]
    seed = int(seed[0] if isinstance(seed, list) else seed)
    manager = DataManager(
        config["dataset"], config["shuffle"], seed,
        config["init_cls"], config["increment"], config,
    )
    if args.target_task >= manager.nb_tasks:
        raise ValueError("target-task exceeds dataset task count")
    device = torch.device(args.device)
    concentration = torch.empty(0)
    bias = torch.empty(0)
    results = []
    predictions = {}
    for task in range(args.target_task + 1):
        task_cfg, merged, prototypes, _, _ = load_task(args.run_dir, task)
        if any(task_cfg[key] != config[key] for key in ("dataset", "seed", "shuffle")):
            raise ValueError("task configuration changed across snapshots")
        known = len(concentration)
        total = len(prototypes)
        expected = sum(manager.get_task_size(index) for index in range(task + 1))
        if total != expected or total <= known:
            raise ValueError("snapshot prototype class count is inconsistent")
        head = prototype_matrix(prototypes)
        model = build_backbone(merged, device)
        train_data = manager.get_dataset(np.arange(known, total), source="train", mode="test")
        train_features, train_labels = collect_features(
            model, train_data, args.batch_size, device
        )
        current_rho = class_concentration(
            train_features, train_labels, head, range(known, total)
        )
        concentration = torch.cat((concentration, current_rho))
        bias = torch.cat((bias, torch.zeros(total - known)))
        calibration = None
        if known:
            delta, calibration = crossfit_bias(
                train_features, train_labels, head, concentration, bias,
                known, seed + 1701 + task * 101
            )
            bias[:known] += delta
        else:
            delta = 0.0
        test_data = manager.get_dataset(np.arange(total), source="test", mode="test")
        test_features, test_labels = collect_features(
            model, test_data, args.batch_size, device
        )
        base_logits = score(test_features, head)
        biased_logits = base_logits + bias
        base, base_pred = summarize(base_logits, test_labels, known)
        calibrated, biased_pred = summarize(biased_logits, test_labels, known)
        results.append({
            "task": task,
            "known_classes": known,
            "total_classes": total,
            "concentration_min": float(concentration.min()),
            "concentration_mean": float(concentration.mean()),
            "delta": delta,
            "bias_min": float(bias.min()),
            "bias_max": float(bias.max()),
            "calibration": calibration,
            "baseline": base,
            "bias_only": calibrated,
        })
        predictions[task] = {
            "labels": test_labels,
            "baseline": base_pred,
            "bias_only": biased_pred,
        }
        print(
            f"task={task} base={base['top1']:.2f} "
            f"bias={calibrated['top1']:.2f} delta={delta:.5f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {
        "version": 1,
        "protocol": "offline sequential snapshots; acquisition-only concentration; "
                    "two-fold current-train calibration; old test for reporting only",
        "dataset": config["dataset"],
        "seed": seed,
        "target_task": args.target_task,
        "deployed_head": "prototype cosine",
        "per_class_extra_scalars": 2,
        "tasks": results,
        "baseline_final": results[-1]["baseline"]["top1"],
        "bias_final": results[-1]["bias_only"]["top1"],
        "baseline_aaa": sum(x["baseline"]["top1"] for x in results) / len(results),
        "bias_aaa": sum(x["bias_only"]["top1"] for x in results) / len(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "concentration": concentration,
        "bias": bias,
        "predictions": predictions,
    }, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in (
        "baseline_final", "bias_final", "baseline_aaa", "bias_aaa"
    )}, indent=2))


if __name__ == "__main__":
    main()
