#!/usr/bin/env python3
"""P1 offline decomposition: prototype-coordinate mismatch vs representation drift.

DIAGNOSTIC ONLY: this script re-reads old training data to recompute
prototypes in the final feature space.  It is NOT part of the rehearsal-free
method and oracle results must never enter the main method table.

Paths (guide ccfa_next_experiment_guide_sd.md section 7.2):
  1. final model + stored training-time prototypes  (current real path)
  2. final model + recomputed prototypes            (oracle, diagnostic only)
  3. task-end model + stored prototypes             (from training logs)
  4. task-end model + recomputed prototypes         (unavailable: Live-A
     persists only the final cumulative O(1) state; per-task model snapshots
     were not saved and P1 forbids retraining)

Outputs per task age: final accuracy, saved/recomputed prototype cosine
distance, intra-class compactness, nearest-error-class margin, old/new
accuracy, and the oracle upper bound on Final / AAA / Forgetting.

Usage:
  python scripts/diagnose_prototype_representation_drift.py \
    --config exps/inr_p1_livea_dual_b_seed1995_nccl.json \
    --artifact INR_P1_LIVEA_DUALB_SEED1995_NCCL \
    --log p3_inr_livea_dual_b_seed1995_nccl.log \
    --device cuda:0
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_sa_sdlora import _MergedQKV, build_merged_backbone
from utils.data_manager import DataManager


NUM_WORKERS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--log", required=True, help="training log for path 3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def build_backbone(config, artifact, device):
    base_model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0
    ).to(device)
    base_model.eval()
    merged_state = torch.load(
        str(Path(artifact) / "sa_merged_lora.pt"),
        map_location=device,
        weights_only=True,
    )
    backbone = build_merged_backbone(base_model, merged_state).to(device)
    backbone.eval()
    return backbone


def extract_features(backbone, loader, device):
    feats, targets = [], []
    with torch.no_grad():
        for _, inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            feats.append(backbone(inputs).detach().cpu())
            targets.append(labels)
    return torch.cat(feats), torch.cat(targets)


def recompute_prototypes(features, targets):
    """Per-class L2-normalized mean prototypes (same convention as training)."""
    sums, counts = {}, {}
    for f, c in zip(features, targets):
        c = int(c.item())
        sums[c] = sums.get(c, 0.0) + f
        counts[c] = counts.get(c, 0) + 1
    return {
        c: F.normalize(sums[c] / counts[c], p=2, dim=0)
        for c in sums
    }


def prototype_logits(features, prototypes, class_ids):
    weight = torch.stack(
        [F.normalize(prototypes[c], p=2, dim=0) for c in class_ids]
    )
    feats = F.normalize(features, p=2, dim=1)
    return feats @ weight.t()


def accuracy(features, targets, prototypes, class_ids):
    if len(class_ids) == 0 or len(features) == 0:
        return float("nan")
    logits = prototype_logits(features, prototypes, class_ids)
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return 100.0 * correct / len(targets)


def accuracy_matrix(features_by_task, targets_by_task, prototypes, boundaries):
    """acc[j][k] = accuracy on task j classes at age k (j <= k)."""
    t = len(boundaries) - 1
    matrix = np.full((t, t), np.nan)
    for k in range(t):
        seen = list(range(boundaries[k + 1]))
        for j in range(k + 1):
            matrix[j, k] = accuracy(
                features_by_task[j], targets_by_task[j], prototypes, seen
            )
    return matrix


def full_curve(matrix):
    """Per-age accuracy over all classes seen so far."""
    curve = []
    for k in range(matrix.shape[1]):
        accs = [matrix[j, k] for j in range(k + 1)]
        curve.append(float(np.mean(accs)))
    return curve


def forgetting(matrix):
    """Mean per-task peak-minus-final forgetting (CNN convention)."""
    t = matrix.shape[0]
    values = []
    for j in range(t):
        row = [matrix[j, k] for k in range(j, t)]
        values.append(float(np.max(row) - row[-1]))
    return float(np.mean(values))


def old_new_accuracy(matrix, t):
    old = [matrix[j, t - 1] for j in range(t - 1)]
    return float(np.mean(old)), float(matrix[t - 1, t - 1])


def parse_task_end_proto_curve(log_path):
    """Per-age prototype top1 at task end (path 3) from [DualHead] lines."""
    text = open(log_path, errors="ignore").read()
    matches = re.findall(
        r"\[DualHead\] task (\d+) mode=proto top1=([\d.]+)", text
    )
    by_task = {}
    for task, top1 in matches:
        by_task[int(task)] = float(top1)
    return [by_task[i] for i in sorted(by_task)]


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
    boundaries = [0]
    for task in range(data_manager.nb_tasks):
        boundaries.append(boundaries[-1] + data_manager.get_task_size(task))
    t = data_manager.nb_tasks

    train_loader = DataLoader(
        data_manager.get_dataset(
            list(range(data_manager.nb_classes)), source="train", mode="test"
        ),
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        data_manager.get_dataset(
            list(range(data_manager.nb_classes)), source="test", mode="test"
        ),
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    backbone = build_backbone(config, cli.artifact, device)
    train_feats, train_targets = extract_features(backbone, train_loader, device)
    test_feats, test_targets = extract_features(backbone, test_loader, device)

    stored = torch.load(
        str(Path(cli.artifact) / "sa_prototypes.pt"),
        map_location="cpu",
        weights_only=True,
    )
    recomputed = recompute_prototypes(train_feats, train_targets)

    train_by_task = [
        train_feats[(train_targets >= boundaries[j]) & (train_targets < boundaries[j + 1])]
        for j in range(t)
    ]
    train_t_by_task = [
        train_targets[(train_targets >= boundaries[j]) & (train_targets < boundaries[j + 1])]
        for j in range(t)
    ]
    test_by_task = [
        test_feats[(test_targets >= boundaries[j]) & (test_targets < boundaries[j + 1])]
        for j in range(t)
    ]
    test_t_by_task = [
        test_targets[(test_targets >= boundaries[j]) & (test_targets < boundaries[j + 1])]
        for j in range(t)
    ]

    print("Dataset: {}  seed: {}  tasks: {}  classes: {}".format(
        config["dataset"], seed, t, data_manager.nb_classes
    ))
    print("Boundaries: {}".format(boundaries))

    stored_matrix = accuracy_matrix(
        test_by_task, test_t_by_task, stored, boundaries
    )
    recomputed_matrix = accuracy_matrix(
        test_by_task, test_t_by_task, recomputed, boundaries
    )

    stored_curve = full_curve(stored_matrix)
    recomputed_curve = full_curve(recomputed_matrix)
    stored_old, stored_new = old_new_accuracy(stored_matrix, t)
    recomputed_old, recomputed_new = old_new_accuracy(recomputed_matrix, t)

    print("\n[P1 path 1] final model + stored prototypes")
    print("  per-age final accuracy: {}".format(
        " ".join("{:.2f}".format(v) for v in stored_curve)
    ))
    print("  Final={:.2f}  AAA={:.2f}  Forgetting={:.2f}  old={:.2f}  new={:.2f}".format(
        stored_curve[-1], float(np.mean(stored_curve)), forgetting(stored_matrix),
        stored_old, stored_new,
    ))

    print("\n[P1 path 2] final model + recomputed prototypes (ORACLE, diagnostic only)")
    print("  per-age final accuracy: {}".format(
        " ".join("{:.2f}".format(v) for v in recomputed_curve)
    ))
    print("  Final={:.2f}  AAA={:.2f}  Forgetting={:.2f}  old={:.2f}  new={:.2f}".format(
        recomputed_curve[-1], float(np.mean(recomputed_curve)), forgetting(recomputed_matrix),
        recomputed_old, recomputed_new,
    ))
    print("  oracle deltas: Final {:+.2f}  AAA {:+.2f}  Forgetting {:+.2f}  old {:+.2f}  new {:+.2f}".format(
        recomputed_curve[-1] - stored_curve[-1],
        float(np.mean(recomputed_curve)) - float(np.mean(stored_curve)),
        forgetting(recomputed_matrix) - forgetting(stored_matrix),
        recomputed_old - stored_old,
        recomputed_new - stored_new,
    ))

    common = sorted(set(stored) & set(recomputed))
    cosines = [float((stored[c] * recomputed[c]).sum()) for c in common]
    per_age_cosine = []
    for k in range(t):
        age_cos = [
            float((stored[c] * recomputed[c]).sum())
            for c in common if c < boundaries[k + 1]
        ]
        per_age_cosine.append(float(np.mean(age_cos)))
    print("\n[P1] stored vs recomputed prototype cosine (final model)")
    print("  mean={:.4f} min={:.4f} max={:.4f}  per-age: {}".format(
        float(np.mean(cosines)), float(np.min(cosines)), float(np.max(cosines)),
        " ".join("{:.4f}".format(v) for v in per_age_cosine),
    ))

    # Intra-class compactness and nearest-error-class margin in the final space.
    compactness, margins = {}, {}
    for c in common:
        mask = train_targets == c
        if mask.sum() == 0:
            continue
        feats = F.normalize(train_feats[mask], p=2, dim=1)
        own = F.normalize(recomputed[c], p=2, dim=0)
        compactness[c] = float((feats @ own).mean())
        test_mask = test_targets == c
        if test_mask.sum() == 0:
            continue
        tfeats = F.normalize(test_feats[test_mask], p=2, dim=1)
        sims = tfeats @ torch.stack(
            [F.normalize(recomputed[o], p=2, dim=0) for o in common]
        ).t()
        own_idx = common.index(c)
        others = sims.clone()
        others[:, own_idx] = -1.0
        margins[c] = float((sims[:, own_idx] - others.max(dim=1).values).mean())
    print("\n[P1] intra-class compactness (recomputed protos, train): "
          "mean={:.4f} min={:.4f} max={:.4f}".format(
              float(np.mean(list(compactness.values()))),
              float(np.min(list(compactness.values()))),
              float(np.max(list(compactness.values()))),
          ))
    print("[P1] nearest-error-class margin (recomputed protos, test): "
          "mean={:.4f} min={:.4f} max={:.4f}".format(
              float(np.mean(list(margins.values()))),
              float(np.min(list(margins.values()))),
              float(np.max(list(margins.values()))),
          ))

    task_end = parse_task_end_proto_curve(cli.log)
    print("\n[P1 path 3] task-end model + stored prototypes (from training log)")
    print("  per-age prototype top1: {}".format(
        " ".join("{:.2f}".format(v) for v in task_end)
    ))
    if len(task_end) == len(stored_curve):
        drift = [a - b for a, b in zip(task_end, stored_curve)]
        print("  task-end minus final-model per-age deltas: {}".format(
            " ".join("{:+.2f}".format(v) for v in drift)
        ))
        print("  mean delta = {:+.2f} (positive => representation drift "
              "between task end and final model)".format(float(np.mean(drift))))
    else:
        print("  WARNING: task-end curve length {} != final per-age {}".format(
            len(task_end), len(stored_curve)
        ))

    print("\n[P1 path 4] task-end model + recomputed prototypes: UNAVAILABLE")
    print("  Live-A persists only the final cumulative O(1) state; per-task "
          "model snapshots were not saved and P1 forbids retraining.")

    print("\n[P1 decision inputs]")
    print("  oracle old-class improvement   = {:+.2f} (threshold >= 1.00)".format(
        recomputed_old - stored_old
    ))
    print("  oracle Forgetting improvement  = {:+.2f} (threshold >= 0.75)".format(
        forgetting(stored_matrix) - forgetting(recomputed_matrix)
    ))
    task_delta = (
        float(np.mean(task_end)) - float(np.mean(stored_curve))
        if len(task_end) == len(stored_curve) else float("nan")
    )
    print("  task-end vs final (stored protos) = {:+.2f}".format(task_delta))


if __name__ == "__main__":
    main()
