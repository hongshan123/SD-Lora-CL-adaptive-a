#!/usr/bin/env python3
"""Offline, train-data-only diagnostic for fixed-rank LoRA subspace allocation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


WEIGHTS = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)


def normalized_residual(operator: torch.Tensor, basis: torch.Tensor) -> float:
    total = operator.square().sum()
    if total <= 0:
        raise ValueError("operator has zero energy")
    captured = (operator @ basis.T).square().sum()
    return float((1 - captured / total).clamp(0, 1))


def spectral_basis(
    historical: torch.Tensor,
    demand: torch.Tensor,
    rank: int,
    current_weight: float,
) -> torch.Tensor:
    """Return top-r right singular vectors of unit-energy weighted operators."""
    if current_weight < 0 or not math.isfinite(current_weight):
        raise ValueError("current_weight must be finite and nonnegative")
    historical = historical / torch.linalg.vector_norm(historical)
    demand = demand / torch.linalg.vector_norm(demand)
    stacked = torch.cat((historical, math.sqrt(current_weight) * demand), dim=0)
    return torch.linalg.svd(stacked, full_matrices=False).Vh[:rank]


def analyze_branch(
    down: torch.Tensor,
    up: torch.Tensor,
    gradient: torch.Tensor,
    holdout_gradient: torch.Tensor | None = None,
    weights=WEIGHTS,
    budget=0.05,
    sketch_rank=32,
) -> dict:
    """Compare old and new row spaces without modifying any model tensor."""
    down = down.detach().double()
    up = up.detach().double()
    gradient = gradient.detach().double()
    if holdout_gradient is not None:
        holdout_gradient = holdout_gradient.detach().double()
    rank, dim = down.shape
    if up.shape[1] != rank or gradient.shape != (up.shape[0], dim):
        raise ValueError("incompatible operator shapes")
    if not all(bool(torch.isfinite(x).all()) for x in (down, up, gradient)):
        raise ValueError("non-finite operator")
    q, _ = torch.linalg.qr(down.T, mode="reduced")
    old_basis = q.T
    historical = up @ down
    old_current_loss = normalized_residual(gradient, old_basis)
    old_holdout_loss = (
        normalized_residual(holdout_gradient, old_basis)
        if holdout_gradient is not None else None
    )
    alignment_scale = (
        torch.linalg.vector_norm(gradient) * torch.linalg.vector_norm(holdout_gradient)
        if holdout_gradient is not None else None
    )

    def descent_alignment(basis):
        if holdout_gradient is None:
            return None
        return float(
            ((holdout_gradient @ basis.T) * (gradient @ basis.T)).sum()
            / alignment_scale
        )

    old_descent_alignment = descent_alignment(old_basis)

    # The small factor has the same right Gram matrix as up @ down.
    gram = up.T @ up
    eigval, eigvec = torch.linalg.eigh(gram)
    root = (eigvec * eigval.clamp_min(0).sqrt().unsqueeze(0)) @ eigvec.T
    history_factor = root @ down

    # A fixed-rank sketch changes candidate construction, not risk evaluation.
    qrank = min(sketch_rank + 8, dim)
    u, singular, v = torch.svd_lowrank(gradient, q=qrank, niter=2)
    demand_factor = singular[:sketch_rank, None] * v[:, :sketch_rank].T
    retained = float(
        demand_factor.square().sum() / gradient.square().sum()
    )
    candidates = []
    for weight in weights:
        basis = (
            old_basis
            if weight == 0
            else spectral_basis(history_factor, demand_factor, rank, weight)
        )
        history_risk = normalized_residual(historical, basis)
        current_risk = normalized_residual(gradient, basis)
        heldout_alignment = descent_alignment(basis)
        candidates.append(
            {
                "current_weight": weight,
                "history_risk": history_risk,
                "current_risk": current_risk,
                "current_coverage_gain": old_current_loss - current_risk,
                "holdout_coverage_gain": (
                    old_holdout_loss - normalized_residual(holdout_gradient, basis)
                    if holdout_gradient is not None else None
                ),
                "holdout_descent_alignment_gain": (
                    heldout_alignment - old_descent_alignment
                    if holdout_gradient is not None else None
                ),
            }
        )
    feasible = [row for row in candidates if row["history_risk"] <= budget + 1e-8]
    best = min(feasible, key=lambda row: row["current_risk"])
    return {
        "rank": rank,
        "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "old_offspace_fraction": old_current_loss,
        "old_holdout_offspace_fraction": old_holdout_loss,
        "old_descent_alignment": old_descent_alignment,
        "train_holdout_cosine": (
            float((gradient * holdout_gradient).sum() / alignment_scale)
            if holdout_gradient is not None else None
        ),
        "sketch_energy_retained": retained,
        "best_feasible": best,
        "frontier": candidates,
    }


class HistoricalQKV(nn.Module):
    def __init__(self, qkv, down_q, up_q, down_v, up_v, sample_tokens):
        super().__init__()
        self.qkv = qkv
        self.register_buffer("down_q", down_q)
        self.register_buffer("up_q", up_q)
        self.register_buffer("down_v", down_v)
        self.register_buffer("up_v", up_v)
        self.sample_tokens = sample_tokens
        self.capture = False
        self.register_buffer("gradient_q", torch.zeros_like(up_q @ down_q))
        self.register_buffer("gradient_v", torch.zeros_like(up_v @ down_v))

    def forward(self, x):
        width = x.shape[-1]
        raw = self.qkv(x)
        q = raw[..., :width] + F.linear(F.linear(x, self.down_q), self.up_q)
        k = raw[..., width : 2 * width]
        v = raw[..., -width:] + F.linear(F.linear(x, self.down_v), self.up_v)
        output = torch.cat((q, k, v), dim=-1)
        if self.capture:
            flat_x = x.detach().reshape(-1, width)
            indices = torch.linspace(
                0, flat_x.shape[0] - 1,
                steps=min(self.sample_tokens, flat_x.shape[0]),
                device=x.device,
            ).long()
            selected_x = flat_x[indices]
            multiplier = flat_x.shape[0] / len(indices)

            def record(grad):
                sampled = grad.detach().reshape(-1, 3 * width)[indices]
                self.gradient_q.add_(
                    multiplier * sampled[:, :width].T @ selected_x
                )
                self.gradient_v.add_(
                    multiplier * sampled[:, -width:].T @ selected_x
                )

            output.requires_grad_(True)
            output.register_hook(record)
        return output


def balanced_indices(dataset, first_class, class_count, per_class, seed):
    rng = np.random.default_rng(seed)
    by_class = {}
    for index, label in enumerate(dataset.labels):
        label = int(label)
        if first_class <= label < first_class + class_count:
            by_class.setdefault(label, []).append(index)
    prototype, gradient, holdout = [], [], []
    for label in range(first_class, first_class + class_count):
        indices = np.asarray(by_class.get(label, []))
        if len(indices) < 3:
            raise ValueError(f"class {label} has fewer than three train samples")
        rng.shuffle(indices)
        n = min(per_class, len(indices) // 3)
        prototype.extend(indices[:n].tolist())
        gradient.extend(indices[n : 2 * n].tolist())
        holdout.extend(indices[2 * n : 3 * n].tolist())
    return prototype, gradient, holdout


def feature_batch(model, images):
    output = model(images)
    return output["features"] if isinstance(output, dict) else output


def run(args):
    import timm
    from utils.data_manager import DataManager

    config = json.loads(Path(args.config).read_text())
    state = torch.load(args.state, map_location="cpu", weights_only=False)
    if state.get("merge_mode") != "sensitivity_budgeted_g":
        raise ValueError("expected a persisted fixed-P SBGC checkpoint")
    completed_tasks = int(state["task_id"])
    seed = int(config["seed"][0])
    torch.manual_seed(seed)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    manager = DataManager(
        config["dataset"], config["shuffle"], seed,
        config["init_cls"], config["increment"], config,
    )
    first_class = sum(manager.get_task_size(t) for t in range(completed_tasks))
    class_count = manager.get_task_size(completed_tasks)
    dataset = manager.get_dataset(
        np.arange(first_class, first_class + class_count),
        source="train", mode="test",
    )
    proto_indices, gradient_indices, holdout_indices = balanced_indices(
        dataset, first_class, class_count, args.per_class, seed + 91,
    )

    model = timm.create_model(
        config["backbone_type"], pretrained=True, num_classes=0
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrappers = []
    for layer, block in enumerate(model.blocks):
        offset = 2 * layer
        wrapper = HistoricalQKV(
            block.attn.qkv,
            state["projection_down"][offset].float(),
            state["unified_up"][offset].float(),
            state["projection_down"][offset + 1].float(),
            state["unified_up"][offset + 1].float(),
            args.sample_tokens,
        )
        block.attn.qkv = wrapper
        wrappers.append(wrapper)
    model.to(device)

    loader_kwargs = {"batch_size": args.batch_size, "shuffle": False, "num_workers": 2}
    prototypes = torch.zeros(class_count, model.num_features, device=device)
    counts = torch.zeros(class_count, device=device)
    with torch.no_grad():
        for _, images, labels in DataLoader(Subset(dataset, proto_indices), **loader_kwargs):
            features = F.normalize(feature_batch(model, images.to(device)), dim=-1)
            local_labels = labels.to(device) - first_class
            prototypes.index_add_(0, local_labels, features)
            counts.index_add_(0, local_labels, torch.ones_like(local_labels, dtype=counts.dtype))
    prototypes = F.normalize(prototypes / counts[:, None], dim=-1).detach()

    def collect_gradients(indices):
        for wrapper in wrappers:
            wrapper.gradient_q.zero_()
            wrapper.gradient_v.zero_()
            wrapper.capture = True
        losses = []
        for _, images, labels in DataLoader(Subset(dataset, indices), **loader_kwargs):
            features = F.normalize(feature_batch(model, images.to(device)), dim=-1)
            logits = 20 * features @ prototypes.T
            loss = F.cross_entropy(logits, labels.to(device) - first_class)
            losses.append(float(loss.detach()))
            loss.backward()
        for wrapper in wrappers:
            wrapper.capture = False
        gradients = [
            (wrapper.gradient_q.clone(), wrapper.gradient_v.clone())
            for wrapper in wrappers
        ]
        return gradients, float(np.mean(losses))

    design_gradients, design_loss = collect_gradients(gradient_indices)
    holdout_gradients, holdout_loss = collect_gradients(holdout_indices)

    results = []
    for layer, (design_pair, holdout_pair) in enumerate(zip(design_gradients, holdout_gradients)):
        for branch_index, suffix in enumerate(("q", "v")):
            offset = 2 * layer + (suffix == "v")
            result = analyze_branch(
                state["projection_down"][offset].to(device),
                state["unified_up"][offset].to(device),
                design_pair[branch_index],
                holdout_pair[branch_index],
                budget=args.budget,
                sketch_rank=args.sketch_rank,
            )
            result.update({"layer": layer, "branch": suffix})
            results.append(result)
    offspace = [row["old_offspace_fraction"] for row in results]
    gain = [row["best_feasible"]["current_coverage_gain"] for row in results]
    holdout_gain = [row["best_feasible"]["holdout_coverage_gain"] for row in results]
    descent_gain = [row["best_feasible"]["holdout_descent_alignment_gain"] for row in results]
    output = {
        "protocol": "train-only frozen-feature prototype CE gradient; no parameter updates",
        "dataset": config["dataset"],
        "seed": seed,
        "completed_tasks": completed_tasks,
        "next_task": completed_tasks,
        "first_class": first_class,
        "class_count": class_count,
        "prototype_samples": len(proto_indices),
        "gradient_samples": len(gradient_indices),
        "holdout_samples": len(holdout_indices),
        "design_loss_mean": design_loss,
        "holdout_loss_mean": holdout_loss,
        "budget": args.budget,
        "median_offspace_fraction": float(np.median(offspace)),
        "median_feasible_current_coverage_gain": float(np.median(gain)),
        "median_holdout_coverage_gain": float(np.median(holdout_gain)),
        "median_holdout_descent_alignment_gain": float(np.median(descent_gain)),
        "branches_with_positive_holdout_gain": sum(value > 1e-3 for value in holdout_gain),
        "branches_with_positive_descent_gain": sum(value > 1e-3 for value in descent_gain),
        "branches_with_positive_gain": sum(value > 1e-3 for value in gain),
        "branches": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "branches"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--sample-tokens", type=int, default=256)
    parser.add_argument("--sketch-rank", type=int, default=32)
    parser.add_argument("--budget", type=float, default=0.05)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
