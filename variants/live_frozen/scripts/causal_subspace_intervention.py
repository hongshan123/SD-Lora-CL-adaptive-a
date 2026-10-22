#!/usr/bin/env python3
"""One-task, matched CUB intervention on the shared LoRA input subspace.

The historical effective operator is frozen while the new task trains. At
deployment it is least-squares projected into the arm's final row space.
This is a controlled mechanism test, not a replacement CIL training method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_subspace_allocation import (  # noqa: E402
    HistoricalQKV,
    WEIGHTS,
    balanced_indices,
    normalized_residual,
    spectral_basis,
)
from utils.data_manager import DataManager  # noqa: E402


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_rows(a):
    return torch.linalg.qr(a.double().T, mode="reduced").Q.T


def recoverability(old_down, old_up, new_down):
    old_operator = old_up.double() @ old_down.double()
    basis = normalized_rows(new_down)
    return normalized_residual(old_operator, basis)


def choose_spectral_down(old_down, old_up, gradient, budget=0.05, sketch_rank=32):
    """Select by design-gradient coverage, with an exact historical risk check."""
    old_down = old_down.detach().double()
    old_up = old_up.detach().double()
    gradient = gradient.detach().double()
    rank = old_down.shape[0]
    if not torch.isfinite(gradient).all() or gradient.square().sum() <= 0:
        raise ValueError("effective-weight gradient must be finite and nonzero")
    gram = old_up.T @ old_up
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    gram_root = (
        eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)
    ) @ eigenvectors.T
    history_factor = gram_root @ old_down
    sketch_width = min(sketch_rank + 8, gradient.shape[1])
    _, singular, right = torch.svd_lowrank(gradient, q=sketch_width, niter=2)
    demand_factor = singular[:sketch_rank, None] * right[:, :sketch_rank].T
    historical = old_up @ old_down
    candidates = []
    for weight in WEIGHTS:
        basis = (
            normalized_rows(old_down)
            if weight == 0
            else spectral_basis(history_factor, demand_factor, rank, weight)
        )
        historical_risk = normalized_residual(historical, basis)
        current_risk = normalized_residual(gradient, basis)
        candidates.append((historical_risk, current_risk, weight, basis))
    feasible = [item for item in candidates if item[0] <= budget + 1e-8]
    if not feasible:
        raise RuntimeError("even the old basis violates the historical budget")
    selected = min(feasible, key=lambda item: item[1])
    return selected[3].float(), {
        "historical_risk": selected[0],
        "current_risk": selected[1],
        "current_weight": selected[2],
        "old_current_risk": normalized_residual(gradient, normalized_rows(old_down)),
        "sketch_energy_retained": float(demand_factor.square().sum() / gradient.square().sum()),
    }


class TaskQKV(nn.Module):
    def __init__(self, qkv, old_down_q, old_up_q, old_down_v, old_up_v, scale_getter, live):
        super().__init__()
        self.qkv = qkv
        self.register_buffer("old_down_q", old_down_q.detach().clone())
        self.register_buffer("old_down_v", old_down_v.detach().clone())
        self.register_buffer("old_up_q", old_up_q.detach().clone())
        self.register_buffer("old_up_v", old_up_v.detach().clone())
        self.a_q = nn.Parameter(old_down_q.detach().clone(), requires_grad=live)
        self.a_v = nn.Parameter(old_down_v.detach().clone(), requires_grad=live)
        self.b_q = nn.Parameter(torch.zeros_like(old_up_q))
        self.b_v = nn.Parameter(torch.zeros_like(old_up_v))
        self.scale_getter = scale_getter
        self.deployed = False
        self.current_enabled = True
        self.register_buffer("deployment_down_q", old_down_q.detach().clone())
        self.register_buffer("deployment_down_v", old_down_v.detach().clone())
        self.register_buffer("deployment_up_q", old_up_q.detach().clone())
        self.register_buffer("deployment_up_v", old_up_v.detach().clone())

    def forward(self, x):
        width = x.shape[-1]
        raw = self.qkv(x)
        if self.deployed:
            history_q = F.linear(F.linear(x, self.deployment_down_q), self.deployment_up_q)
            history_v = F.linear(F.linear(x, self.deployment_down_v), self.deployment_up_v)
        else:
            history_q = F.linear(F.linear(x, self.old_down_q), self.old_up_q)
            history_v = F.linear(F.linear(x, self.old_down_v), self.old_up_v)
        if self.current_enabled:
            scale = self.scale_getter()
            current_q = scale * F.linear(F.linear(x, self.a_q), self.b_q)
            current_v = scale * F.linear(F.linear(x, self.a_v), self.b_v)
        else:
            current_q = torch.zeros_like(history_q)
            current_v = torch.zeros_like(history_v)
        return torch.cat(
            (raw[..., :width] + history_q + current_q,
             raw[..., width : 2 * width],
             raw[..., -width:] + history_v + current_v),
            dim=-1,
        )

    @torch.no_grad()
    def deploy(self):
        diagnostics = []
        for suffix in ("q", "v"):
            old_down = getattr(self, f"old_down_{suffix}")
            old_up = getattr(self, f"old_up_{suffix}")
            basis = normalized_rows(getattr(self, f"a_{suffix}")).to(old_down)
            aligned_up = (old_up @ old_down) @ basis.T
            getattr(self, f"deployment_down_{suffix}").copy_(basis)
            getattr(self, f"deployment_up_{suffix}").copy_(aligned_up)
            diagnostics.append(recoverability(old_down, old_up, basis))
        self.deployed = True
        return diagnostics


class TaskModel(nn.Module):
    def __init__(self, state, old_fc_weight, old_fc_bias, new_classes, live):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        for parameter in self.vit.parameters():
            parameter.requires_grad_(False)
        self.scale = nn.Parameter(torch.tensor(0.8))
        self.wrappers = []
        for layer, block in enumerate(self.vit.blocks):
            offset = 2 * layer
            wrapper = TaskQKV(
                block.attn.qkv,
                state["projection_down"][offset].float(),
                state["unified_up"][offset].float(),
                state["projection_down"][offset + 1].float(),
                state["unified_up"][offset + 1].float(),
                lambda: self.scale,
                live,
            )
            block.attn.qkv = wrapper
            self.wrappers.append(wrapper)
        old_classes = int(old_fc_weight.shape[0])
        self.fc = nn.Linear(self.vit.num_features, old_classes + new_classes)
        with torch.no_grad():
            self.fc.weight[:old_classes].copy_(old_fc_weight.float())
            self.fc.bias[:old_classes].copy_(old_fc_bias.float())

    def forward(self, images):
        features = self.vit(images)
        return self.fc(features), features

    def set_basis(self, bases):
        with torch.no_grad():
            for layer, wrapper in enumerate(self.wrappers):
                wrapper.a_q.copy_(bases[2 * layer].to(wrapper.a_q))
                wrapper.a_v.copy_(bases[2 * layer + 1].to(wrapper.a_v))

    def deploy(self):
        return [risk for wrapper in self.wrappers for risk in wrapper.deploy()]


def make_data_manager(config):
    seed = int(config["seed"][0])
    return DataManager(
        config["dataset"], config["shuffle"], seed,
        config["init_cls"], config["increment"], config,
    )


def current_task_dataset(manager, task_index, mode):
    first = sum(manager.get_task_size(t) for t in range(task_index))
    last = first + manager.get_task_size(task_index)
    return manager.get_dataset(np.arange(first, last), source="train", mode=mode)


def split_current_calibration(dataset, first_class, class_count, seed):
    rng = np.random.default_rng(seed)
    prototype, calibration = [], []
    labels = np.asarray(dataset.labels)
    for label in range(first_class, first_class + class_count):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 5:
            raise ValueError(f"class {label} has fewer than five training samples")
        rng.shuffle(indices)
        holdout = max(1, len(indices) // 5)
        calibration.extend(indices[:holdout].tolist())
        prototype.extend(indices[holdout:].tolist())
    return prototype, calibration


def gate_score(history_features, full_features, old_prototypes, new_prototypes):
    old_similarity = F.normalize(history_features, dim=-1) @ F.normalize(old_prototypes, dim=-1).T
    new_similarity = F.normalize(full_features, dim=-1) @ F.normalize(new_prototypes, dim=-1).T
    return new_similarity.max(dim=1).values - old_similarity.max(dim=1).values


def calibrate_new_recall(new_scores, target_recall):
    if not 0 < target_recall <= 1 or len(new_scores) == 0:
        raise ValueError("target recall must be in (0, 1] and calibration cannot be empty")
    ordered = torch.sort(new_scores.detach().float().cpu()).values
    return float(ordered[int(np.floor((1 - target_recall) * len(ordered)))])


def gate_metrics(old, new, threshold):
    old_to_new = old["score"] >= threshold
    new_to_new = new["score"] >= threshold
    old_history = old["history_correct"].float()
    old_full = old["full_correct"].float()
    new_history = new["history_correct"].float()
    new_full = new["full_correct"].float()
    old_gated = torch.where(old_to_new, old_full, old_history)
    new_gated = torch.where(new_to_new, new_full, new_history)
    old_count, new_count = len(old_gated), len(new_gated)
    if not old_count or not new_count:
        raise ValueError("both evaluation groups must be nonempty")
    total = old_count + new_count
    old_scores = np.sort(old["score"].detach().cpu().numpy())
    new_scores = new["score"].detach().cpu().numpy()
    below = np.searchsorted(old_scores, new_scores, side="left")
    equal = np.searchsorted(old_scores, new_scores, side="right") - below
    auc = float(np.mean((below + 0.5 * equal) / old_count))
    weighted = lambda a, b: float(100 * (a.sum() + b.sum()) / total)
    return {
        "threshold": float(threshold),
        "gate_auc": auc,
        "routing_old_to_new_rate": float(old_to_new.float().mean()),
        "routing_new_to_old_rate": float((~new_to_new).float().mean()),
        "old_gated_top1": float(100 * old_gated.mean()),
        "new_gated_top1": float(100 * new_gated.mean()),
        "full_top1": weighted(old_full, new_full),
        "history_only_top1": weighted(old_history, new_history),
        "oracle_top1": weighted(old_history, new_full),
        "gated_top1": weighted(old_gated, new_gated),
        "oracle_gap_pp": weighted(old_history, new_full) - weighted(old_gated, new_gated),
        "old_misroute_cost_pp": float(100 * ((old_history - old_full) * old_to_new).mean()),
        "new_misroute_cost_pp": float(100 * ((new_full - new_history) * (~new_to_new)).mean()),
        "old_samples": old_count,
        "new_samples": new_count,
    }


def collect_gate_predictions(model, loader, old_prototypes, new_prototypes, device):
    weights = F.normalize(torch.cat((old_prototypes.to(device), new_prototypes), dim=0), dim=-1)
    old_count = len(old_prototypes)
    rows = {name: [] for name in ("score", "history_correct", "full_correct")}
    model.eval()
    try:
        with torch.no_grad():
            for _, images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                for wrapper in model.wrappers:
                    wrapper.current_enabled = True
                _, full_features = model(images)
                for wrapper in model.wrappers:
                    wrapper.current_enabled = False
                _, history_features = model(images)
                full_logits = F.normalize(full_features, dim=-1) @ weights.T
                history_logits = F.normalize(history_features, dim=-1) @ weights.T
                rows["score"].append(gate_score(
                    history_features, full_features, weights[:old_count], weights[old_count:]
                ).cpu())
                rows["full_correct"].append((full_logits.argmax(dim=1) == labels).cpu())
                rows["history_correct"].append((history_logits.argmax(dim=1) == labels).cpu())
    finally:
        for wrapper in model.wrappers:
            wrapper.current_enabled = True
    return {name: torch.cat(chunks) for name, chunks in rows.items()}


def evaluate_gating(model, manager, task_index, first_class, class_count,
                    old_prototypes, old_loader, new_loader, seed, device):
    current = current_task_dataset(manager, task_index, "test")
    prototype_indices, calibration_indices = split_current_calibration(
        current, first_class, class_count, seed + 1701
    )
    prototype_loader = DataLoader(Subset(current, prototype_indices), batch_size=64, num_workers=2)
    calibration_loader = DataLoader(Subset(current, calibration_indices), batch_size=64, num_workers=2)
    new_prototypes = prototypes_from_current(
        model, prototype_loader, first_class, class_count, device
    )
    calibration = collect_gate_predictions(
        model, calibration_loader, old_prototypes, new_prototypes, device
    )
    thresholds = {
        str(recall): calibrate_new_recall(calibration["score"], recall)
        for recall in (0.90, 0.95, 0.99)
    }
    old = collect_gate_predictions(model, old_loader, old_prototypes, new_prototypes, device)
    new = collect_gate_predictions(model, new_loader, old_prototypes, new_prototypes, device)
    return {
        "protocol": "current-task train only; deterministic test transform; 80/20 per-class split",
        "prototype_train_samples": len(prototype_indices),
        "calibration_train_samples": len(calibration_indices),
        "calibration_seed": seed + 1701,
        "score": "max_new_cosine(full) - max_old_cosine(history_only)",
        "calibration_score_quantiles": np.quantile(
            calibration["score"].numpy(), (0, 0.05, 0.5, 0.95, 1)
        ).tolist(),
        "old_test_score_quantiles": np.quantile(old["score"].numpy(), (0, 0.05, 0.5, 0.95, 1)).tolist(),
        "new_test_score_quantiles": np.quantile(new["score"].numpy(), (0, 0.05, 0.5, 0.95, 1)).tolist(),
        "target_new_recall": {
            recall: {
                **gate_metrics(old, new, threshold),
                "calibration_new_recall": float((calibration["score"] >= threshold).float().mean()),
            }
            for recall, threshold in thresholds.items()
        },
    }


def prototypes_from_current(model, loader, first_class, new_classes, device):
    sums = torch.zeros(new_classes, 768, device=device)
    counts = torch.zeros(new_classes, device=device)
    model.eval()
    with torch.no_grad():
        for _, images, labels in loader:
            _, features = model(images.to(device, non_blocking=True))
            features = F.normalize(features, dim=-1)
            indices = labels.to(device) - first_class
            sums.index_add_(0, indices, features)
            counts.index_add_(0, indices, torch.ones_like(indices, dtype=counts.dtype))
    if not bool((counts > 0).all()):
        raise RuntimeError("missing current-task class while building prototypes")
    return F.normalize(sums / counts[:, None], dim=-1)


def evaluate_groups(model, old_prototypes, new_prototypes, old_loader, new_loader, device):
    weights = F.normalize(torch.cat((old_prototypes.to(device), new_prototypes), dim=0), dim=-1)
    model.eval()
    output = {}
    with torch.no_grad():
        for name, loader in (("old", old_loader), ("new", new_loader)):
            total = 0
            correct = 0
            restricted_correct = 0
            loss_sum = 0.0
            fc_correct = 0
            fc_loss_sum = 0.0
            for _, images, labels in loader:
                fc_logits, features = model(images.to(device, non_blocking=True))
                logits = F.normalize(features, dim=-1) @ weights.T
                labels = labels.to(device)
                loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
                fc_loss_sum += float(F.cross_entropy(fc_logits, labels, reduction="sum"))
                correct += int((logits.argmax(dim=1) == labels).sum())
                if name == "old":
                    restricted_logits = logits[:, :len(old_prototypes)]
                    restricted_labels = labels
                else:
                    restricted_logits = logits[:, len(old_prototypes):]
                    restricted_labels = labels - len(old_prototypes)
                restricted_correct += int(
                    (restricted_logits.argmax(dim=1) == restricted_labels).sum()
                )
                fc_correct += int((fc_logits.argmax(dim=1) == labels).sum())
                total += len(labels)
            output[name] = {
                "prototype_top1": 100 * correct / total,
                "restricted_prototype_top1": 100 * restricted_correct / total,
                "prototype_ce": loss_sum / total,
                "fc_top1": 100 * fc_correct / total,
                "fc_ce": fc_loss_sum / total,
                "samples": total,
            }
    return output


def estimate_demand(config, source, output, device, prototype_per_class=8, demand_per_class=8):
    state_path = source / "sa_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    task_index = int(state["task_id"])
    manager = make_data_manager(config)
    first_class = sum(manager.get_task_size(t) for t in range(task_index))
    class_count = manager.get_task_size(task_index)
    dataset = current_task_dataset(manager, task_index, "test")
    selected_per_class = min(prototype_per_class, demand_per_class)
    prototype_indices, design_indices, _ = balanced_indices(
        dataset, first_class, class_count, selected_per_class,
        int(config["seed"][0]) + 91,
    )
    design_size = len(design_indices) // class_count
    design_indices = np.array(design_indices).reshape(class_count, design_size).T.reshape(-1).tolist()
    torch.manual_seed(int(config["seed"][0]))
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).eval()
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
            256,
        )
        block.attn.qkv = wrapper
        wrappers.append(wrapper)
    model.to(device)
    from torch.utils.data import Subset
    prototype_loader = DataLoader(Subset(dataset, prototype_indices), batch_size=16, num_workers=2)
    sums = torch.zeros(class_count, 768, device=device)
    counts = torch.zeros(class_count, device=device)
    with torch.no_grad():
        for _, images, labels in prototype_loader:
            features = F.normalize(model(images.to(device)), dim=-1)
            local = labels.to(device) - first_class
            sums.index_add_(0, local, features)
            counts.index_add_(0, local, torch.ones_like(local, dtype=counts.dtype))
    prototypes = F.normalize(sums / counts[:, None], dim=-1).detach()
    for wrapper in wrappers:
        wrapper.capture = True
    raw_gradients = None
    demand_loader = DataLoader(Subset(dataset, design_indices), batch_size=32, num_workers=2)
    for batch_index, (_, images, labels) in enumerate(demand_loader):
        features = F.normalize(model(images.to(device)), dim=-1)
        logits = 20 * features @ prototypes.T
        loss = F.cross_entropy(logits, labels.to(device) - first_class)
        loss.backward()
        if batch_index == 0:
            raw_gradients = [
                tensor.detach().clone()
                for wrapper in wrappers
                for tensor in (wrapper.gradient_q, wrapper.gradient_v)
            ]
    stable_gradients = [
        tensor.detach().clone()
        for wrapper in wrappers
        for tensor in (wrapper.gradient_q, wrapper.gradient_v)
    ]
    if raw_gradients is None:
        raise RuntimeError("empty demand loader")
    bases = {}
    diagnostics = {}
    for name, gradients in (("raw", raw_gradients), ("stable", stable_gradients)):
        bases[name] = []
        diagnostics[name] = []
        for branch, gradient in enumerate(gradients):
            basis, result = choose_spectral_down(
                state["projection_down"][branch].to(device),
                state["unified_up"][branch].to(device),
                gradient,
            )
            bases[name].append(basis.cpu())
            diagnostics[name].append(result)
    payload = {
        "source_sha256": file_sha256(state_path),
        "task_index": task_index,
        "first_class": first_class,
        "class_count": class_count,
        "prototype_samples": len(prototype_indices),
        "demand_samples": len(design_indices),
        "raw_batches": 1,
        "stable_batches": len(demand_loader),
        "bases": bases,
        "diagnostics": diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({
        "task_index": task_index,
        "demand_samples": len(design_indices),
        "raw_mean_risk": float(np.mean([x["historical_risk"] for x in diagnostics["raw"]])),
        "stable_mean_risk": float(np.mean([x["historical_risk"] for x in diagnostics["stable"]])),
    }))


def train_one_task(config, source, demand_path, mode, output, device, epochs, gate_diagnostic=False):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("causal experiment requires two ranks, batch 64 each")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    seed = int(config["seed"][0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    state_path = source / "sa_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    demand = torch.load(demand_path, map_location="cpu", weights_only=False)
    if demand["source_sha256"] != file_sha256(state_path):
        raise RuntimeError("demand and historical checkpoint differ")
    task_index = int(state["task_id"])
    if task_index != demand["task_index"]:
        raise RuntimeError("demand task index mismatch")
    manager = make_data_manager(config)
    first_class = sum(manager.get_task_size(t) for t in range(task_index))
    class_count = manager.get_task_size(task_index)
    if first_class != demand["first_class"] or class_count != demand["class_count"]:
        raise RuntimeError("demand class range mismatch")
    old_weight = torch.load(source / f"CLs_weight{task_index - 1}.pt", map_location="cpu", weights_only=True)
    old_bias = torch.load(source / f"CLs_bias{task_index - 1}.pt", map_location="cpu", weights_only=True)
    old_prototypes = torch.load(source / "sa_prototypes.pt", map_location="cpu", weights_only=True)
    if sorted(int(k) for k in old_prototypes) != list(range(first_class)):
        raise RuntimeError("historical prototypes are incomplete")
    old_prototype_tensor = torch.stack([old_prototypes[c].float() for c in range(first_class)])

    model = TaskModel(state, old_weight, old_bias, class_count, live=(mode == "live"))
    if mode in ("raw", "stable"):
        model.set_basis(demand["bases"][mode])
    model.to(device)
    train_dataset = current_task_dataset(manager, task_index, "train")
    train_sampler = DistributedSampler(train_dataset, num_replicas=2, rank=rank, shuffle=True, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=2, pin_memory=True)
    proto_loader = DataLoader(current_task_dataset(manager, task_index, "test"), batch_size=64, num_workers=2)
    old_test = DataLoader(manager.get_dataset(np.arange(first_class), "test", "test"), batch_size=64, num_workers=2)
    new_test = DataLoader(manager.get_dataset(np.arange(first_class, first_class + class_count), "test", "test"), batch_size=64, num_workers=2)
    pre = None
    if rank == 0:
        pre_prototypes = prototypes_from_current(model, proto_loader, first_class, class_count, device)
        pre = evaluate_groups(model, old_prototype_tensor, pre_prototypes, old_test, new_test, device)
    dist.barrier()
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.SGD(
        (p for p in wrapped.parameters() if p.requires_grad),
        lr=float(config["lrate"]), momentum=0.9,
    )
    start = time.monotonic()
    history = []
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        wrapped.train()
        loss_sum = torch.zeros(2, device=device)
        for _, images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = wrapped(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum[0] += loss.detach() * len(labels)
            loss_sum[1] += len(labels)
        dist.all_reduce(loss_sum)
        if rank == 0:
            history.append(float(loss_sum[0] / loss_sum[1]))
            print(f"mode={mode} task={task_index} epoch={epoch+1}/{epochs} train_ce={history[-1]:.5f}", flush=True)
    dist.barrier()
    if rank == 0:
        model = wrapped.module
        premerge_new_prototypes = prototypes_from_current(model, proto_loader, first_class, class_count, device)
        premerge = evaluate_groups(model, old_prototype_tensor, premerge_new_prototypes, old_test, new_test, device)
        risks = model.deploy()
        new_prototypes = prototypes_from_current(model, proto_loader, first_class, class_count, device)
        post = evaluate_groups(model, old_prototype_tensor, new_prototypes, old_test, new_test, device)
        for wrapper in model.wrappers:
            wrapper.current_enabled = False
        aligned_history_only = evaluate_groups(
            model, old_prototype_tensor, new_prototypes, old_test, new_test, device
        )
        for wrapper in model.wrappers:
            wrapper.deployed = False
        exact_history_only = evaluate_groups(
            model, old_prototype_tensor, new_prototypes, old_test, new_test, device
        )
        for wrapper in model.wrappers:
            wrapper.deployed = True
            wrapper.current_enabled = True
        old_energy = []
        for wrapper in model.wrappers:
            for suffix in ("q", "v"):
                old_energy.append(float(torch.linalg.vector_norm(
                    getattr(wrapper, f"old_up_{suffix}") @ getattr(wrapper, f"old_down_{suffix}")
                ).square()))
        global_risk = float(np.dot(risks, old_energy) / np.sum(old_energy))
        result = {
            "mode": mode,
            "dataset": config["dataset"],
            "seed": seed,
            "epochs": epochs,
            "task_index": task_index,
            "source_sha256": demand["source_sha256"],
            "demand_samples": demand["demand_samples"],
            "before": pre,
            "premerge": premerge,
            "after": post,
            "counterfactual": {
                "aligned_history_only": aligned_history_only,
                "exact_history_only": exact_history_only,
            },
            "delta_new_ce": post["new"]["fc_ce"] - pre["new"]["fc_ce"],
            "delta_new_top1": post["new"]["prototype_top1"] - pre["new"]["prototype_top1"],
            "delta_old_top1": post["old"]["prototype_top1"] - pre["old"]["prototype_top1"],
            "historical_risk_global": global_risk,
            "historical_risk_max": max(risks),
            "historical_risk_per_branch": risks,
            "train_ce_curve": history,
            "seconds": time.monotonic() - start,
        }
        if gate_diagnostic:
            result["gate_diagnostic"] = evaluate_gating(
                model, manager, task_index, first_class, class_count,
                old_prototype_tensor, old_test, new_test, seed, device,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("historical_risk_per_branch", "train_ce_curve")}, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--demand", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("estimate", "frozen", "live", "raw", "stable"), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gate-diagnostic", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["dataset"] != "cub" or int(config["lora_rank"]) != 10:
        raise ValueError("this preregistered intervention only supports CUB-200 rank10")
    if args.mode == "estimate":
        if "RANK" in os.environ:
            raise RuntimeError("demand estimation is single-GPU and must precede torchrun")
        torch.cuda.set_device(0)
        estimate_demand(config, args.source, args.demand, torch.device("cuda:0"))
    else:
        epochs = int(config["epochs"]) if args.epochs is None else args.epochs
        if epochs < 1:
            raise ValueError("epochs must be positive")
        train_one_task(
            config, args.source, args.demand, args.mode, args.output,
            torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"), epochs,
            gate_diagnostic=args.gate_diagnostic,
        )


if __name__ == "__main__":
    main()
