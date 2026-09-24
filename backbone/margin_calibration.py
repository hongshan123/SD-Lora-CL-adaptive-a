"""Deterministic, exemplar-free calibration for a prototype cosine head."""

import torch
from torch.nn import functional as F


def class_concentration(features, labels, prototypes, class_ids):
    """Mean cosine to each class prototype at that class's acquisition task."""
    features = F.normalize(features.float(), dim=1)
    prototypes = F.normalize(prototypes.float(), dim=1)
    values = []
    for class_id in class_ids:
        selected = features[labels == class_id]
        if selected.numel() == 0:
            raise ValueError(f"class {class_id} has no acquisition features")
        value = (selected @ prototypes[class_id]).mean()
        if not torch.isfinite(value) or not (0 < value <= 1.00001):
            raise ValueError(f"class {class_id} has invalid concentration")
        values.append(value.clamp(max=1.0))
    return torch.stack(values)


def synthetic_old_features(prototypes, concentration, samples_per_class=32, seed=0):
    """Isotropic tangent samples with class-prototype cosine equal to rho."""
    if prototypes.ndim != 2 or concentration.shape != (len(prototypes),):
        raise ValueError("prototype/concentration shape mismatch")
    if samples_per_class < 1 or not torch.isfinite(concentration).all():
        raise ValueError("invalid sample count or concentration")
    if bool(((concentration <= 0) | (concentration > 1)).any()):
        raise ValueError("concentration must be in (0, 1]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    p = F.normalize(prototypes.float().cpu(), dim=1)
    noise = torch.randn(
        len(p), samples_per_class, p.shape[1], generator=generator
    )
    noise = noise - (noise * p[:, None, :]).sum(dim=2, keepdim=True) * p[:, None, :]
    noise = F.normalize(noise, dim=2)
    rho = concentration.float().cpu()
    output = rho[:, None, None] * p[:, None, :] + (
        1 - rho.square()
    ).clamp_min(0).sqrt()[:, None, None] * noise
    labels = torch.arange(len(p)).repeat_interleave(samples_per_class)
    return F.normalize(output.reshape(-1, p.shape[1]), dim=1), labels


def stratified_two_folds(labels, class_ids, seed):
    generator = torch.Generator().manual_seed(int(seed))
    folds = torch.empty(len(labels), dtype=torch.long)
    for class_id in class_ids:
        indices = torch.where(labels == class_id)[0]
        if len(indices) < 2:
            raise ValueError(f"class {class_id} needs at least two train samples")
        order = indices[torch.randperm(len(indices), generator=generator)]
        folds[order] = torch.arange(len(order)) % 2
    return folds


def class_balanced_weights(labels, total_classes):
    counts = torch.bincount(labels.long(), minlength=total_classes).float()
    if bool((counts == 0).any()):
        raise ValueError("calibration requires all seen classes")
    weights = counts[labels.long()].reciprocal()
    return weights / weights.sum()


def fit_old_group_bias(logits, labels, old_count, bounds=(-0.5, 0.5), steps=50):
    """Exact one-dimensional convex CE optimum within fixed cosine-score bounds."""
    if logits.ndim != 2 or logits.shape[0] != len(labels):
        raise ValueError("logits/labels shape mismatch")
    if not 0 < old_count < logits.shape[1]:
        raise ValueError("both old and current classes are required")
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite calibration logits")
    weights = class_balanced_weights(labels, logits.shape[1])
    old_logsum = torch.logsumexp(logits[:, :old_count].double(), dim=1)
    new_logsum = torch.logsumexp(logits[:, old_count:].double(), dim=1)
    target_old = (labels < old_count).double()

    def derivative(delta):
        p_old = torch.sigmoid(old_logsum - new_logsum + delta)
        return float((weights.double() * (p_old - target_old)).sum())

    left, right = map(float, bounds)
    if not left < right:
        raise ValueError("invalid bias bounds")
    if derivative(left) >= 0:
        return left
    if derivative(right) <= 0:
        return right
    for _ in range(steps):
        middle = (left + right) / 2
        if derivative(middle) < 0:
            left = middle
        else:
            right = middle
    return (left + right) / 2
