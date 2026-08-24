"""Closed-form coordinate stabilization for cumulative Shared-A LoRA."""

import torch
from torch.nn import functional as F


def normalized_effective_operator(aggregate_up, shared_a, eps=1e-8):
    """Return ``G A / ||A||_F`` in float64 for stable diagnostics."""
    aggregate = aggregate_up.detach().double()
    down = shared_a.detach().double()
    return aggregate @ (down / (torch.linalg.vector_norm(down) + eps))


def align_live_a_aggregate(aggregate_up, old_a, new_a, eps=1e-8):
    """Align ``G`` to a changed shared-A coordinate system.

    Solves ``min_X ||X A_new_hat - G A_old_hat||_F`` in closed form.  The
    historical effective operator is preserved exactly when the old and new
    row spaces coincide.  No alignment matrix is persisted.
    """
    if aggregate_up.ndim != 2 or old_a.ndim != 2 or new_a.ndim != 2:
        raise ValueError("aggregate_up, old_a and new_a must be matrices")
    if old_a.shape != new_a.shape:
        raise ValueError("old_a and new_a must have identical shapes")
    if aggregate_up.shape[1] != old_a.shape[0]:
        raise ValueError("aggregate_up rank must match shared-A rank")

    old_operator = normalized_effective_operator(aggregate_up, old_a, eps=eps)
    new_down = new_a.detach().double()
    new_down = new_down / (torch.linalg.vector_norm(new_down) + eps)
    before_operator = aggregate_up.detach().double() @ new_down
    old_norm = torch.linalg.vector_norm(old_operator)

    # X A_new = O_old => A_new^T X^T = O_old^T.
    solution = torch.linalg.lstsq(new_down.t(), old_operator.t()).solution
    aligned = solution.t().to(dtype=aggregate_up.dtype)
    after_operator = aligned.double() @ new_down

    denominator = old_norm + eps
    before_error = torch.linalg.vector_norm(
        before_operator - old_operator
    ) / denominator
    after_error = torch.linalg.vector_norm(
        after_operator - old_operator
    ) / denominator
    singular_values = torch.linalg.svdvals(new_down)
    condition = singular_values.max() / (singular_values.min() + eps)
    return aligned, {
        "before_relative_error": float(before_error),
        "after_relative_error": float(after_error),
        "condition": float(condition),
    }


def _apply_matrix(features, basis, rotation):
    if basis.numel() == 0:
        return features
    projected = features @ basis
    return features + (projected @ (rotation - torch.eye(
        rotation.shape[0], dtype=rotation.dtype, device=rotation.device
    ))) @ basis.t()


def fit_residual_orthogonal_transport(
    source,
    target,
    rank,
    identity_reg=1e-4,
    min_validation_gain=0.0,
):
    """Fit a gated low-rank orthogonal transport from paired features.

    The top right-singular directions of the observed feature displacement
    define the only subspace that may rotate.  A deterministic 80/20 split
    gates the transport: when the held-out paired-feature error does not
    improve, the returned transform is the identity.
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must be equal-shaped matrices")
    if source.shape[0] < 5:
        raise ValueError("at least five paired features are required")
    if rank <= 0:
        raise ValueError("rank must be positive")

    source = F.normalize(source.detach().float(), p=2, dim=1)
    target = F.normalize(target.detach().float(), p=2, dim=1)
    indices = torch.arange(source.shape[0])
    val_mask = indices.remainder(5) == 0
    train_mask = ~val_mask
    source_train, target_train = source[train_mask], target[train_mask]
    source_val, target_val = source[val_mask], target[val_mask]

    drift = target_train - source_train
    _, singular_values, vh = torch.linalg.svd(drift, full_matrices=False)
    active_rank = min(int(rank), vh.shape[0], source.shape[1])
    basis = vh[:active_rank].t().contiguous()

    x_projected = source_train @ basis
    y_projected = target_train @ basis
    cross = x_projected.t() @ y_projected / max(source_train.shape[0], 1)
    cross = cross + float(identity_reg) * torch.eye(active_rank)
    u, _, vh_cross = torch.linalg.svd(cross, full_matrices=False)
    rotation = u @ vh_cross

    before_val = (source_val - target_val).square().mean()
    transported_val = F.normalize(
        _apply_matrix(source_val, basis, rotation), p=2, dim=1
    )
    after_val = (transported_val - target_val).square().mean()
    validation_gain = (before_val - after_val) / (before_val + 1e-12)
    enabled = bool(float(validation_gain) > float(min_validation_gain))
    if not enabled:
        rotation = torch.eye(active_rank)

    total_energy = singular_values.square().sum()
    explained = singular_values[:active_rank].square().sum() / (
        total_energy + 1e-12
    )
    transported_train = F.normalize(
        _apply_matrix(source_train, basis, rotation), p=2, dim=1
    )
    return {
        "basis": basis.cpu(),
        "rotation": rotation.cpu(),
        "enabled": enabled,
        "rank": active_rank,
        "explained_drift": float(explained),
        "validation_before": float(before_val),
        "validation_after": float(after_val),
        "validation_gain": float(validation_gain),
        "train_before": float(
            (source_train - target_train).square().mean()
        ),
        "train_after": float(
            (transported_train - target_train).square().mean()
        ),
    }


def apply_residual_orthogonal_transport(prototypes, transport):
    """Apply a fitted transport to tensor or list-valued prototypes."""
    basis = transport["basis"].float()
    rotation = transport["rotation"].float()

    def apply_one(vector):
        row = vector.detach().float().reshape(1, -1)
        moved = _apply_matrix(row, basis, rotation).reshape(-1)
        return F.normalize(moved, p=2, dim=0).to(vector.dtype)

    updated = {}
    for class_id, value in prototypes.items():
        if isinstance(value, (list, tuple)):
            updated[int(class_id)] = [apply_one(vector) for vector in value]
        else:
            updated[int(class_id)] = apply_one(value)
    return updated
