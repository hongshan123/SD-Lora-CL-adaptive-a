"""Projected CUO normal-equation kernel for a fixed low-rank coordinate system."""

from __future__ import annotations

import torch


def _validate_damping(damping: float) -> None:
    if not isinstance(damping, (int, float)) or not torch.isfinite(torch.tensor(damping)) or damping <= 0:
        raise ValueError("damping must be positive and finite")


def _validate_matrix(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor")
    if tensor.numel() == 0 or not tensor.is_floating_point():
        raise ValueError(f"{name} must be a non-empty floating-point tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_gram(gram: torch.Tensor) -> None:
    _validate_matrix(gram, "gram")
    if gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    if not torch.allclose(gram, gram.T, rtol=1e-5, atol=1e-6):
        raise ValueError("gram must be symmetric")


def row_orthonormal_projection(weight: torch.Tensor) -> torch.Tensor:
    """Return the row-orthonormal basis obtained by reduced QR of ``weight.T``."""
    _validate_matrix(weight, "weight")
    rows, columns = weight.shape
    if rows > columns:
        raise ValueError("weight must have no more rows than columns")
    q, _ = torch.linalg.qr(weight.T, mode="reduced")
    return q.T


def solve_projected_cuo(
    gram: torch.Tensor,
    cross: torch.Tensor,
    damping: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Solve ``up @ (gram + damping I) = cross`` without a dense state."""
    _validate_damping(damping)
    _validate_gram(gram)
    _validate_matrix(cross, "cross")
    if cross.shape[1] != gram.shape[0]:
        raise ValueError("cross shape must be [d, r] for gram shape [r, r]")
    if cross.device != gram.device:
        raise ValueError("gram and cross must be on the same device")
    if cross.dtype != gram.dtype:
        raise ValueError("gram and cross must have the same dtype")

    system = gram.to(torch.float64) + damping * torch.eye(
        gram.shape[0], dtype=torch.float64, device=gram.device
    )
    solved = torch.linalg.solve(system, cross.to(torch.float64).T).T
    up = solved.to(dtype=cross.dtype)
    diagnostics = {
        "condition_number": torch.linalg.cond(system),
        "residual_norm": torch.linalg.norm(up.to(torch.float64) @ system - cross.to(torch.float64)),
    }
    return up, diagnostics


def advance_projected_cuo(
    previous_up: torch.Tensor,
    previous_gram: torch.Tensor,
    batch_z: torch.Tensor,
    batch_y: torch.Tensor,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Add one batch of projected observations and solve the updated system."""
    _validate_damping(damping)
    _validate_matrix(previous_up, "previous_up")
    _validate_gram(previous_gram)
    _validate_matrix(batch_z, "batch_z")
    _validate_matrix(batch_y, "batch_y")
    if previous_up.shape[1] != previous_gram.shape[0]:
        raise ValueError("previous_up shape must be [d, r] for previous_gram shape [r, r]")
    if batch_z.shape[1] != previous_gram.shape[0] or batch_y.shape[0] != batch_z.shape[0]:
        raise ValueError("batch shapes must be batch_z [n, r] and batch_y [n, d]")
    if batch_y.shape[1] != previous_up.shape[0]:
        raise ValueError("batch_y feature dimension must match previous_up")
    tensors = (previous_up, previous_gram, batch_z, batch_y)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all CUO tensors must be on the same device")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("all CUO tensors must have the same dtype")

    identity = torch.eye(previous_gram.shape[0], dtype=previous_gram.dtype, device=previous_gram.device)
    previous_cross = previous_up @ (previous_gram + damping * identity)
    new_gram = previous_gram + batch_z.T @ batch_z
    new_cross = previous_cross + batch_y.T @ batch_z
    new_up, diagnostics = solve_projected_cuo(new_gram, new_cross, damping)
    return new_up, new_gram, diagnostics
