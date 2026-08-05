"""Low-Rank Prototype Transport (LRPT).

Given paired features of the *current* task before and after the shared-A
LoRA update, estimate a low-rank transport

    p' = p + U (V^T p),      U, V in R^{d x r},

so that p' approximates where an old prototype p (computed in the previous
feature space) would live in the updated feature space.  The transport rank is
tied to the LoRA rank r of the shared down-projection A, and the fit is a
closed-form rank-r least-squares problem:

    W* = argmin_W ||Y - X - W X||_F^2 + reg * ||W||_F^2
    W  = rank-r SVD truncation of W*

No old-task data is read, saved, or replayed: X and Y are extracted from the
current task's own training data at the two model states.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def fit_low_rank_transport(
    z_old: torch.Tensor,
    z_new: torch.Tensor,
    rank: int,
    reg: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit p' = p + U (V^T p) from paired rows (N, d) of pre/post features.

    Args:
        z_old: (N, d) features extracted with the model state before update.
        z_new: (N, d) features of the same inputs after update.
        rank: rank of the transport; should be tied to the LoRA rank.
        reg: ridge regularizer on the unconstrained least-squares map.

    Returns:
        U, V: (d, rank) tensors such that p' = p + U @ (V^T @ p).
    """
    if z_old.shape != z_new.shape:
        raise ValueError("z_old and z_new must have the same shape")
    if z_old.ndim != 2:
        raise ValueError("expected 2D feature matrices (N, d)")
    n, d = z_old.shape
    if rank <= 0 or rank > d:
        raise ValueError("rank must satisfy 0 < rank <= d")
    if n < 1:
        raise ValueError("need at least one paired sample")

    x = z_old.to(torch.float64).t().contiguous()  # (d, N)
    y = z_new.to(torch.float64).t().contiguous()  # (d, N)
    residual = y - x
    cov = x @ x.t() + max(reg, 0.0) * torch.eye(d, dtype=torch.float64)
    w = residual @ x.t() @ torch.linalg.inv(cov)  # (d, d)

    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    sqrt_s = torch.sqrt(s[:rank].clamp_min(0.0))
    u_r = u[:, :rank] * sqrt_s
    v_r = vh.t()[:, :rank] * sqrt_s
    return u_r.to(torch.float32), v_r.to(torch.float32)


def apply_transport(
    prototypes: dict[int, torch.Tensor],
    u: torch.Tensor,
    v: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Apply p' = normalize(p + U (V^T p)) to every stored prototype."""
    u64 = u.to(torch.float64)
    v64 = v.to(torch.float64)
    moved = {}
    for class_id, proto in prototypes.items():
        p = proto.detach().to(torch.float64).reshape(-1)
        p_new = p + u64 @ (v64.t() @ p)
        moved[int(class_id)] = F.normalize(p_new.to(torch.float32), p=2, dim=0)
    return moved


def transport_prediction_error(
    z_old: torch.Tensor,
    z_new: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[float, float]:
    """Relative residual of the fitted transport on the fitting features."""
    pred = z_old + (z_old @ v) @ u.t()
    drift = z_new - z_old
    residual = z_new - pred
    denom = torch.linalg.norm(drift.double())
    if denom <= 0:
        return 0.0, 0.0
    rel = float(torch.linalg.norm(residual.double()) / denom)
    frac = float(torch.linalg.norm(pred.double()) / torch.linalg.norm(z_new.double()))
    return rel, frac


def fit_rank_residuals(
    z_old: torch.Tensor,
    z_new: torch.Tensor,
    ranks: tuple[int, ...],
    reg: float = 1e-2,
) -> dict[int, float]:
    """Relative drift residual after fitting a rank-r transport for each rank."""
    out = {}
    for rank in ranks:
        if rank <= 0 or rank > z_old.shape[1]:
            continue
        u, v = fit_low_rank_transport(z_old, z_new, rank=rank, reg=reg)
        rel, _ = transport_prediction_error(z_old, z_new, u, v)
        out[rank] = rel
    return out


def bias_relative_error(
    z_old: torch.Tensor,
    z_new: torch.Tensor,
) -> float:
    """Relative drift residual after subtracting only the global mean shift."""
    drift = z_new - z_old
    denom = torch.linalg.norm(drift.double())
    if denom <= 0:
        return 0.0
    centered = drift - drift.mean(dim=0, keepdim=True)
    return float(torch.linalg.norm(centered.double()) / denom)
