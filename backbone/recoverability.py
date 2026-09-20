"""Low-rank geometry for recoverability-constrained Shared-A updates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor


def _validate_matrix(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError("{} must be a matrix".format(name))
    if not value.is_floating_point():
        raise ValueError("{} must be floating point".format(name))


def _symmetric_pinv(matrix: Tensor, rcond: float = 1e-10) -> Tensor:
    if rcond <= 0:
        raise ValueError("rcond must be positive")
    work = 0.5 * (matrix.double() + matrix.double().t())
    eigenvalues, eigenvectors = torch.linalg.eigh(work)
    largest = torch.clamp_min(eigenvalues.abs().max(), torch.finfo(work.dtype).tiny)
    keep = eigenvalues > rcond * largest
    inverse = torch.where(
        keep,
        torch.reciprocal(torch.clamp_min(eigenvalues, torch.finfo(work.dtype).tiny)),
        torch.zeros_like(eigenvalues),
    )
    return (eigenvectors * inverse.unsqueeze(0)) @ eigenvectors.t()


def row_polar_retraction(candidate: Tensor, rcond: float = 1e-10) -> Tensor:
    """Return the row-orthonormal polar factor of a full-row-rank matrix."""
    _validate_matrix("candidate", candidate)
    if candidate.shape[0] > candidate.shape[1]:
        raise ValueError("row retraction requires rows <= columns")
    work = candidate.double()
    gram = work @ work.t()
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.t()))
    largest = torch.clamp_min(eigenvalues.max(), torch.finfo(work.dtype).tiny)
    if bool(torch.any(eigenvalues <= rcond * largest)):
        raise ValueError("candidate is rank deficient under the configured rcond")
    inverse_sqrt = (
        eigenvectors
        * torch.rsqrt(torch.clamp_min(eigenvalues, torch.finfo(work.dtype).tiny)).unsqueeze(0)
    ) @ eigenvectors.t()
    result = inverse_sqrt @ work
    return result.to(device=candidate.device, dtype=candidate.dtype)


def recoverability_energies(
    anchor_up: Tensor,
    anchor_a: Tensor,
    candidate_a: Tensor,
    *,
    rcond: float = 1e-10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return historical total, captured, and irrecoverable operator energy.

    The computation is the exact reduced form of
    ``min_X ||X candidate_a - anchor_up anchor_a||_F^2``.
    """
    _validate_matrix("anchor_up", anchor_up)
    _validate_matrix("anchor_a", anchor_a)
    _validate_matrix("candidate_a", candidate_a)
    if anchor_up.shape[1] != anchor_a.shape[0]:
        raise ValueError("anchor_up and anchor_a ranks must match")
    if candidate_a.shape[1] != anchor_a.shape[1]:
        raise ValueError("anchor and candidate input dimensions must match")

    device = candidate_a.device
    g = anchor_up.detach().to(device=device, dtype=torch.float64)
    old = anchor_a.detach().to(device=device, dtype=torch.float64)
    new = candidate_a.detach().to(device=device, dtype=torch.float64)
    s = g.t() @ g
    h_old = old @ old.t()
    cross = old @ new.t()
    h_new_pinv = _symmetric_pinv(new @ new.t(), rcond=rcond)
    total = torch.sum(s * h_old.t())
    captured = torch.trace(s @ cross @ h_new_pinv @ cross.t())
    total = torch.clamp_min(total, 0.0)
    captured = torch.clamp(captured, min=0.0, max=total)
    residual = torch.clamp_min(total - captured, 0.0)
    return total, captured, residual


def operator_weighted_recoverability(
    anchor_up: Tensor,
    anchor_a: Tensor,
    candidate_a: Tensor,
    *,
    eps: float = 1e-12,
    rcond: float = 1e-10,
) -> Tensor:
    if eps <= 0:
        raise ValueError("eps must be positive")
    total, _, residual = recoverability_energies(
        anchor_up, anchor_a, candidate_a, rcond=rcond
    )
    if float(total) == 0.0:
        return total.new_zeros(())
    return residual / (total + eps)


def align_to_anchor(
    anchor_up: Tensor,
    anchor_a: Tensor,
    candidate_a: Tensor,
    *,
    rcond: float = 1e-10,
) -> Tensor:
    """Return the least-squares up projection of a fixed anchor operator."""
    _validate_matrix("anchor_up", anchor_up)
    _validate_matrix("anchor_a", anchor_a)
    _validate_matrix("candidate_a", candidate_a)
    if anchor_up.shape[1] != anchor_a.shape[0]:
        raise ValueError("anchor_up and anchor_a ranks must match")
    if candidate_a.shape[1] != anchor_a.shape[1]:
        raise ValueError("anchor and candidate input dimensions must match")
    device = candidate_a.device
    work_up = anchor_up.detach().to(device=device, dtype=torch.float64)
    old = anchor_a.detach().to(device=device, dtype=torch.float64)
    new = candidate_a.detach().to(device=device, dtype=torch.float64)
    aligned = work_up @ old @ new.t() @ _symmetric_pinv(
        new @ new.t(), rcond=rcond
    )
    return aligned.to(device=candidate_a.device, dtype=candidate_a.dtype)


def _flatten_samples(value: Tensor, feature_dim: int, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim < 2:
        raise ValueError("{} must have samples and features".format(name))
    if value.shape[-1] != feature_dim:
        raise ValueError("{} feature dimension mismatch".format(name))
    return value.reshape(-1, feature_dim)


def effective_weight_gradient(inputs: Tensor, output_grad: Tensor) -> Tensor:
    """Return ``H = dL/dW`` for a linear map ``y = x W^T``."""
    x = _flatten_samples(inputs, inputs.shape[-1], "inputs")
    z = _flatten_samples(output_grad, output_grad.shape[-1], "output_grad")
    if x.shape[0] != z.shape[0]:
        raise ValueError("inputs and output_grad sample counts must match")
    return z.t() @ x


def effective_gradient_cross(
    inputs: Tensor, output_grad: Tensor, basis: Tensor
) -> Tensor:
    _validate_matrix("basis", basis)
    x = _flatten_samples(inputs, basis.shape[1], "inputs")
    z = _flatten_samples(output_grad, output_grad.shape[-1], "output_grad")
    if x.shape[0] != z.shape[0]:
        raise ValueError("inputs and output_grad sample counts must match")
    return z.t() @ (x @ basis.t())


def effective_gradient_energy(inputs: Tensor, output_grad: Tensor) -> Tensor:
    gradient = effective_weight_gradient(inputs, output_grad)
    return gradient.square().sum()


def accessibility_energy(
    gradient_cross: Tensor,
    basis: Tensor,
    *,
    rcond: float = 1e-10,
) -> Tensor:
    """Return ``||H P_A||_F^2`` from ``gradient_cross = H A^T``."""
    _validate_matrix("gradient_cross", gradient_cross)
    _validate_matrix("basis", basis)
    if gradient_cross.shape[1] != basis.shape[0]:
        raise ValueError("gradient_cross rank must match basis rows")
    cross = gradient_cross.double()
    gram_pinv = _symmetric_pinv(
        basis.double() @ basis.double().t(), rcond=rcond
    )
    value = torch.trace(cross.t() @ cross @ gram_pinv)
    return torch.clamp_min(value, 0.0)


def grassmann_accessibility_direction(
    inputs: Tensor,
    output_grad: Tensor,
    basis: Tensor,
) -> Tensor:
    """Return ``A H^T H (I-P_A)`` for a row-orthonormal basis."""
    _validate_matrix("basis", basis)
    h = effective_weight_gradient(inputs, output_grad)
    work_basis = basis.to(device=h.device, dtype=h.dtype)
    raw = (h.t() @ (h @ work_basis.t())).t()
    normal = raw - (raw @ work_basis.t()) @ work_basis
    return normal.to(device=basis.device, dtype=basis.dtype)


def accessibility_candidates(
    basis: Tensor,
    direction: Tensor,
    *,
    gradient_energy: Tensor | float,
    gammas: Iterable[float],
    step_size: float,
    eps: float = 1e-12,
    rcond: float = 1e-10,
) -> list[tuple[float, Tensor]]:
    if basis.shape != direction.shape:
        raise ValueError("basis and direction must have the same shape")
    if step_size < 0:
        raise ValueError("step_size must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    values = [float(gamma) for gamma in gammas]
    if not values or any(not 0.0 <= gamma <= 1.0 for gamma in values):
        raise ValueError("gammas must be non-empty values in [0, 1]")
    if values != sorted(set(values)):
        raise ValueError("gammas must be sorted and unique")
    normalizer = torch.as_tensor(
        gradient_energy, device=basis.device, dtype=basis.dtype
    )
    scaled = step_size * direction / (normalizer + eps)
    candidates = []
    for gamma in values:
        if gamma == 0.0:
            candidate = basis.detach().clone()
        else:
            candidate = row_polar_retraction(
                basis + gamma * scaled, rcond=rcond
            )
        candidates.append((gamma, candidate))
    return candidates


def _validate_candidate_layers(
    layers: Sequence[Sequence[dict[str, float]]],
) -> None:
    if not layers or any(not layer for layer in layers):
        raise ValueError("candidate layers must be non-empty")
    required = {"gamma", "utility", "residual", "history"}
    for layer in layers:
        for candidate in layer:
            if set(candidate) < required:
                raise ValueError("candidate is missing required fields")
            if candidate["residual"] < 0 or candidate["history"] < 0:
                raise ValueError("candidate energies must be non-negative")


def choose_local_recoverability_candidates(
    layers: Sequence[Sequence[dict[str, float]]],
    *,
    risk_budget: float,
    eps: float = 1e-12,
) -> dict:
    """Choose each branch independently under the same normalized tolerance."""
    _validate_candidate_layers(layers)
    if risk_budget < 0:
        raise ValueError("risk_budget must be non-negative")
    indices = []
    utility = 0.0
    residual = 0.0
    history = 0.0
    for layer in layers:
        feasible = []
        for index, candidate in enumerate(layer):
            denominator = float(candidate["history"])
            risk = (
                float(candidate["residual"]) / (denominator + eps)
                if denominator > 0
                else 0.0
            )
            if risk <= risk_budget + eps:
                feasible.append((float(candidate["utility"]), -risk, -index, index))
        if not feasible:
            raise RuntimeError("no local recoverability candidate satisfies the budget")
        index = max(feasible)[-1]
        selected = layer[index]
        indices.append(index)
        utility += float(selected["utility"])
        residual += float(selected["residual"])
        history += float(selected["history"])
    return {
        "indices": indices,
        "utility": utility,
        "risk": residual / (history + eps) if history > 0 else 0.0,
    }


def choose_global_recoverability_candidates(
    layers: Sequence[Sequence[dict[str, float]]],
    *,
    risk_budget: float,
    eps: float = 1e-12,
) -> dict:
    """Maximize summed accessibility under one operator-energy budget."""
    _validate_candidate_layers(layers)
    if risk_budget < 0:
        raise ValueError("risk_budget must be non-negative")
    total_history = sum(float(layer[0]["history"]) for layer in layers)
    allowed_residual = risk_budget * total_history
    frontier: list[tuple[float, float, tuple[int, ...]]] = [(0.0, 0.0, ())]
    for layer in layers:
        expanded = []
        for accumulated_residual, accumulated_utility, indices in frontier:
            for index, candidate in enumerate(layer):
                next_residual = accumulated_residual + float(candidate["residual"])
                if total_history == 0.0 or next_residual <= allowed_residual + eps:
                    expanded.append(
                        (
                            next_residual,
                            accumulated_utility + float(candidate["utility"]),
                            indices + (index,),
                        )
                    )
        expanded.sort(key=lambda item: (item[0], -item[1], item[2]))
        frontier = []
        best_utility = float("-inf")
        for state in expanded:
            if state[1] > best_utility + eps:
                frontier.append(state)
                best_utility = state[1]
        if not frontier:
            raise RuntimeError("no global recoverability candidate satisfies the budget")
    selected = max(frontier, key=lambda item: (item[1], -item[0], item[2]))
    return {
        "indices": list(selected[2]),
        "utility": selected[1],
        "risk": selected[0] / (total_history + eps) if total_history > 0 else 0.0,
    }

