"""Sensitivity-budgeted consolidation in a fixed low-rank coordinate system."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _matrix(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, Tensor) or tensor.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain finite floating-point values")


def _vector(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, Tensor) or tensor.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain finite floating-point values")


def normalize_sensitivity(sensitivity: Tensor, floor: float = 1e-4) -> Tensor:
    """Return a finite, unit-mean diagonal sensitivity with a positive floor."""
    _vector(sensitivity, "sensitivity")
    if not math.isfinite(float(floor)) or not 0.0 < float(floor) <= 1.0:
        raise ValueError("floor must be finite and in (0, 1]")
    work = sensitivity.detach().to(dtype=torch.float64)
    if bool((work < 0).any()):
        raise ValueError("sensitivity must be non-negative")
    mean = work.mean()
    if not bool(mean > torch.finfo(work.dtype).eps):
        raise ValueError("sensitivity must have strictly positive mean")
    normalized = (work / mean).clamp_min(float(floor))
    normalized = normalized / normalized.mean()
    return normalized.to(device=sensitivity.device, dtype=sensitivity.dtype)


def weighted_response_energy(
    operator: Tensor,
    covariance: Tensor,
    sensitivity: Tensor,
) -> Tensor:
    """Compute ``sum_j f_j g_j C g_j^T`` without dense activations."""
    _matrix(operator, "operator")
    _matrix(covariance, "covariance")
    _vector(sensitivity, "sensitivity")
    if covariance.shape != (operator.shape[1], operator.shape[1]):
        raise ValueError("covariance shape must match operator rank")
    if sensitivity.shape[0] != operator.shape[0]:
        raise ValueError("sensitivity shape must match operator output dimension")
    if len({operator.device, covariance.device, sensitivity.device}) != 1:
        raise ValueError("operator, covariance and sensitivity must share a device")
    work = operator.to(dtype=torch.float64)
    cov = covariance.to(dtype=torch.float64)
    sens = sensitivity.to(dtype=torch.float64)
    row_energy = ((work @ cov) * work).sum(dim=1)
    return (sens * row_energy).sum().clamp_min(0.0)


def historical_response_risk(
    candidate: Tensor,
    historical: Tensor,
    covariance: Tensor,
    sensitivity: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Normalized historical branch-response drift of one G candidate."""
    if candidate.shape != historical.shape:
        raise ValueError("candidate and historical operators must have equal shape")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    numerator = weighted_response_energy(
        candidate.to(dtype=torch.float64) - historical.to(dtype=torch.float64),
        covariance,
        sensitivity,
    )
    denominator = weighted_response_energy(
        historical, covariance, sensitivity
    )
    return numerator / (denominator + float(eps))


def _validate_problem(
    historical: Tensor,
    target: Tensor,
    historical_covariance: Tensor,
    current_covariance: Tensor,
    historical_sensitivity: Tensor,
    current_sensitivity: Tensor,
) -> None:
    _matrix(historical, "historical")
    _matrix(target, "target")
    _matrix(historical_covariance, "historical_covariance")
    _matrix(current_covariance, "current_covariance")
    _vector(historical_sensitivity, "historical_sensitivity")
    _vector(current_sensitivity, "current_sensitivity")
    if historical.shape != target.shape:
        raise ValueError("historical and target operators must have equal shape")
    output_dim, rank = historical.shape
    if historical_covariance.shape != (rank, rank):
        raise ValueError("historical_covariance shape must be [rank, rank]")
    if current_covariance.shape != (rank, rank):
        raise ValueError("current_covariance shape must be [rank, rank]")
    if historical_sensitivity.shape != (output_dim,):
        raise ValueError("historical_sensitivity shape must match output dimension")
    if current_sensitivity.shape != (output_dim,):
        raise ValueError("current_sensitivity shape must match output dimension")
    tensors = (
        historical,
        target,
        historical_covariance,
        current_covariance,
        historical_sensitivity,
        current_sensitivity,
    )
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all SBGC tensors must share a device")
    if bool((historical_sensitivity < 0).any()) or bool(
        (current_sensitivity < 0).any()
    ):
        raise ValueError("SBGC sensitivities must be non-negative")


def solve_lagrangian_candidate(
    historical: Tensor,
    target: Tensor,
    historical_covariance: Tensor,
    current_covariance: Tensor,
    historical_sensitivity: Tensor,
    current_sensitivity: Tensor,
    eta: float,
    ridge: float = 1e-6,
) -> Tensor:
    """Solve the row-wise SBGC Lagrangian for one non-negative multiplier."""
    _validate_problem(
        historical,
        target,
        historical_covariance,
        current_covariance,
        historical_sensitivity,
        current_sensitivity,
    )
    if not math.isfinite(float(eta)) or eta < 0:
        raise ValueError("eta must be non-negative and finite")
    if not math.isfinite(float(ridge)) or ridge <= 0:
        raise ValueError("ridge must be positive and finite")

    work_dtype = torch.float64
    old = historical.to(dtype=work_dtype)
    goal = target.to(dtype=work_dtype)
    c_old = historical_covariance.to(dtype=work_dtype)
    c_cur = current_covariance.to(dtype=work_dtype)
    f_old = historical_sensitivity.to(dtype=work_dtype)
    f_cur = current_sensitivity.to(dtype=work_dtype)
    c_old = 0.5 * (c_old + c_old.t())
    c_cur = 0.5 * (c_cur + c_cur.t())

    rank = old.shape[1]
    covariance_scale = torch.stack(
        (c_old.diag().mean(), c_cur.diag().mean(), c_cur.new_tensor(1.0))
    ).max()
    ridge_value = float(ridge) * covariance_scale
    identity = torch.eye(rank, dtype=work_dtype, device=old.device)

    systems = (
        f_cur[:, None, None] * c_cur[None, :, :]
        + float(eta) * f_old[:, None, None] * c_old[None, :, :]
        + ridge_value * identity[None, :, :]
    )
    right = (
        f_cur[:, None] * (goal @ c_cur)
        + float(eta) * f_old[:, None] * (old @ c_old)
        + ridge_value * goal
    )
    solved = torch.linalg.solve(systems, right.unsqueeze(-1)).squeeze(-1)
    if not bool(torch.isfinite(solved).all()):
        raise RuntimeError("SBGC solve produced non-finite values")
    return solved.to(device=historical.device, dtype=historical.dtype)


def solve_sensitivity_budgeted_g(
    historical: Tensor,
    target: Tensor,
    historical_covariance: Tensor,
    current_covariance: Tensor,
    historical_sensitivity: Tensor,
    current_sensitivity: Tensor,
    risk_budget: float = 0.05,
    ridge: float = 1e-6,
    bisection_steps: int = 40,
    eps: float = 1e-12,
    max_bracket_steps: int = 80,
) -> tuple[Tensor, dict[str, float | bool | int]]:
    """Return the closest current target satisfying the historical risk budget."""
    _validate_problem(
        historical,
        target,
        historical_covariance,
        current_covariance,
        historical_sensitivity,
        current_sensitivity,
    )
    if not math.isfinite(float(risk_budget)) or risk_budget < 0:
        raise ValueError("risk_budget must be non-negative and finite")
    if isinstance(bisection_steps, bool) or int(bisection_steps) <= 0:
        raise ValueError("bisection_steps must be a positive integer")
    if isinstance(max_bracket_steps, bool) or int(max_bracket_steps) <= 0:
        raise ValueError("max_bracket_steps must be a positive integer")

    target_risk = historical_response_risk(
        target,
        historical,
        historical_covariance,
        historical_sensitivity,
        eps=eps,
    )
    update = target.to(torch.float64) - historical.to(torch.float64)
    target_update_energy = weighted_response_energy(
        update, current_covariance, current_sensitivity
    )

    if float(target_risk) <= float(risk_budget):
        return target.detach().clone(), {
            "eta": 0.0,
            "target_risk": float(target_risk),
            "achieved_risk": float(target_risk),
            "current_distortion": 0.0,
            "constraint_active": False,
            "bracket_steps": 0,
            "bisection_steps": 0,
        }

    lower = 0.0
    upper = 1.0
    candidate = None
    achieved = None
    bracket_steps = 0
    for bracket_steps in range(1, int(max_bracket_steps) + 1):
        candidate = solve_lagrangian_candidate(
            historical,
            target,
            historical_covariance,
            current_covariance,
            historical_sensitivity,
            current_sensitivity,
            eta=upper,
            ridge=ridge,
        )
        achieved = historical_response_risk(
            candidate,
            historical,
            historical_covariance,
            historical_sensitivity,
            eps=eps,
        )
        if float(achieved) <= float(risk_budget):
            break
        lower = upper
        upper *= 2.0
    else:
        raise RuntimeError("failed to bracket a feasible SBGC multiplier")

    for _ in range(int(bisection_steps)):
        midpoint = 0.5 * (lower + upper)
        trial = solve_lagrangian_candidate(
            historical,
            target,
            historical_covariance,
            current_covariance,
            historical_sensitivity,
            current_sensitivity,
            eta=midpoint,
            ridge=ridge,
        )
        trial_risk = historical_response_risk(
            trial,
            historical,
            historical_covariance,
            historical_sensitivity,
            eps=eps,
        )
        if float(trial_risk) <= float(risk_budget):
            upper = midpoint
            candidate = trial
            achieved = trial_risk
        else:
            lower = midpoint

    if candidate is None or achieved is None:
        raise RuntimeError("SBGC bisection did not produce a candidate")
    distortion = weighted_response_energy(
        candidate.to(torch.float64) - target.to(torch.float64),
        current_covariance,
        current_sensitivity,
    ) / (target_update_energy + float(eps))
    return candidate, {
        "eta": float(upper),
        "target_risk": float(target_risk),
        "achieved_risk": float(achieved),
        "current_distortion": float(distortion),
        "constraint_active": True,
        "bracket_steps": int(bracket_steps),
        "bisection_steps": int(bisection_steps),
    }


def update_running_moment(
    historical: Tensor,
    historical_count: float,
    current: Tensor,
    current_count: float,
) -> tuple[Tensor, float]:
    """Count-weighted update for a task-constant covariance or sensitivity."""
    if historical.shape != current.shape:
        raise ValueError("running moments must have equal shapes")
    if not bool(torch.isfinite(historical).all()) or not bool(
        torch.isfinite(current).all()
    ):
        raise ValueError("running moments must be finite")
    if not math.isfinite(float(historical_count)) or historical_count < 0:
        raise ValueError("historical_count must be finite and non-negative")
    if not math.isfinite(float(current_count)) or current_count <= 0:
        raise ValueError("current_count must be finite and positive")
    total = float(historical_count) + float(current_count)
    if historical_count == 0:
        return current.detach().clone(), total
    merged = (
        historical * float(historical_count) + current * float(current_count)
    ) / total
    return merged, total


def sbgc_state_scalar_counts(num_blocks: int, rank: int, dim: int) -> dict[str, int]:
    """Separate deployed factors from task-constant SBGC bookkeeping."""
    if not all(
        isinstance(value, int) and value > 0
        for value in (num_blocks, rank, dim)
    ):
        raise ValueError("num_blocks, rank and dim must be positive integers")
    branches = 2 * num_blocks
    factors = branches * 2 * dim * rank
    covariance = branches * rank * rank
    sensitivity = branches * dim
    counts = 2 * branches
    return {
        "lora_factor_scalars": factors,
        "covariance_scalars": covariance,
        "sensitivity_scalars": sensitivity,
        "count_scalars": counts,
        "persistent_scalar_total": factors + covariance + sensitivity + counts,
    }
