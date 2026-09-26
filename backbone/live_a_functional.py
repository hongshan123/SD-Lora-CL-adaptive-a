"""Task-boundary G consolidation after a live shared-A update."""

import math

import torch
from torch import Tensor

from backbone.sbgc import solve_sensitivity_budgeted_g


def solve_live_a_functional_merge(
    old_a: Tensor,
    old_g: Tensor,
    new_a: Tensor,
    current_up: Tensor,
    historical_input_second_moment: Tensor,
    current_input_second_moment: Tensor,
    historical_output_sensitivity: Tensor,
    current_output_sensitivity: Tensor,
    *,
    risk_budget: float = 0.05,
    ridge: float = 1e-6,
    bisection_steps: int = 40,
) -> tuple[Tensor, dict[str, float | bool]]:
    """Minimize current response distortion under total old-operator risk.

    The current branch must already be absorbed into ``current_up``, so
    ``current_up @ (new_a / ||new_a||)`` is its committed effective operator.
    Input moments use a diagonal second-moment approximation; output
    sensitivities use a diagonal Fisher approximation.
    """
    if not math.isfinite(float(risk_budget)) or not 0 < risk_budget <= 1:
        raise ValueError("risk_budget must be finite and in (0, 1]")
    matrices = (old_a, old_g, new_a, current_up)
    vectors = (
        historical_input_second_moment,
        current_input_second_moment,
        historical_output_sensitivity,
        current_output_sensitivity,
    )
    if any(t.ndim != 2 or not bool(torch.isfinite(t).all()) for t in matrices):
        raise ValueError("A and G factors must be finite matrices")
    if any(t.ndim != 1 or not bool(torch.isfinite(t).all()) for t in vectors):
        raise ValueError("input moments and sensitivities must be finite vectors")
    rank, input_dim = old_a.shape
    output_dim = old_g.shape[0]
    if (
        new_a.shape != (rank, input_dim)
        or old_g.shape != (output_dim, rank)
        or current_up.shape != old_g.shape
        or historical_input_second_moment.shape != (input_dim,)
        or current_input_second_moment.shape != (input_dim,)
        or historical_output_sensitivity.shape != (output_dim,)
        or current_output_sensitivity.shape != (output_dim,)
    ):
        raise ValueError("factor and statistic dimensions do not match")
    if len({t.device for t in matrices + vectors}) != 1:
        raise ValueError("all factors and statistics must share a device")
    if bool((historical_input_second_moment <= 0).any()) or bool(
        (current_input_second_moment <= 0).any()
    ):
        raise ValueError("input second moments must be strictly positive")
    if any(bool((t < 0).any()) or not bool(t.sum() > 0) for t in vectors[2:]):
        raise ValueError("output sensitivities must be nonnegative and nonzero")

    old_down = old_a.double()
    new_down = new_a.double()
    old_norm = old_down.norm()
    new_norm = new_down.norm()
    if not bool(old_norm > 1e-12) or not bool(new_norm > 1e-12):
        raise ValueError("old and new A must have nonzero norm")
    old_down = old_down / old_norm
    new_down = new_down / new_norm
    historical_operator = old_g.double() @ old_down
    v_old = historical_input_second_moment.double()
    v_new = current_input_second_moment.double()
    f_old = historical_output_sensitivity.double()
    f_new = current_output_sensitivity.double()
    old_cov = (new_down * v_old.unsqueeze(0)) @ new_down.T
    new_cov = (new_down * v_new.unsqueeze(0)) @ new_down.T
    old_cov = 0.5 * (old_cov + old_cov.T)
    new_cov = 0.5 * (new_cov + new_cov.T)
    if not bool(torch.linalg.cond(old_cov) < 1e10):
        raise ValueError("historical projected covariance is ill-conditioned")

    cross = (historical_operator * v_old.unsqueeze(0)) @ new_down.T
    aligned = torch.linalg.solve(old_cov, cross.T).T
    old_energy = (
        f_old[:, None] * historical_operator.square() * v_old.unsqueeze(0)
    ).sum()
    recovered_energy = (f_old[:, None] * (aligned @ old_cov) * aligned).sum()
    irrecoverable = (old_energy - recovered_energy).clamp_min(0.0)
    if not bool(old_energy > 1e-12):
        raise ValueError("historical response energy must be nonzero")
    allowance = float(risk_budget) * old_energy - irrecoverable
    if bool(allowance < -1e-9 * old_energy):
        raise ValueError("new A has irrecoverable historical risk above budget")
    allowance = allowance.clamp_min(0.0)
    target = aligned + current_up.double()
    if bool(allowance <= 1e-12 * old_energy):
        chosen = aligned
        solver = {"eta": float("inf"), "current_distortion": 1.0,
                  "constraint_active": True}
    else:
        effective_budget = float(allowance / recovered_energy.clamp_min(1e-12))
        chosen, solver = solve_sensitivity_budgeted_g(
            aligned, target, old_cov, new_cov, f_old, f_new,
            risk_budget=effective_budget, ridge=ridge,
            bisection_steps=bisection_steps,
        )

    deployed = chosen.to(device=old_g.device, dtype=old_g.dtype)
    difference = deployed.double() @ new_down - historical_operator
    deployed_risk = (
        f_old[:, None] * difference.square() * v_old.unsqueeze(0)
    ).sum() / old_energy
    if not bool(torch.isfinite(deployed_risk)) or bool(
        deployed_risk > float(risk_budget) + 1e-6
    ):
        raise RuntimeError("deployed Live-A functional merge exceeds risk budget")
    return deployed, {
        "irrecoverable_risk": float(irrecoverable / old_energy),
        "total_historical_risk": float(deployed_risk),
        "current_distortion": float(solver["current_distortion"]),
        "eta": float(solver["eta"]),
        "constraint_active": bool(solver["constraint_active"]),
    }
