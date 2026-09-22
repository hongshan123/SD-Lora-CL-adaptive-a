import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sbgc import (
    historical_response_risk,
    normalize_sensitivity,
    sbgc_state_scalar_counts,
    solve_lagrangian_candidate,
    solve_sensitivity_budgeted_g,
    update_running_moment,
    weighted_response_energy,
)


def _problem(dtype=torch.float32):
    torch.manual_seed(71)
    output_dim, rank = 9, 3
    historical = torch.randn(output_dim, rank, dtype=dtype)
    target = historical + 0.8 * torch.randn(output_dim, rank, dtype=dtype)
    old_data = torch.randn(80, rank, dtype=torch.float64)
    new_data = torch.randn(70, rank, dtype=torch.float64)
    old_covariance = (old_data.T @ old_data / len(old_data)).to(dtype)
    new_covariance = (new_data.T @ new_data / len(new_data)).to(dtype)
    old_sensitivity = normalize_sensitivity(
        torch.linspace(0.1, 2.0, output_dim, dtype=dtype)
    )
    new_sensitivity = normalize_sensitivity(
        torch.linspace(2.0, 0.1, output_dim, dtype=dtype)
    )
    return (
        historical,
        target,
        old_covariance,
        new_covariance,
        old_sensitivity,
        new_sensitivity,
    )


def test_weighted_response_energy_matches_explicit_activations():
    torch.manual_seed(3)
    operator = torch.randn(7, 3)
    activation = torch.randn(31, 3)
    covariance = activation.double().T @ activation.double() / len(activation)
    sensitivity = torch.rand(7)
    direct = (
        (activation.double() @ operator.double().T).square()
        * sensitivity.double().unsqueeze(0)
    ).sum() / len(activation)
    actual = weighted_response_energy(operator, covariance, sensitivity)
    assert torch.allclose(actual, direct, atol=1e-10, rtol=1e-10)


def test_zero_multiplier_recovers_target_exactly():
    problem = _problem()
    candidate = solve_lagrangian_candidate(*problem, eta=0.0)
    assert torch.allclose(candidate, problem[1], atol=2e-6, rtol=2e-6)


def test_budgeted_solution_satisfies_constraint_and_kkt_stationarity():
    problem = _problem()
    candidate, diagnostics = solve_sensitivity_budgeted_g(
        *problem, risk_budget=0.03, bisection_steps=48
    )
    assert diagnostics["constraint_active"]
    assert diagnostics["target_risk"] > 0.03
    assert diagnostics["achieved_risk"] <= 0.03 + 1e-8
    assert diagnostics["eta"] > 0

    historical, target, c_old, c_cur, f_old, f_cur = problem
    eta = diagnostics["eta"]
    residual = (
        f_cur[:, None] * ((candidate - target) @ c_cur)
        + eta * f_old[:, None] * ((candidate - historical) @ c_old)
    )
    scale = max(float(torch.linalg.norm(target)), 1.0)
    assert float(torch.linalg.norm(residual)) / scale < 2e-5


def test_risk_decreases_monotonically_with_multiplier():
    problem = _problem()
    historical, target, c_old, _, f_old, _ = problem
    risks = []
    for eta in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0):
        candidate = solve_lagrangian_candidate(*problem, eta=eta)
        risks.append(
            float(
                historical_response_risk(
                    candidate, historical, c_old, f_old
                )
            )
        )
    assert all(left + 1e-9 >= right for left, right in zip(risks, risks[1:]))


def test_target_inside_budget_is_returned_without_modification():
    problem = _problem()
    target_risk = float(
        historical_response_risk(
            problem[1], problem[0], problem[2], problem[4]
        )
    )
    candidate, diagnostics = solve_sensitivity_budgeted_g(
        *problem, risk_budget=target_risk + 0.1
    )
    assert torch.equal(candidate, problem[1])
    assert diagnostics["eta"] == 0.0
    assert not diagnostics["constraint_active"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_solver_accepts_low_precision_inputs_and_uses_stable_internal_math(dtype):
    problem = _problem(dtype=dtype)
    candidate, diagnostics = solve_sensitivity_budgeted_g(
        *problem, risk_budget=0.05
    )
    assert candidate.dtype == dtype
    assert torch.isfinite(candidate).all()
    assert diagnostics["achieved_risk"] <= 0.05 + 5e-3


def test_singular_covariances_remain_finite():
    problem = list(_problem())
    problem[2] = torch.diag(torch.tensor([2.0, 0.0, 0.0]))
    problem[3] = torch.diag(torch.tensor([0.0, 3.0, 0.0]))
    candidate, diagnostics = solve_sensitivity_budgeted_g(
        *problem, risk_budget=0.02
    )
    assert torch.isfinite(candidate).all()
    assert diagnostics["achieved_risk"] <= 0.02 + 1e-7


def test_sensitivity_normalization_and_invalid_zero_signal():
    value = normalize_sensitivity(torch.tensor([0.0, 1.0, 100.0]), floor=1e-4)
    assert torch.allclose(value.mean(), torch.tensor(1.0))
    assert value.min() > 0
    with pytest.raises(ValueError, match="strictly positive mean"):
        normalize_sensitivity(torch.zeros(4))


def test_running_moment_is_count_weighted_and_task_constant():
    old = torch.tensor([1.0, 3.0])
    current = torch.tensor([5.0, 7.0])
    merged, count = update_running_moment(old, 2.0, current, 1.0)
    assert count == 3.0
    assert torch.allclose(merged, torch.tensor([7.0 / 3.0, 13.0 / 3.0]))


def test_state_accounting_matches_rank10_vit_base_contract():
    counts = sbgc_state_scalar_counts(num_blocks=12, rank=10, dim=768)
    assert counts == {
        "lora_factor_scalars": 368640,
        "covariance_scalars": 2400,
        "sensitivity_scalars": 18432,
        "count_scalars": 48,
        "persistent_scalar_total": 389520,
    }
