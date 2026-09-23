import torch

from scripts.offline_subspace_allocation import (
    analyze_branch,
    normalized_residual,
    spectral_basis,
)


def test_current_operator_from_old_basis_cannot_request_new_space():
    down = torch.eye(2, 4, dtype=torch.float64)
    up = torch.eye(2, dtype=torch.float64)
    current = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64) @ down
    assert normalized_residual(current, down) < 1e-12
    result = analyze_branch(down, up, current, sketch_rank=2)
    assert result["old_offspace_fraction"] < 1e-12
    assert abs(result["best_feasible"]["current_coverage_gain"]) < 1e-12


def test_independent_demand_can_replace_low_energy_history_direction():
    torch.manual_seed(7)
    down = torch.eye(2, 4, dtype=torch.float64)
    up = torch.diag(torch.tensor([10.0, 1.0], dtype=torch.float64))
    demand = torch.zeros(2, 4, dtype=torch.float64)
    demand[0, 2] = 1.0
    result = analyze_branch(down, up, demand, demand, sketch_rank=2)
    assert result["old_offspace_fraction"] > 0.999999
    selected = result["best_feasible"]
    assert selected["history_risk"] <= 0.05 + 1e-8
    assert selected["current_coverage_gain"] > 0.9
    assert selected["holdout_coverage_gain"] > 0.9
    assert selected["holdout_descent_alignment_gain"] > 0.9


def test_opposed_holdout_gradient_does_not_count_as_descent_gain():
    torch.manual_seed(7)
    down = torch.eye(2, 4, dtype=torch.float64)
    up = torch.diag(torch.tensor([10.0, 1.0], dtype=torch.float64))
    demand = torch.zeros(2, 4, dtype=torch.float64)
    demand[0, 2] = 1.0
    result = analyze_branch(down, up, demand, -demand, sketch_rank=2)
    assert result["best_feasible"]["holdout_coverage_gain"] > 0.9
    assert result["best_feasible"]["holdout_descent_alignment_gain"] < -0.9


def test_spectral_basis_solves_weighted_coverage():
    history = torch.diag(torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64))
    demand = torch.diag(torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64))
    basis = spectral_basis(history, demand, rank=1, current_weight=2.0)
    assert normalized_residual(demand, basis) < 1e-12
