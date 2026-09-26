import pytest
import torch

from backbone.live_a_functional import solve_live_a_functional_merge


def problem(new_a=None):
    old_a = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    old_g = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    if new_a is None:
        new_a = torch.tensor([[1.0, 0.1]], dtype=torch.float32)
    current_up = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    v_old = torch.ones(2, dtype=torch.float32)
    v_new = torch.ones(2, dtype=torch.float32)
    f_old = torch.ones(2, dtype=torch.float32)
    f_new = torch.ones(2, dtype=torch.float32)
    return old_a, old_g, new_a, current_up, v_old, v_new, f_old, f_new


def test_live_a_merge_accounts_for_unrecoverable_old_rowspace_loss():
    args = problem()
    new_g, stats = solve_live_a_functional_merge(*args, risk_budget=0.05)
    old_a, old_g, new_a = args[:3]
    old_m = old_g.double() @ (old_a.double() / old_a.double().norm())
    new_m = new_g.double() @ (new_a.double() / new_a.double().norm())
    actual = (new_m - old_m).square().sum() / old_m.square().sum()
    assert new_g.dtype == torch.float32
    assert abs(float(actual) - stats["total_historical_risk"]) < 1e-6
    assert float(actual) <= 0.050001
    assert 0 < stats["irrecoverable_risk"] < stats["total_historical_risk"]
    assert stats["constraint_active"]


def test_live_a_merge_returns_additive_target_when_budget_allows_it():
    old_a, old_g, _, current_up, v_old, v_new, f_old, f_new = problem()
    new_g, stats = solve_live_a_functional_merge(
        old_a, old_g, old_a, 0.1 * current_up,
        v_old, v_new, f_old, f_new, risk_budget=0.05,
    )
    assert torch.allclose(new_g, old_g + 0.1 * current_up, atol=1e-6)
    assert stats["constraint_active"] is False
    assert stats["irrecoverable_risk"] < 1e-8


def test_live_a_merge_rejects_rowspace_with_no_feasible_historical_operator():
    args = problem(new_a=torch.tensor([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="irrecoverable"):
        solve_live_a_functional_merge(*args, risk_budget=0.05)


def test_live_a_merge_is_invariant_to_positive_a_gauge_scaling():
    args = problem()
    reference, _ = solve_live_a_functional_merge(*args, risk_budget=0.05)
    old_a, old_g, new_a, current_up, *stats = args
    scaled, _ = solve_live_a_functional_merge(
        3 * old_a, old_g, 7 * new_a, current_up, *stats, risk_budget=0.05
    )
    assert torch.allclose(reference, scaled, atol=1e-6)


def test_live_a_merge_uses_anisotropic_input_and_output_weights():
    old_a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    new_a = torch.tensor([[1.0, 0.0, 0.2], [0.0, 1.0, 0.1]])
    old_g = torch.tensor([[1.0, 0.2], [0.1, 0.8]])
    current_up = torch.tensor([[0.3, 0.2], [0.1, 0.1]])
    v_old = torch.tensor([10.0, 0.5, 2.0])
    v_new = torch.tensor([0.2, 5.0, 1.0])
    f_old = torch.tensor([2.0, 0.1])
    f_new = torch.tensor([0.2, 3.0])
    merged, stats = solve_live_a_functional_merge(
        old_a, old_g, new_a, current_up,
        v_old, v_new, f_old, f_new, risk_budget=0.10,
    )
    old_m = old_g.double() @ (old_a.double() / old_a.double().norm())
    new_m = merged.double() @ (new_a.double() / new_a.double().norm())
    difference = new_m - old_m
    numerator = (f_old.double()[:, None] * difference.square() * v_old.double()).sum()
    denominator = (f_old.double()[:, None] * old_m.square() * v_old.double()).sum()
    assert abs(float(numerator / denominator) - stats["total_historical_risk"]) < 1e-6
    assert stats["total_historical_risk"] <= 0.100001
    assert stats["irrecoverable_risk"] > 0


def test_live_a_merge_rejects_rank_deficient_new_basis():
    args = problem(new_a=torch.tensor([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="nonzero norm"):
        solve_live_a_functional_merge(*args, risk_budget=0.05)
