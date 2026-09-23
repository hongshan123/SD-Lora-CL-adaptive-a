import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from causal_subspace_intervention import (
    TaskQKV,
    calibrate_new_recall,
    choose_spectral_down,
    evaluate_groups,
    gate_metrics,
    gate_score,
    recoverability,
    split_current_calibration,
)


def test_spectral_candidate_obeys_exact_historical_budget():
    torch.manual_seed(7)
    down = torch.randn(3, 16)
    up = torch.randn(16, 3)
    gradient = torch.randn(16, 16)
    basis, diagnostics = choose_spectral_down(down, up, gradient)
    assert basis.shape == down.shape
    assert diagnostics["historical_risk"] <= 0.05 + 1e-8
    assert recoverability(down, up, basis) <= 0.05 + 1e-6
    assert torch.allclose(basis @ basis.T, torch.eye(3), atol=1e-6)


def test_fixed_basis_deployment_preserves_current_output():
    torch.manual_seed(11)
    down = torch.randn(3, 16)
    up = torch.randn(16, 3)
    wrapper = TaskQKV(
        nn.Linear(16, 48), down, up, down, up,
        lambda: torch.tensor(0.8), live=False,
    )
    with torch.no_grad():
        wrapper.b_q.normal_()
        wrapper.b_v.normal_()
    inputs = torch.randn(2, 5, 16)
    before = wrapper(inputs)
    risks = wrapper.deploy()
    after = wrapper(inputs)
    assert max(risks) < 1e-12
    assert torch.allclose(before, after, atol=4e-6, rtol=1e-6)


def test_counterfactual_current_branch_can_be_disabled_without_changing_weights():
    torch.manual_seed(17)
    down = torch.randn(3, 16)
    up = torch.randn(16, 3)
    wrapper = TaskQKV(
        nn.Linear(16, 48), down, up, down, up,
        lambda: torch.tensor(0.8), live=False,
    )
    with torch.no_grad():
        wrapper.b_q.normal_()
        wrapper.b_v.normal_()
    inputs = torch.randn(2, 5, 16)
    full = wrapper(inputs)
    wrapper.current_enabled = False
    without_current = wrapper(inputs)
    expected_q = 0.8 * nn.functional.linear(
        nn.functional.linear(inputs, wrapper.a_q), wrapper.b_q
    )
    assert torch.allclose(full[..., :16] - without_current[..., :16], expected_q, atol=2e-6)
    wrapper.current_enabled = True
    assert torch.allclose(wrapper(inputs), full)


def test_old_only_accuracy_separates_new_prototype_competition():
    class IdentityFeatures(nn.Module):
        def forward(self, inputs):
            return torch.zeros(len(inputs), 3), inputs

    old = torch.tensor([[0.8, 0.6], [0.0, 1.0]])
    new = torch.tensor([[1.0, 0.0]])
    old_loader = DataLoader(TensorDataset(
        torch.arange(1), torch.tensor([[1.0, 0.0]]), torch.tensor([0]),
    ))
    new_loader = DataLoader(TensorDataset(
        torch.arange(1), torch.tensor([[1.0, 0.0]]), torch.tensor([2]),
    ))
    scores = evaluate_groups(
        IdentityFeatures(), old, new, old_loader, new_loader, torch.device("cpu")
    )
    assert scores["old"]["prototype_top1"] == 0.0
    assert scores["old"]["restricted_prototype_top1"] == 100.0


def test_calibration_split_is_disjoint_and_class_balanced():
    class Data:
        labels = torch.tensor([20] * 10 + [21] * 10)

    prototype, calibration = split_current_calibration(Data(), 20, 2, seed=7)
    assert len(prototype) == 16
    assert len(calibration) == 4
    assert set(prototype).isdisjoint(calibration)
    assert sum(i < 10 for i in calibration) == 2
    assert (prototype, calibration) == split_current_calibration(Data(), 20, 2, seed=7)


def test_gate_score_compares_history_old_with_full_new():
    old = torch.tensor([[1.0, 0.0]])
    new = torch.tensor([[0.0, 1.0]])
    history = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    full = history.clone()
    assert torch.allclose(gate_score(history, full, old, new), torch.tensor([-1.0, 1.0]))


def test_threshold_uses_only_new_calibration_scores():
    calibration = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    threshold = calibrate_new_recall(calibration, 0.75)
    assert threshold == 0.0
    assert (calibration >= threshold).float().mean() >= 0.75


def test_gate_metrics_decomposes_misrouting_cost_and_oracle_gap():
    old = {
        "score": torch.tensor([-1.0, 1.0]),
        "history_correct": torch.tensor([True, True]),
        "full_correct": torch.tensor([False, False]),
    }
    new = {
        "score": torch.tensor([2.0, -2.0]),
        "history_correct": torch.tensor([False, False]),
        "full_correct": torch.tensor([True, True]),
    }
    result = gate_metrics(old, new, threshold=0.0)
    assert result["routing_old_to_new_rate"] == 0.5
    assert result["routing_new_to_old_rate"] == 0.5
    assert result["oracle_top1"] == 100.0
    assert result["gated_top1"] == 50.0
    assert result["old_misroute_cost_pp"] == 50.0
    assert result["new_misroute_cost_pp"] == 50.0
    weighted_cost = 0.5 * result["old_misroute_cost_pp"] + 0.5 * result["new_misroute_cost_pp"]
    assert result["oracle_gap_pp"] == weighted_cost
