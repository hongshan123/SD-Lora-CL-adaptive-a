import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from causal_subspace_intervention import TaskQKV, choose_spectral_down, recoverability


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
