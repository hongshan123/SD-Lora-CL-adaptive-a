import torch

from scripts.live_frozen_stages import (
    build_stage_factors,
    effective_operator,
    recoverability_diagnostics,
    weighted_response_energy,
)


def test_stage_factors_separate_a_alignment_exact_absorb_and_normcap():
    old_a = torch.tensor([[1.0, 0.0]])
    old_g = torch.tensor([[2.0], [0.0]])
    new_a = torch.tensor([[1.0, 1.0]])
    current_b = torch.tensor([[0.0], [1.0]])
    previous = {"shared_a": [old_a], "aggregate_up": [old_g]}
    pre = {"task_id": 1, "current_scale": torch.tensor(0.5),
           "normalize_current_branch": False,
           "branches": [{"down": new_a, "current_up": current_b,
                         "historical_up": old_g}]}
    states, diagnostics = build_stage_factors(previous, pre)
    actual = {name: effective_operator(state, 0) for name, state in states.items()}
    assert torch.allclose(actual["S0"], torch.tensor([[2., 0.], [0., 0.]]))
    assert torch.allclose(actual["S1"], torch.tensor([[2**0.5, 2**0.5], [0., 0.]]))
    assert torch.allclose(actual["S2"], torch.tensor([[2**0.5, 2**0.5], [.5, .5]]))
    assert torch.allclose(actual["S3"], torch.tensor([[1., 1.], [.5, .5]]))
    assert torch.allclose(actual["S4"], actual["S3"], atol=1e-6)
    assert torch.allclose(actual["S5"], torch.tensor([[1., 1.], [2**-1.5, 2**-1.5]]))
    assert torch.allclose(actual["H_LS"], torch.tensor([[1., 1.], [0., 0.]]))
    assert abs(diagnostics[0]["normcap_gain"] - 2**-0.5) < 1e-6


def test_recoverability_uses_historical_operator_and_old_layer_inputs():
    old_a = torch.tensor([[1.0, 0.0]])
    old_g = torch.tensor([[2.0], [0.0]])
    new_a = torch.tensor([[1.0, 1.0]])
    diagnostics = recoverability_diagnostics(old_g, old_a, new_a)
    assert abs(diagnostics["r_rec"] - 0.5) < 1e-6
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    lost, history = weighted_response_energy(old_g, old_a, new_a, x)
    assert abs(lost - 2.0) < 1e-6
    assert abs(history - 4.0) < 1e-6
