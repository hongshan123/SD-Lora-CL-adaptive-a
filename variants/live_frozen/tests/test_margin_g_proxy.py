import torch

from scripts.evaluate_margin_g_proxy import current_data_old_kl, interpolate_g


def test_interpolation_keeps_a_and_hits_deployed_endpoint():
    previous = {"shared_a": [torch.eye(2)], "merged_b": [torch.ones(2, 2)]}
    target = {"shared_a": [torch.eye(2)], "merged_b": [torch.full((2, 2), 3.)]}
    midpoint = interpolate_g(previous, target, 0.5)
    final = interpolate_g(previous, target, 1.0)
    assert torch.equal(midpoint["shared_a"][0], previous["shared_a"][0])
    assert torch.equal(midpoint["merged_b"][0], torch.full((2, 2), 2.))
    assert torch.equal(final["merged_b"][0], target["merged_b"][0])


def test_old_logit_kl_is_zero_for_identical_models():
    features = torch.tensor([[1., 0.], [0., 1.], [0.5, 0.5], [1., 1.]])
    prototypes = torch.eye(2)
    plain, high = current_data_old_kl(
        features, features, prototypes, prototypes
    )
    assert abs(plain) < 1e-7
    assert abs(high) < 1e-7


def test_interpolation_rejects_changed_shared_basis():
    previous = {"shared_a": [torch.eye(2)], "merged_b": [torch.zeros(2, 2)]}
    target = {"shared_a": [2 * torch.eye(2)], "merged_b": [torch.ones(2, 2)]}
    try:
        interpolate_g(previous, target, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("changed A was accepted as a G-only intervention")
