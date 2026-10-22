import torch
from torch.nn import functional as F

from backbone.margin_calibration import (
    class_balanced_weights,
    class_concentration,
    fit_old_group_bias,
    stratified_two_folds,
    synthetic_old_features,
)


def test_concentration_and_tangent_cosine_are_exact():
    prototypes = torch.eye(3)
    features = torch.tensor([[1., 0., 0.], [0.8, 0.6, 0.], [0., 1., 0.]])
    labels = torch.tensor([0, 0, 1])
    rho = class_concentration(features, labels, prototypes, [0, 1])
    assert torch.allclose(rho, torch.tensor([0.9, 1.0]))
    samples, synthetic_labels = synthetic_old_features(
        prototypes[:2], rho, samples_per_class=16, seed=7
    )
    similarities = (samples * prototypes[synthetic_labels]).sum(dim=1)
    assert torch.allclose(similarities, rho[synthetic_labels], atol=1e-6)
    repeated, _ = synthetic_old_features(prototypes[:2], rho, 16, 7)
    assert torch.equal(samples, repeated)


def test_folds_keep_every_class_in_fit_and_holdout():
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    folds = stratified_two_folds(labels, [0, 1], 1993)
    for cls in (0, 1):
        assert set(folds[labels == cls].tolist()) == {0, 1}


def test_scalar_bias_minimizes_balanced_cross_entropy():
    logits = torch.tensor([[0.5, 0.1], [0.3, 0.2], [0.6, 0.1], [0.2, 0.1]])
    labels = torch.tensor([0, 1, 0, 1])
    delta = fit_old_group_bias(logits, labels, 1)
    weight = class_balanced_weights(labels, 2)

    def loss(value):
        shifted = logits + torch.tensor([value, 0.])
        return (F.cross_entropy(shifted, labels, reduction="none") * weight).sum()

    assert -0.5 <= delta <= 0.5
    assert loss(delta) <= loss(0) + 1e-7
    assert loss(delta) <= loss(delta + 0.05) + 1e-7
    assert loss(delta) <= loss(delta - 0.05) + 1e-7


def test_invalid_concentration_and_missing_class_fail():
    try:
        synthetic_old_features(torch.eye(2), torch.tensor([0.5, 0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("zero concentration was accepted")
    try:
        class_concentration(torch.eye(2), torch.tensor([0, 1]), torch.eye(2), [2])
    except ValueError:
        pass
    else:
        raise AssertionError("missing class was accepted")
