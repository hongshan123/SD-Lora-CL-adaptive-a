import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.lrpt import (
    apply_transport,
    bias_relative_error,
    fit_affine_low_rank_transport,
    fit_affine_rank_residuals,
    fit_low_rank_transport,
    fit_rank_residuals,
    transport_prediction_error,
)


def _make_paired_features(n=400, d=24, rank=4, noise=1e-6, seed=7):
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    u0 = torch.randn(d, rank)
    v0 = torch.randn(d, rank)
    y = x + (x @ v0) @ u0.t() + noise * torch.randn(n, d)
    return x, y, u0, v0


def test_fit_recovers_exact_low_rank_transport():
    x, y, u0, v0 = _make_paired_features(noise=0.0)
    u, v = fit_low_rank_transport(x, y, rank=4, reg=1e-6)

    assert u.shape == (24, 4)
    assert v.shape == (24, 4)

    # The fitted map should reproduce the ground-truth drift on held-out points.
    held = torch.randn(50, 24)
    pred = held + (held @ v) @ u.t()
    true = held + (held @ v0) @ u0.t()
    assert torch.allclose(pred, true, atol=1e-4)


def test_fit_rank_limits_spectrum():
    x, y, _, _ = _make_paired_features(noise=1e-6)
    u, v = fit_low_rank_transport(x, y, rank=3, reg=1e-6)

    w = u @ v.t()
    s = torch.linalg.svdvals(w)
    assert len(s) == 24
    assert s[:3].min() > 1e-6  # rank-3 map has 3 significant singular values
    assert s[3:].max() < 1e-5  # remaining spectrum is numerical noise

    # A rank-3 fit must be strictly worse than the true rank-4 drift.
    rel, _ = transport_prediction_error(x, y, u, v)
    assert rel > 1e-3


def test_apply_transport_updates_and_normalizes_prototypes():
    _, _, u0, v0 = _make_paired_features(noise=0.0)
    prototypes = {
        0: torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 18),
        1: torch.randn(24),
    }
    prototypes[1] = torch.nn.functional.normalize(prototypes[1], p=2, dim=0)

    moved = apply_transport(prototypes, u0, v0)

    assert set(moved.keys()) == {0, 1}
    assert moved[0].shape == (24,)
    assert torch.allclose(torch.linalg.norm(moved[0]), torch.tensor(1.0), atol=1e-6)
    assert moved[1].abs().max() > 0


def test_fit_requires_matching_shapes_and_valid_rank():
    x = torch.randn(20, 8)
    with pytest.raises(ValueError):
        fit_low_rank_transport(x, torch.randn(20, 9), rank=2)
    with pytest.raises(ValueError):
        fit_low_rank_transport(x, torch.randn(20, 8), rank=0)
    with pytest.raises(ValueError):
        fit_low_rank_transport(x, torch.randn(20, 8), rank=9)


def test_transport_prediction_error_is_zero_for_exact_fit():
    x, y, _, _ = _make_paired_features(noise=0.0)
    u_fit, v_fit = fit_low_rank_transport(x, y, rank=4, reg=1e-6)
    rel, frac = transport_prediction_error(x, y, u_fit, v_fit)
    assert rel == pytest.approx(0.0, abs=1e-6)
    assert frac == pytest.approx(1.0, abs=1e-6)


def test_fit_rank_residuals_decrease_with_rank():
    x, y, _, _ = _make_paired_features(noise=1e-5)
    residuals = fit_rank_residuals(x, y, (2, 4, 6), reg=1e-6)
    assert set(residuals.keys()) == {2, 4, 6}
    assert residuals[2] > residuals[4] > residuals[6]


def test_bias_relative_error_is_between_zero_and_one():
    x = torch.randn(100, 16)
    y = x + 0.3 + 0.1 * torch.randn(100, 16)
    rel = bias_relative_error(x, y)
    assert 0.0 <= rel <= 1.0
    assert rel < 1.0


def test_affine_fit_recovers_bias_plus_low_rank_map():
    torch.manual_seed(11)
    n, d, rank = 300, 20, 3
    x = torch.randn(n, d)
    x = x - x.mean(dim=0, keepdim=True)  # centered input -> exact recovery
    u0 = torch.randn(d, rank)
    v0 = torch.randn(d, rank)
    b0 = 0.5 * torch.randn(d)
    y = x + b0 + (x @ v0) @ u0.t()

    u, v, b = fit_affine_low_rank_transport(x, y, rank=rank, reg=1e-6)
    pred = x + b + (x @ v) @ u.t()
    assert torch.allclose(pred, y, atol=1e-4)
    assert torch.allclose(b, b0, atol=1e-4)


def test_apply_transport_with_bias_normalizes():
    torch.manual_seed(12)
    u = torch.randn(8, 2)
    v = torch.randn(8, 2)
    bias = torch.randn(8)
    prototypes = {0: torch.nn.functional.normalize(torch.randn(8), p=2, dim=0)}
    moved = apply_transport(prototypes, u, v, bias=bias)
    assert torch.allclose(torch.linalg.norm(moved[0]), torch.tensor(1.0), atol=1e-6)


def test_apply_transport_damping_scales_movement():
    torch.manual_seed(14)
    u = torch.randn(8, 2)
    v = torch.randn(8, 2)
    bias = torch.randn(8)
    proto = torch.nn.functional.normalize(torch.randn(8), p=2, dim=0)
    full = apply_transport({0: proto}, u, v, bias=bias, damping=1.0)[0]
    half = apply_transport({0: proto}, u, v, bias=bias, damping=0.5)[0]
    zero = apply_transport({0: proto}, u, v, bias=bias, damping=0.0)[0]
    assert torch.allclose(zero, proto, atol=1e-6)
    assert torch.linalg.norm(half - proto) < torch.linalg.norm(full - proto)


def test_affine_rank_residuals_improve_over_plain():
    torch.manual_seed(13)
    n, d, rank = 200, 16, 3
    x = torch.randn(n, d)
    u0 = torch.randn(d, rank)
    v0 = torch.randn(d, rank)
    b0 = 0.8 * torch.randn(d)
    y = x + b0 + (x @ v0) @ u0.t()
    plain = fit_rank_residuals(x, y, (rank,), reg=1e-6)
    affine = fit_affine_rank_residuals(x, y, (rank,), reg=1e-6)
    assert affine[rank] < plain[rank]
    assert affine[rank] < 0.1
