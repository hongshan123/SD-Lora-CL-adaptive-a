import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.cuo_lowrank import (
    advance_projected_cuo,
    row_orthonormal_projection,
    solve_projected_cuo,
)


def test_projected_cuo_matches_direct_concatenated_ridge_solution():
    z0, y0 = torch.randn(13, 3), torch.randn(13, 7)
    z1, y1 = torch.randn(17, 3), torch.randn(17, 7)
    up0, gram0, _ = advance_projected_cuo(
        torch.zeros(7, 3), torch.zeros(3, 3), z0, y0, damping=0.2
    )
    up1, gram1, _ = advance_projected_cuo(up0, gram0, z1, y1, damping=0.2)
    z, y = torch.cat([z0, z1]), torch.cat([y0, y1])
    direct = torch.linalg.solve(
        z.T @ z + 0.2 * torch.eye(3), (y.T @ z).T
    ).T
    assert torch.allclose(up1, direct, atol=1e-6, rtol=1e-6)
    assert torch.allclose(gram1, z.T @ z, atol=1e-6, rtol=1e-6)


def test_projected_cuo_is_additive_over_batches():
    torch.manual_seed(3)
    z0, y0 = torch.randn(8, 4), torch.randn(8, 6)
    z1, y1 = torch.randn(11, 4), torch.randn(11, 6)
    sequential, gram, _ = advance_projected_cuo(
        *advance_projected_cuo(
            torch.zeros(6, 4), torch.zeros(4, 4), z0, y0, damping=0.4
        )[:2],
        z1,
        y1,
        damping=0.4,
    )
    combined, combined_gram, _ = advance_projected_cuo(
        torch.zeros(6, 4), torch.zeros(4, 4), torch.cat([z0, z1]), torch.cat([y0, y1]), damping=0.4
    )
    assert torch.allclose(sequential, combined, atol=1e-6, rtol=1e-6)
    assert torch.allclose(gram, combined_gram, atol=1e-6, rtol=1e-6)


def test_row_orthonormal_projection_uses_reduced_qr():
    projection = row_orthonormal_projection(torch.randn(3, 9))
    assert projection.shape == (3, 9)
    assert torch.allclose(projection @ projection.T, torch.eye(3), atol=1e-6)


@pytest.mark.parametrize("damping", [0.0, -0.1, float("nan"), float("inf")])
def test_projected_cuo_requires_positive_finite_damping(damping):
    with pytest.raises(ValueError, match="damping"):
        solve_projected_cuo(torch.eye(2), torch.ones(3, 2), damping)


def test_projected_cuo_rejects_invalid_shapes_and_nonsymmetric_gram():
    with pytest.raises(ValueError, match="shape"):
        solve_projected_cuo(torch.eye(2), torch.ones(3, 3), damping=0.1)
    with pytest.raises(ValueError, match="symmetric"):
        solve_projected_cuo(torch.tensor([[1.0, 2.0], [0.0, 1.0]]), torch.ones(3, 2), damping=0.1)


def test_projected_cuo_preserves_cross_dtype_and_device():
    gram = torch.eye(2, dtype=torch.float32)
    cross = torch.randn(5, 2, dtype=torch.float32)
    up, diagnostics = solve_projected_cuo(gram, cross, damping=0.1)
    assert up.dtype == cross.dtype
    assert up.device == cross.device
    assert diagnostics["condition_number"] > 0


def test_projected_cuo_rejects_nonfinite_statistics():
    with pytest.raises(ValueError, match="finite"):
        solve_projected_cuo(torch.tensor([[float("nan")]]), torch.ones(2, 1), damping=0.1)


def test_projected_cuo_handles_ill_conditioned_gram_with_damping():
    gram = torch.diag(torch.tensor([1e-12, 1.0]))
    cross = torch.ones(3, 2)
    up, _ = solve_projected_cuo(gram, cross, damping=0.2)
    assert torch.isfinite(up).all()
