import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.cuo_lowrank import (
    advance_projected_cuo,
    row_orthonormal_projection,
    solve_projected_cuo,
)
from backbone.lora import ParameterWrapper
from backbone.sa_lora import _CUOLowRankQKV
from models.sa_sdlora import validate_cuo_lowrank_config


def make_cuo_wrapper(dim=8, rank=3):
    qkv = nn.Linear(dim, 3 * dim, bias=False)
    a_q = nn.Linear(dim, rank, bias=False)
    a_v = nn.Linear(dim, rank, bias=False)
    b_q = nn.Linear(rank, dim, bias=False)
    b_v = nn.Linear(rank, dim, bias=False)
    projection_q = torch.randn(rank, dim)
    projection_v = torch.randn(rank, dim)
    unified_up_q = torch.randn(dim, rank)
    unified_up_v = torch.randn(dim, rank)
    scaling = nn.ModuleList(
        [ParameterWrapper(nn.Parameter(torch.tensor([0.7])))]
    )
    return _CUOLowRankQKV(
        qkv,
        a_q,
        a_v,
        b_q,
        b_v,
        projection_q,
        unified_up_q,
        projection_v,
        unified_up_v,
        scaling,
        layer_index=0,
    )


def test_cuo_historical_branch_uses_fixed_projection_after_a_changes():
    wrapper = make_cuo_wrapper(dim=8, rank=3)
    inputs = torch.randn(2, 5, 8)
    before_q, _ = wrapper.historical_output(inputs)
    with torch.no_grad():
        wrapper.a_q.weight.add_(torch.randn_like(wrapper.a_q.weight))
    after_q, _ = wrapper.historical_output(inputs)
    assert torch.allclose(before_q, after_q, atol=1e-6, rtol=1e-6)


def test_cuo_wrapper_collects_fp64_projected_total_residual_statistics():
    torch.manual_seed(19)
    wrapper = make_cuo_wrapper(dim=6, rank=2)
    inputs = torch.randn(3, 4, 6)
    wrapper.begin_cuo_calibration()
    wrapper(inputs)

    (gram_q, cross_q, count_q), (gram_v, cross_v, count_v) = (
        wrapper.consume_cuo_statistics()
    )
    historical_q, historical_v = wrapper.historical_output(inputs)
    current_q, current_v = wrapper.current_output(inputs)
    z_q = inputs.double().reshape(-1, 6) @ wrapper.projection_q.double().T
    z_v = inputs.double().reshape(-1, 6) @ wrapper.projection_v.double().T
    y_q = (historical_q + current_q).double().reshape(-1, 6)
    y_v = (historical_v + current_v).double().reshape(-1, 6)

    assert gram_q.dtype == torch.float64
    assert cross_q.dtype == torch.float64
    assert count_q == 12
    assert torch.allclose(gram_q, z_q.T @ z_q)
    assert torch.allclose(cross_q, y_q.T @ z_q)
    assert gram_v.dtype == torch.float64
    assert cross_v.dtype == torch.float64
    assert count_v == 12
    assert torch.allclose(gram_v, z_v.T @ z_v)
    assert torch.allclose(cross_v, y_v.T @ z_v)


def test_cuo_wrapper_does_not_collect_in_normal_training():
    wrapper = make_cuo_wrapper(dim=6, rank=2).train()
    wrapper(torch.randn(2, 3, 6))
    (gram_q, cross_q, count_q), (gram_v, cross_v, count_v) = (
        wrapper.consume_cuo_statistics()
    )
    assert count_q == 0
    assert count_v == 0
    assert torch.count_nonzero(gram_q) == 0
    assert torch.count_nonzero(cross_q) == 0
    assert torch.count_nonzero(gram_v) == 0
    assert torch.count_nonzero(cross_v) == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sa_cumulative_state": False}, "sa_cumulative_state"),
        ({"sa_train_a_all_tasks": False}, "sa_train_a_all_tasks"),
        ({"sa_cumulative_rank": 2}, "sa_cumulative_rank"),
        ({"sa_cuo_lambda": 0.0}, "sa_cuo_lambda"),
        ({"sa_cuo_lambda": float("nan")}, "sa_cuo_lambda"),
        ({"sa_coordinate_stable_transport": True}, "CoordinateStable"),
        ({"sa_live_a_coordinate_align": True}, "CoordinateStable"),
        ({"sa_hbd_enabled": True}, "HBD"),
        ({"sa_adaptive_a_enabled": True}, "Adaptive-A"),
        ({"sa_adaptive_a_strategy": "function_safe_pareto"}, "function-safe"),
    ],
)
def test_cuo_lowrank_rejects_incompatible_configuration(overrides, message):
    config = {
        "sa_cumulative_merge": "cuo_lowrank",
        "sa_cumulative_state": True,
        "sa_train_a_all_tasks": True,
        "lora_rank": 3,
        "sa_cumulative_rank": 3,
        "sa_cuo_lambda": 0.1,
    }
    config.update(overrides)
    with pytest.raises(ValueError, match=message):
        validate_cuo_lowrank_config(config)


def test_cuo_lowrank_accepts_fixed_projection_configuration():
    config = {
        "sa_cumulative_merge": "cuo_lowrank",
        "sa_cumulative_state": True,
        "sa_train_a_all_tasks": True,
        "lora_rank": 3,
        "sa_cumulative_rank": 3,
        "sa_cuo_lambda": 0.1,
    }
    assert validate_cuo_lowrank_config(config) == pytest.approx(0.1)


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
