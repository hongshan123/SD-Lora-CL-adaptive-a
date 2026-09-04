"""Tests for Adaptive-A coordinate-plasticity gradient gating."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import (
    SA_STATE_FILENAME,
    SharedALoRA_ViT_timm,
    adaptive_a_layer_gradient,
    canonical_down_projection,
    low_rank_product_frobenius_norm,
)


class _TinyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x):
        return self.qkv(x)


class _TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _TinyAttention(dim)

    def forward(self, x):
        return self.attn(x)


class _TinyViT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dim)])
        self.head = nn.Identity()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _layer_gradient(
    gradient_q,
    gradient_v,
    a_q,
    a_v,
    b_q,
    b_v,
    g_q=None,
    g_v=None,
    **kwargs,
):
    return adaptive_a_layer_gradient(
        gradient_q=gradient_q,
        gradient_v=gradient_v,
        shared_a_q=a_q,
        shared_a_v=a_v,
        current_up_q=b_q,
        current_up_v=b_v,
        historical_up_q=g_q,
        historical_up_v=g_v,
        scale=torch.tensor(1.0),
        stability_weight=kwargs.pop("stability_weight", 1.0),
        gate_floor=kwargs.pop("gate_floor", 0.05),
        momentum=kwargs.pop("momentum", 0.9),
        eps=kwargs.pop("eps", 1e-8),
        **kwargs,
    )


def _adaptive_model(tmp_path, **kwargs):
    settings = {
        "train_a_all_tasks": True,
        "cumulative_state": True,
        "cumulative_merge": "live_a_aggregate_b",
        "live_a_coordinate_align": True,
        "adaptive_a_enabled": True,
    }
    settings.update(kwargs)
    return SharedALoRA_ViT_timm(
        _TinyViT(4), r=2, filepath=str(tmp_path / "run"),
        cur_task_index=0, **settings,
    )


def test_low_rank_product_frobenius_norm_matches_dense_product():
    torch.manual_seed(1)
    up = torch.randn(7, 3)
    down = torch.randn(3, 5)

    actual = low_rank_product_frobenius_norm(up, down)

    assert torch.allclose(actual, torch.linalg.matrix_norm(up @ down))


def test_adaptive_layer_gradient_decomposes_each_gradient_orthogonally():
    torch.manual_seed(2)
    a_q = torch.randn(2, 5)
    a_v = torch.randn(2, 5)
    gradient_q = torch.randn_like(a_q)
    gradient_v = torch.randn_like(a_v)
    result = _layer_gradient(
        gradient_q, gradient_v, a_q, a_v,
        torch.randn(5, 2), torch.randn(5, 2),
    )

    for raw, branch in ((gradient_q, "q"), (gradient_v, "v")):
        parallel = result["parallel_{}_gradient".format(branch)]
        perpendicular = result["perpendicular_{}_gradient".format(branch)]
        basis, _ = canonical_down_projection(
            a_q if branch == "q" else a_v
        )
        assert torch.allclose(parallel + perpendicular, raw, atol=1e-6)
        assert torch.allclose(
            perpendicular @ basis.t(),
            torch.zeros_like(perpendicular @ basis.t()),
            atol=1e-6,
        )


def test_adaptive_layer_gradient_keeps_no_history_gradient_exactly():
    torch.manual_seed(3)
    a_q = torch.randn(2, 4)
    a_v = torch.randn(2, 4)
    gradient_q = torch.randn_like(a_q)
    gradient_v = torch.randn_like(a_v)
    result = _layer_gradient(
        gradient_q, gradient_v, a_q, a_v,
        torch.randn(4, 2), torch.randn(4, 2),
    )

    assert torch.equal(result["gradient_q"], gradient_q)
    assert torch.equal(result["gradient_v"], gradient_v)
    assert result["raw_gate"].item() == 1.0
    assert result["gate"].item() == 1.0


def test_adaptive_layer_gradient_applies_floor_and_ema_limits():
    a_q = torch.tensor([[1.0, 0.0]])
    a_v = torch.tensor([[1.0, 0.0]])
    gradient_q = torch.tensor([[0.0, 1.0]])
    gradient_v = torch.tensor([[0.0, 1.0]])
    zero_up = torch.zeros(2, 1)
    large_up = torch.full((2, 1), 100.0)

    history_dominates = _layer_gradient(
        gradient_q, gradient_v, a_q, a_v, zero_up, zero_up,
        large_up, large_up, gate_floor=0.2, momentum=0.75,
    )
    current_dominates = _layer_gradient(
        gradient_q, gradient_v, a_q, a_v, large_up, large_up,
        zero_up, zero_up, gate_floor=0.2, momentum=0.75,
        previous_gate=history_dominates["gate"],
    )

    assert history_dominates["raw_gate"].item() == pytest.approx(0.2)
    assert history_dominates["gate"].item() == pytest.approx(0.2)
    assert current_dominates["raw_gate"].item() == pytest.approx(1.0)
    assert current_dominates["gate"].item() == pytest.approx(0.4)


def test_model_applies_one_layer_gate_to_q_and_v_gradients(tmp_path):
    model = _adaptive_model(tmp_path, adaptive_a_gate_floor=0.1)
    wrapper = model.lora_vit.blocks[0].attn.qkv
    with torch.no_grad():
        wrapper.aggregate_q.fill_(50.0)
        wrapper.aggregate_v.fill_(50.0)
        wrapper.b_q.weight.fill_(1.0)
        wrapper.b_v.weight.fill_(1.0)
    model.task_id = 1
    raw_q = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    raw_v = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
    model.w_As[0].weight.grad = raw_q.clone()
    model.w_As[1].weight.grad = raw_v.clone()

    applied = model.apply_adaptive_a_gradients()

    assert applied is not None
    assert applied["per_layer_mean_gate"] == pytest.approx(
        [applied["mean_gate"]]
    )
    assert torch.allclose(
        model.w_As[0].weight.grad,
        applied["layer_gradients"][0]["gradient_q"],
    )
    assert torch.allclose(
        model.w_As[1].weight.grad,
        applied["layer_gradients"][0]["gradient_v"],
    )
    layer = applied["layer_gradients"][0]
    assert layer["gate"].item() < 1.0


def test_adaptive_model_reports_task_local_diagnostics_without_state(tmp_path):
    model = _adaptive_model(tmp_path, adaptive_a_gate_floor=0.05)
    wrapper = model.lora_vit.blocks[0].attn.qkv
    with torch.no_grad():
        wrapper.aggregate_q.fill_(20.0)
        wrapper.aggregate_v.fill_(20.0)
    model.task_id = 1
    for _ in range(2):
        model.w_As[0].weight.grad = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        model.w_As[1].weight.grad = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
        )
        model.apply_adaptive_a_gradients()

    diagnostics = model.adaptive_a_diagnostics()
    assert diagnostics["observations"] == 2
    assert diagnostics["min_gate"] <= diagnostics["mean_gate"]
    assert diagnostics["max_gate"] >= diagnostics["mean_gate"]
    assert diagnostics["fraction_gate_below_0_1"] == pytest.approx(1.0)
    assert diagnostics["per_layer_mean_gate"] == pytest.approx(
        [diagnostics["mean_gate"]]
    )

    model.task_id = 0
    model.save_lora_parameters(str(tmp_path / "run"), task_id=0)
    state = torch.load(
        tmp_path / "run" / SA_STATE_FILENAME,
        map_location="cpu", weights_only=True,
    )
    assert not any("adaptive" in key for key in state)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"train_a_all_tasks": False}, "sa_train_a_all_tasks"),
        ({"cumulative_state": False}, "sa_cumulative_state"),
        ({"cumulative_merge": "gauge"}, "live_a_aggregate_b"),
        ({"live_a_coordinate_align": False}, "coordinate_align"),
    ],
)
def test_adaptive_a_rejects_incompatible_configuration(tmp_path, settings, message):
    with pytest.raises(ValueError, match=message):
        _adaptive_model(tmp_path, **settings)


def test_disabled_adaptive_a_hook_is_a_noop(tmp_path):
    model = _adaptive_model(tmp_path, adaptive_a_enabled=False)
    raw = torch.randn_like(model.w_As[0].weight)
    model.w_As[0].weight.grad = raw.clone()

    assert model.apply_adaptive_a_gradients() is None
    assert torch.equal(model.w_As[0].weight.grad, raw)
    assert model.adaptive_a_diagnostics() is None
