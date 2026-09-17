"""Tests for Adaptive-A coordinate-plasticity gradient gating."""

from datetime import timedelta
import multiprocessing as mp
import sys
import logging
from pathlib import Path
import traceback
from uuid import uuid4

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import (
    SA_STATE_FILENAME,
    SharedALoRA_ViT_timm,
    adaptive_a_layer_gradient,
    canonical_down_projection,
    choose_pareto_knee_modes,
    choose_risk_budgeted_modes,
    crossfit_dot_utility,
    decompose_adaptive_a_gradient,
    effective_risk_budget,
    low_rank_product_frobenius_norm,
    signed_gradient_utility,
    weighted_operator_risk,
)
from models.sdlora import Learner as SDLoraLearner
from models import sa_sdlora as sa_sdlora_module
from models.sa_sdlora import (
    Learner as SharedALearner,
    crossfit_classification_gradients,
    validate_adaptive_a_config,
)
from utils.inc_net import get_backbone


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

    assert actual.item() == pytest.approx(
        torch.linalg.matrix_norm(up @ down).item()
    )


def test_risk_budgeted_candidates_are_exact_frozen_tangent_and_live():
    shared_a = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    gradient = torch.tensor(
        [[2.0, -1.0, 3.0, 4.0], [0.5, 1.5, -2.0, 1.0]]
    )

    candidates = decompose_adaptive_a_gradient(gradient, shared_a)

    assert torch.count_nonzero(candidates["frozen"]) == 0
    assert torch.equal(
        candidates["tangent"],
        torch.tensor([[2.0, -1.0, 0.0, 0.0], [0.5, 1.5, 0.0, 0.0]]),
    )
    assert torch.equal(candidates["live"], gradient)
    assert torch.allclose(
        candidates["live"],
        candidates["tangent"] + candidates["perpendicular"],
    )


def test_signed_utility_preserves_helpful_and_harmful_gradient_direction():
    candidate = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

    assert signed_gradient_utility(candidate, candidate).item() == pytest.approx(
        1.0
    )
    assert signed_gradient_utility(candidate, -candidate).item() == pytest.approx(
        -1.0
    )
    assert (
        signed_gradient_utility(torch.zeros_like(candidate), candidate).item()
        == 0.0
    )


def test_weighted_operator_risk_is_scale_invariant_in_historical_up():
    torch.manual_seed(17)
    historical_up = torch.randn(5, 2)
    shared_a = torch.randn(2, 4)
    delta_a = torch.randn(2, 4)
    input_rms = torch.tensor([0.25, 0.5, 1.0, 2.0])

    risk = weighted_operator_risk(
        historical_up, shared_a, delta_a, input_rms
    )
    scaled_risk = weighted_operator_risk(
        13.0 * historical_up, shared_a, delta_a, input_rms
    )

    assert risk.item() > 0
    assert scaled_risk.item() == pytest.approx(risk.item(), rel=1e-6)


def test_global_selector_uses_true_frozen_endpoint_for_negative_utility():
    layers = [
        {
            "frozen": {"utility": 0.0, "risk": 0.0},
            "tangent": {"utility": -0.2, "risk": 0.01},
            "live": {"utility": -0.5, "risk": 0.02},
        },
        {
            "frozen": {"utility": 0.0, "risk": 0.0},
            "tangent": {"utility": 0.4, "risk": 0.03},
            "live": {"utility": 0.9, "risk": 0.20},
        },
    ]

    selected = choose_risk_budgeted_modes(layers, risk_budget=0.05)

    assert selected["modes"] == ["frozen", "tangent"]
    assert selected["selected_risk"] <= 0.05 + 1e-8


def test_global_selector_selects_live_when_budget_and_utility_support_it():
    layers = [
        {
            "frozen": {"utility": 0.0, "risk": 0.0},
            "tangent": {"utility": 0.2, "risk": 0.01},
            "live": {"utility": 0.8, "risk": 0.04},
        },
        {
            "frozen": {"utility": 0.0, "risk": 0.0},
            "tangent": {"utility": 0.1, "risk": 0.01},
            "live": {"utility": 0.7, "risk": 0.03},
        },
    ]

    first = choose_risk_budgeted_modes(layers, risk_budget=0.08)
    second = choose_risk_budgeted_modes(layers, risk_budget=0.08)

    assert first == second
    assert first["modes"] == ["live", "live"]
    assert first["selected_risk"] == pytest.approx(0.07)

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


def test_adaptive_layer_gradient_emas_history_zero_perpendicular_drift():
    """Removing the history EMA on gauge-only updates must fail this test."""
    a_q = torch.tensor([[1.0, 0.0]])
    a_v = torch.tensor([[1.0, 0.0]])
    gradient_q = torch.tensor([[3.0, 0.0]])
    gradient_v = torch.tensor([[-2.0, 0.0]])
    up = torch.ones(2, 1)
    history = torch.full((2, 1), 5.0)

    first = _layer_gradient(
        gradient_q, gradient_v, a_q, a_v, up, up, history, history,
    )
    result = _layer_gradient(
        gradient_q, gradient_v, a_q, a_v, up, up, history, history,
        momentum=0.9, previous_gate=0.2,
    )

    assert first["raw_gate"].item() == 1.0
    assert first["gate"].item() == 1.0
    assert result["raw_gate"].item() == 1.0
    assert result["gate"].item() == pytest.approx(0.28)
    assert torch.equal(result["gradient_q"], gradient_q)
    assert torch.equal(result["gradient_v"], gradient_v)


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


@pytest.mark.parametrize(
    "gate_formula, expected_gate",
    [("ratio", 1.0 / 3.0), ("squared_ratio", 1.0 / 5.0)],
)
def test_adaptive_layer_gradient_supports_ratio_gate_formulas(
    gate_formula, expected_gate
):
    shared_a = torch.tensor([[1.0, 0.0]])
    perpendicular = torch.tensor([[0.0, 1.0]])
    current_q = torch.tensor([[2.0], [0.0]])
    historical_q = torch.tensor([[4.0], [0.0]])
    zero_up = torch.zeros(2, 1)

    result = _layer_gradient(
        perpendicular,
        perpendicular,
        shared_a,
        shared_a,
        current_q,
        zero_up,
        historical_q,
        zero_up,
        gate_floor=0.0,
        momentum=0.0,
        gate_formula=gate_formula,
    )

    assert result["current_impact"].item() == pytest.approx(2.0)
    assert result["historical_impact"].item() == pytest.approx(4.0)
    assert result["raw_gate"].item() == pytest.approx(expected_gate)
    assert result["gate"].item() == pytest.approx(expected_gate)


def test_adaptive_layer_gradient_rejects_unknown_gate_formula():
    shared_a = torch.tensor([[1.0, 0.0]])
    gradient = torch.tensor([[0.0, 1.0]])
    up = torch.ones(2, 1)

    with pytest.raises(ValueError, match="gate_formula"):
        _layer_gradient(
            gradient,
            gradient,
            shared_a,
            shared_a,
            up,
            up,
            up,
            up,
            gate_formula="cubic_ratio",
        )


def test_tangent_strategy_cancels_perpendicular_sgd_momentum(tmp_path):
    model = _adaptive_model(
        tmp_path,
        adaptive_a_strategy="tangent",
        adaptive_a_gate_floor=0.0,
        adaptive_a_gate_momentum=0.0,
    )
    model.task_id = 1
    with torch.no_grad():
        for module in model.w_As:
            module.weight.zero_()
            module.weight[0, 0] = 1.0
            module.weight[1, 1] = 1.0

    buffers = []
    for module in model.w_As:
        module.weight.grad = torch.zeros_like(module.weight)
        buffer = torch.zeros_like(module.weight)
        buffer[0, 2] = 2.0
        buffer[1, 3] = -3.0
        buffers.append(buffer)

    result = model.apply_adaptive_a_gradients(
        momentum_buffers=buffers,
        momentum=0.9,
    )

    assert result is not None
    assert result["mean_gate"] == pytest.approx(0.0)
    for module, buffer in zip(model.w_As, buffers):
        effective_direction = module.weight.grad + 0.9 * buffer
        assert torch.count_nonzero(effective_direction) == 0


def test_impact_ratio_gates_effective_sgd_momentum_direction(tmp_path):
    model = _adaptive_model(
        tmp_path,
        adaptive_a_gate_floor=0.0,
        adaptive_a_gate_momentum=0.0,
        adaptive_a_gate_formula="ratio",
    )
    model.task_id = 1
    wrapper = model.lora_vit.blocks[0].attn.qkv
    with torch.no_grad():
        for module in model.w_As:
            module.weight.zero_()
            module.weight[0, 0] = 1.0
            module.weight[1, 1] = 1.0
        wrapper.b_q.weight.fill_(1.0)
        wrapper.b_v.weight.fill_(1.0)
        wrapper.aggregate_q.fill_(4.0)
        wrapper.aggregate_v.fill_(4.0)

    buffers = []
    for module in model.w_As:
        module.weight.grad = torch.zeros_like(module.weight)
        buffer = torch.zeros_like(module.weight)
        buffer[:, 2:] = 1.0
        buffers.append(buffer)

    result = model.apply_adaptive_a_gradients(
        momentum_buffers=buffers,
        momentum=0.9,
    )

    layer = result["layer_gradients"][0]
    gate = layer["gate"]
    for branch, module, buffer in zip(("q", "v"), model.w_As, buffers):
        effective_direction = module.weight.grad + 0.9 * buffer
        expected = gate * 0.9 * buffer
        assert torch.allclose(effective_direction, expected, atol=1e-7)
        assert torch.allclose(
            effective_direction,
            layer["gradient_{}".format(branch)],
            atol=1e-7,
        )


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


def test_adaptive_model_resets_statistics_when_task_id_changes(tmp_path):
    model = _adaptive_model(tmp_path, adaptive_a_gate_floor=0.05)
    wrapper = model.lora_vit.blocks[0].attn.qkv
    task_zero_q = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    task_zero_v = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
    )
    model.w_As[0].weight.grad = task_zero_q.clone()
    model.w_As[1].weight.grad = task_zero_v.clone()

    model.apply_adaptive_a_gradients()

    assert torch.equal(model.w_As[0].weight.grad, task_zero_q)
    assert torch.equal(model.w_As[1].weight.grad, task_zero_v)

    model.task_id = 1
    with torch.no_grad():
        wrapper.aggregate_q.fill_(20.0)
        wrapper.aggregate_v.fill_(20.0)
    model.w_As[0].weight.grad = task_zero_q.clone()
    model.w_As[1].weight.grad = task_zero_v.clone()

    applied = model.apply_adaptive_a_gradients()

    layer = applied["layer_gradients"][0]
    diagnostics = model.adaptive_a_diagnostics()
    assert layer["gate"].item() == pytest.approx(layer["raw_gate"].item())
    assert diagnostics["observations"] == 1
    assert diagnostics["mean_gate"] == pytest.approx(layer["gate"].item())


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


def test_risk_budgeted_model_can_select_exact_frozen_gradient(tmp_path):
    model = _adaptive_model(
        tmp_path,
        adaptive_a_strategy="risk_budgeted",
        adaptive_a_risk_budget=1.0,
    )
    model.task_id = 1
    wrapper = model.lora_vit.blocks[0].attn.qkv
    with torch.no_grad():
        wrapper.aggregate_q.fill_(1.0)
        wrapper.aggregate_v.fill_(1.0)
    raw_q = torch.tensor(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]
    )
    raw_v = torch.tensor(
        [[0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 1.0]]
    )
    model.w_As[0].weight.grad = raw_q.clone()
    model.w_As[1].weight.grad = raw_v.clone()
    momentum_q = torch.full_like(raw_q, 0.25)
    momentum_v = torch.full_like(raw_v, -0.5)
    proposed_q = raw_q + 0.9 * momentum_q
    proposed_v = raw_v + 0.9 * momentum_v

    result = model.apply_adaptive_a_gradients(
        [-proposed_q, -proposed_v],
        momentum_buffers=[momentum_q, momentum_v],
        momentum=0.9,
    )

    assert result["selection"]["modes"] == ["frozen"]
    assert torch.allclose(
        0.9 * momentum_q + model.w_As[0].weight.grad,
        torch.zeros_like(raw_q),
    )
    assert torch.allclose(
        0.9 * momentum_v + model.w_As[1].weight.grad,
        torch.zeros_like(raw_v),
    )
    assert result["mode_fractions"]["frozen"] == pytest.approx(1.0)


def test_relative_risk_budget_scales_with_live_candidate_risk():
    layers = [
        {
            "frozen": {"risk": 0.0},
            "tangent": {"risk": 0.01},
            "live": {"risk": 0.04},
        },
        {
            "frozen": {"risk": 0.0},
            "tangent": {"risk": 0.02},
            "live": {"risk": 0.06},
        },
    ]

    assert effective_risk_budget(layers, 0.25, "absolute") == pytest.approx(0.25)
    assert effective_risk_budget(layers, 0.25, "relative") == pytest.approx(0.025)


def test_relative_risk_budget_requires_supported_mode():
    with pytest.raises(ValueError, match="risk_budget_mode"):
        effective_risk_budget([], 0.5, "unknown")


def test_pareto_knee_selects_balanced_nondominated_mode():
    layers = [
        {
            "frozen": {"risk": 0.0, "utility": 0.0},
            "tangent": {"risk": 0.2, "utility": 0.8},
            "live": {"risk": 1.0, "utility": 1.0},
        }
    ]

    selected = choose_pareto_knee_modes(layers)

    assert selected["modes"] == ["tangent"]
    assert selected["selected_risk"] == pytest.approx(0.2)
    assert selected["selected_utility"] == pytest.approx(0.8)
    assert selected["implied_risk_ratio"] == pytest.approx(0.2)


def test_pareto_knee_is_invariant_to_positive_affine_objective_scaling():
    layers = [
        {
            "frozen": {"risk": 2.0, "utility": -4.0},
            "tangent": {"risk": 4.0, "utility": 4.0},
            "live": {"risk": 12.0, "utility": 6.0},
        }
    ]
    scaled = [
        {
            mode: {
                "risk": 100.0 * values["risk"] + 7.0,
                "utility": 0.01 * values["utility"] - 3.0,
            }
            for mode, values in layers[0].items()
        }
    ]

    assert choose_pareto_knee_modes(layers)["modes"] == ["tangent"]
    assert choose_pareto_knee_modes(scaled)["modes"] == ["tangent"]


def test_pareto_knee_requires_explicit_crossfit_gradients(tmp_path):
    model = _adaptive_model(tmp_path, adaptive_a_strategy="pareto_knee")
    model.task_id = 1
    model.w_As[0].weight.grad = torch.ones_like(model.w_As[0].weight)
    model.w_As[1].weight.grad = torch.ones_like(model.w_As[1].weight)

    with pytest.raises(ValueError, match="crossfit"):
        model.apply_adaptive_a_gradients()


def test_pareto_balanced_crossfit_utility_selects_tangent_update(tmp_path):
    model = _adaptive_model(tmp_path, adaptive_a_strategy="pareto_knee")
    model.task_id = 1
    wrapper = model.lora_vit.blocks[0].attn.qkv
    with torch.no_grad():
        shared = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        model.w_As[0].weight.copy_(shared)
        model.w_As[1].weight.copy_(shared)
        wrapper.aggregate_q.fill_(1.0)
        wrapper.aggregate_v.fill_(1.0)
    full = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
    )
    model.w_As[0].weight.grad = full.clone()
    model.w_As[1].weight.grad = full.clone()
    crossfit = ([full.clone(), full.clone()], [full.clone(), full.clone()])

    selected = model.apply_adaptive_a_gradients(
        crossfit_gradients=crossfit,
        step_size=0.1,
    )

    expected_tangent = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    assert selected["selection"]["modes"] == ["tangent"]
    assert torch.allclose(model.w_As[0].weight.grad, expected_tangent)
    assert torch.allclose(model.w_As[1].weight.grad, expected_tangent)
    assert 0.0 <= selected["selection"]["implied_risk_ratio"] <= 1.0
    assert selected["implied_risk_ratio"] == pytest.approx(
        selected["selection"]["implied_risk_ratio"]
    )
    assert selected["mean_implied_risk_ratio"] == pytest.approx(
        selected["selection"]["implied_risk_ratio"]
    )
    assert selected["pareto_points"] >= 2


def test_crossfit_dot_utility_matches_symmetric_first_order_gain():
    train_gradient = torch.tensor([[1.0, 2.0]])
    control_gradient = torch.tensor([[3.0, -1.0]])
    train_candidate = torch.tensor([[0.5, 1.0]])
    control_candidate = torch.tensor([[2.0, -0.5]])

    utility = crossfit_dot_utility(
        train_candidate,
        control_gradient,
        control_candidate,
        train_gradient,
        step_size=0.2,
    )

    expected = 0.1 * (
        torch.sum(train_candidate * control_gradient)
        + torch.sum(control_candidate * train_gradient)
    )
    assert utility == pytest.approx(float(expected))


def test_crossfit_classification_gradients_use_disjoint_alternating_folds():
    network = _TrainingNet()
    with torch.no_grad():
        network.weight.weight.zero_()
        network.weight.bias.zero_()
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
    )
    targets = torch.tensor([0, 1, 0, 1])

    train, control = crossfit_classification_gradients(
        network,
        [network.weight.weight],
        inputs,
        targets,
        known_classes=0,
    )

    assert torch.allclose(
        train[0], torch.tensor([[-0.75, 0.0], [0.75, 0.0]])
    )
    assert torch.allclose(
        control[0], torch.tensor([[0.0, 0.75], [0.0, -0.75]])
    )


def test_pareto_learner_passes_crossfit_gradients_to_backbone():
    class CrossfitBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.w_As = nn.ModuleList([nn.Linear(2, 2, bias=False)])
            self.adaptive_a_strategy = "pareto_knee"
            self.received = None

        def apply_adaptive_a_gradients(self, **kwargs):
            self.received = kwargs

    class CrossfitNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = CrossfitBackbone()

        def forward(self, inputs, ortho_loss=False):
            logits = self.backbone.w_As[0](inputs)
            output = {"logits": logits, "features": logits}
            if ortho_loss:
                return output, torch.zeros((), device=inputs.device)
            return output

    learner = object.__new__(SharedALearner)
    learner._sa_adaptive_a_enabled = True
    learner._sa_adaptive_a_strategy = "pareto_knee"
    learner._sa_adaptive_a_crossfit_interval = 4
    learner._cur_task = 1
    learner._network = CrossfitNetwork()
    learner._known_classes = 0
    learner._raw_network = lambda: learner._network
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
    )
    targets = torch.tensor([0, 1, 0, 1])
    optimizer = optim.SGD(learner._network.parameters(), lr=0.1, momentum=0.9)
    F.cross_entropy(learner._network(inputs)["logits"], targets).backward()

    learner._after_backward(inputs, targets, optimizer)

    received = learner._network.backbone.received
    assert received is not None
    assert "crossfit_gradients" in received
    assert len(received["crossfit_gradients"]) == 2
    assert len(received["crossfit_gradients"][0]) == 1


@pytest.mark.parametrize(
    "task_id, step, interval, has_cache, expected",
    [
        (0, 0, 4, False, False),
        (0, 8, 4, True, False),
        (1, 0, 4, False, True),
        (1, 1, 4, True, False),
        (1, 4, 4, True, True),
        (2, 7, 8, True, False),
        (2, 8, 8, True, True),
        (1, 3, 1, True, True),
    ],
)
def test_pareto_crossfit_refresh_schedule(
    task_id, step, interval, has_cache, expected
):
    assert (
        sa_sdlora_module.should_refresh_pareto_crossfit(
            task_id, step, interval, has_cache
        )
        is expected
    )


def test_pareto_crossfit_interval_defaults_to_four_and_requires_positive_integer():
    validator = sa_sdlora_module.validate_pareto_crossfit_interval

    assert validator({}) == 4
    assert validator({"sa_adaptive_a_crossfit_interval": 8}) == 8
    with pytest.raises(ValueError, match="crossfit_interval"):
        validator({"sa_adaptive_a_crossfit_interval": 0})
    with pytest.raises(ValueError, match="crossfit_interval"):
        validator({"sa_adaptive_a_crossfit_interval": 4.5})


def test_pareto_learner_skips_task_zero_and_reuses_modes_until_refresh():
    class ScheduledBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.w_As = nn.ModuleList([nn.Linear(2, 2, bias=False)])
            self.adaptive_a_strategy = "pareto_knee"
            self.calls = []

        def apply_adaptive_a_gradients(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("crossfit_gradients") is not None:
                return {
                    "selection": {
                        "modes": ["tangent"],
                        "selected_utilities": [0.75],
                    }
                }
            return {
                "selection": {
                    "modes": kwargs.get("cached_modes", ["live"]),
                    "selected_utilities": kwargs.get(
                        "cached_utilities", [1.0]
                    ),
                }
            }

    class ScheduledNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = ScheduledBackbone()

    learner = object.__new__(SharedALearner)
    learner._sa_adaptive_a_enabled = True
    learner._sa_adaptive_a_strategy = "pareto_knee"
    learner._sa_adaptive_a_crossfit_interval = 4
    learner._network = ScheduledNetwork()
    learner._raw_network = lambda: learner._network
    learner._known_classes = 0
    crossfit_calls = []

    def record_crossfit(inputs, targets):
        crossfit_calls.append((inputs, targets))
        gradient = torch.ones_like(learner._network.backbone.w_As[0].weight)
        return ([gradient], [gradient])

    learner._crossfit_adaptive_a_gradients = record_crossfit
    inputs = torch.ones(4, 2)
    targets = torch.zeros(4, dtype=torch.long)
    optimizer = optim.SGD(learner._network.parameters(), lr=0.1)

    learner._cur_task = 0
    learner._after_backward(optimizer=optimizer)
    assert len(crossfit_calls) == 0
    assert learner._network.backbone.calls[-1]["crossfit_gradients"] is None

    learner._cur_task = 1
    for _ in range(5):
        learner._after_backward(inputs, targets, optimizer)

    assert len(crossfit_calls) == 2
    task_one_calls = learner._network.backbone.calls[1:]
    assert task_one_calls[0]["crossfit_gradients"] is not None
    assert task_one_calls[1]["cached_modes"] == ["tangent"]
    assert task_one_calls[1]["cached_utilities"] == [0.75]
    assert task_one_calls[4]["crossfit_gradients"] is not None

    learner._cur_task = 2
    learner._after_backward(inputs, targets, optimizer)
    assert len(crossfit_calls) == 3


def test_pareto_task_zero_keeps_current_shared_a_gradients_live(tmp_path):
    model = _adaptive_model(tmp_path, adaptive_a_strategy="pareto_knee")
    model.task_id = 0
    raw_q = torch.randn_like(model.w_As[0].weight)
    raw_v = torch.randn_like(model.w_As[1].weight)
    model.w_As[0].weight.grad = raw_q.clone()
    model.w_As[1].weight.grad = raw_v.clone()

    result = model.apply_adaptive_a_gradients()

    assert result["selection"]["modes"] == ["live"]
    assert torch.equal(model.w_As[0].weight.grad, raw_q)
    assert torch.equal(model.w_As[1].weight.grad, raw_v)


def test_pareto_cached_mode_projects_the_current_minibatch_gradient(tmp_path):
    model = _adaptive_model(tmp_path, adaptive_a_strategy="pareto_knee")
    model.task_id = 1
    wrapper = model.lora_vit.blocks[0].attn.qkv
    with torch.no_grad():
        wrapper.aggregate_q.fill_(1.0)
        wrapper.aggregate_v.fill_(1.0)
    raw_q = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    )
    raw_v = torch.tensor(
        [[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]]
    )
    model.w_As[0].weight.grad = raw_q.clone()
    model.w_As[1].weight.grad = raw_v.clone()
    expected_q = decompose_adaptive_a_gradient(
        raw_q, wrapper.a_q.weight
    )["tangent"]
    expected_v = decompose_adaptive_a_gradient(
        raw_v, wrapper.a_v.weight
    )["tangent"]

    result = model.apply_adaptive_a_gradients(
        cached_modes=["tangent"], cached_utilities=[0.75]
    )

    assert torch.allclose(model.w_As[0].weight.grad, expected_q)
    assert torch.allclose(model.w_As[1].weight.grad, expected_v)
    assert result["selection"]["modes"] == ["tangent"]
    assert result["selection"]["selected_utilities"] == [0.75]


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_pareto_crossfit_interval_rejects_non_positive_or_non_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        sa_sdlora_module.validate_pareto_crossfit_interval(
            {"sa_adaptive_a_crossfit_interval": value}
        )


def test_pareto_crossfit_interval_defaults_to_four_and_accepts_eight():
    assert sa_sdlora_module.validate_pareto_crossfit_interval({}) == 4
    assert (
        sa_sdlora_module.validate_pareto_crossfit_interval(
            {"sa_adaptive_a_crossfit_interval": 8}
        )
        == 8
    )


def test_risk_budgeted_model_uses_previous_minibatch_update_as_control(tmp_path):
    model = _adaptive_model(
        tmp_path,
        adaptive_a_strategy="risk_budgeted",
        adaptive_a_risk_budget=1.0,
    )
    model.task_id = 1
    wrapper = model.lora_vit.blocks[0].attn.qkv
    with torch.no_grad():
        wrapper.aggregate_q.fill_(1.0)
        wrapper.aggregate_v.fill_(1.0)
    first_q = torch.randn_like(model.w_As[0].weight)
    first_v = torch.randn_like(model.w_As[1].weight)
    model.w_As[0].weight.grad = first_q.clone()
    model.w_As[1].weight.grad = first_v.clone()

    warmup = model.apply_adaptive_a_gradients()
    assert warmup["selection"]["modes"] == ["frozen"]

    model.w_As[0].weight.grad = -first_q
    model.w_As[1].weight.grad = -first_v
    selected = model.apply_adaptive_a_gradients()

    assert selected["selection"]["modes"] == ["frozen"]
    assert torch.count_nonzero(model.w_As[0].weight.grad) == 0
    assert torch.count_nonzero(model.w_As[1].weight.grad) == 0


@pytest.mark.parametrize(
    "strategy", ["risk_budgeted", "pareto_knee", "function_safe_pareto"]
)
def test_adaptive_a_input_sketch_is_fixed_size_and_reloaded(tmp_path, strategy):
    run_dir = tmp_path / "run"
    model = _adaptive_model(
        tmp_path,
        adaptive_a_strategy=strategy,
    )
    model.train()
    model(torch.randn(2, 3, 4))
    model.save_lora_parameters(str(run_dir), task_id=0)

    state = torch.load(
        run_dir / SA_STATE_FILENAME, map_location="cpu", weights_only=True
    )
    assert len(state["adaptive_a_input_rms"]) == 1
    assert state["adaptive_a_input_rms"][0].shape == (4,)
    assert state["adaptive_a_input_counts"].shape == (1,)
    assert state["adaptive_a_input_counts"][0].item() == 6

    restored = SharedALoRA_ViT_timm(
        _TinyViT(4),
        r=2,
        filepath=str(run_dir),
        cur_task_index=1,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        live_a_coordinate_align=True,
        adaptive_a_enabled=True,
        adaptive_a_strategy=strategy,
    )
    wrapper = restored.lora_vit.blocks[0].attn.qkv
    assert torch.allclose(
        wrapper.historical_input_rms.cpu(), state["adaptive_a_input_rms"][0]
    )


class _TrainingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Linear(2, 2)

    def forward(self, inputs, ortho_loss=False):
        logits = self.weight(inputs)
        output = {"logits": logits, "features": logits}
        if ortho_loss:
            return output, torch.zeros((), device=inputs.device)
        return output


class _SingleBatchLoader:
    sampler = None

    def __init__(self):
        self.batch = (
            torch.tensor([0, 1]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
        )

    def __iter__(self):
        return iter([self.batch])


class _LoopBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_vit = nn.Module()
        self.lora_vit.patch_embed = nn.Identity()
        self.lora_vit.pos_drop = nn.Identity()


class _RecordingSGD(optim.SGD):
    def __init__(self, params, events):
        super().__init__(params, lr=0.1)
        self.events = events

    def step(self, closure=None):
        self.events.append("step")
        return super().step(closure)


def _loop_learner(events):
    learner = object.__new__(SDLoraLearner)
    learner.args = {
        "init_epoch": 1,
        "epochs": 1,
    }
    learner._network = _TrainingNet()
    learner._device = torch.device("cpu")
    learner._cur_task = 0
    learner._known_classes = 0
    learner._is_main_process = lambda: False
    learner._barrier = lambda: None
    learner._sync_sum = lambda value: float(value)
    learner._additional_training_losses = lambda *args, **kwargs: {}
    learner._after_backward = (
        lambda inputs=None, targets=None, optimizer=None: events.append("hook")
    )
    learner._raw_network = lambda: _BackboneHolder(_LoopBackbone())
    return learner


@pytest.mark.parametrize("loop_name", ["_init_train", "_update_representation"])
def test_training_loops_run_post_backward_hook_before_optimizer_step(loop_name):
    """Removing the hook call between backward and step must fail this test."""
    events = []
    learner = _loop_learner(events)
    optimizer = _RecordingSGD(learner._network.parameters(), events)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loader = _SingleBatchLoader()

    getattr(learner, loop_name)(loader, loader, optimizer, scheduler)

    assert events == ["hook", "step"]


class _AdaptiveBackbone(nn.Module):
    def __init__(self, events=None):
        super().__init__()
        self.events = events if events is not None else []

    def apply_adaptive_a_gradients(self, **kwargs):
        self.events.append("adaptive")

    def adaptive_a_diagnostics(self):
        self.events.append("diagnostics")
        return {
            "observations": 2,
            "mean_gate": 0.4,
            "min_gate": 0.2,
            "max_gate": 0.7,
            "fraction_gate_below_0_1": 0.0,
            "fraction_gate_above_0_9": 0.0,
            "mean_current_impact": 1.5,
            "mean_historical_impact": 3.0,
            "mean_perpendicular_retention": 0.4,
            "per_layer_mean_gate": [0.3, 0.5],
        }


class _BackboneHolder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs):
        return self.backbone(inputs)


class _GradientBearingAdaptiveBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[2.0, -1.0]]))
        self.apply_calls = 0

    def forward(self, inputs):
        return inputs @ self.weight.t()

    def apply_adaptive_a_gradients(self, **kwargs):
        self.apply_calls += 1
        self.weight.grad.zero_()


class _MomentumCapturingAdaptiveBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_As = nn.ModuleList(
            [nn.Linear(2, 1, bias=False), nn.Linear(2, 1, bias=False)]
        )
        self.received = None

    def apply_adaptive_a_gradients(self, **kwargs):
        self.received = kwargs


def _two_rank_adaptive_a_ddp_worker(rank, init_file, result_queue):
    """Exercise DDP reduction before the real learner post-backward hook."""
    try:
        dist.init_process_group(
            backend="gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=20),
        )
        torch.manual_seed(29)
        model = _adaptive_model(
            Path(init_file).parent / "rank-{}".format(rank),
            adaptive_a_gate_momentum=0.9,
        )
        model.task_id = 1
        wrapper = model.lora_vit.blocks[0].attn.qkv
        with torch.no_grad():
            wrapper.aggregate_q.fill_(3.0)
            wrapper.aggregate_v.fill_(2.0)
            wrapper.b_q.weight.fill_(1.0)
            wrapper.b_v.weight.fill_(0.5)

        learner = object.__new__(SharedALearner)
        learner._sa_adaptive_a_enabled = True
        learner._network = DDP(
            _BackboneHolder(model),
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        optimizer = optim.SGD(learner._network.parameters(), lr=0.05)
        inputs = torch.tensor(
            [[[1.0 + rank, -2.0, 0.5, 3.0 - rank]]]
        )
        output_weight = torch.linspace(0.25, 1.5, 12).reshape(1, 1, 12)
        local_loss = (learner._network(inputs) * output_weight).sum()
        local_loss.backward()
        raw_q = model.w_As[0].weight.grad.detach().clone()
        raw_v = model.w_As[1].weight.grad.detach().clone()

        assert learner._after_backward() is None
        gated_q = model.w_As[0].weight.grad.detach().clone()
        gated_v = model.w_As[1].weight.grad.detach().clone()
        gate = model._adaptive_a_gate_ema[wrapper.layer_index]
        optimizer.step()
        result_queue.put(
            {
                "rank": rank,
                "local_loss": float(local_loss.detach()),
                "raw_q": raw_q.tolist(),
                "raw_v": raw_v.tolist(),
                "gated_q": gated_q.tolist(),
                "gated_v": gated_v.tolist(),
                "a_q": model.w_As[0].weight.detach().tolist(),
                "a_v": model.w_As[1].weight.detach().tolist(),
                "gate": gate,
            }
        )
    except Exception:
        result_queue.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_disabled_shared_a_hook_leaves_actual_backbone_gradient_unchanged():
    """Disabled Adaptive-A must not invoke or mutate the learner backbone hook."""
    learner = object.__new__(SharedALearner)
    learner._sa_adaptive_a_enabled = False
    backbone = _GradientBearingAdaptiveBackbone()
    learner._network = _BackboneHolder(backbone)

    learner._network(torch.tensor([[3.0, -2.0]])).sum().backward()
    gradient_before_hook = backbone.weight.grad.detach().clone()

    assert learner._after_backward() is None
    assert backbone.apply_calls == 0
    assert torch.equal(backbone.weight.grad, gradient_before_hook)


@pytest.mark.parametrize("wrapped", [False, True])
def test_shared_a_hook_reaches_backbone_through_local_network_wrappers(wrapped):
    """Bypassing raw-network unwrapping would skip DataParallel backbones."""
    events = []
    learner = object.__new__(SharedALearner)
    learner._sa_adaptive_a_enabled = True
    holder = _BackboneHolder(_AdaptiveBackbone(events))
    learner._network = nn.DataParallel(holder) if wrapped else holder

    assert learner._after_backward() is None
    assert events == ["adaptive"]


@pytest.mark.parametrize("strategy", ["impact_ratio", "tangent"])
def test_shared_a_hook_forwards_optimizer_momentum_to_impact_paths(strategy):
    learner = object.__new__(SharedALearner)
    learner._sa_adaptive_a_enabled = True
    learner._sa_adaptive_a_strategy = strategy
    backbone = _MomentumCapturingAdaptiveBackbone()
    learner._network = _BackboneHolder(backbone)
    optimizer = optim.SGD(backbone.parameters(), lr=0.1, momentum=0.9)
    buffers = []
    for module in backbone.w_As:
        buffer = torch.full_like(module.weight, 2.0)
        optimizer.state[module.weight]["momentum_buffer"] = buffer
        buffers.append(buffer)

    assert learner._after_backward(optimizer=optimizer) is None
    assert backbone.received["momentum"] == pytest.approx(0.9)
    assert backbone.received["momentum_buffers"] == buffers


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_shared_a_hook_unwraps_real_single_rank_gloo_ddp(tmp_path):
    """The post-backward hook must reach the real module inside CPU DDP."""
    init_file = tmp_path / "adaptive-a-ddp-{}.init".format(uuid4().hex)
    dist.init_process_group(
        backend="gloo",
        init_method="file://{}".format(init_file),
        rank=0,
        world_size=1,
    )
    try:
        learner = object.__new__(SharedALearner)
        learner._sa_adaptive_a_enabled = True
        backbone = _GradientBearingAdaptiveBackbone()
        learner._network = DDP(_BackboneHolder(backbone))

        learner._network(torch.tensor([[3.0, -2.0]])).sum().backward()
        gradient_before_hook = backbone.weight.grad.detach().clone()

        assert learner._raw_network() is learner._network.module
        assert learner._after_backward() is None
        assert backbone.apply_calls == 1
        assert not torch.equal(backbone.weight.grad, gradient_before_hook)
        assert torch.count_nonzero(backbone.weight.grad) == 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_shared_a_hook_gates_synchronized_gradients_on_two_gloo_ranks(tmp_path):
    """Different local losses must yield equal gated Shared-A updates on DDP."""
    init_file = tmp_path / "adaptive-a-ddp-two-rank-{}.init".format(uuid4().hex)
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_two_rank_adaptive_a_ddp_worker,
            args=(rank, str(init_file), result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert all(not process.is_alive() for process in processes)
    assert all(process.exitcode == 0 for process in processes)

    results = [result_queue.get(timeout=5) for _ in processes]
    assert not [result["error"] for result in results if "error" in result]
    results.sort(key=lambda result: result["rank"])
    assert results[0]["local_loss"] != pytest.approx(results[1]["local_loss"])
    for key in ("raw_q", "raw_v", "gated_q", "gated_v", "a_q", "a_v"):
        assert torch.equal(
            torch.tensor(results[0][key]), torch.tensor(results[1][key])
        )
    assert results[0]["gate"] == pytest.approx(results[1]["gate"])


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"sa_train_a_all_tasks": False}, "sa_train_a_all_tasks"),
        ({"sa_cumulative_state": False}, "sa_cumulative_state"),
        ({"sa_cumulative_merge": "gauge"}, "live_a_aggregate_b"),
        ({"sa_live_a_coordinate_align": False}, "coordinate_align"),
        ({"sa_adaptive_a_gate_floor": 1.1}, "gate_floor"),
        ({"sa_adaptive_a_gate_formula": "cubic_ratio"}, "gate_formula"),
    ],
)
def test_learner_validates_enabled_adaptive_a_configuration(settings, message):
    """Dropping learner validation would allow an unsupported training path."""
    args = {
        "sa_adaptive_a_enabled": True,
        "sa_train_a_all_tasks": True,
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
    }
    args.update(settings)

    with pytest.raises(ValueError, match=message):
        validate_adaptive_a_config(args)


def test_adaptive_a_defaults_and_factory_forwarding(tmp_path, monkeypatch):
    """Omitting a setting must preserve disabled defaults in both constructors."""
    defaults = validate_adaptive_a_config({})
    assert defaults == {
        "adaptive_a_enabled": False,
        "adaptive_a_stability_weight": 1.0,
        "adaptive_a_gate_floor": 0.05,
        "adaptive_a_gate_momentum": 0.9,
        "adaptive_a_gate_formula": "ratio",
        "adaptive_a_eps": 1e-8,
        "adaptive_a_strategy": "impact_ratio",
        "adaptive_a_risk_budget": 0.05,
        "adaptive_a_risk_budget_mode": "absolute",
    }

    monkeypatch.setattr(
        "utils.inc_net.timm.create_model", lambda *args, **kwargs: _TinyViT(4)
    )
    backbone = get_backbone(
        {
            "backbone_type": "vit_base_patch16_224",
            "model_name": "sa_sdlora",
            "lora_rank": 2,
            "increment": 2,
            "filepath": str(tmp_path / "factory"),
            "sa_train_a_all_tasks": True,
            "sa_cumulative_state": True,
            "sa_cumulative_merge": "live_a_aggregate_b",
            "sa_live_a_coordinate_align": True,
            "sa_adaptive_a_enabled": True,
            "sa_adaptive_a_stability_weight": 2.5,
            "sa_adaptive_a_gate_floor": 0.2,
            "sa_adaptive_a_gate_momentum": 0.6,
            "sa_adaptive_a_gate_formula": "squared_ratio",
            "sa_adaptive_a_eps": 1e-6,
            "sa_adaptive_a_strategy": "risk_budgeted",
            "sa_adaptive_a_risk_budget": 0.025,
            "sa_adaptive_a_risk_budget_mode": "relative",
        }
    )

    assert backbone.adaptive_a_enabled is True
    assert backbone.adaptive_a_stability_weight == pytest.approx(2.5)
    assert backbone.adaptive_a_gate_floor == pytest.approx(0.2)
    assert backbone.adaptive_a_gate_momentum == pytest.approx(0.6)
    assert backbone.adaptive_a_gate_formula == "squared_ratio"
    assert backbone.adaptive_a_eps == pytest.approx(1e-6)
    assert backbone.adaptive_a_strategy == "risk_budgeted"
    assert backbone.adaptive_a_risk_budget == pytest.approx(0.025)
    assert backbone.adaptive_a_risk_budget_mode == "relative"


def test_direct_shared_a_constructor_forwards_adaptive_a_settings(monkeypatch):
    """Removing direct-constructor forwarding would silently disable the rule."""
    captured = {}

    class _CapturedBackbone:
        out_dim = 768

        def __init__(self, **kwargs):
            captured.update(kwargs)

    learner = object.__new__(SharedALearner)
    learner.args = {
        "lora_rank": 2,
        "increment": 2,
        "filepath": "run",
        "sa_adaptive_a_enabled": True,
        "sa_adaptive_a_stability_weight": 2.5,
        "sa_adaptive_a_gate_floor": 0.2,
        "sa_adaptive_a_gate_momentum": 0.6,
        "sa_adaptive_a_gate_formula": "squared_ratio",
        "sa_adaptive_a_eps": 1e-6,
        "sa_adaptive_a_strategy": "risk_budgeted",
        "sa_adaptive_a_risk_budget": 0.025,
        "sa_adaptive_a_risk_budget_mode": "relative",
        "sa_train_a_all_tasks": True,
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
    }
    learner._cur_task = 3
    monkeypatch.setattr(
        "models.sa_sdlora.timm.create_model", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr("models.sa_sdlora.SharedALoRA_ViT_timm", _CapturedBackbone)

    learner.update_network()

    assert {
        key: captured[key]
        for key in (
            "adaptive_a_enabled",
            "adaptive_a_stability_weight",
            "adaptive_a_gate_floor",
            "adaptive_a_gate_momentum",
            "adaptive_a_gate_formula",
            "adaptive_a_eps",
            "adaptive_a_strategy",
            "adaptive_a_risk_budget",
            "adaptive_a_risk_budget_mode",
        )
    } == {
        "adaptive_a_enabled": True,
        "adaptive_a_stability_weight": 2.5,
        "adaptive_a_gate_floor": 0.2,
        "adaptive_a_gate_momentum": 0.6,
        "adaptive_a_gate_formula": "squared_ratio",
        "adaptive_a_eps": 1e-6,
        "adaptive_a_strategy": "risk_budgeted",
        "adaptive_a_risk_budget": 0.025,
        "adaptive_a_risk_budget_mode": "relative",
    }


def test_adaptive_a_diagnostics_log_immediately_after_training_boundary(
    monkeypatch, caplog
):
    """Moving diagnostics before training completes would report stale gates."""
    events = []
    learner = object.__new__(SharedALearner)
    learner.args = {"sa_use_prototype_classifier": False, "filepath": "run"}
    learner._cur_task = 0
    learner._dual_head = False
    learner._hbd_enabled = False
    learner._hbd_teacher = None
    learner._hbd_teacher_handles = []
    learner._coordinate_stable_transport = False
    learner._lrpt_enabled = False
    learner._coordinate_pre_features = None
    learner._is_main_process = lambda: True
    backbone = _AdaptiveBackbone(events)
    backbone.cumulative_state = False
    holder = _BackboneHolder(backbone)
    backbone.save_merged_lora = lambda filepath: events.append("saved")
    learner._raw_network = lambda: holder
    learner._log_post_train_hash = lambda task_id: None
    monkeypatch.setattr(
        SDLoraLearner,
        "incremental_train",
        lambda self, data_manager: events.append("training_boundary"),
    )

    caplog.set_level(logging.INFO)
    learner.incremental_train(type("DataManager", (), {"nb_tasks": 2})())

    assert events[:2] == ["training_boundary", "diagnostics"]
    assert "AdaptiveA-SDLoRA" in caplog.text
    assert "per_layer_mean_gate=[0.3, 0.5]" in caplog.text
