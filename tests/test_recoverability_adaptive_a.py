import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.recoverability import align_to_anchor
from backbone.sa_lora import SA_STATE_FILENAME, SharedALoRA_ViT_timm
from models.sa_sdlora import validate_adaptive_a_config
from utils import inc_net


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
    def __init__(self, dim=4):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dim)])
        self.head = nn.Identity()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _model(tmp_path, stage="global_budget", task_id=0, **kwargs):
    settings = {
        "train_a_all_tasks": True,
        "cumulative_state": True,
        "cumulative_merge": "live_a_aggregate_b",
        "live_a_coordinate_align": True,
        "adaptive_a_enabled": True,
        "adaptive_a_strategy": "recoverability",
        "recoverability_stage": stage,
        "recoverability_budget": 1.0,
        "recoverability_step_size": 0.2,
        "recoverability_interval": 1,
        "recoverability_gammas": (0.0, 0.5, 1.0),
    }
    settings.update(kwargs)
    model = SharedALoRA_ViT_timm(
        _TinyViT(),
        r=2,
        filepath=str(tmp_path / "run"),
        cur_task_index=0,
        **settings,
    )
    model.task_id = task_id
    return model


def _backward_once(model):
    torch.manual_seed(41)
    inputs = torch.randn(3, 5, 4)
    output = model(inputs)
    output.square().mean().backward()
    return inputs


def test_recoverability_configuration_exposes_ordered_stages():
    base = {
        "sa_adaptive_a_enabled": True,
        "sa_adaptive_a_strategy": "recoverability",
        "sa_train_a_all_tasks": True,
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
        "optimizer": "sgd",
    }
    for stage in (
        "exact_risk",
        "accessibility",
        "anchor_realign",
        "global_budget",
    ):
        settings = validate_adaptive_a_config(
            {**base, "sa_recoverability_stage": stage}
        )
        assert settings["recoverability_stage"] == stage

    with pytest.raises(ValueError, match="sa_recoverability_stage"):
        validate_adaptive_a_config(
            {**base, "sa_recoverability_stage": "unknown"}
        )


def test_task_zero_backbone_receives_recoverability_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(
        inc_net.timm,
        "create_model",
        lambda *args, **kwargs: _TinyViT(),
    )
    args = {
        "backbone_type": "vit_base_patch16_224",
        "model_name": "sa_sdlora",
        "increment": 10,
        "filepath": str(tmp_path / "run"),
        "lora_rank": 2,
        "sa_train_a_all_tasks": True,
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
        "sa_adaptive_a_enabled": True,
        "sa_adaptive_a_strategy": "recoverability",
        "sa_recoverability_stage": "exact_risk",
        "sa_recoverability_budget": 0.02,
        "sa_recoverability_step_size": 0.15,
        "sa_recoverability_interval": 8,
        "sa_recoverability_sketch_rank": 3,
        "sa_recoverability_gammas": [0.0, 0.5, 1.0],
    }

    model = inc_net.get_backbone(args, pretrained=False)
    wrapper = model.lora_vit.blocks[0].attn.qkv

    assert model.recoverability_stage == "exact_risk"
    assert model.recoverability_budget == pytest.approx(0.02)
    assert model.recoverability_step_size == pytest.approx(0.15)
    assert model.recoverability_interval == 8
    assert model.recoverability_sketch_rank == 3
    assert model.recoverability_gammas == (0.0, 0.5, 1.0)
    assert wrapper.capture_effective_gradient is False


def test_effective_weight_gradient_is_captured_when_current_b_is_zero(tmp_path):
    model = _model(tmp_path, stage="accessibility")
    wrapper = model.lora_vit.blocks[0].attn.qkv
    assert torch.count_nonzero(wrapper.b_q.weight) == 0
    assert torch.count_nonzero(wrapper.b_v.weight) == 0

    _backward_once(model)
    q_gradient, v_gradient = wrapper.consume_effective_weight_gradients()

    assert q_gradient is not None and v_gradient is not None
    assert q_gradient.shape == (4, 4)
    assert v_gradient.shape == (4, 4)
    assert torch.linalg.vector_norm(q_gradient) > 0
    assert torch.linalg.vector_norm(v_gradient) > 0


def test_effective_weight_gradient_accumulates_multiple_loss_forwards(tmp_path):
    model = _model(tmp_path, stage="accessibility")
    wrapper = model.lora_vit.blocks[0].attn.qkv
    torch.manual_seed(42)
    first = torch.randn(2, 3, 4)
    second = torch.randn(2, 3, 4)
    loss = model(first).sum() + 2.0 * model(second).sum()

    loss.backward()
    q_gradient, v_gradient = wrapper.consume_effective_weight_gradients()
    expected = (
        torch.ones(first.numel() // 4, 4).t() @ first.reshape(-1, 4)
        + 2.0
        * (torch.ones(second.numel() // 4, 4).t() @ second.reshape(-1, 4))
    )

    assert torch.allclose(q_gradient, expected)
    assert torch.allclose(v_gradient, expected)


def test_canonicalization_preserves_historical_and_current_operators(tmp_path):
    model = _model(tmp_path, stage="exact_risk")
    wrapper = model.lora_vit.blocks[0].attn.qkv
    torch.manual_seed(43)
    with torch.no_grad():
        wrapper.a_q.weight.copy_(torch.randn_like(wrapper.a_q.weight))
        wrapper.a_v.weight.copy_(torch.randn_like(wrapper.a_v.weight))
        wrapper.b_q.weight.copy_(torch.randn_like(wrapper.b_q.weight))
        wrapper.b_v.weight.copy_(torch.randn_like(wrapper.b_v.weight))
        wrapper.aggregate_q.copy_(torch.randn_like(wrapper.aggregate_q))
        wrapper.aggregate_v.copy_(torch.randn_like(wrapper.aggregate_v))
    inputs = torch.randn(2, 3, 4)
    historical_before = tuple(value.detach().clone() for value in wrapper.historical_output(inputs))
    current_before = tuple(value.detach().clone() for value in wrapper.current_output(inputs))

    model._initialize_recoverability_state()
    historical_after = wrapper.historical_output(inputs)
    current_after = wrapper.current_output(inputs)

    for before, after in zip(historical_before, historical_after):
        assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)
    for before, after in zip(current_before, current_after):
        assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)
    for module in model.w_As:
        assert torch.allclose(
            module.weight @ module.weight.t(), torch.eye(2), atol=1e-5
        )


def test_accessibility_stage_prepares_full_task_zero_rotation_with_zero_b(tmp_path):
    model = _model(tmp_path, stage="accessibility", task_id=0)
    before = [module.weight.detach().clone() for module in model.w_As]
    _backward_once(model)

    result = model.prepare_recoverability_step(step_size=0.01)

    assert result["selected_gammas"] == [1.0, 1.0]
    assert all(torch.count_nonzero(module.weight.grad) == 0 for module in model.w_As)
    assert any(
        not torch.allclose(candidate, old)
        for candidate, old in zip(model._recoverability_pending_bases, before)
    )


def test_task_zero_forces_full_candidate_even_when_large_step_reduces_utility(tmp_path):
    model = _model(
        tmp_path,
        stage="accessibility",
        task_id=0,
        recoverability_step_size=1000.0,
    )
    _backward_once(model)

    result = model.prepare_recoverability_step(step_size=0.01)

    assert result["selected_gammas"] == [1.0, 1.0]


def test_exact_risk_stage_uses_original_gradient_direction(tmp_path):
    model = _model(tmp_path, stage="exact_risk", task_id=1)
    wrapper = model.lora_vit.blocks[0].attn.qkv
    with torch.no_grad():
        wrapper.recoverability_anchor_up_q.fill_(1.0)
        wrapper.recoverability_anchor_up_v.fill_(1.0)
        wrapper.aggregate_q.fill_(1.0)
        wrapper.aggregate_v.fill_(1.0)
    for module in model.w_As:
        module.weight.grad = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
    before = [module.weight.detach().clone() for module in model.w_As]

    result = model.prepare_recoverability_step(step_size=0.1)

    assert result["stage"] == "exact_risk"
    assert result["selected_gammas"] == [1.0, 1.0]
    for old, candidate in zip(before, model._recoverability_pending_bases):
        assert not torch.allclose(old, candidate)
        assert torch.allclose(
            candidate @ candidate.t(), torch.eye(2), atol=1e-5
        )


def test_anchor_realign_always_targets_original_task_start_operator(tmp_path):
    model = _model(tmp_path, stage="anchor_realign", task_id=1)
    wrapper = model.lora_vit.blocks[0].attn.qkv
    anchor_a = wrapper.recoverability_anchor_a_q.detach().clone()
    with torch.no_grad():
        anchor_up = torch.tensor(
            [[3.0, 0.0], [0.0, 1.0], [2.0, -1.0], [0.5, 0.25]]
        )
        wrapper.recoverability_anchor_up_q.copy_(anchor_up)

        first = anchor_a.clone()
        first[0, 2] = 0.3
        first = torch.linalg.qr(first.t(), mode="reduced").Q.t()
        model.w_As[0].weight.copy_(first)
    model.realign_recoverability_history()
    first_effective = wrapper.aggregate_q / torch.linalg.vector_norm(first)
    expected_first = align_to_anchor(anchor_up, anchor_a, first)
    assert torch.allclose(first_effective, expected_first, atol=1e-5)

    with torch.no_grad():
        second = anchor_a.clone()
        second[0, 3] = -0.4
        second = torch.linalg.qr(second.t(), mode="reduced").Q.t()
        model.w_As[0].weight.copy_(second)
    model.realign_recoverability_history()
    second_effective = wrapper.aggregate_q / torch.linalg.vector_norm(second)
    expected_second = align_to_anchor(anchor_up, anchor_a, second)

    assert torch.allclose(second_effective, expected_second, atol=1e-5)
    assert not torch.allclose(second_effective, align_to_anchor(first_effective, first, second))


def test_apply_step_retracts_basis_clears_momentum_and_realigns(tmp_path):
    model = _model(tmp_path, stage="anchor_realign", task_id=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    _backward_once(model)
    model.prepare_recoverability_step(step_size=0.01)
    expected = [candidate.clone() for candidate in model._recoverability_pending_bases]
    optimizer.step()

    model.apply_recoverability_step(optimizer)

    for module, candidate in zip(model.w_As, expected):
        assert torch.allclose(module.weight, candidate)
        assert torch.allclose(
            module.weight @ module.weight.t(), torch.eye(2), atol=1e-5
        )
        assert "momentum_buffer" not in optimizer.state[module.weight]
    assert model._recoverability_pending_bases is None


def test_recoverability_diagnostics_report_stage_budget_and_gate_distribution(tmp_path):
    model = _model(tmp_path, stage="global_budget", task_id=0)
    _backward_once(model)
    model.prepare_recoverability_step(step_size=0.01)

    diagnostics = model.adaptive_a_diagnostics()

    assert diagnostics["strategy"] == "recoverability"
    assert diagnostics["stage"] == "global_budget"
    assert diagnostics["observations"] == 1
    assert diagnostics["budget"] == pytest.approx(1.0)
    assert diagnostics["mean_gamma"] == pytest.approx(1.0)
    assert diagnostics["mean_risk"] == pytest.approx(0.0)


def test_recoverability_anchor_is_not_serialized(tmp_path):
    model = _model(tmp_path, stage="global_budget", task_id=0)
    model.save_lora_parameters(str(tmp_path / "run"), task_id=0)
    state = torch.load(
        tmp_path / "run" / SA_STATE_FILENAME,
        map_location="cpu",
        weights_only=True,
    )

    assert not any("anchor" in key or "recoverability" in key for key in state)


def test_online_aligned_state_is_not_aligned_a_second_time_when_saved(tmp_path):
    model = _model(tmp_path, stage="anchor_realign", task_id=1)
    wrapper = model.lora_vit.blocks[0].attn.qkv
    with torch.no_grad():
        wrapper.recoverability_anchor_up_q.copy_(torch.randn_like(wrapper.aggregate_q))
        wrapper.recoverability_anchor_up_v.copy_(torch.randn_like(wrapper.aggregate_v))
        moved_q = wrapper.a_q.weight.detach().clone()
        moved_v = wrapper.a_v.weight.detach().clone()
        moved_q[0, 2] = 0.25
        moved_v[1, 3] = -0.35
        wrapper.a_q.weight.copy_(torch.linalg.qr(moved_q.t(), mode="reduced").Q.t())
        wrapper.a_v.weight.copy_(torch.linalg.qr(moved_v.t(), mode="reduced").Q.t())
    model.realign_recoverability_history()
    expected = [
        wrapper.aggregate_q.detach().cpu().clone(),
        wrapper.aggregate_v.detach().cpu().clone(),
    ]

    model.save_lora_parameters(str(tmp_path / "run"), task_id=1)
    state = torch.load(
        tmp_path / "run" / SA_STATE_FILENAME,
        map_location="cpu",
        weights_only=True,
    )

    assert torch.allclose(state["aggregate_up"][0], expected[0], atol=1e-6)
    assert torch.allclose(state["aggregate_up"][1], expected[1], atol=1e-6)
