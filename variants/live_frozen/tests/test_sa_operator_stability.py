import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_operator_stability import (
    aggregate_normalized_up_projections,
    relative_effective_operator_drift,
)


def _make_operator_state(seed=23):
    torch.manual_seed(seed)
    reference_down = torch.randn(2, 5)
    up_weights = [torch.randn(7, 2), torch.randn(7, 2)]
    scales = [torch.tensor([0.6]), torch.tensor([1.1])]
    reference_up = aggregate_normalized_up_projections(up_weights, scales)
    return reference_down, up_weights, scales, reference_up


def test_effective_operator_loss_is_zero_at_task_boundary():
    reference_down, up_weights, scales, reference_up = _make_operator_state()
    current_down = reference_down.clone().requires_grad_()
    current_up = aggregate_normalized_up_projections(up_weights, scales)

    loss = relative_effective_operator_drift(
        current_down, reference_down, current_up, reference_up
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_effective_operator_is_invariant_to_down_projection_rescaling():
    reference_down, up_weights, scales, reference_up = _make_operator_state()
    current_up = aggregate_normalized_up_projections(up_weights, scales)

    loss = relative_effective_operator_drift(
        9.0 * reference_down, reference_down, current_up, reference_up
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_effective_operator_penalizes_shared_a_and_historical_scale_drift():
    reference_down, up_weights, scales, reference_up = _make_operator_state()
    current_down = (reference_down + 0.3 * torch.randn_like(reference_down)).requires_grad_()
    live_scales = [
        torch.nn.Parameter(scales[0].clone()),
        torch.nn.Parameter(0.7 * scales[1].clone()),
    ]
    current_up = aggregate_normalized_up_projections(up_weights, live_scales)

    loss = relative_effective_operator_drift(
        current_down, reference_down, current_up, reference_up
    )
    loss.backward()

    assert loss.item() > 0.0
    assert current_down.grad is not None
    assert torch.isfinite(current_down.grad).all()
    assert live_scales[0].grad is not None
    assert live_scales[1].grad is not None
    assert torch.isfinite(live_scales[0].grad).all()
    assert torch.isfinite(live_scales[1].grad).all()


def test_aggregate_rejects_invalid_historical_state():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_normalized_up_projections([], [])
    with pytest.raises(ValueError, match="counts must match"):
        aggregate_normalized_up_projections([torch.randn(4, 2)], [])
    with pytest.raises(ValueError, match="scalar"):
        aggregate_normalized_up_projections(
            [torch.randn(4, 2)], [torch.ones(2)]
        )


class _TinyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _TinyAttention(dim)


class _TinyViT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dim)])
        self.head = nn.Identity()


def test_shared_a_history_penalty_starts_at_zero_and_is_not_persistent(tmp_path):
    pytest.importorskip("timm")
    from backbone.sa_lora import SA_STATE_FILENAME, SharedALoRA_ViT_timm

    dim, rank = 6, 2
    task0 = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(tmp_path),
        cur_task_index=0,
        train_a_all_tasks=True,
    )
    assert task0.old_operator_stability_loss().item() == pytest.approx(0.0)

    shared_a = [torch.randn(rank, dim), torch.randn(rank, dim)]
    torch.save(
        {"version": 1, "shared_a": shared_a, "scales": {0: torch.tensor([0.8])}},
        tmp_path / SA_STATE_FILENAME,
    )
    torch.save(
        [torch.randn(dim, rank), torch.randn(dim, rank)],
        tmp_path / "sa_lora_w_b_0.pt",
    )
    task1 = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(tmp_path),
        cur_task_index=1,
        train_a_all_tasks=True,
    )

    assert task1.old_operator_stability_loss().item() == pytest.approx(0.0, abs=1e-7)
    with torch.no_grad():
        task1.w_As[0].weight.add_(0.25)
    assert task1.old_operator_stability_loss().item() > 0.0
    task1.save_lora_parameters(str(tmp_path), task_id=1)
    assert task1.old_operator_stability_loss().item() > 0.0
    assert not any("_operator_reference" in key for key in task1.state_dict())
