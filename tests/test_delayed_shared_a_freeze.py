"""A task-count schedule must only change the shared-A trainability."""

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import SA_STATE_FILENAME, SharedALoRA_ViT_timm
from tests.test_live_a_aggregate_backbone import _TinyViT


def _settings(path):
    return dict(
        r=3, filepath=str(path), train_a_all_tasks=True,
        cumulative_state=True, cumulative_merge="live_a_aggregate_b",
        live_a_coordinate_align=True,
        live_a_absorb_mode="bounded_norm_calibrated_absorb",
    )


def test_delayed_freeze_keeps_a_trainable_for_exactly_three_tasks(tmp_path):
    torch.manual_seed(281)
    pristine = _TinyViT(8)
    anchor = None
    state_keys = None
    for task in range(5):
        model = SharedALoRA_ViT_timm(
            copy.deepcopy(pristine), cur_task_index=task,
            freeze_a_after_tasks=3, **_settings(tmp_path),
        )
        assert all(module.weight.requires_grad == (task < 3) for module in model.w_As)
        assert all(module.weight.requires_grad for module in model.w_Bs)
        assert model.wrapped_param[0].param.requires_grad
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(2, 4, 8)).square().mean().backward()
            if task >= 3:
                assert all(module.weight.grad is None for module in model.w_As)
            optimizer.step()
        if task >= 3:
            assert all(torch.equal(module.weight, old) for module, old in zip(model.w_As, anchor))
        if task == 2:
            anchor = [module.weight.detach().clone() for module in model.w_As]
        model.save_lora_parameters(str(tmp_path), task)
        state = torch.load(tmp_path / SA_STATE_FILENAME, weights_only=True)
        if state_keys is None:
            state_keys = state.keys()
        assert state.keys() == state_keys
        assert sum(tensor.numel() for key in ("shared_a", "aggregate_up")
                   for tensor in state[key]) == 4 * 8 * 3


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "3"])
def test_invalid_freeze_task_count_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="freeze_a_after_tasks"):
        SharedALoRA_ViT_timm(
            _TinyViT(8), cur_task_index=0,
            freeze_a_after_tasks=value, **_settings(tmp_path),
        )


def test_delayed_freeze_rejects_adaptive_controller(tmp_path):
    with pytest.raises(ValueError, match="freeze_a_after_tasks"):
        SharedALoRA_ViT_timm(
            _TinyViT(8), cur_task_index=0, freeze_a_after_tasks=3,
            adaptive_a_enabled=True, **_settings(tmp_path),
        )


def test_delayed_freeze_requires_explicit_live_prefix(tmp_path):
    with pytest.raises(ValueError, match="freeze_a_after_tasks"):
        SharedALoRA_ViT_timm(
            _TinyViT(8), cur_task_index=0, freeze_a_after_tasks=3,
            **{**_settings(tmp_path), "train_a_all_tasks": False},
        )


def test_no_schedule_preserves_existing_frozen_and_live_endpoints(tmp_path):
    for live in (False, True):
        directory = tmp_path / str(live)
        torch.manual_seed(282)
        pristine = _TinyViT(8)
        settings = {**_settings(directory), "train_a_all_tasks": live}
        task0 = SharedALoRA_ViT_timm(copy.deepcopy(pristine), cur_task_index=0, **settings)
        task0.save_lora_parameters(str(directory), 0)
        task1 = SharedALoRA_ViT_timm(copy.deepcopy(pristine), cur_task_index=1, **settings)
        assert all(module.weight.requires_grad == live for module in task1.w_As)

