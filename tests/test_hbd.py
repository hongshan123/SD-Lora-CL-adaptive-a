"""Unit tests for Historical-Branch Activation Distillation (HBD)."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import (
    SA_STATE_FILENAME,
    SharedALoRA_ViT_timm,
    _LiveAAggregateQKV,
    hbd_historical_branch_distance,
    live_a_historical_outputs,
    register_live_a_historical_capture_hooks,
)
from models.sa_sdlora import Learner


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
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def _outputs(seed=0, shape=(4, 6, 8), layers=3):
    torch.manual_seed(seed)
    return [
        (torch.randn(*shape), torch.randn(*shape))
        for _ in range(layers)
    ]


def test_hbd_distance_zero_for_identical_outputs():
    outputs = _outputs()
    distance = hbd_historical_branch_distance(outputs, outputs)
    assert torch.allclose(distance, torch.zeros(()), atol=1e-6)


def test_hbd_distance_one_for_orthogonal_outputs():
    left = [
        (torch.ones(4, 6, 8), torch.ones(4, 6, 8))
        for _ in range(3)
    ]
    right = [
        (torch.zeros(4, 6, 8), torch.zeros(4, 6, 8))
        for _ in range(3)
    ]
    # Zero vectors normalize to zero; cosine = 0, distance = 1.
    distance = hbd_historical_branch_distance(left, right)
    assert torch.allclose(distance, torch.ones(()), atol=1e-6)


def test_hbd_distance_rejects_layer_count_mismatch():
    left = _outputs(layers=3)
    right = _outputs(layers=2)
    with pytest.raises(ValueError):
        hbd_historical_branch_distance(left, right)


def _run_tasks(tmp_path, tasks):
    dim, rank = 6, 2
    torch.manual_seed(31)
    run = tmp_path / "run"
    for task in range(tasks):
        model = SharedALoRA_ViT_timm(
            _TinyViT(dim),
            r=rank,
            filepath=str(run),
            cur_task_index=task,
            train_a_all_tasks=True,
            cumulative_state=True,
            cumulative_merge="live_a_aggregate_b",
        )
        with torch.no_grad():
            for w in model.w_Bs:
                w.weight.copy_(torch.randn_like(w.weight))
        model.save_lora_parameters(str(run), task_id=task)
    return run


def test_hbd_teacher_folds_g_prev_and_freezes(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(37)
    run = tmp_path / "run"
    model0 = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    with torch.no_grad():
        for w in model0.w_Bs:
            w.weight.copy_(torch.randn_like(w.weight))
    model0.save_lora_parameters(str(run), task_id=0)
    model1 = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=1,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    with torch.no_grad():
        for w in model1.w_Bs:
            w.weight.copy_(torch.randn_like(w.weight))
    # Saving keeps the in-memory aggregate as G_0 + raw B_1; the state file
    # stores the folded G_1. The teacher must reproduce that folded G_1.
    model1.save_lora_parameters(str(run), task_id=1)
    state = torch.load(
        run / SA_STATE_FILENAME, map_location="cpu", weights_only=True
    )
    teacher = model1.build_hbd_teacher()
    wrapper = teacher.lora_vit.blocks[0].attn.qkv
    assert isinstance(wrapper, _LiveAAggregateQKV)
    assert torch.allclose(
        wrapper.aggregate_q, state["aggregate_up"][0], atol=1e-5
    )
    assert torch.allclose(
        wrapper.aggregate_v, state["aggregate_up"][1], atol=1e-5
    )
    # Shared A is frozen at the task-start value and teacher B is zero.
    assert torch.allclose(
        wrapper.a_q.weight, model1.w_As[0].weight, atol=1e-6
    )
    assert all(torch.all(w.weight == 0) for w in teacher.w_Bs)
    assert all(not p.requires_grad for p in teacher.parameters())


def test_hbd_teacher_adds_no_persistent_parameters(tmp_path):
    dim, rank = 6, 2
    run = _run_tasks(tmp_path, tasks=1)
    torch.manual_seed(41)
    model = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=1,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    before = len(model.state_dict())
    teacher = model.build_hbd_teacher()
    after = len(model.state_dict())
    assert before == after
    assert teacher is not None


def test_hbd_capture_hooks_collect_historical_outputs(tmp_path):
    dim, rank = 6, 2
    run = _run_tasks(tmp_path, tasks=2)
    torch.manual_seed(43)
    model = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=2,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    captures = []
    handles = register_live_a_historical_capture_hooks(model, captures)
    x = torch.randn(4, 6, dim)
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert len(captures) == 1
    wrapper = model.lora_vit.blocks[0].attn.qkv
    expected_q, expected_v = wrapper.historical_output(x)
    assert torch.allclose(captures[0][0], expected_q, atol=1e-6)
    assert torch.allclose(captures[0][1], expected_v, atol=1e-6)


def test_hbd_loss_gradients_flow_only_to_shared_a(tmp_path):
    dim, rank = 6, 2
    run = _run_tasks(tmp_path, tasks=2)
    torch.manual_seed(47)
    model = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=2,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    teacher = model.build_hbd_teacher()
    teacher_captures = []
    teacher_handles = register_live_a_historical_capture_hooks(
        teacher, teacher_captures
    )
    x = torch.randn(4, 6, dim)
    with torch.no_grad():
        teacher_outputs = live_a_historical_outputs(teacher, x, teacher_captures)
    for handle in teacher_handles:
        handle.remove()

    # Perturb the student shared A; fresh B stays zero.
    with torch.no_grad():
        for w_a in model.w_As:
            w_a.weight.add_(0.1 * torch.randn_like(w_a.weight))
    captures = []
    handles = register_live_a_historical_capture_hooks(model, captures)
    model.train()
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    loss = hbd_historical_branch_distance(captures, teacher_outputs)
    loss.backward()
    a_grads = [w.weight.grad for w in model.w_As]
    b_grads = [w.weight.grad for w in model.w_Bs]
    assert all(g is not None and torch.any(g != 0) for g in a_grads)
    assert all(g is None or torch.all(g == 0) for g in b_grads)


def test_hbd_teacher_hash_unchanged_by_student_training(tmp_path):
    dim, rank = 6, 2
    run = _run_tasks(tmp_path, tasks=2)
    torch.manual_seed(53)
    model = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=2,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    teacher = model.build_hbd_teacher()
    teacher_a_hash = [
        w.weight.detach().clone() for w in teacher.w_As
    ]
    optimizer = torch.optim.SGD(
        [w.weight for w in model.w_As], lr=0.1
    )
    x = torch.randn(4, 6, dim)
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
    for saved, current in zip(teacher_a_hash, teacher.w_As):
        assert torch.equal(saved, current.weight.detach())


def test_live_a_gradient_diagnostics_none_when_a_frozen(tmp_path):
    dim, rank = 6, 2
    run = _run_tasks(tmp_path, tasks=2)
    model = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=2,
        train_a_all_tasks=False,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    x = torch.randn(4, 6, dim)
    assert model.live_a_gradient_diagnostics(x) is None


def test_hbd_task0_creates_no_loss_and_missing_teacher_raises():
    learner = object.__new__(Learner)
    learner._hbd_enabled = True
    learner._cur_task = 0
    assert learner._hbd_training_loss() is None
    learner._cur_task = 1
    learner._hbd_teacher = None
    with pytest.raises(RuntimeError):
        learner._hbd_training_loss()
