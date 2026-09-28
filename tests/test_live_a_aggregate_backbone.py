"""Backbone-level tests for Live-A Aggregate-B (v4 state)."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.lora import ParameterWrapper
from backbone.sa_lora import (
    SA_MERGED_FILENAME,
    SA_STATE_FILENAME,
    SA_STATE_VERSION_LEGACY,
    SA_STATE_VERSION,
    SharedALoRA_ViT_timm,
    _LiveAAggregateQKV,
    _SharedAQKV,
    migrate_sa_state_v1_to_v4,
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
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def _make_qkv(dim, rank):
    qkv = nn.Linear(dim, dim * 3, bias=False)
    nn.init.zeros_(qkv.weight)
    a_q = nn.Linear(dim, rank, bias=False)
    a_v = nn.Linear(dim, rank, bias=False)
    b_q = nn.Linear(rank, dim, bias=False)
    b_v = nn.Linear(rank, dim, bias=False)
    nn.init.zeros_(b_q.weight)
    nn.init.zeros_(b_v.weight)
    return qkv, a_q, a_v, b_q, b_v


def _make_g(b_list_q, b_list_v, scales):
    g_q = torch.zeros_like(b_list_q[0])
    g_v = torch.zeros_like(b_list_v[0])
    for (bq, bv, s) in zip(b_list_q, b_list_v, scales):
        g_q = g_q + s * bq / (torch.linalg.vector_norm(bq) + 1e-8)
        g_v = g_v + s * bv / (torch.linalg.vector_norm(bv) + 1e-8)
    return g_q, g_v


def test_anchored_history_is_fixed_but_preserves_input_gradients():
    torch.manual_seed(73)
    dim, rank = 8, 3
    qkv, a_q, a_v, b_q, b_v = _make_qkv(dim, rank)
    scales = nn.ModuleList([ParameterWrapper(nn.Parameter(torch.tensor([0.8])))])
    wrapper = _LiveAAggregateQKV(
        qkv, a_q, a_v, b_q, b_v, torch.randn(dim, rank),
        torch.randn(dim, rank), scales, 0, history_forward="anchored",
    )
    x = torch.randn(2, 4, dim, requires_grad=True)
    before = tuple(value.detach().clone() for value in wrapper.historical_output(x))
    with torch.no_grad():
        a_q.weight.add_(torch.randn_like(a_q.weight))
        a_v.weight.add_(torch.randn_like(a_v.weight))
        wrapper.aggregate_q.add_(torch.randn_like(wrapper.aggregate_q))
    after = wrapper.historical_output(x)
    for expected, actual in zip(before, after):
        assert torch.equal(expected, actual.detach())
    sum(value.square().mean() for value in after).backward()
    assert a_q.weight.grad is None
    assert a_v.weight.grad is None
    assert b_q.weight.grad is None
    assert x.grad is not None and x.grad.norm() > 0
    with torch.no_grad():
        b_q.weight.copy_(torch.randn_like(b_q.weight))
        b_v.weight.copy_(torch.randn_like(b_v.weight))
    wrapper(x).square().mean().backward()
    for parameter in (a_q.weight, a_v.weight, b_q.weight, b_v.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0


def test_anchored_history_task0_matches_shared_forward_and_gradients():
    import copy

    torch.manual_seed(74)
    dim, rank = 8, 3
    qkv, a_q, a_v, b_q, b_v = _make_qkv(dim, rank)
    with torch.no_grad():
        b_q.weight.copy_(torch.randn_like(b_q.weight))
        b_v.weight.copy_(torch.randn_like(b_v.weight))
    scales = nn.ModuleList([ParameterWrapper(nn.Parameter(torch.tensor([0.8])))])
    shared = _LiveAAggregateQKV(
        qkv, a_q, a_v, b_q, b_v, torch.zeros(dim, rank),
        torch.zeros(dim, rank), scales, 0,
    )
    anchored = copy.deepcopy(shared)
    anchored.history_forward = "anchored"
    x = torch.randn(2, 4, dim)
    assert torch.equal(shared(x), anchored(x))
    shared(x).square().mean().backward()
    anchored(x).square().mean().backward()
    for old, new in zip(shared.parameters(), anchored.parameters()):
        assert torch.equal(old.grad, new.grad)


def test_anchored_history_state_rebuild_and_mode_guard(tmp_path):
    import copy

    torch.manual_seed(75)
    dim, rank = 8, 3
    pristine = _TinyViT(dim)
    settings = dict(
        r=rank, filepath=str(tmp_path), train_a_all_tasks=True,
        cumulative_state=True, cumulative_merge="live_a_aggregate_b",
        live_a_coordinate_align=True, live_a_absorb_mode="bounded_norm_calibrated_absorb",
        live_a_boundary_merge="joint_svd", live_a_history_forward="anchored",
    )
    task0 = SharedALoRA_ViT_timm(copy.deepcopy(pristine), cur_task_index=0, **settings)
    with torch.no_grad():
        for module in task0.w_Bs:
            module.weight.copy_(torch.randn_like(module.weight))
    task0.save_lora_parameters(str(tmp_path), task_id=0)
    state0 = torch.load(tmp_path / SA_STATE_FILENAME, weights_only=True)
    assert state0["history_forward"] == "anchored"
    task1 = SharedALoRA_ViT_timm(copy.deepcopy(pristine), cur_task_index=1, **settings)
    wrapper = task1.lora_vit.blocks[0].attn.qkv
    x = torch.randn(2, 4, dim)
    original = tuple(value.detach().clone() for value in wrapper.historical_output(x))
    with torch.no_grad():
        for module in task1.w_As:
            module.weight.add_(0.2 * torch.randn_like(module.weight))
        for module in task1.w_Bs:
            module.weight.copy_(torch.randn_like(module.weight))
    for old, new in zip(original, wrapper.historical_output(x)):
        assert torch.equal(old, new.detach())
    task1.save_lora_parameters(str(tmp_path), task_id=1)
    state1 = torch.load(tmp_path / SA_STATE_FILENAME, weights_only=True)
    merged = torch.load(tmp_path / SA_MERGED_FILENAME, weights_only=True)
    assert merged["history_forward"] == state1["history_forward"] == "anchored"
    assert state0.keys() == state1.keys()
    for state in (state0, state1):
        assert sum(t.numel() for key in ("shared_a", "aggregate_up")
                   for t in state[key]) == 4 * dim * rank
    rebuilt = SharedALoRA_ViT_timm(copy.deepcopy(pristine), cur_task_index=2, **settings)
    restored = rebuilt.lora_vit.blocks[0].attn.qkv
    for index, actual in enumerate(restored.historical_output(x)):
        expected = x @ (state1["aggregate_up"][index] @ state1["shared_a"][index]
                        / (state1["shared_a"][index].norm() + 1e-8)).T
        assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="history forward differs"):
        SharedALoRA_ViT_timm(
            copy.deepcopy(pristine), cur_task_index=2,
            **{**settings, "live_a_history_forward": "shared"},
        )


def test_anchored_history_requires_joint_boundary(tmp_path):
    with pytest.raises(ValueError, match="anchored history requires joint_svd"):
        SharedALoRA_ViT_timm(
            _TinyViT(8), r=3, filepath=str(tmp_path),
            cumulative_state=True, cumulative_merge="live_a_aggregate_b",
            train_a_all_tasks=True, live_a_coordinate_align=True,
            live_a_history_forward="anchored",
        )


def test_live_a_bank_vs_aggregate_forward_and_a_grad():
    torch.manual_seed(11)
    dim, rank, tasks = 8, 3, 3
    qkv, a_q, a_v, b_q, b_v = _make_qkv(dim, rank)
    saved_b_q = [torch.randn(dim, rank) for _ in range(tasks)]
    saved_b_v = [torch.randn(dim, rank) for _ in range(tasks)]
    scales = [torch.tensor([0.6 + 0.2 * i]) for i in range(tasks)]
    scaling_prev = nn.ModuleList(
        [ParameterWrapper(nn.Parameter(s.clone())) for s in scales]
    )
    scaling_cur = nn.ModuleList(
        [ParameterWrapper(nn.Parameter(torch.tensor([0.8])))]
    )
    bank = _SharedAQKV(
        qkv, a_q, a_v, b_q, b_v, saved_b_q, saved_b_v,
        scaling_cur, scaling_prev, 0,
    )
    g_q, g_v = _make_g(saved_b_q, saved_b_v, scales)
    agg = _LiveAAggregateQKV(
        qkv, a_q, a_v, b_q, b_v, g_q, g_v, scaling_cur, 0,
        history_groups=1,
    )
    x = torch.randn(4, 6, dim)
    with torch.no_grad():
        out_bank = bank(x)
        out_agg = agg(x)
    assert torch.allclose(out_agg, out_bank, atol=1e-5)

    # Gradient w.r.t. shared A (Q branch) must match when scales are frozen.
    y = torch.randn_like(out_bank)

    def a_grad(module):
        loss = ((module(x) - y) ** 2).mean()
        loss.backward()
        return module.a_q.weight.grad.clone()

    a_q.weight.grad = None
    grad_bank = a_grad(bank)
    a_q.weight.grad = None
    grad_agg = a_grad(agg)
    assert torch.allclose(grad_agg, grad_bank, atol=1e-5)


def test_normalized_current_bank_matches_exact_aggregate_absorption():
    """A normalized task branch must fold exactly into the fixed-size G state."""
    torch.manual_seed(17)
    dim, rank = 8, 3
    qkv, a_q, a_v, b_q, b_v = _make_qkv(dim, rank)
    with torch.no_grad():
        b_q.weight.copy_(torch.randn_like(b_q.weight))
        b_v.weight.copy_(torch.randn_like(b_v.weight))
    scale = torch.tensor([0.7])
    scaling_cur = nn.ModuleList(
        [ParameterWrapper(nn.Parameter(scale.clone()))]
    )
    bank = _SharedAQKV(
        qkv,
        a_q,
        a_v,
        b_q,
        b_v,
        [],
        [],
        scaling_cur,
        nn.ModuleList(),
        0,
        normalize_current_branch=True,
    )
    aggregate_q = scale * b_q.weight.detach() / (
        torch.linalg.vector_norm(b_q.weight.detach()) + 1e-8
    )
    aggregate_v = scale * b_v.weight.detach() / (
        torch.linalg.vector_norm(b_v.weight.detach()) + 1e-8
    )
    aggregate = _LiveAAggregateQKV(
        qkv,
        a_q,
        a_v,
        b_q,
        b_v,
        aggregate_q,
        aggregate_v,
        scaling_cur,
        0,
        normalize_current_branch=True,
    )
    aggregate.current_branch_enabled = False

    x = torch.randn(4, 6, dim)
    with torch.no_grad():
        bank_output = bank(x)
        aggregate_output = aggregate(x)
    assert torch.allclose(bank_output, aggregate_output, atol=1e-6, rtol=1e-6)


def test_normalized_zero_initialized_b_has_finite_first_gradient():
    dim, rank = 6, 2
    torch.manual_seed(19)
    qkv, a_q, a_v, b_q, b_v = _make_qkv(dim, rank)
    scaling_cur = nn.ModuleList(
        [ParameterWrapper(nn.Parameter(torch.tensor([0.8])))]
    )
    bank = _SharedAQKV(
        qkv,
        a_q,
        a_v,
        b_q,
        b_v,
        [],
        [],
        scaling_cur,
        nn.ModuleList(),
        0,
        normalize_current_branch=True,
    )
    bank(torch.randn(2, 3, dim)).sum().backward()
    assert torch.isfinite(b_q.weight.grad).all()
    assert torch.isfinite(b_v.weight.grad).all()
    assert torch.linalg.vector_norm(b_q.weight.grad) > 0
    assert torch.linalg.vector_norm(b_v.weight.grad) > 0


def test_historical_only_forward_removes_current_branch_and_restores_it(tmp_path):
    """Leaking the diagnostic scope would silently change ordinary training."""
    torch.manual_seed(13)
    model = SharedALoRA_ViT_timm(
        _TinyViT(4),
        r=2,
        filepath=str(tmp_path / "run"),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    wrapper = model.lora_vit.blocks[0].attn.qkv
    with torch.no_grad():
        wrapper.aggregate_q.copy_(torch.randn_like(wrapper.aggregate_q))
        wrapper.aggregate_v.copy_(torch.randn_like(wrapper.aggregate_v))
        wrapper.b_q.weight.copy_(torch.randn_like(wrapper.b_q.weight))
        wrapper.b_v.weight.copy_(torch.randn_like(wrapper.b_v.weight))
    inputs = torch.randn(3, 2, 4)

    full_before = model(inputs)
    with model.historical_only_forward():
        historical = model(inputs)
    full_after = model(inputs)

    current_q, current_v = wrapper.current_output(inputs)
    expected_difference = torch.zeros_like(full_before)
    expected_difference[:, :, :4] = current_q
    expected_difference[:, :, -4:] = current_v
    assert torch.allclose(full_before - historical, expected_difference, atol=1e-6)
    assert torch.allclose(full_after, full_before, atol=1e-6)


def test_historical_only_forward_restores_current_branch_after_exception(tmp_path):
    """An exception in stability-gradient computation must not poison training."""
    model = SharedALoRA_ViT_timm(
        _TinyViT(4),
        r=2,
        filepath=str(tmp_path / "run"),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    wrapper = model.lora_vit.blocks[0].attn.qkv

    with pytest.raises(RuntimeError, match="probe"):
        with model.historical_only_forward():
            raise RuntimeError("probe")

    assert wrapper.current_branch_enabled is True


def test_live_a_fresh_roundtrip_keeps_raw_current_semantics(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(17)
    run = tmp_path / "run"
    model = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    with torch.no_grad():
        for w in model.w_Bs:
            w.weight.copy_(torch.randn_like(w.weight))
    model.save_lora_parameters(str(run), task_id=0)

    assert not (run / "sa_lora_w_b_0.pt").exists()
    state = torch.load(run / SA_STATE_FILENAME, map_location="cpu", weights_only=True)
    assert state["version"] == 4
    assert state["merge_mode"] == "live_a_aggregate_b"
    assert len(state["aggregate_up"]) == 2
    # In-memory semantics stay "old G (empty) + current raw B".
    assert len(model.aggregate_up) == 0
    assert any(torch.any(w.weight != 0) for w in model.w_Bs)

    reloaded = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=1,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    assert len(reloaded.aggregate_up) == 2
    assert all(torch.all(w.weight == 0) for w in reloaded.w_Bs)
    x = torch.randn(4, 6, dim)
    assert torch.isfinite(reloaded(x)).all()


def test_live_a_multi_task_aggregate_matches_explicit_g(tmp_path):
    dim, rank, tasks = 6, 2, 3
    torch.manual_seed(23)
    run = tmp_path / "run"
    all_b_q = []
    all_b_v = []
    all_scales = []
    for task in range(tasks):
        model = SharedALoRA_ViT_timm(
            _TinyViT(dim),
            r=rank,
            filepath=str(run),
            cur_task_index=task,
            train_a_all_tasks=True,
            cumulative_state=True,
            cumulative_merge="live_a_aggregate_b",
            live_a_absorb_mode="normalized_absorb",
        )
        with torch.no_grad():
            for w in model.w_Bs:
                w.weight.copy_(torch.randn_like(w.weight))
        all_b_q.append(model.w_Bs[0].weight.detach().cpu().float())
        all_b_v.append(model.w_Bs[1].weight.detach().cpu().float())
        all_scales.append(
            model.wrapped_param[0].param.detach().cpu().reshape(())
        )
        model.save_lora_parameters(str(run), task_id=task)

    state = torch.load(run / SA_STATE_FILENAME, map_location="cpu", weights_only=True)
    assert len(state["aggregate_up"]) == 2
    expected_q, expected_v = _make_g(all_b_q, all_b_v, all_scales)
    assert torch.allclose(state["aggregate_up"][0], expected_q, atol=1e-6)
    assert torch.allclose(state["aggregate_up"][1], expected_v, atol=1e-6)

    reloaded = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=tasks,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        live_a_absorb_mode="normalized_absorb",
    )
    # Historical-only forward (current B = 0) equals explicit G A / ||A||.
    x = torch.randn(4, 6, dim)
    with torch.no_grad():
        out = reloaded(x)
    wrapper = reloaded.lora_vit.blocks[0].attn.qkv
    assert isinstance(wrapper, _LiveAAggregateQKV)
    a_q = wrapper.a_q.weight.detach().cpu()
    z = x @ a_q.t()
    new_q = z @ (
        expected_q / torch.linalg.vector_norm(a_q)
    ).t()
    a_v = wrapper.a_v.weight.detach().cpu()
    zv = x @ a_v.t()
    new_v = zv @ (
        expected_v / torch.linalg.vector_norm(a_v)
    ).t()
    expected = wrapper.qkv(x).clone()
    expected[:, :, :dim] += new_q
    expected[:, :, -dim:] += new_v
    assert torch.allclose(out, expected, atol=1e-5)


def test_live_a_final_rebuild_matches_merged(tmp_path):
    dim, rank, tasks = 6, 2, 2
    torch.manual_seed(29)
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

    merged = torch.load(
        run / SA_MERGED_FILENAME, map_location="cpu", weights_only=True
    )
    assert merged["version"] == 4
    # Rebuild from state == deployed operator G_final A / ||A||.
    rebuilt = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=tasks,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    x = torch.randn(4, 6, dim)
    with torch.no_grad():
        out_rebuilt = rebuilt(x)
    wrapper = rebuilt.lora_vit.blocks[0].attn.qkv
    a_q = merged["shared_a"][0]
    b_q = merged["merged_b"][0]
    a_v = merged["shared_a"][1]
    b_v = merged["merged_b"][1]
    z = x @ a_q.t()
    zv = x @ a_v.t()
    expected = wrapper.qkv(x).clone()
    expected[:, :, :dim] += z @ b_q.t()
    expected[:, :, -dim:] += zv @ b_v.t()
    assert torch.allclose(out_rebuilt, expected, atol=1e-5)


def test_live_a_rejects_old_versions(tmp_path):
    dim, rank = 6, 2
    run = tmp_path / "v1"
    model = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=0,
        train_a_all_tasks=False,
    )
    with torch.no_grad():
        for w in model.w_Bs:
            w.weight.copy_(torch.randn_like(w.weight))
    model.save_lora_parameters(str(run), task_id=0)
    with pytest.raises(ValueError, match="migrate"):
        SharedALoRA_ViT_timm(
            _TinyViT(dim),
            r=rank,
            filepath=str(run),
            cur_task_index=1,
            train_a_all_tasks=True,
            cumulative_state=True,
            cumulative_merge="live_a_aggregate_b",
        )


def test_migrate_v1_to_v4_preserves_operator(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(31)
    run = tmp_path / "v1"
    for task in range(2):
        model = SharedALoRA_ViT_timm(
            _TinyViT(dim),
            r=rank,
            filepath=str(run),
            cur_task_index=task,
            train_a_all_tasks=True,
        )
        with torch.no_grad():
            for w in model.w_Bs:
                w.weight.copy_(torch.randn_like(w.weight))
        model.save_lora_parameters(str(run), task_id=task)
    v1_state = torch.load(
        run / SA_STATE_FILENAME, map_location="cpu", weights_only=True
    )
    migrated = migrate_sa_state_v1_to_v4(str(run))
    assert migrated["version"] == 4
    assert (run / "sa_state.pt.v1").exists()
    v4_state = torch.load(
        run / SA_STATE_FILENAME, map_location="cpu", weights_only=True
    )
    a = migrated["shared_a"][0]
    g = migrated["aggregate_up"][0]
    # Compare with the v1 merged artifact operator.
    # Compare with the explicit v1 bank operator (sum s b/(||a|| ||b||)).
    bank_op = torch.zeros_like(g)
    for task_id in sorted(v1_state["scales"]):
        b_list = torch.load(
            run / "sa_lora_w_b_{}.pt".format(task_id),
            map_location="cpu",
            weights_only=True,
        )
        b = b_list[0].float()
        s = v1_state["scales"][task_id].reshape(())
        bank_op = bank_op + s * b / (
            torch.linalg.vector_norm(a)
            * torch.linalg.vector_norm(b)
            + 1e-8
        )
    assert torch.allclose(
        g / (torch.linalg.vector_norm(a) + 1e-8),
        bank_op,
        atol=1e-5,
    )
    assert v4_state["task_id"] == len(v1_state["scales"])


def test_freeze_old_scales_disables_historical_scale_grad(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(37)
    run = tmp_path / "run"
    model = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=0,
        train_a_all_tasks=True,
    )
    with torch.no_grad():
        for w in model.w_Bs:
            w.weight.copy_(torch.randn_like(w.weight))
    model.save_lora_parameters(str(run), task_id=0)

    reloaded = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=1,
        train_a_all_tasks=True,
        freeze_old_scales=True,
    )
    assert reloaded.wrapped_param_prev[0].param.requires_grad is False
    assert reloaded.wrapped_param[0].param.requires_grad is True


def test_live_a_gradient_diagnostics_reports_both_branches(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(41)
    run = tmp_path / "run"
    model = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    with torch.no_grad():
        for w in model.w_Bs:
            w.weight.copy_(torch.randn_like(w.weight))
    model.save_lora_parameters(str(run), task_id=0)

    reloaded = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=1,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
    )
    with torch.no_grad():
        for w in reloaded.w_Bs:
            w.weight.copy_(torch.randn_like(w.weight))
    x = torch.randn(4, 6, dim)
    diag = reloaded.live_a_gradient_diagnostics(x)
    assert diag is not None
    assert diag["historical_dL_dA"] > 0
    assert diag["current_dL_dA"] > 0
    assert diag["mean_G_norm"] > 0
    assert diag["scale"] > 0
    # Normal forward/backward still works after the diagnostic autograd pass.
    out = reloaded(x)
    loss = out.square().mean()
    loss.backward()
    assert all(w.weight.grad is not None for w in reloaded.w_As)
