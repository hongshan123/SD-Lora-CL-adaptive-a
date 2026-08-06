import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.lora import ParameterWrapper
from backbone.sa_lora import (
    SA_MERGED_FILENAME,
    SharedALoRA_ViT_timm,
    _SharedAQKV,
    fold_all_cumulative_up_projections,
    fold_cumulative_up_projection,
)


def _make_bank_state(dim=6, rank=2, tasks=3, seed=7):
    torch.manual_seed(seed)
    a = torch.randn(rank, dim)
    b = [torch.randn(dim, rank) for _ in range(tasks)]
    scales = [torch.tensor([0.6 + 0.2 * i]) for i in range(tasks)]
    return a, b, scales


def _bank_operator(shared_a, up_weights, scales, eps=1e-8):
    return sum(
        scale.reshape(()) * (up @ shared_a)
        / (
            torch.linalg.vector_norm(shared_a)
            * torch.linalg.vector_norm(up)
            + eps
        )
        for up, scale in zip(up_weights, scales)
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


def _make_qkv_wrapper(dim, rank, saved_b_q, saved_b_v, scales):
    qkv = nn.Linear(dim, dim * 3, bias=False)
    nn.init.zeros_(qkv.weight)
    a_q = nn.Linear(dim, rank, bias=False)
    a_v = nn.Linear(dim, rank, bias=False)
    b_q = nn.Linear(rank, dim, bias=False)
    b_v = nn.Linear(rank, dim, bias=False)
    nn.init.zeros_(b_q.weight)
    nn.init.zeros_(b_v.weight)
    scaling_prev = nn.ModuleList(
        [ParameterWrapper(nn.Parameter(s.clone())) for s in scales]
    )
    scaling_cur = nn.ModuleList(
        [ParameterWrapper(nn.Parameter(torch.tensor([0.8])))]
    )
    return _SharedAQKV(
        qkv,
        a_q,
        a_v,
        b_q,
        b_v,
        saved_b_q,
        saved_b_v,
        scaling_cur,
        scaling_prev,
        0,
    )


def test_fold_matches_bank_operator():
    a, b, scales = _make_bank_state()
    h = fold_cumulative_up_projection(a, b, scales)
    bank = _bank_operator(a, b, scales)
    rel = (h @ a - bank).square().sum().sqrt() / bank.square().sum().sqrt()
    assert rel.item() < 1e-5


def test_fold_rejects_invalid_state():
    a, b, scales = _make_bank_state()
    with pytest.raises(ValueError, match="at least one historical"):
        fold_cumulative_up_projection(a, [], [])
    with pytest.raises(ValueError, match="counts must match"):
        fold_cumulative_up_projection(a, b[:2], scales)
    with pytest.raises(ValueError, match="scalar"):
        fold_cumulative_up_projection(
            a, b, [torch.ones(2), torch.ones(2), torch.ones(2)]
        )
    with pytest.raises(ValueError, match="at least one historical task"):
        fold_all_cumulative_up_projections([a], {}, {})
    with pytest.raises(ValueError, match="missing scale"):
        fold_all_cumulative_up_projections([a], {0: [b[0]]}, {})


def test_fold_all_matches_saved_merged_artifact(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(11)
    run = tmp_path / "run"
    for task in range(2):
        model = SharedALoRA_ViT_timm(
            _TinyViT(dim),
            r=rank,
            filepath=str(run),
            cur_task_index=task,
            train_a_all_tasks=False,
        )
        with torch.no_grad():
            for w in model.w_Bs:
                w.weight.copy_(torch.randn_like(w.weight))
        model.save_lora_parameters(str(run), task_id=task)

    model = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=2,
        train_a_all_tasks=False,
    )
    scales = {
        task: model.wrapped_param_prev[task].param.detach().cpu()
        for task in range(2)
    }
    shared_a = [w.weight.detach().cpu() for w in model.w_As]
    cumulative = fold_all_cumulative_up_projections(
        shared_a, model.saved_b_tasks, scales
    )

    merged_dir = tmp_path / "merged"
    model.save_merged_lora(str(merged_dir))
    merged = torch.load(
        merged_dir / SA_MERGED_FILENAME, map_location="cpu", weights_only=True
    )
    assert len(cumulative) == len(merged["merged_b"]) == len(shared_a)
    for h, m in zip(cumulative, merged["merged_b"]):
        assert torch.allclose(h, m, atol=1e-6)


def test_cumulative_reproduces_bank_forward_and_logits(tmp_path):
    dim, rank = 8, 3
    torch.manual_seed(17)
    run = tmp_path / "run"
    for task in range(2):
        model = SharedALoRA_ViT_timm(
            _TinyViT(dim),
            r=rank,
            filepath=str(run),
            cur_task_index=task,
            train_a_all_tasks=False,
        )
        with torch.no_grad():
            for w in model.w_Bs:
                w.weight.copy_(torch.randn_like(w.weight))
        model.save_lora_parameters(str(run), task_id=task)

    bank = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=2,
        train_a_all_tasks=False,
    )
    x = torch.randn(4, 6, dim)
    feat_bank = bank(x)

    scales = [
        bank.wrapped_param_prev[task].param.detach().cpu()
        for task in range(2)
    ]
    wrapper = bank.lora_vit.blocks[0].attn.qkv
    assert isinstance(wrapper, _SharedAQKV)
    h_q = fold_cumulative_up_projection(
        wrapper.a_q.weight.detach().cpu(),
        [bank.saved_b_tasks[t][0].cpu() for t in range(2)],
        scales,
    )
    h_v = fold_cumulative_up_projection(
        wrapper.a_v.weight.detach().cpu(),
        [bank.saved_b_tasks[t][1].cpu() for t in range(2)],
        scales,
    )

    feat_cum = wrapper.qkv(x).clone()
    zq = x @ wrapper.a_q.weight.t()
    zv = x @ wrapper.a_v.weight.t()
    feat_cum[:, :, :dim] += zq @ h_q.t()
    feat_cum[:, :, -dim:] += zv @ h_v.t()

    assert torch.allclose(feat_bank, feat_cum, atol=1e-5)

    head = nn.Linear(dim * 3, 4)
    with torch.no_grad():
        logits_bank = head(feat_bank.reshape(-1, dim * 3))
        logits_cum = head(feat_cum.reshape(-1, dim * 3))
    assert torch.allclose(logits_bank, logits_cum, atol=1e-5)


def test_direct_qkv_wrapper_bank_matches_cumulative_math():
    dim, rank, tasks = 8, 3, 3
    torch.manual_seed(23)
    saved_b_q = [torch.randn(dim, rank) for _ in range(tasks)]
    saved_b_v = [torch.randn(dim, rank) for _ in range(tasks)]
    scales = [torch.tensor([0.7]), torch.tensor([1.1]), torch.tensor([0.9])]
    wrapper = _make_qkv_wrapper(dim, rank, saved_b_q, saved_b_v, scales)
    x = torch.randn(4, 6, dim)
    out_bank = wrapper(x)

    h_q = fold_cumulative_up_projection(
        wrapper.a_q.weight, saved_b_q, scales
    )
    h_v = fold_cumulative_up_projection(
        wrapper.a_v.weight, saved_b_v, scales
    )
    out_cum = wrapper.qkv(x).clone()
    zq = x @ wrapper.a_q.weight.t()
    zv = x @ wrapper.a_v.weight.t()
    out_cum[:, :, :dim] += zq @ h_q.t()
    out_cum[:, :, -dim:] += zv @ h_v.t()

    assert torch.allclose(out_bank, out_cum, atol=1e-5)
