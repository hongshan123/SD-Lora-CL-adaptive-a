import sys
import shutil
import copy
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
    canonical_down_projection,
    canonicalize_effective_up_projection,
    fold_all_cumulative_up_projections,
    fold_cumulative_up_projection,
    gauge_align_up_projection,
    gauge_projection_residual,
    migrate_sa_state_v1_to_v2,
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


def test_canonical_decomposition_preserves_operator():
    torch.manual_seed(31)
    a = torch.randn(3, 8)
    h = torch.randn(8, 3)
    q, r = canonical_down_projection(a)
    assert q.shape == (3, 8)
    assert r.shape == (3, 3)
    assert torch.allclose(r.t() @ q, a, atol=1e-6)

    h_can = canonicalize_effective_up_projection(h, r)
    assert torch.allclose(h_can @ q, h @ a, atol=1e-6)


def test_canonical_cumulative_matches_bank_operator():
    a, b, scales = _make_bank_state(seed=37)
    h_raw = fold_cumulative_up_projection(a, b, scales)
    bank = _bank_operator(a, b, scales)
    q, r = canonical_down_projection(a)
    h_can = canonicalize_effective_up_projection(h_raw, r)
    assert torch.allclose(h_can @ q, bank, atol=1e-6)


def test_gauge_alignment_exact_when_spans_equal():
    torch.manual_seed(41)
    a_old = torch.randn(3, 8)
    h = torch.randn(8, 3)
    q_old, _ = canonical_down_projection(a_old)
    o = torch.linalg.qr(torch.randn(3, 3), mode="reduced")[0]
    q_new = o.t() @ q_old

    h_aligned = gauge_align_up_projection(h, q_old, q_new)
    assert h_aligned.shape == h.shape
    assert torch.allclose(
        h_aligned @ q_new, h @ q_old, atol=1e-5
    )


def test_gauge_alignment_projects_old_operator():
    torch.manual_seed(43)
    q_old = torch.linalg.qr(torch.randn(8, 3), mode="reduced")[0].t()
    q_new = torch.linalg.qr(torch.randn(8, 3), mode="reduced")[0].t()
    h = torch.randn(8, 3)

    h_aligned = gauge_align_up_projection(h, q_old, q_new)
    old_operator = h @ q_old
    projected = old_operator @ (q_new.t() @ q_new)
    assert torch.allclose(h_aligned @ q_new, projected, atol=1e-5)

    residual = gauge_projection_residual(h, q_old, q_new)
    expected = (old_operator - projected).norm()
    assert residual.item() == pytest.approx(expected.item(), abs=1e-5)


def test_v2_fresh_run_roundtrip_without_per_task_files(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(47)
    run = tmp_path / "run"
    model = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
    )
    with torch.no_grad():
        for w in model.w_Bs:
            w.weight.copy_(torch.randn_like(w.weight))
    model.save_lora_parameters(str(run), task_id=0)

    assert not (run / "sa_lora_w_b_0.pt").exists()
    state = torch.load(
        run / "sa_state.pt", map_location="cpu", weights_only=True
    )
    assert state["version"] == 2
    assert state["task_id"] == 1
    assert len(state["canonical_down"]) == 2
    assert len(state["cumulative_up"]) == 2
    assert len(state["triangular_r"]) == 2

    reloaded = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=1,
        train_a_all_tasks=True,
    )
    assert reloaded.cumulative_state
    x = torch.randn(4, 6, dim)
    out = reloaded(x)
    assert out.shape == (4, 6, dim * 3)
    assert torch.isfinite(out).all()


def test_v2_canonical_forward_matches_v1_bank(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(53)
    v1_dir = tmp_path / "v1"
    v2_dir = tmp_path / "v2"
    tiny = _TinyViT(dim)
    v1 = SharedALoRA_ViT_timm(
        copy.deepcopy(tiny),
        r=rank,
        filepath=str(v1_dir),
        cur_task_index=0,
        train_a_all_tasks=False,
    )
    v2 = SharedALoRA_ViT_timm(
        copy.deepcopy(tiny),
        r=rank,
        filepath=str(v2_dir),
        cur_task_index=0,
        train_a_all_tasks=False,
        cumulative_state=True,
    )
    with torch.no_grad():
        for a1, a2 in zip(v1.w_As, v2.w_As):
            a2.weight.copy_(a1.weight)
        for b1, b2 in zip(v1.w_Bs, v2.w_Bs):
            b1.weight.copy_(torch.randn_like(b1.weight))
            b2.weight.copy_(b1.weight)
        v2.wrapped_param[0].param.copy_(v1.wrapped_param[0].param)
    v1.save_lora_parameters(str(v1_dir), task_id=0)
    v2.save_lora_parameters(str(v2_dir), task_id=0)

    m1 = SharedALoRA_ViT_timm(
        copy.deepcopy(tiny),
        r=rank,
        filepath=str(v1_dir),
        cur_task_index=1,
        train_a_all_tasks=False,
    )
    m2 = SharedALoRA_ViT_timm(
        copy.deepcopy(tiny),
        r=rank,
        filepath=str(v2_dir),
        cur_task_index=1,
        train_a_all_tasks=False,
    )
    x = torch.randn(4, 6, dim)
    assert torch.allclose(m1(x), m2(x), atol=1e-5)


def test_v2_save_accumulates_gauge_aligned_operator(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(59)
    run = tmp_path / "run"
    model0 = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
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
    )
    with torch.no_grad():
        for w_a in model1.w_As:
            w_a.weight.add_(0.2 * torch.randn_like(w_a.weight))
        for w_b in model1.w_Bs:
            w_b.weight.copy_(torch.randn_like(w_b.weight))
    model1.save_lora_parameters(str(run), task_id=1)

    state = torch.load(
        run / "sa_state.pt", map_location="cpu", weights_only=True
    )
    assert state["task_id"] == 2
    for idx, (w_a, w_b) in enumerate(zip(model1.w_As, model1.w_Bs)):
        a1 = w_a.weight.detach().cpu().float()
        b1 = w_b.weight.detach().cpu().float()
        s1 = model1.wrapped_param[0].param.detach().cpu().float().reshape(())
        q_new, r_new = canonical_down_projection(a1)

        h_hist = gauge_align_up_projection(
            model0.cumulative_up[idx],
            model0.canonical_down[idx],
            q_new,
        )
        norm_a = torch.linalg.vector_norm(a1)
        norm_b = torch.linalg.vector_norm(b1) + 1e-8
        h_cur = canonicalize_effective_up_projection(
            s1 * b1 / (norm_a * norm_b), r_new
        )
        assert torch.allclose(
            state["cumulative_up"][idx], h_hist + h_cur, atol=1e-5
        )
        # The historical part is the projection of the old operator.
        projector = q_new.t() @ q_new
        old_operator = model0.cumulative_up[idx] @ model0.canonical_down[idx]
        assert torch.allclose(
            h_hist @ q_new, old_operator @ projector, atol=1e-5
        )


def test_v2_cumulative_only_keeps_old_h_unaligned(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(73)
    run = tmp_path / "run"
    model0 = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
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
        cumulative_gauge=False,
    )
    with torch.no_grad():
        for w_a in model1.w_As:
            w_a.weight.add_(0.2 * torch.randn_like(w_a.weight))
        for w_b in model1.w_Bs:
            w_b.weight.copy_(torch.randn_like(w_b.weight))
    model1.save_lora_parameters(str(run), task_id=1)

    state = torch.load(
        run / "sa_state.pt", map_location="cpu", weights_only=True
    )
    for idx, (w_a, w_b) in enumerate(zip(model1.w_As, model1.w_Bs)):
        a1 = w_a.weight.detach().cpu().float()
        b1 = w_b.weight.detach().cpu().float()
        s1 = model1.wrapped_param[0].param.detach().cpu().float().reshape(())
        _, r_new = canonical_down_projection(a1)
        norm_a = torch.linalg.vector_norm(a1)
        norm_b = torch.linalg.vector_norm(b1) + 1e-8
        h_cur = canonicalize_effective_up_projection(
            s1 * b1 / (norm_a * norm_b), r_new
        )
        # No gauge: old H is carried over unchanged (no projection).
        assert torch.allclose(
            state["cumulative_up"][idx],
            model0.cumulative_up[idx] + h_cur,
            atol=1e-5,
        )


def test_migrate_v1_to_v2_is_forward_equivalent(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(61)
    run = tmp_path / "run"
    tiny = _TinyViT(dim)
    for task in range(2):
        model = SharedALoRA_ViT_timm(
            copy.deepcopy(tiny),
            r=rank,
            filepath=str(run),
            cur_task_index=task,
            train_a_all_tasks=False,
        )
        with torch.no_grad():
            for w in model.w_Bs:
                w.weight.copy_(torch.randn_like(w.weight))
        model.save_lora_parameters(str(run), task_id=task)

    legacy_dir = tmp_path / "legacy"
    shutil.copytree(run, legacy_dir)
    state = torch.load(
        legacy_dir / "sa_state.pt", map_location="cpu", weights_only=True
    )
    assert state["version"] == 1

    new_state = migrate_sa_state_v1_to_v2(str(run))
    assert new_state["version"] == 2
    assert (run / "sa_state.pt.v1").exists()
    assert (run / "sa_lora_w_b_0.pt").exists()
    assert (run / "sa_lora_w_b_1.pt").exists()

    legacy = SharedALoRA_ViT_timm(
        copy.deepcopy(tiny),
        r=rank,
        filepath=str(legacy_dir),
        cur_task_index=2,
        train_a_all_tasks=False,
    )
    migrated = SharedALoRA_ViT_timm(
        copy.deepcopy(tiny),
        r=rank,
        filepath=str(run),
        cur_task_index=2,
        train_a_all_tasks=False,
    )
    assert migrated.cumulative_state
    x = torch.randn(4, 6, dim)
    assert torch.allclose(legacy(x), migrated(x), atol=1e-5)


def test_v2_rejects_legacy_flag_mismatch(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(67)
    run = tmp_path / "run"
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
            train_a_all_tasks=False,
            cumulative_state=True,
        )


def test_v2_gauge_residual_tracks_projection_error(tmp_path):
    dim, rank = 6, 2
    torch.manual_seed(71)
    run = tmp_path / "run"
    model0 = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
    )
    with torch.no_grad():
        for w in model0.w_Bs:
            w.weight.copy_(torch.randn_like(w.weight))
    model0.save_lora_parameters(str(run), task_id=0)
    assert model0.cumulative_gauge_residual() < 1e-6

    model1 = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=1,
        train_a_all_tasks=True,
        cumulative_state=True,
    )
    assert model1.cumulative_gauge_residual() < 1e-4
    with torch.no_grad():
        model1.w_As[0].weight.add_(0.3 * torch.randn_like(model1.w_As[0].weight))
    assert model1.cumulative_gauge_residual() > 1e-4
