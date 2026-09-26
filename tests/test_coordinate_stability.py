import copy
import math
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.coordinate_stability import (
    align_live_a_aggregate,
    apply_residual_orthogonal_transport,
    fit_residual_orthogonal_transport,
    normalized_effective_operator,
)
from utils.inc_net import get_backbone
from backbone.sa_lora import (
    SA_MERGED_FILENAME,
    SA_STATE_FILENAME,
    SharedALoRA_ViT_timm,
    absorb_live_a_current_projection,
)
from backbone import sa_lora
from models.sa_sdlora import validate_coordinate_transport_config


class _TinyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, inputs):
        return self.qkv(inputs)


class _TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _TinyAttention(dim)

    def forward(self, inputs):
        return self.attn(inputs)


class _TinyViT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dim)])
        self.head = nn.Identity()

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return self.head(inputs)


def _plane_rotation(dim, angle):
    matrix = torch.eye(dim)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    matrix[0, 0] = cosine
    matrix[0, 1] = -sine
    matrix[1, 0] = sine
    matrix[1, 1] = cosine
    return matrix


def test_live_a_alignment_preserves_same_row_space_operator():
    torch.manual_seed(7)
    rank, dim = 4, 12
    old_a = torch.randn(rank, dim)
    coordinate_change = torch.randn(rank, rank)
    coordinate_change = coordinate_change + 2.0 * torch.eye(rank)
    new_a = coordinate_change @ old_a
    aggregate = torch.randn(dim, rank)

    aligned, diagnostics = align_live_a_aggregate(
        aggregate, old_a, new_a
    )
    old_operator = normalized_effective_operator(aggregate, old_a)
    new_operator = normalized_effective_operator(aligned, new_a)

    assert torch.allclose(old_operator, new_operator, atol=2e-6, rtol=2e-6)
    assert diagnostics["after_relative_error"] < 2e-6
    assert diagnostics["after_relative_error"] < diagnostics[
        "before_relative_error"
    ]


def test_operator_preserving_absorption_matches_current_operator():
    torch.manual_seed(9)
    rank, dim = 4, 12
    shared_a = torch.randn(rank, dim, dtype=torch.float32)
    current_b = torch.randn(dim, rank, dtype=torch.float32)
    scale = torch.tensor(0.73, dtype=torch.float32)

    absorbed_up, diagnostics = absorb_live_a_current_projection(
        shared_a,
        current_b,
        scale,
        mode="operator_preserving_absorb",
    )
    before = scale * (current_b @ shared_a)
    after = absorbed_up @ (
        shared_a / (torch.linalg.vector_norm(shared_a) + 1e-8)
    )
    relative_error = torch.linalg.vector_norm(after - before) / (
        torch.linalg.vector_norm(before) + 1e-8
    )

    assert relative_error < 1e-6
    assert diagnostics["absorption_relative_error"] < 1e-6


def test_normalized_absorption_reproduces_legacy_rule():
    torch.manual_seed(91)
    shared_a = torch.randn(3, 8)
    current_b = torch.randn(8, 3)
    scale = torch.tensor(-0.42)

    absorbed_up, diagnostics = absorb_live_a_current_projection(
        shared_a,
        current_b,
        scale,
        mode="normalized_absorb",
    )
    expected = scale * current_b / (
        torch.linalg.vector_norm(current_b) + 1e-8
    )

    assert torch.allclose(absorbed_up, expected, atol=1e-7, rtol=1e-7)
    assert diagnostics["absorption_relative_error"] > 1e-2


def test_bounded_norm_calibration_attenuates_oversized_operator():
    shared_a = torch.eye(3) * 2.0
    current_b = torch.eye(3)
    scale = torch.tensor(0.7)

    absorbed_up, diagnostics = absorb_live_a_current_projection(
        shared_a,
        current_b,
        scale,
        mode="bounded_norm_calibrated_absorb",
    )
    expected = scale * current_b / (
        torch.linalg.vector_norm(current_b) + 1e-8
    )
    expected_gain = 1.0 / (
        torch.linalg.vector_norm(shared_a)
        * torch.linalg.vector_norm(current_b)
    )

    assert torch.allclose(absorbed_up, expected, atol=1e-7, rtol=1e-7)
    assert diagnostics["consolidation_gain"] == pytest.approx(
        float(expected_gain), rel=1e-6
    )
    assert diagnostics["consolidation_gain"] < 1.0


def test_bounded_norm_calibration_never_amplifies_small_operator():
    shared_a = torch.eye(3) * 0.2
    current_b = torch.eye(3) * 0.2
    scale = torch.tensor(0.7)

    absorbed_up, diagnostics = absorb_live_a_current_projection(
        shared_a,
        current_b,
        scale,
        mode="bounded_norm_calibrated_absorb",
    )
    before = scale * (current_b @ shared_a)
    after = absorbed_up @ (
        shared_a / (torch.linalg.vector_norm(shared_a) + 1e-8)
    )

    assert diagnostics["norm_product"] < 1.0
    assert diagnostics["consolidation_gain"] == pytest.approx(1.0)
    assert torch.allclose(after, before, atol=1e-7, rtol=1e-6)
    assert diagnostics["absorption_relative_error"] < 1e-6


def test_operator_preserving_absorption_survives_backbone_rebuild(tmp_path):
    torch.manual_seed(10)
    dim, rank = 8, 3
    run = tmp_path / "absorption-rebuild"
    pristine = _TinyViT(dim)
    before_model = SharedALoRA_ViT_timm(
        copy.deepcopy(pristine),
        r=rank,
        filepath=str(run),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        live_a_absorb_mode="operator_preserving_absorb",
    )
    with torch.no_grad():
        for weight in before_model.w_Bs:
            weight.weight.copy_(torch.randn_like(weight.weight))
        before_model.wrapped_param[0].param.fill_(0.83)
    inputs = torch.randn(2, 5, dim)
    with torch.no_grad():
        before_output = before_model(inputs)
    before_model.save_lora_parameters(str(run), task_id=0)

    rebuilt_model = SharedALoRA_ViT_timm(
        copy.deepcopy(pristine),
        r=rank,
        filepath=str(run),
        cur_task_index=1,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        live_a_absorb_mode="operator_preserving_absorb",
    )
    with torch.no_grad():
        rebuilt_output = rebuilt_model(inputs)
    relative_error = torch.linalg.vector_norm(
        rebuilt_output - before_output
    ) / (torch.linalg.vector_norm(before_output) + 1e-8)
    max_abs_diff = (rebuilt_output - before_output).abs().max()

    assert relative_error < 1e-6, (
        "pre-save/post-rebuild relative error={:.3e}, max_abs_diff={:.3e}".format(
            float(relative_error), float(max_abs_diff)
        )
    )


def test_joint_live_a_boundary_factors_best_rank_for_capped_sum():
    torch.manual_seed(31)
    dim, rank = 8, 3
    old_a = torch.randn(rank, dim)
    old_g = torch.randn(dim, rank)
    current_a = torch.randn(rank, dim)
    current_b = 3.0 * torch.randn(dim, rank)
    scale = torch.tensor(0.81)
    current_up, absorb = absorb_live_a_current_projection(
        current_a, current_b, scale,
        mode="bounded_norm_calibrated_absorb",
    )
    target = (
        old_g.double() @ old_a.double() / old_a.double().norm()
        + current_up.double() @ current_a.double() / current_a.double().norm()
    )
    shared_a, merged_g, diagnostics = sa_lora.joint_live_a_boundary_factors(
        old_a, old_g, current_a, current_up
    )
    deployed = merged_g.double() @ shared_a.double() / shared_a.double().norm()
    optimal_tail = torch.linalg.svdvals(target)[rank:].square().sum()
    actual_error = (deployed - target).square().sum()

    assert shared_a.shape == old_a.shape
    assert merged_g.shape == old_g.shape
    assert shared_a.norm() == pytest.approx(current_a.norm(), rel=1e-6)
    assert actual_error == pytest.approx(optimal_tail, rel=1e-4, abs=1e-6)
    assert diagnostics["relative_truncation_error"] == pytest.approx(
        float((optimal_tail / target.square().sum()).sqrt()), rel=1e-4
    )
    assert absorb["consolidation_gain"] < 1.0


def test_joint_live_a_boundary_save_and_rebuild_uses_chosen_basis(tmp_path):
    torch.manual_seed(32)
    dim, rank = 8, 3
    run = tmp_path / "joint-boundary"
    pristine = _TinyViT(dim)
    settings = {
        "r": rank,
        "filepath": str(run),
        "train_a_all_tasks": True,
        "cumulative_state": True,
        "cumulative_merge": "live_a_aggregate_b",
        "live_a_coordinate_align": True,
        "live_a_absorb_mode": "bounded_norm_calibrated_absorb",
        "live_a_boundary_merge": "joint_svd",
    }
    task0 = SharedALoRA_ViT_timm(
        copy.deepcopy(pristine), cur_task_index=0, **settings
    )
    with torch.no_grad():
        for w_b in task0.w_Bs:
            w_b.weight.copy_(torch.randn_like(w_b.weight))
    task0.save_lora_parameters(str(run), task_id=0)
    old_state = torch.load(run / SA_STATE_FILENAME, map_location="cpu", weights_only=True)
    assert old_state["boundary_merge"] == "joint_svd"

    task1 = SharedALoRA_ViT_timm(
        copy.deepcopy(pristine), cur_task_index=1, **settings
    )
    with torch.no_grad():
        for w_a, w_b in zip(task1.w_As, task1.w_Bs):
            w_a.weight.add_(0.2 * torch.randn_like(w_a.weight))
            w_b.weight.copy_(2.0 * torch.randn_like(w_b.weight))
    scale = task1.wrapped_param[0].param.detach().cpu().float().reshape(())
    targets = []
    for idx, (w_a, w_b) in enumerate(zip(task1.w_As, task1.w_Bs)):
        a = w_a.weight.detach().cpu().float()
        b = w_b.weight.detach().cpu().float()
        current_up, _ = absorb_live_a_current_projection(
            a, b, scale, mode="bounded_norm_calibrated_absorb"
        )
        old_a = old_state["shared_a"][idx]
        old_g = old_state["aggregate_up"][idx]
        targets.append(
            old_g.double() @ old_a.double() / old_a.double().norm()
            + current_up.double() @ a.double() / a.double().norm()
        )
    task1.save_lora_parameters(str(run), task_id=1)
    state = torch.load(run / SA_STATE_FILENAME, map_location="cpu", weights_only=True)
    merged = torch.load(run / SA_MERGED_FILENAME, map_location="cpu", weights_only=True)
    assert state["boundary_merge"] == "joint_svd"
    for idx, target in enumerate(targets):
        a, g = state["shared_a"][idx], state["aggregate_up"][idx]
        deployed = g.double() @ a.double() / a.double().norm()
        optimal_tail = torch.linalg.svdvals(target)[rank:].square().sum()
        assert (deployed - target).square().sum() == pytest.approx(
            optimal_tail, rel=1e-4, abs=1e-6
        )
        assert torch.allclose(merged["shared_a"][idx], a)
        assert torch.allclose(
            merged["merged_b"][idx] @ a,
            (g / a.norm()) @ a,
            atol=1e-6,
        )
    reloaded = SharedALoRA_ViT_timm(
        copy.deepcopy(pristine), cur_task_index=2, **settings
    )
    x = torch.randn(2, 4, dim)
    wrapper = reloaded.lora_vit.blocks[0].attn.qkv
    with torch.no_grad():
        output = reloaded(x)
        expected = wrapper.qkv(x)
        expected[..., :dim] += x @ (
            state["aggregate_up"][0] @ state["shared_a"][0]
            / state["shared_a"][0].norm()
        ).t()
        expected[..., -dim:] += x @ (
            state["aggregate_up"][1] @ state["shared_a"][1]
            / state["shared_a"][1].norm()
        ).t()
    assert torch.allclose(output, expected, atol=2e-6)
    with pytest.raises(ValueError, match="boundary merge differs"):
        SharedALoRA_ViT_timm(
            copy.deepcopy(pristine), cur_task_index=2,
            **{**settings, "live_a_boundary_merge": "aligned"},
        )


def test_residual_transport_recovers_low_rank_rotation_and_unit_norms():
    torch.manual_seed(11)
    source = F.normalize(torch.randn(200, 8), p=2, dim=1)
    expected_rotation = _plane_rotation(8, angle=0.35)
    target = source @ expected_rotation

    transport = fit_residual_orthogonal_transport(
        source, target, rank=2, identity_reg=0.0
    )
    prototypes = {
        class_id: source[class_id].clone() for class_id in range(10)
    }
    moved = apply_residual_orthogonal_transport(prototypes, transport)
    expected = target[:10]
    actual = torch.stack([moved[class_id] for class_id in range(10)])

    assert transport["enabled"]
    assert transport["rank"] == 2
    assert transport["validation_after"] < transport["validation_before"]
    assert torch.allclose(actual.norm(dim=1), torch.ones(10), atol=1e-6)
    assert (actual * expected).sum(dim=1).mean() > 0.995


def test_residual_transport_validation_gate_returns_identity():
    torch.manual_seed(19)
    source = F.normalize(torch.randn(100, 8), p=2, dim=1)
    positive = _plane_rotation(8, angle=0.5)
    negative = _plane_rotation(8, angle=-0.5)
    target = source @ positive
    val_mask = torch.arange(source.shape[0]).remainder(5) == 0
    target[val_mask] = source[val_mask] @ negative

    transport = fit_residual_orthogonal_transport(
        source, target, rank=2, identity_reg=0.0
    )
    prototype = {0: source[0].clone()}
    moved = apply_residual_orthogonal_transport(prototype, transport)

    assert not transport["enabled"]
    assert torch.allclose(moved[0], prototype[0], atol=1e-6)


def test_coordinate_transport_config_allows_transport_without_alignment():
    args = {
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": False,
    }

    validate_coordinate_transport_config(
        args,
        use_prototypes=True,
        lrpt_enabled=False,
        transport_rank=10,
    )

    args["sa_cumulative_merge"] = "sensitivity_budgeted_g"
    validate_coordinate_transport_config(
        args,
        use_prototypes=True,
        lrpt_enabled=False,
        transport_rank=10,
    )

    args["sa_cumulative_merge"] = "gauge"
    with pytest.raises(ValueError, match="live_a_aggregate_b"):
        validate_coordinate_transport_config(
            args,
            use_prototypes=True,
            lrpt_enabled=False,
            transport_rank=10,
        )


def test_coordinate_aligned_live_a_state_roundtrip(tmp_path):
    torch.manual_seed(31)
    dim, rank = 8, 3
    run = tmp_path / "run"
    task0 = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        live_a_coordinate_align=True,
    )
    with torch.no_grad():
        for weight in task0.w_Bs:
            weight.weight.copy_(torch.randn_like(weight.weight))
    task0.save_lora_parameters(str(run), task_id=0)
    old_state = torch.load(
        run / SA_STATE_FILENAME, map_location="cpu", weights_only=True
    )
    assert old_state["coordinate_aligned"] is True

    task1 = SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=1,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        live_a_coordinate_align=True,
    )
    with torch.no_grad():
        coordinate_change = torch.randn(rank, rank) + 2.0 * torch.eye(rank)
        for weight in task1.w_As:
            weight.weight.copy_(coordinate_change @ weight.weight)
        for weight in task1.w_Bs:
            weight.weight.copy_(torch.randn_like(weight.weight))
    current_b = [weight.weight.detach().cpu().clone() for weight in task1.w_Bs]
    current_scale = task1.wrapped_param[0].param.detach().cpu().reshape(())
    task1.save_lora_parameters(str(run), task_id=1)

    new_state = torch.load(
        run / SA_STATE_FILENAME, map_location="cpu", weights_only=True
    )
    merged = torch.load(
        run / SA_MERGED_FILENAME, map_location="cpu", weights_only=True
    )
    assert new_state["coordinate_aligned"] is True
    for index, new_aggregate in enumerate(new_state["aggregate_up"]):
        current_term, absorption = absorb_live_a_current_projection(
            new_state["shared_a"][index],
            current_b[index],
            current_scale,
            mode="operator_preserving_absorb",
        )
        assert absorption["absorption_relative_error"] < 1e-6
        aligned_history = new_aggregate - current_term
        old_operator = normalized_effective_operator(
            old_state["aggregate_up"][index], old_state["shared_a"][index]
        )
        new_operator = normalized_effective_operator(
            aligned_history, new_state["shared_a"][index]
        )
        assert torch.allclose(
            old_operator, new_operator, atol=3e-6, rtol=3e-6
        )
        assert torch.allclose(
            merged["aggregate_up"][index], new_aggregate, atol=1e-7
        )

    SharedALoRA_ViT_timm(
        _TinyViT(dim),
        r=rank,
        filepath=str(run),
        cur_task_index=2,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        live_a_coordinate_align=True,
    )


def test_backbone_factory_preserves_coordinate_alignment_flag(
    tmp_path, monkeypatch
):
    run = tmp_path / "factory-run"
    monkeypatch.setattr(
        "utils.inc_net.timm.create_model",
        lambda *args, **kwargs: _TinyViT(8),
    )
    backbone = get_backbone(
        {
            "backbone_type": "vit_base_patch16_224",
            "model_name": "sa_sdlora",
            "lora_rank": 3,
            "increment": 10,
            "filepath": str(run),
            "sa_train_a_all_tasks": True,
            "sa_cumulative_state": True,
            "sa_cumulative_merge": "live_a_aggregate_b",
            "sa_live_a_coordinate_align": True,
        },
        pretrained=True,
    )

    assert backbone.live_a_coordinate_align is True
    assert backbone.live_a_absorb_mode == "operator_preserving_absorb"
    backbone.save_lora_parameters(str(run), task_id=0)
    state = torch.load(
        run / SA_STATE_FILENAME, map_location="cpu", weights_only=True
    )
    assert state["coordinate_aligned"] is True
    assert state["absorb_mode"] == "operator_preserving_absorb"


def test_backbone_factory_preserves_joint_boundary_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "utils.inc_net.timm.create_model",
        lambda *args, **kwargs: _TinyViT(8),
    )
    backbone = get_backbone(
        {
            "backbone_type": "vit_base_patch16_224",
            "model_name": "sa_sdlora",
            "lora_rank": 3,
            "increment": 10,
            "filepath": str(tmp_path / "factory-joint"),
            "sa_train_a_all_tasks": True,
            "sa_cumulative_state": True,
            "sa_cumulative_merge": "live_a_aggregate_b",
            "sa_live_a_coordinate_align": True,
            "sa_live_a_boundary_merge": "joint_svd",
        },
        pretrained=True,
    )
    assert backbone.live_a_boundary_merge == "joint_svd"
