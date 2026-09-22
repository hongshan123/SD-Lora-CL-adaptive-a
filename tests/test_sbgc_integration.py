import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import (
    SA_MERGED_FILENAME,
    SA_STATE_FILENAME,
    SA_STATE_VERSION_SBGC,
    SharedALoRA_ViT_timm,
    _SensitivityBudgetedGQKV,
)
from models.sa_sdlora import validate_sbgc_config
from models.sa_sdlora import Learner as SharedALearner
from utils.inc_net import get_backbone
from utils.rng_utils import rng_state_hash


class _TinyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _TinyAttention(dim)


class _TinyViT(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dim)])
        self.head = nn.Identity()

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block.attn.qkv(inputs)
        return self.head(inputs.mean(dim=1))


class _CalibrationDataset(Dataset):
    def __init__(self, count=5):
        generator = torch.Generator().manual_seed(811)
        self.inputs = torch.randn(count, 4, 6, generator=generator)
        self.targets = torch.arange(count) % 2

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return index, self.inputs[index], self.targets[index]


class _CalibrationDataManager:
    def __init__(self):
        self.calls = []
        self.dataset = _CalibrationDataset()

    def get_dataset(self, classes, source, mode):
        self.calls.append((tuple(classes.tolist()), source, mode))
        return self.dataset


class _DictLinear(nn.Linear):
    def forward(self, inputs):
        return {"logits": super().forward(inputs)}


class _RawNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = _DictLinear(18, 2)

    def forward(self, inputs):
        features = self.backbone(inputs)
        output = self.fc(features)
        output["features"] = features
        return output


def _make_model(run, task, vit=None, metric="fisher_diag", shadow=False):
    return SharedALoRA_ViT_timm(
        copy.deepcopy(vit) if vit is not None else _TinyViT(),
        r=2,
        filepath=str(run),
        cur_task_index=task,
        train_a_all_tasks=False,
        cumulative_state=True,
        cumulative_merge="sensitivity_budgeted_g",
        cumulative_rank=2,
        g_risk_budget=0.05,
        g_sensitivity_metric=metric,
        g_sensitivity_floor=1e-4,
        g_solver_ridge=1e-6,
        g_bisection_steps=40,
        g_shadow_only=shadow,
    )


def _collect_and_finalize(model, inputs):
    outputs = model(inputs)
    branch_outputs = model.sbgc_calibration_outputs()
    weights = torch.linspace(
        0.2,
        2.0,
        branch_outputs[0].shape[-1],
        device=inputs.device,
        dtype=inputs.dtype,
    )
    loss = sum(
        (branch * weights).sum(dim=(1, 2)).mean()
        for branch in branch_outputs
    )
    gradients = torch.autograd.grad(loss, branch_outputs)
    model.accumulate_sbgc_sensitivities(gradients, inputs.shape[0])
    stats = model.finalize_sbgc_calibration()
    return outputs.detach(), stats


def _train_and_save_task_zero(run, vit, shadow=False):
    torch.manual_seed(701)
    model = _make_model(run, task=0, vit=vit, shadow=shadow)
    with torch.no_grad():
        for module in model.w_As + model.w_Bs:
            module.weight.copy_(torch.randn_like(module.weight))
        model.wrapped_param[0].param.fill_(0.63)
    model.prepare_sbgc_calibration()
    _collect_and_finalize(model, torch.randn(5, 4, 6))
    model.save_lora_parameters(str(run), task_id=0)
    return model


def test_task_zero_qr_and_absorption_preserve_effective_operator(tmp_path):
    model = _make_model(tmp_path / "run", task=0)
    with torch.no_grad():
        for module in model.w_As + model.w_Bs:
            module.weight.copy_(torch.randn_like(module.weight))
        model.wrapped_param[0].param.fill_(0.57)
    scale = model.wrapped_param[0].param.detach().reshape(())
    before = [
        scale * b.weight.detach() @ a.weight.detach()
        for a, b in zip(model.w_As, model.w_Bs)
    ]

    model.prepare_sbgc_calibration()

    wrappers = model._sbgc_wrappers()
    projections = [
        tensor
        for wrapper in wrappers
        for tensor in (wrapper.projection_q, wrapper.projection_v)
    ]
    canonical = [
        scale * b.weight.detach() @ a.weight.detach()
        for a, b in zip(model.w_As, model.w_Bs)
    ]
    for expected, actual, projection in zip(before, canonical, projections):
        relative = torch.linalg.matrix_norm(actual - expected) / torch.linalg.matrix_norm(
            expected
        ).clamp_min(1e-12)
        assert float(relative) < 1e-6
        assert torch.allclose(
            projection @ projection.t(),
            torch.eye(projection.shape[0]),
            atol=2e-6,
            rtol=2e-6,
        )

    _, stats = _collect_and_finalize(model, torch.randn(4, 3, 6))
    absorbed = [
        up @ down
        for wrapper in wrappers
        for up, down in (
            (wrapper.unified_up_q, wrapper.projection_q),
            (wrapper.unified_up_v, wrapper.projection_v),
        )
    ]
    for expected, actual in zip(before, absorbed):
        relative = torch.linalg.matrix_norm(actual - expected) / torch.linalg.matrix_norm(
            expected
        ).clamp_min(1e-12)
        assert float(relative) < 1e-6
    assert stats["task0_operator_error"] < 1e-6
    assert all(torch.count_nonzero(module.weight) == 0 for module in model.w_Bs)


def test_state_roundtrip_preserves_deployed_logits_and_fixed_p(tmp_path):
    run = tmp_path / "run"
    pristine = _TinyViT()
    model = _train_and_save_task_zero(run, pristine)
    inputs = torch.randn(2, 5, 6)
    with torch.no_grad():
        before = model(inputs)

    rebuilt = _make_model(run, task=1, vit=pristine)
    with torch.no_grad():
        after = rebuilt(inputs)

    assert torch.allclose(after, before, atol=2e-6, rtol=2e-6)
    assert all(not module.weight.requires_grad for module in rebuilt.w_As)
    state = torch.load(run / SA_STATE_FILENAME, weights_only=True)
    assert set(state) == {
        "version",
        "task_id",
        "rank",
        "merge_mode",
        "projection_down",
        "unified_up",
        "projected_covariance",
        "sensitivity_diag",
        "covariance_count",
        "sensitivity_count",
        "risk_budget",
        "sensitivity_metric",
        "sensitivity_floor",
        "solver_ridge",
        "bisection_steps",
        "shadow_only",
    }
    assert state["version"] == SA_STATE_VERSION_SBGC
    assert len(state["projection_down"]) == 2
    assert len(state["projected_covariance"]) == 2
    assert bool((state["covariance_count"] > 0).all())
    assert bool((state["sensitivity_count"] > 0).all())
    merged = torch.load(run / SA_MERGED_FILENAME, weights_only=True)
    assert set(merged) == {
        "version",
        "task_id",
        "rank",
        "merge_mode",
        "projection_down",
        "unified_up",
    }
    assert not list(run.glob("sa_lora_w_b_*.pt"))


def test_task_one_keeps_p_fixed_and_meets_deployed_risk_budget(tmp_path):
    run = tmp_path / "run"
    pristine = _TinyViT()
    _train_and_save_task_zero(run, pristine)
    model = _make_model(run, task=1, vit=pristine)
    wrapper = model.lora_vit.blocks[0].attn.qkv
    assert isinstance(wrapper, _SensitivityBudgetedGQKV)
    before_p = [wrapper.projection_q.clone(), wrapper.projection_v.clone()]
    with torch.no_grad():
        model.w_Bs[0].weight.copy_(8.0 * wrapper.unified_up_q)
        model.w_Bs[1].weight.copy_(8.0 * wrapper.unified_up_v)

    model.prepare_sbgc_calibration()
    _, stats = _collect_and_finalize(model, torch.randn(6, 4, 6))

    assert torch.equal(wrapper.projection_q, before_p[0])
    assert torch.equal(wrapper.projection_v, before_p[1])
    assert torch.equal(model.w_As[0].weight, before_p[0])
    assert torch.equal(model.w_As[1].weight, before_p[1])
    assert stats["active_constraints"] == 2
    assert stats["max_achieved_risk"] <= 0.05 + 1e-6
    assert all(
        branch["deployed_historical_risk"] <= 0.05 + 1e-6
        for branch in model._last_sbgc_branch_diagnostics
    )


def test_shadow_mode_computes_candidates_but_deploys_additive_target(tmp_path):
    run = tmp_path / "run"
    pristine = _TinyViT()
    _train_and_save_task_zero(run, pristine, shadow=True)
    model = _make_model(run, task=1, vit=pristine, shadow=True)
    wrapper = model.lora_vit.blocks[0].attn.qkv
    scale = model.wrapped_param[0].param.detach().reshape(())
    old_q = wrapper.unified_up_q.clone()
    old_v = wrapper.unified_up_v.clone()
    with torch.no_grad():
        wrapper.b_q.weight.copy_(5.0 * old_q / scale)
        wrapper.b_v.weight.copy_(5.0 * old_v / scale)
    target_q = old_q + scale * wrapper.b_q.weight.detach()
    target_v = old_v + scale * wrapper.b_v.weight.detach()

    model.prepare_sbgc_calibration()
    _collect_and_finalize(model, torch.randn(5, 3, 6))

    assert torch.allclose(wrapper.unified_up_q, target_q, atol=1e-6, rtol=1e-6)
    assert torch.allclose(wrapper.unified_up_v, target_v, atol=1e-6, rtol=1e-6)
    diagnostics = model._last_sbgc_branch_diagnostics
    assert all(item["fisher"]["constraint_active"] for item in diagnostics)
    assert all(item["deployed_historical_risk"] > 0.05 for item in diagnostics)


def test_learner_boundary_pass_preserves_rng_head_and_existing_gradients(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    backbone = _make_model(run, task=0)
    with torch.no_grad():
        for module in backbone.w_As + backbone.w_Bs:
            module.weight.copy_(torch.randn_like(module.weight))
    raw_network = _RawNetwork(backbone).train()
    for parameter in raw_network.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.randn_like(parameter)
    head_before = {
        name: tensor.detach().clone()
        for name, tensor in raw_network.fc.state_dict().items()
    }
    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in raw_network.named_parameters()
    }
    data_manager = _CalibrationDataManager()
    learner = object.__new__(SharedALearner)
    learner._cur_task = 0
    learner._known_classes = 0
    learner._total_classes = 2
    learner._device = torch.device("cpu")
    learner.args = {"batch_size": 2, "filepath": str(run)}
    learner._cuo_calibration_data_manager = data_manager
    learner._sa_g_calibration_batch_size = 2
    learner._loader_workers = lambda: 0
    learner._is_main_process = lambda: True
    rng_before = rng_state_hash()

    SharedALearner._before_task_save(learner, raw_network, train_loader=None)

    assert rng_state_hash() == rng_before
    assert raw_network.training is True
    assert data_manager.calls == [((0, 1), "train", "test")]
    assert backbone._sbgc_calibration_finalized
    for name, tensor in raw_network.fc.state_dict().items():
        assert torch.equal(tensor, head_before[name])
    for name, parameter in raw_network.named_parameters():
        expected = gradients_before[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, expected)
    diagnostics = torch.load(
        run / "sbgc_diagnostics.pt", map_location="cpu", weights_only=False
    )
    assert diagnostics["version"] == 1
    assert diagnostics["tasks"][0]["task_id"] == 0
    stats = diagnostics["tasks"][0]["stats"]
    assert stats["calibration_seconds"] >= 0.0
    assert stats["solver_seconds"] >= 0.0
    assert stats["boundary_seconds"] >= stats["calibration_seconds"]
    assert stats["peak_cuda_allocated_mib"] == 0.0
    assert stats["additional_peak_cuda_allocated_mib"] == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sa_cumulative_state": False}, "sa_cumulative_state"),
        ({"sa_train_a_all_tasks": True}, "sa_train_a_all_tasks"),
        ({"sa_cumulative_rank": 2}, "sa_cumulative_rank"),
        ({"sa_g_risk_budget": 1.1}, "sa_g_risk_budget"),
        ({"sa_g_sensitivity_metric": "dense"}, "sensitivity_metric"),
        ({"sa_g_sensitivity_floor": 0.0}, "sensitivity_floor"),
        ({"sa_g_solver_ridge": 0.0}, "solver_ridge"),
        ({"sa_g_bisection_steps": 0}, "bisection_steps"),
        ({"sa_coordinate_stable_transport": True}, "coordinate transport"),
        ({"sa_hbd_enabled": True}, "HBD"),
        ({"sa_adaptive_a_enabled": True}, "Adaptive-A"),
        ({"sa_dual_head": True}, "Dual-B"),
        (
            {"sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb"},
            "NormCap",
        ),
    ],
)
def test_sbgc_rejects_incompatible_configuration(overrides, message):
    config = {
        "sa_cumulative_merge": "sensitivity_budgeted_g",
        "sa_cumulative_state": True,
        "sa_train_a_all_tasks": False,
        "lora_rank": 3,
        "sa_cumulative_rank": 3,
    }
    config.update(overrides)
    with pytest.raises(ValueError, match=message):
        validate_sbgc_config(config)


def test_sbgc_accepts_isolated_fixed_projection_configuration():
    config = {
        "sa_cumulative_merge": "sensitivity_budgeted_g",
        "sa_cumulative_state": True,
        "sa_train_a_all_tasks": False,
        "lora_rank": 3,
        "sa_cumulative_rank": 3,
    }
    settings = validate_sbgc_config(config)
    assert settings == {
        "g_risk_budget": 0.05,
        "g_sensitivity_metric": "fisher_diag",
        "g_sensitivity_floor": 1e-4,
        "g_solver_ridge": 1e-6,
        "g_bisection_steps": 40,
        "g_shadow_only": False,
    }


def test_initial_backbone_factory_forwards_all_sbgc_settings(
    tmp_path, monkeypatch
):
    """Task 0 and task-boundary rebuilds must use identical SBGC settings."""
    monkeypatch.setattr(
        "utils.inc_net.timm.create_model", lambda *args, **kwargs: _TinyViT()
    )
    backbone = get_backbone(
        {
            "backbone_type": "vit_base_patch16_224",
            "model_name": "sa_sdlora",
            "lora_rank": 2,
            "increment": 2,
            "filepath": str(tmp_path),
            "sa_cumulative_state": True,
            "sa_cumulative_merge": "sensitivity_budgeted_g",
            "sa_cumulative_rank": 2,
            "sa_train_a_all_tasks": False,
            "sa_g_risk_budget": 0.075,
            "sa_g_sensitivity_metric": "uniform",
            "sa_g_sensitivity_floor": 2e-4,
            "sa_g_solver_ridge": 3e-6,
            "sa_g_bisection_steps": 17,
            "sa_g_shadow_only": True,
        }
    )

    assert backbone.g_risk_budget == pytest.approx(0.075)
    assert backbone.g_sensitivity_metric == "uniform"
    assert backbone.g_sensitivity_floor == pytest.approx(2e-4)
    assert backbone.g_solver_ridge == pytest.approx(3e-6)
    assert backbone.g_bisection_steps == 17
    assert backbone.g_shadow_only is True
