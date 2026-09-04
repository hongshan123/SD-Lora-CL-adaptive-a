"""Tests for Adaptive-A coordinate-plasticity gradient gating."""

import sys
import logging
from pathlib import Path
from uuid import uuid4

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import (
    SA_STATE_FILENAME,
    SharedALoRA_ViT_timm,
    adaptive_a_layer_gradient,
    canonical_down_projection,
    low_rank_product_frobenius_norm,
)
from models.sdlora import Learner as SDLoraLearner
from models.sa_sdlora import (
    Learner as SharedALearner,
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

    assert torch.allclose(actual, torch.linalg.matrix_norm(up @ down))


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
    learner._after_backward = lambda: events.append("hook")
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

    def apply_adaptive_a_gradients(self):
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

    def apply_adaptive_a_gradients(self):
        self.apply_calls += 1
        self.weight.grad.zero_()


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


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"sa_train_a_all_tasks": False}, "sa_train_a_all_tasks"),
        ({"sa_cumulative_state": False}, "sa_cumulative_state"),
        ({"sa_cumulative_merge": "gauge"}, "live_a_aggregate_b"),
        ({"sa_live_a_coordinate_align": False}, "coordinate_align"),
        ({"sa_adaptive_a_gate_floor": 1.1}, "gate_floor"),
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
        "adaptive_a_eps": 1e-8,
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
            "sa_adaptive_a_eps": 1e-6,
        }
    )

    assert backbone.adaptive_a_enabled is True
    assert backbone.adaptive_a_stability_weight == pytest.approx(2.5)
    assert backbone.adaptive_a_gate_floor == pytest.approx(0.2)
    assert backbone.adaptive_a_gate_momentum == pytest.approx(0.6)
    assert backbone.adaptive_a_eps == pytest.approx(1e-6)


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
        "sa_adaptive_a_eps": 1e-6,
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
            "adaptive_a_eps",
        )
    } == {
        "adaptive_a_enabled": True,
        "adaptive_a_stability_weight": 2.5,
        "adaptive_a_gate_floor": 0.2,
        "adaptive_a_gate_momentum": 0.6,
        "adaptive_a_eps": 1e-6,
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
