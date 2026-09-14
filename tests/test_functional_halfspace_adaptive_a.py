"""Tests for the pure global functional-halfspace projection kernel."""

import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch import optim


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone import sa_lora as sa_lora_module  # noqa: E402
from backbone.sa_lora import (  # noqa: E402
    SharedALoRA_ViT_timm,
    decompose_adaptive_a_gradient,
    project_functional_halfspace_directions,
)
from models import sa_sdlora as sa_sdlora_module  # noqa: E402
from models.sa_sdlora import Learner as SharedALearner  # noqa: E402
from models.sa_sdlora import validate_adaptive_a_config  # noqa: E402
from models.sa_sdlora import (  # noqa: E402
    should_capture_functional_diagnostic,
    validate_functional_diagnostics_interval,
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
    def __init__(self, dim, blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dim) for _ in range(blocks)])
        self.head = nn.Identity()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _functional_model(tmp_path, strategy="functional_halfspace", blocks=1):
    return SharedALoRA_ViT_timm(
        _TinyViT(4, blocks=blocks),
        r=2,
        filepath=str(tmp_path / "run"),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        live_a_coordinate_align=True,
        adaptive_a_enabled=True,
        adaptive_a_strategy=strategy,
    )


@pytest.mark.parametrize(
    "proposed, stability, shared_as",
    [
        ([], [], []),
        ([torch.ones(1, 2)], [], [torch.ones(1, 2)]),
        (
            [torch.ones(1, 2)],
            [torch.ones(1, 2)],
            [torch.ones(2, 1)],
        ),
    ],
)
def test_functional_halfspace_rejects_invalid_tensor_lists(
    proposed, stability, shared_as
):
    """A missing or mismatched branch must never get a partial projection."""
    with pytest.raises(ValueError):
        project_functional_halfspace_directions(proposed, stability, shared_as)


def test_functional_halfspace_keeps_nonconflicting_directions_exactly():
    """Changing the non-conflict branch to project would be a regression."""
    proposed = [torch.tensor([[2.0, 3.0]])]
    stability = [torch.tensor([[1.0, 4.0]])]
    shared_as = [torch.tensor([[1.0, 0.0]])]

    result = project_functional_halfspace_directions(
        proposed, stability, shared_as
    )

    assert result["mode"] == "identity"
    assert result["pre_inner"] == pytest.approx(14.0)
    assert torch.equal(result["directions"][0], proposed[0])
    assert result["directions"][0].data_ptr() != proposed[0].data_ptr()


def test_functional_halfspace_corrects_one_global_normal_across_matrices():
    """Per-matrix projection would produce different directions for this case."""
    proposed = [
        torch.tensor([[-2.0, -3.0, 0.0]]),
        torch.tensor([[1.0, -4.0, 0.0]]),
    ]
    stability = [
        torch.tensor([[2.0, 3.0, 0.0]]),
        torch.tensor([[-1.0, 4.0, 0.0]]),
    ]
    shared_as = [
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    ]

    result = project_functional_halfspace_directions(
        proposed, stability, shared_as
    )

    assert result["mode"] == "normal"
    assert torch.equal(
        result["directions"][0], torch.tensor([[-2.0, 6.0, 0.0]])
    )
    assert torch.equal(
        result["directions"][1], torch.tensor([[-2.0, -4.0, 0.0]])
    )
    assert result["pre_inner"] == pytest.approx(-30.0)
    assert result["post_inner"] >= -1e-6 * max(1.0, abs(result["pre_inner"]))
    assert result["normal_fraction"] == pytest.approx(1.0 / 3.0)
    assert result["correction_ratio"] == pytest.approx(3.0 ** 0.5)


def test_functional_halfspace_preserves_each_row_space_parallel_component():
    """Replacing the normal correction with full-space projection loses this invariant."""
    proposed = [
        torch.tensor([[-2.0, -3.0, 0.0]]),
        torch.tensor([[1.0, -4.0, 0.0]]),
    ]
    stability = [
        torch.tensor([[2.0, 3.0, 0.0]]),
        torch.tensor([[-1.0, 4.0, 0.0]]),
    ]
    shared_as = [
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    ]

    result = project_functional_halfspace_directions(
        proposed, stability, shared_as
    )

    for original, projected, shared_a in zip(
        proposed, result["directions"], shared_as
    ):
        original_parallel = decompose_adaptive_a_gradient(original, shared_a)[
            "tangent"
        ]
        projected_parallel = decompose_adaptive_a_gradient(projected, shared_a)[
            "tangent"
        ]
        assert torch.allclose(projected_parallel, original_parallel, atol=1e-6)


def test_functional_halfspace_uses_full_projection_for_row_space_stability():
    """A zero normal channel must fall back to the complete stability gradient."""
    proposed = [torch.tensor([[-2.0, 7.0]])]
    stability = [torch.tensor([[2.0, 0.0]])]
    shared_as = [torch.tensor([[1.0, 0.0]])]

    result = project_functional_halfspace_directions(
        proposed, stability, shared_as
    )

    assert result["mode"] == "full"
    assert torch.equal(result["directions"][0], torch.tensor([[0.0, 7.0]]))
    assert result["post_inner"] >= -1e-6 * max(1.0, abs(result["pre_inner"]))
    assert result["normal_fraction"] == pytest.approx(0.0)


def test_functional_halfspace_uses_full_projection_for_small_normal_fraction():
    """Ignoring the fraction threshold would wrongly select normal projection."""
    proposed = [torch.tensor([[-10.0, -1.0]])]
    stability = [torch.tensor([[10.0, 1.0]])]
    shared_as = [torch.tensor([[1.0, 0.0]])]

    result = project_functional_halfspace_directions(
        proposed, stability, shared_as, min_normal_fraction=0.1
    )

    assert result["mode"] == "full"
    assert torch.equal(result["directions"][0], torch.zeros_like(proposed[0]))
    assert result["normal_fraction"] == pytest.approx(1.0 / 101.0)


def test_functional_halfspace_preserves_same_dtype_output_and_halfspace():
    """Same-dtype correction must retain direction storage and the constraint."""
    proposed = [torch.tensor([[-2.0, -2.0, 0.0]], dtype=torch.float32)]
    stability = [torch.tensor([[2.0, 2.0, 0.0]], dtype=torch.float32)]
    shared_as = [torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)]

    result = project_functional_halfspace_directions(
        proposed, stability, shared_as
    )

    projected = result["directions"][0]
    assert result["mode"] == "normal"
    assert projected.dtype == proposed[0].dtype
    assert projected.device == proposed[0].device
    assert torch.equal(
        projected, torch.tensor([[-2.0, 2.0, 0.0]], dtype=torch.float32)
    )
    assert result["post_inner"] >= -1e-6 * max(1.0, abs(result["pre_inner"]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_functional_halfspace_rejects_low_precision_projection_tensors(dtype):
    """Storage quantization must not weaken the hard halfspace guarantee."""
    tensors = [
        torch.tensor([[-2.0, -2.0, 0.0]], dtype=dtype),
        torch.tensor([[2.0, 2.0, 0.0]], dtype=dtype),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype),
    ]

    with pytest.raises(ValueError, match="float32 or torch.float64"):
        project_functional_halfspace_directions(*[[tensor] for tensor in tensors])


def test_functional_halfspace_rejects_per_branch_dtype_mismatches():
    """Mixed precision cannot preserve the hard halfspace after correction."""
    with pytest.raises(ValueError, match="dtypes must match"):
        project_functional_halfspace_directions(
            [torch.tensor([[-2.0, -2.0, 0.0]], dtype=torch.float16)],
            [torch.tensor([[2.0, 2.0, 0.0]], dtype=torch.float32)],
            [torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)],
        )


def test_functional_halfspace_rejects_per_branch_device_mismatches():
    """Paired tensors on different devices must fail before projection math."""
    with pytest.raises(ValueError, match="devices must match"):
        project_functional_halfspace_directions(
            [torch.tensor([[-2.0, 2.0]])],
            [torch.empty((1, 2), device="meta")],
            [torch.tensor([[1.0, 0.0]])],
        )


def test_functional_halfspace_aggregates_float32_products_in_float64():
    """Float32 product overflow must not turn a valid full projection into NaNs."""
    proposed = [torch.tensor([[-1e20, 3.0]], dtype=torch.float32)]
    stability = [torch.tensor([[1e20, 0.0]], dtype=torch.float32)]
    shared_as = [torch.tensor([[1.0, 0.0]], dtype=torch.float32)]

    result = project_functional_halfspace_directions(
        proposed, stability, shared_as
    )

    assert result["mode"] == "full"
    assert torch.equal(result["directions"][0], torch.tensor([[0.0, 3.0]]))
    assert result["pre_inner"] == pytest.approx(-1e40)
    assert result["post_inner"] >= -1e-6 * max(1.0, abs(result["pre_inner"]))


def test_functional_halfspace_keeps_degenerate_stability_direction_exactly():
    """A zero stability gradient must be a cloned no-op, not a perturbed divide."""
    proposed = [torch.tensor([[2.0, -3.0]]), torch.tensor([[4.0, 5.0]])]
    stability = [torch.zeros_like(proposed[0]), torch.zeros_like(proposed[1])]
    shared_as = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]

    result = project_functional_halfspace_directions(
        proposed, stability, shared_as
    )

    assert result["mode"] == "degenerate"
    assert result["pre_inner"] == pytest.approx(0.0)
    assert result["post_inner"] == pytest.approx(0.0)
    for original, projected in zip(proposed, result["directions"]):
        assert torch.equal(projected, original)
        assert projected.data_ptr() != original.data_ptr()


def test_functional_halfspace_config_has_exact_defaults_and_bounds():
    """Changing the functional thresholds or accepting invalid values is a bug."""
    base = {
        "sa_adaptive_a_enabled": True,
        "sa_adaptive_a_strategy": "functional_halfspace",
        "sa_train_a_all_tasks": True,
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
        "optimizer": "sgd",
    }
    defaults = validate_adaptive_a_config(base)
    assert defaults["adaptive_a_strategy"] == "functional_halfspace"
    assert defaults["functional_conflict_tol"] == pytest.approx(1e-12)
    assert defaults["functional_normal_tol"] == pytest.approx(1e-12)
    assert defaults["functional_min_normal_fraction"] == pytest.approx(1e-4)
    for key, value, message in (
        ("sa_functional_conflict_tol", -1.0, "conflict_tol"),
        ("sa_functional_normal_tol", 0.0, "normal_tol"),
        ("sa_functional_min_normal_fraction", -0.1, "min_normal_fraction"),
        ("sa_functional_min_normal_fraction", 1.1, "min_normal_fraction"),
        ("optimizer", "adam", "require SGD"),
    ):
        invalid = {**base, key: value}
        with pytest.raises(ValueError, match=message):
            validate_adaptive_a_config(invalid)


def test_functional_halfspace_rejects_ordinary_hbd_scalar_loss():
    """Combining the controller with HBD would add its distance to CE loss."""
    functional = {
        "sa_adaptive_a_enabled": True,
        "sa_adaptive_a_strategy": "functional_halfspace",
        "sa_train_a_all_tasks": True,
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
        "optimizer": "sgd",
        "sa_hbd_enabled": True,
    }

    with pytest.raises(ValueError, match="functional_halfspace.*sa_hbd_enabled"):
        validate_adaptive_a_config(functional)

    ordinary_hbd = {**functional, "sa_adaptive_a_strategy": "impact_ratio"}
    assert validate_adaptive_a_config(ordinary_hbd)["adaptive_a_strategy"] == "impact_ratio"


def test_functional_halfspace_task_zero_keeps_shared_a_gradients_exactly(tmp_path):
    """Projecting task zero would violate the no-history lifecycle."""
    model = _functional_model(tmp_path)
    model.task_id = 0
    original = [
        torch.randn_like(module.weight) for module in model.w_As
    ]
    stability = [torch.randn_like(module.weight) for module in model.w_As]
    for module, gradient in zip(model.w_As, original):
        module.weight.grad = gradient.clone()

    model.apply_adaptive_a_gradients(stability_gradients=stability)

    for module, gradient in zip(model.w_As, original):
        assert torch.equal(module.weight.grad, gradient)


def test_functional_halfspace_reconstructs_sgd_direction_and_preserves_b(tmp_path):
    """Writing d* directly to grad would make SGD apply d* + mu*v instead."""
    model = _functional_model(tmp_path)
    model.task_id = 1
    with torch.no_grad():
        model.w_As[0].weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                  [0.0, 1.0, 0.0, 0.0]]))
        model.w_As[1].weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                  [0.0, 1.0, 0.0, 0.0]]))
    gradients = [
        torch.tensor([[0.0, 0.0, -2.0, -3.0], [0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0, -4.0], [0.0, 0.0, 0.0, 0.0]]),
    ]
    stability = [
        torch.tensor([[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, -1.0, 4.0], [0.0, 0.0, 0.0, 0.0]]),
    ]
    buffers = [torch.full_like(gradient, 0.5) for gradient in gradients]
    momentum = 0.9
    for module, gradient in zip(model.w_As, gradients):
        module.weight.grad = gradient.clone()
    b_gradients = [
        torch.randn_like(module.weight) for module in model.w_Bs
    ]
    for module, gradient in zip(model.w_Bs, b_gradients):
        module.weight.grad = gradient.clone()
    classifier = nn.Linear(4, 3, bias=False)
    classifier_gradient = torch.randn_like(classifier.weight)
    classifier.weight.grad = classifier_gradient.clone()
    network = type(
        "Network", (), {"backbone": model, "classifier": classifier}
    )()
    optimizer = optim.SGD(
        list(model.parameters()) + list(classifier.parameters()),
        lr=0.1,
        momentum=momentum,
    )
    for module, buffer in zip(model.w_As, buffers):
        optimizer.state[module.weight]["momentum_buffer"] = buffer.clone()
    learner = object.__new__(SharedALearner)
    learner._sa_adaptive_a_enabled = True
    learner._sa_adaptive_a_strategy = "functional_halfspace"
    learner._cur_task = 1
    learner._network = network
    learner._raw_network = lambda: network
    learner._functional_stability_gradients = lambda inputs: stability

    expected = project_functional_halfspace_directions(
        [gradient + momentum * buffer for gradient, buffer in zip(gradients, buffers)],
        stability,
        [module.weight for module in model.w_As],
    )
    learner._after_backward(torch.ones(1, 4), optimizer=optimizer)

    for module, buffer, direction in zip(
        model.w_As, buffers, expected["directions"]
    ):
        assert torch.allclose(module.weight.grad + momentum * buffer, direction)
    for module, gradient in zip(model.w_Bs, b_gradients):
        assert torch.equal(module.weight.grad, gradient)
    assert torch.equal(classifier.weight.grad, classifier_gradient)

    diagnostics = model.adaptive_a_diagnostics()
    assert diagnostics["normal_projections"] == 1
    assert diagnostics["full_fallbacks"] == 0
    assert diagnostics["degenerate_noops"] == 0


def test_functional_halfspace_records_full_and_degenerate_diagnostics(tmp_path):
    """Mode accounting must distinguish fallback safety from a no-op."""
    model = _functional_model(tmp_path)
    model.task_id = 1
    with torch.no_grad():
        for module in model.w_As:
            module.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0, 0.0]]))
    for module in model.w_As:
        module.weight.grad = torch.tensor(
            [[-2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
    model.apply_adaptive_a_gradients(
        stability_gradients=[
            torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            torch.zeros_like(model.w_As[1].weight),
        ]
    )
    model.apply_adaptive_a_gradients(
        stability_gradients=[torch.zeros_like(module.weight) for module in model.w_As]
    )

    diagnostics = model.adaptive_a_diagnostics()
    assert diagnostics["observations"] == 2
    assert diagnostics["conflicts"] == 1
    assert diagnostics["normal_projections"] == 0
    assert diagnostics["full_fallbacks"] == 1
    assert diagnostics["degenerate_noops"] == 1


def test_functional_halfspace_diagnostics_ignore_tolerated_negative_inner(tmp_path):
    """Counting a within-tolerance no-op as conflict misstates controller activity."""
    model = _functional_model(tmp_path)
    model.task_id = 1
    with torch.no_grad():
        for module in model.w_As:
            module.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0, 0.0]]))
    for module in model.w_As:
        module.weight.grad = torch.zeros_like(module.weight)
    model.w_As[0].weight.grad[0, 2] = -5e-13

    model.apply_adaptive_a_gradients(
        stability_gradients=[
            torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            torch.zeros_like(model.w_As[1].weight),
        ]
    )

    diagnostics = model.adaptive_a_diagnostics()
    assert diagnostics["conflicts"] == 0
    assert diagnostics["normal_projections"] == 0
    assert diagnostics["full_fallbacks"] == 0


def test_functional_stability_gradients_are_manually_ddp_averaged(monkeypatch):
    """Removing the manual reducer would leave rank-local teacher gradients."""
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.w_As = nn.ModuleList([nn.Linear(2, 1, bias=False)])

        def forward(self, inputs):
            return inputs

    class Network(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

    backbone = Backbone()
    network = Network(backbone)
    teacher = nn.Identity()
    learner = object.__new__(SharedALearner)
    learner._network = network
    learner._raw_network = lambda: network
    learner._hbd_teacher = teacher
    learner._hbd_teacher_captures = []
    learner._cur_task = 1

    monkeypatch.setattr(
        sa_sdlora_module,
        "register_live_a_historical_capture_hooks",
        lambda model, captures: [],
    )
    monkeypatch.setattr(
        sa_sdlora_module,
        "live_a_historical_outputs",
        lambda model, inputs, captures: [(inputs, inputs)],
    )
    monkeypatch.setattr(
        sa_sdlora_module,
        "hbd_historical_branch_distance",
        lambda student, teacher_outputs: 3.0 * backbone.w_As[0].weight.sum(),
    )
    monkeypatch.setattr(sa_sdlora_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sa_sdlora_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        sa_sdlora_module.dist,
        "all_reduce",
        lambda tensor, op: tensor.add_(5.0),
    )

    gradients = learner._functional_stability_gradients(torch.ones(1, 2))

    assert torch.equal(gradients[0], torch.full_like(gradients[0], 4.0))


def test_functional_halfspace_uses_teacher_without_ordinary_hbd_loss():
    """Teacher-gradient control must not quietly become a scalar HBD penalty."""
    learner = object.__new__(SharedALearner)
    learner._hbd_enabled = False
    learner._sa_adaptive_a_enabled = True
    learner._sa_adaptive_a_strategy = "functional_halfspace"
    learner._cur_task = 1
    learner._sa_operator_stability_lambda = 0.0
    learner._sa_prototype_consistency_weight = 0.0

    assert learner._additional_training_losses() == {}


def test_functional_halfspace_creates_and_releases_teacher_without_hbd(
    monkeypatch,
):
    """Functional control needs the existing teacher lifecycle even with HBD off."""
    events = []

    class Handle:
        def remove(self):
            events.append("released")

    class Backbone:
        cumulative_state = False
        cumulative_merge = "live_a_aggregate_b"

        def build_hbd_teacher(self):
            events.append("created")
            return object()

        def adaptive_a_diagnostics(self):
            return None

    class Network:
        def __init__(self):
            self.backbone = Backbone()

    learner = object.__new__(SharedALearner)
    learner._network = Network()
    learner._raw_network = lambda: learner._network
    learner.args = {"sa_use_prototype_classifier": False}
    learner._cur_task = 0
    learner._hbd_enabled = False
    learner._hbd_teacher = None
    learner._hbd_teacher_handles = []
    learner._sa_functional_halfspace_enabled = True
    learner._dual_head = False
    learner._lrpt_enabled = False
    learner._coordinate_stable_transport = False
    learner._is_main_process = lambda: False
    monkeypatch.setattr(
        sa_sdlora_module,
        "register_live_a_historical_capture_hooks",
        lambda teacher, captures: [Handle()],
    )
    monkeypatch.setattr(
        sa_sdlora_module.SDLoraLearner,
        "incremental_train",
        lambda self, data_manager: events.append("trained"),
    )

    learner.incremental_train(type("DataManager", (), {"nb_tasks": 2})())

    assert events == ["created", "trained", "released"]
    assert learner._hbd_teacher is None
    assert learner._hbd_teacher_handles == []


def test_functional_halfspace_logs_its_task_summary(caplog):
    """The generic gate log omits functional projection safety statistics."""
    class Backbone:
        def adaptive_a_diagnostics(self):
            return {
                "strategy": "functional_halfspace",
                "observations": 3,
                "conflicts": 2,
                "normal_projections": 1,
                "full_fallbacks": 1,
                "degenerate_noops": 1,
                "mean_pre_inner": -2.0,
                "mean_post_inner": 0.0,
                "mean_normal_fraction": 0.25,
                "mean_correction_ratio": 0.5,
            }

    learner = object.__new__(SharedALearner)
    learner._cur_task = 1
    learner._raw_network = lambda: type(
        "Network", (), {"backbone": Backbone()}
    )()

    caplog.set_level(sa_sdlora_module.logging.INFO)
    learner._log_adaptive_a_diagnostics()

    assert "FunctionalHalfspace-AdaptiveA" in caplog.text
    assert "normal_projections=1" in caplog.text


def test_function_safe_pareto_config_defaults_to_logit_kl_and_cosine_margin():
    """Rejecting the new strategy or silently using raw-dot noise is a bug."""
    settings = validate_adaptive_a_config(
        {
            "sa_adaptive_a_enabled": True,
            "sa_adaptive_a_strategy": "function_safe_pareto",
            "sa_train_a_all_tasks": True,
            "sa_cumulative_state": True,
            "sa_cumulative_merge": "live_a_aggregate_b",
            "sa_live_a_coordinate_align": True,
            "optimizer": "sgd",
        }
    )

    assert settings["adaptive_a_strategy"] == "function_safe_pareto"
    assert settings["functional_conflict_cosine"] == pytest.approx(0.05)
    assert settings["functional_temperature"] == pytest.approx(2.0)


def test_function_safe_pareto_rejects_dual_head_teacher_ambiguity():
    config = {
        "sa_adaptive_a_enabled": True,
        "sa_adaptive_a_strategy": "function_safe_pareto",
        "sa_train_a_all_tasks": True,
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
        "sa_dual_head": True,
        "optimizer": "sgd",
    }

    with pytest.raises(ValueError, match="function_safe_pareto.*sa_dual_head"):
        validate_adaptive_a_config(config)


def test_function_safe_pareto_requires_transport_for_non_equivalent_absorption():
    config = {
        "sa_adaptive_a_enabled": True,
        "sa_adaptive_a_strategy": "function_safe_pareto",
        "sa_train_a_all_tasks": True,
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
        "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
        "sa_use_prototype_classifier": True,
        "optimizer": "sgd",
    }

    with pytest.raises(ValueError, match="non-operator-preserving absorption"):
        validate_adaptive_a_config(config)

    config["sa_coordinate_stable_transport"] = True
    settings = validate_adaptive_a_config(config)
    assert settings["adaptive_a_strategy"] == "function_safe_pareto"


def test_functional_halfspace_cosine_margin_ignores_weak_negative_alignment():
    """A tiny normalized conflict must not trigger the safety projection."""
    proposed = [torch.tensor([[1.0, 0.0]])]
    stability = [torch.tensor([[-0.01, 1.0]])]
    shared_as = [torch.tensor([[1.0, 0.0]])]

    result = project_functional_halfspace_directions(
        proposed,
        stability,
        shared_as,
        conflict_cosine=0.05,
    )

    assert result["pre_inner"] < 0.0
    assert result["pre_cosine"] == pytest.approx(-0.0099995, rel=1e-4)
    assert result["mode"] == "identity"
    assert torch.equal(result["directions"][0], proposed[0])


def test_old_logits_kl_is_zero_at_teacher_and_backpropagates_only_to_student():
    """Using branch cosine or allowing teacher gradients would break the signal."""
    old_logits_kl = getattr(sa_sdlora_module, "old_logits_kl")
    student = torch.tensor(
        [[2.0, -1.0, 0.5], [0.0, 1.0, -2.0]], requires_grad=True
    )
    teacher = student.detach().clone().requires_grad_(True)

    equal_loss = old_logits_kl(student, teacher, temperature=2.0)
    assert equal_loss.item() == pytest.approx(0.0, abs=1e-7)

    shifted_student = (student + torch.tensor([[0.0, 1.0, 0.0]])).clone()
    loss = old_logits_kl(shifted_student, teacher, temperature=2.0)
    loss.backward()

    assert loss.item() > 0.0
    assert student.grad is not None
    assert torch.linalg.vector_norm(student.grad).item() > 0.0
    assert teacher.grad is None


def test_function_safe_pareto_projects_live_per_block_without_global_cancellation(
    tmp_path,
):
    """A safe second block must not hide a conflict in the first block."""
    model = _functional_model(
        tmp_path, strategy="function_safe_pareto", blocks=2
    )
    model.task_id = 1
    with torch.no_grad():
        for module in model.w_As:
            module.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
                )
            )
        for block in model.lora_vit.blocks:
            block.attn.qkv.aggregate_q.fill_(1.0)
            block.attn.qkv.aggregate_v.fill_(1.0)

    conflicting = torch.zeros_like(model.w_As[0].weight)
    conflicting[0, 2] = -1.0
    safe = torch.zeros_like(model.w_As[2].weight)
    safe[0, 2] = 2.0
    zero = torch.zeros_like(conflicting)
    gradients = [conflicting, zero.clone(), safe, zero.clone()]
    stability = [
        -conflicting,
        zero.clone(),
        safe.clone(),
        zero.clone(),
    ]
    for module, gradient in zip(model.w_As, gradients):
        module.weight.grad = gradient.clone()

    result = model.apply_adaptive_a_gradients(
        stability_gradients=stability,
        cached_modes=["live", "live"],
        cached_utilities=[1.0, 1.0],
    )

    assert result["selection"]["modes"] == ["live", "live"]
    assert torch.allclose(model.w_As[0].weight.grad, zero)
    assert torch.equal(model.w_As[2].weight.grad, safe)
    assert result["functional_safety"]["observations"] == 2
    assert result["functional_safety"]["conflicts"] == 1


def test_function_safe_pareto_refreshes_crossfit_and_stability_together():
    """Refreshing either cache alone would pair stale safety with new utility."""
    class ScheduledBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.w_As = nn.ModuleList([nn.Linear(2, 2, bias=False)])
            self.adaptive_a_strategy = "function_safe_pareto"
            self.calls = []

        def apply_adaptive_a_gradients(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "selection": {
                    "modes": ["live"],
                    "selected_utilities": [0.75],
                }
            }

    class ScheduledNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = ScheduledBackbone()

    learner = object.__new__(SharedALearner)
    learner._sa_adaptive_a_enabled = True
    learner._sa_adaptive_a_strategy = "function_safe_pareto"
    learner._sa_adaptive_a_crossfit_interval = 4
    learner._network = ScheduledNetwork()
    learner._raw_network = lambda: learner._network
    learner._known_classes = 2
    learner._cur_task = 1
    learner._sa_functional_diagnostic_pending = {"stale": True}
    learner._sa_functional_diagnostic_observations = [{"stale": 1.0}]
    crossfit_calls = []
    stability_calls = []

    def crossfit(inputs, targets):
        crossfit_calls.append(1)
        gradient = torch.ones_like(learner._network.backbone.w_As[0].weight)
        return ([gradient], [gradient])

    def stability(inputs):
        stability_calls.append(1)
        return [
            torch.full_like(
                learner._network.backbone.w_As[0].weight,
                float(len(stability_calls)),
            )
        ]

    learner._crossfit_adaptive_a_gradients = crossfit
    learner._functional_old_logit_gradients = stability
    inputs = torch.ones(4, 2)
    targets = torch.full((4,), 2, dtype=torch.long)
    optimizer = optim.SGD(learner._network.parameters(), lr=0.1)

    for _ in range(5):
        learner._after_backward(inputs, targets, optimizer)

    assert learner._sa_functional_diagnostic_pending is None
    assert learner._sa_functional_diagnostic_observations == []
    assert len(crossfit_calls) == 2
    assert len(stability_calls) == 2
    calls = learner._network.backbone.calls
    assert calls[0]["crossfit_gradients"] is not None
    assert calls[0]["stability_gradients"][0][0, 0].item() == 1.0
    for call in calls[1:4]:
        assert call["crossfit_gradients"] is None
        assert call["stability_gradients"][0][0, 0].item() == 1.0
    assert calls[4]["crossfit_gradients"] is not None
    assert calls[4]["stability_gradients"][0][0, 0].item() == 2.0


def test_function_safe_pareto_snapshots_and_releases_frozen_old_head(
    monkeypatch,
):
    """The function teacher must include an immutable old classifier snapshot."""
    events = []

    class Backbone:
        cumulative_state = False
        cumulative_merge = "live_a_aggregate_b"

        def build_hbd_teacher(self):
            return nn.Identity()

        def adaptive_a_diagnostics(self):
            return None

    class Network:
        def __init__(self):
            self.backbone = Backbone()
            self.prototype_head = nn.Linear(2, 3, bias=False)
            self.fc = nn.Linear(2, 3, bias=False)

    learner = object.__new__(SharedALearner)
    learner._network = Network()
    learner._raw_network = lambda: learner._network
    learner.args = {"sa_use_prototype_classifier": False}
    learner._cur_task = 0
    learner._hbd_enabled = False
    learner._hbd_teacher = None
    learner._hbd_teacher_handles = []
    learner._hbd_teacher_captures = []
    learner._functional_old_head = None
    learner._sa_functional_halfspace_enabled = False
    learner._sa_function_safe_pareto_enabled = True
    learner._dual_head = False
    learner._lrpt_enabled = False
    learner._coordinate_stable_transport = False
    learner._is_main_process = lambda: False

    source_head = learner._network.prototype_head

    def train(self, data_manager):
        assert learner._hbd_teacher is not None
        assert learner._functional_old_head is not source_head
        assert not any(
            parameter.requires_grad
            for parameter in learner._functional_old_head.parameters()
        )
        events.append("trained")

    monkeypatch.setattr(sa_sdlora_module.SDLoraLearner, "incremental_train", train)
    monkeypatch.setattr(
        sa_sdlora_module,
        "register_live_a_historical_capture_hooks",
        lambda *args, **kwargs: pytest.fail(
            "function-safe Pareto does not need historical branch hooks"
        ),
    )

    learner.incremental_train(type("DataManager", (), {"nb_tasks": 2})())

    assert events == ["trained"]
    assert learner._hbd_teacher is None
    assert learner._functional_old_head is None


def test_function_safe_pareto_old_logit_gradient_uses_heldout_half_only():
    """Even-index cross-fit samples must not leak into the stability gradient."""
    class Backbone(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.w_As = nn.ModuleList([nn.Linear(2, 2, bias=False)])
            with torch.no_grad():
                self.w_As[0].weight.copy_(weight)

        def forward(self, inputs):
            return self.w_As[0](inputs)

    class DictIdentity(nn.Module):
        def forward(self, features):
            return {"logits": features}

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone(torch.eye(2))

    learner = object.__new__(SharedALearner)
    learner._network = Network()
    learner._raw_network = lambda: learner._network
    learner._cur_task = 1
    learner._hbd_teacher = Backbone(torch.tensor([[1.0, 0.5], [0.0, 1.0]]))
    learner._functional_old_head = DictIdentity()
    learner._sa_functional_temperature = 2.0

    inputs = torch.tensor(
        [[1.0, 7.0], [2.0, 1.0], [-3.0, 5.0], [1.0, 3.0]]
    )
    changed_even_inputs = inputs.clone()
    changed_even_inputs[0] = torch.tensor([100.0, -200.0])
    changed_even_inputs[2] = torch.tensor([-300.0, 400.0])

    first = learner._functional_old_logit_gradients(inputs)
    second = learner._functional_old_logit_gradients(changed_even_inputs)

    assert len(first) == 1
    assert first[0] is not None
    assert torch.linalg.vector_norm(first[0]).item() > 0.0
    assert torch.allclose(first[0], second[0], atol=1e-7, rtol=1e-6)


def test_function_safe_pareto_old_logit_gradient_is_manually_ddp_averaged(
    monkeypatch,
):
    class Backbone(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.w_As = nn.ModuleList([nn.Linear(2, 2, bias=False)])
            with torch.no_grad():
                self.w_As[0].weight.copy_(weight)

        def forward(self, inputs):
            return self.w_As[0](inputs)

    class DictIdentity(nn.Module):
        def forward(self, features):
            return {"logits": features}

    learner = object.__new__(SharedALearner)
    learner._network = type("Network", (), {})()
    learner._network.backbone = Backbone(torch.eye(2))
    learner._raw_network = lambda: learner._network
    learner._cur_task = 1
    learner._hbd_teacher = Backbone(
        torch.tensor([[1.0, 0.5], [0.0, 1.0]])
    )
    learner._functional_old_head = DictIdentity()
    learner._sa_functional_temperature = 2.0
    inputs = torch.tensor([[1.0, 2.0], [2.0, 1.0]])

    monkeypatch.setattr(sa_sdlora_module.dist, "is_initialized", lambda: False)
    reference = learner._functional_old_logit_gradients(inputs)[0]
    collectives = []

    monkeypatch.setattr(sa_sdlora_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sa_sdlora_module.dist, "get_world_size", lambda: 2)

    def all_reduce(tensor, op):
        collectives.append(op)
        tensor.mul_(2.0)

    monkeypatch.setattr(sa_sdlora_module.dist, "all_reduce", all_reduce)
    averaged = learner._functional_old_logit_gradients(inputs)[0]

    assert len(collectives) == 1
    assert torch.allclose(averaged, reference)


@pytest.mark.parametrize("value", [-1, True, 1.5, "4"])
def test_functional_diagnostics_interval_rejects_invalid_values(value):
    """A malformed cadence must not silently alter the training schedule."""
    with pytest.raises(ValueError, match="diagnostics_interval"):
        validate_functional_diagnostics_interval(
            {"sa_functional_diagnostics_interval": value}
        )


def test_functional_diagnostics_interval_defaults_off():
    """Ordinary experiments must remain free of diagnostic forward passes."""
    assert validate_functional_diagnostics_interval({}) == 0


def test_functional_diagnostic_cadence_counts_optimizer_steps():
    """Cross-fit refresh cadence must not shift the requested sample steps."""
    assert not should_capture_functional_diagnostic(0, 0, 20)
    assert not should_capture_functional_diagnostic(1, 0, 0)
    assert should_capture_functional_diagnostic(1, 0, 20)
    assert not should_capture_functional_diagnostic(1, 4, 20)
    assert should_capture_functional_diagnostic(1, 20, 20)


def test_functional_step_diagnostic_attributes_b_only_kl_change_to_b_scale():
    """Changing B alone must not be misreported as an A-induced KL change."""
    class ScaledLinearBackbone(nn.Module):
        def __init__(self, a, b, scale):
            super().__init__()
            self.w_As = nn.ModuleList([nn.Linear(2, 2, bias=False)])
            self.w_Bs = nn.ModuleList([nn.Linear(2, 2, bias=False)])
            self.wrapped_param = nn.ModuleList(
                [sa_lora_module.ParameterWrapper(nn.Parameter(torch.tensor([scale])))]
            )
            with torch.no_grad():
                self.w_As[0].weight.copy_(a)
                self.w_Bs[0].weight.copy_(b)

        def forward(self, inputs):
            features = self.w_Bs[0](self.w_As[0](inputs))
            return self.wrapped_param[0](features)

    class DictIdentity(nn.Module):
        def forward(self, features):
            return {"logits": features}

    learner = object.__new__(SharedALearner)
    learner._network = type("Network", (), {})()
    learner._network.backbone = ScaledLinearBackbone(
        torch.eye(2), torch.eye(2), 1.0
    )
    learner._raw_network = lambda: learner._network
    learner._cur_task = 1
    learner._hbd_teacher = ScaledLinearBackbone(
        torch.eye(2), torch.eye(2), 1.0
    )
    learner._functional_old_head = DictIdentity()
    learner._sa_functional_temperature = 2.0
    learner._sa_functional_diagnostics_interval = 1
    learner._sa_pareto_step = 0
    learner._sa_functional_diagnostic_pending = None
    learner._sa_functional_diagnostic_observations = []
    inputs = torch.tensor(
        [[1.0, 0.0], [1.0, 2.0], [0.0, 1.0], [2.0, 1.0]]
    )

    stability_gradients = learner._functional_old_logit_gradients(inputs)
    learner._prepare_functional_step_diagnostic(inputs, stability_gradients)
    with torch.no_grad():
        learner._network.backbone.w_Bs[0].weight.add_(
            torch.tensor([[0.5, -0.25], [0.0, 0.25]])
        )
    tracked_parameters = [
        learner._network.backbone.w_As[0].weight,
        learner._network.backbone.w_Bs[0].weight,
        learner._network.backbone.wrapped_param[0].param,
    ]
    versions_before_diagnostic = [parameter._version for parameter in tracked_parameters]
    learner._after_optimizer_step(inputs=inputs)

    observation = learner._sa_functional_diagnostic_observations[-1]
    assert observation["full_delta_kl"] > 0.0
    assert observation["a_only_delta_kl"] == pytest.approx(0.0, abs=1e-8)
    assert observation["b_scale_only_delta_kl"] == pytest.approx(
        observation["full_delta_kl"], rel=1e-5, abs=1e-8
    )
    assert observation["interaction_delta_kl"] == pytest.approx(0.0, abs=1e-7)
    assert [parameter._version for parameter in tracked_parameters] == (
        versions_before_diagnostic
    )


def test_functional_diagnostic_preserves_real_backbone_parameter_aliases(tmp_path):
    """Replay must not replace Parameters shared by ViT and QKV wrappers."""
    class FirstTokenHead(nn.Module):
        def forward(self, features):
            return {"logits": features[:, 0, :2]}

    backbone = _functional_model(
        tmp_path, strategy="function_safe_pareto", blocks=1
    )
    backbone.task_id = 1
    learner = object.__new__(SharedALearner)
    learner._network = type("Network", (), {})()
    learner._network.backbone = backbone
    learner._raw_network = lambda: learner._network
    learner._cur_task = 1
    learner._hbd_teacher = copy.deepcopy(backbone).eval()
    learner._functional_old_head = FirstTokenHead()
    learner._sa_functional_temperature = 2.0
    learner._sa_functional_diagnostics_interval = 1
    learner._sa_pareto_step = 0
    learner._sa_functional_diagnostic_pending = None
    learner._sa_functional_diagnostic_observations = []
    inputs = torch.randn(4, 3, 4)
    def current_parameters():
        return [
            *[module.weight for module in backbone.w_As],
            *[module.weight for module in backbone.w_Bs],
            backbone.wrapped_param[0].param,
        ]

    parameters = current_parameters()
    optimizer = optim.SGD(parameters, lr=0.1)
    identities = [id(parameter) for parameter in parameters]
    versions = [parameter._version for parameter in parameters]
    gradients = [torch.ones_like(module.weight) for module in backbone.w_As]

    learner._prepare_functional_step_diagnostic(inputs, gradients)

    live_parameters = current_parameters()
    assert [id(parameter) for parameter in live_parameters] == identities
    assert [parameter._version for parameter in live_parameters] == versions
    assert all(isinstance(parameter, nn.Parameter) for parameter in live_parameters)
    assert all(parameter.requires_grad for parameter in live_parameters)
    assert all(
        any(candidate is parameter for candidate in optimizer.param_groups[0]["params"])
        for parameter in live_parameters
    )

    with torch.no_grad():
        backbone.w_Bs[0].weight.add_(0.01)
    versions_after_update = [parameter._version for parameter in current_parameters()]
    learner._after_optimizer_step(inputs=inputs, optimizer=optimizer)

    replayed_parameters = current_parameters()
    assert [id(parameter) for parameter in replayed_parameters] == identities
    assert [parameter._version for parameter in replayed_parameters] == (
        versions_after_update
    )
    assert all(isinstance(parameter, nn.Parameter) for parameter in replayed_parameters)
    assert all(parameter.requires_grad for parameter in replayed_parameters)
    assert learner._sa_functional_diagnostic_observations
