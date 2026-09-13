"""Tests for the pure global functional-halfspace projection kernel."""

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch import optim


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import (  # noqa: E402
    SharedALoRA_ViT_timm,
    decompose_adaptive_a_gradient,
    project_functional_halfspace_directions,
)
from models import sa_sdlora as sa_sdlora_module  # noqa: E402
from models.sa_sdlora import Learner as SharedALearner  # noqa: E402
from models.sa_sdlora import validate_adaptive_a_config  # noqa: E402


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


def _functional_model(tmp_path):
    return SharedALoRA_ViT_timm(
        _TinyViT(4),
        r=2,
        filepath=str(tmp_path / "run"),
        cur_task_index=0,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        live_a_coordinate_align=True,
        adaptive_a_enabled=True,
        adaptive_a_strategy="functional_halfspace",
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
