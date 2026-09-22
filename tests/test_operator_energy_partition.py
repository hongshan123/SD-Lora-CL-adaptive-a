"""Unit tests for Historical-Operator Energy Partitioned A (HOEP-A)."""

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.operator_energy_partition import (
    canonicalize_operator_coordinates,
    diagonal_functional_energies,
    historical_energy_coordinates,
    mask_jaccard,
    project_partition_gradient,
    retract_partitioned_coordinates,
    rotate_operator_coordinates,
    select_global_low_energy_partition,
    spearman_rank_correlation,
)
from backbone.sa_lora import SA_STATE_FILENAME, SharedALoRA_ViT_timm
from models.sa_sdlora import validate_adaptive_a_config


def _orthonormal_rows(rank=4, dimension=9):
    q, _ = torch.linalg.qr(torch.randn(dimension, rank), mode="reduced")
    return q.t()


class _TinyAttention(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.qkv = nn.Linear(dimension, 3 * dimension, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.attn = _TinyAttention(dimension)

    def forward(self, inputs):
        return self.attn.qkv(inputs)


class _TinyViT(nn.Module):
    def __init__(self, dimension=4):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dimension)])
        self.head = nn.Identity()

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs


def test_historical_energy_spectrum_matches_effective_operator_energy():
    torch.manual_seed(3)
    shared_a = _orthonormal_rows()
    aggregate = torch.randn(7, 4)

    eigenvalues, eigenvectors = historical_energy_coordinates(
        aggregate, shared_a
    )

    effective = aggregate @ shared_a / torch.linalg.vector_norm(shared_a)
    assert torch.all(eigenvalues[:-1] <= eigenvalues[1:])
    assert eigenvalues.sum().item() == pytest.approx(
        effective.square().sum().item(), rel=1e-6, abs=1e-8
    )
    assert torch.allclose(
        eigenvectors.t() @ eigenvectors,
        torch.eye(4, dtype=eigenvectors.dtype),
        atol=1e-10,
        rtol=1e-10,
    )


def test_svd_canonicalization_preserves_both_operators():
    torch.manual_seed(4)
    shared_a = torch.randn(4, 9)
    aggregate = torch.randn(7, 4)
    current = torch.randn(7, 4)
    historical_before = (
        aggregate @ shared_a / torch.linalg.vector_norm(shared_a)
    )
    current_before = current @ shared_a

    canonical_a, canonical_g, canonical_b = canonicalize_operator_coordinates(
        shared_a, aggregate, current
    )

    assert torch.allclose(
        canonical_a @ canonical_a.t(),
        torch.eye(4),
        atol=2e-6,
        rtol=0,
    )
    assert torch.allclose(
        canonical_g @ canonical_a / torch.linalg.vector_norm(canonical_a),
        historical_before,
        atol=3e-6,
        rtol=3e-6,
    )
    assert torch.allclose(
        canonical_b @ canonical_a,
        current_before,
        atol=3e-6,
        rtol=3e-6,
    )


def test_spectral_rotation_preserves_historical_and_current_operators():
    torch.manual_seed(5)
    shared_a = _orthonormal_rows()
    aggregate = torch.randn(9, 4)
    current = torch.randn(9, 4)
    _, eigenvectors = historical_energy_coordinates(aggregate, shared_a)

    rotated_a, rotated_g, rotated_b = rotate_operator_coordinates(
        shared_a, aggregate, current, eigenvectors
    )

    assert torch.allclose(
        aggregate @ shared_a / torch.linalg.vector_norm(shared_a),
        rotated_g @ rotated_a / torch.linalg.vector_norm(rotated_a),
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        current @ shared_a,
        rotated_b @ rotated_a,
        atol=2e-6,
        rtol=2e-6,
    )


def test_global_partition_respects_budget_and_keeps_ties_indivisible():
    spectra = [
        torch.tensor([0.1, 0.1, 2.0]),
        torch.tensor([0.05, 0.4, 3.0]),
    ]

    partition = select_global_low_energy_partition(
        spectra, energy_budget=0.05, tie_relative_tolerance=1e-8
    )

    assert partition.selected_energy <= partition.budget_energy + 1e-12
    assert bool(partition.plastic_masks[0][0]) == bool(
        partition.plastic_masks[0][1]
    )
    assert partition.plastic_masks[1].tolist() == [True, False, False]
    assert partition.selected_directions == 3


def test_global_partition_is_a_low_energy_prefix_not_a_knapsack():
    spectra = [torch.tensor([0.06, 0.06, 0.11])]

    partition = select_global_low_energy_partition(
        spectra, energy_budget=0.5, tie_relative_tolerance=1e-8
    )

    assert partition.budget_energy == pytest.approx(0.115)
    assert partition.plastic_masks[0].tolist() == [False, False, False]
    assert partition.selected_energy == 0.0


def test_diagonal_functional_energy_matches_direct_trace():
    torch.manual_seed(41)
    shared_a = _orthonormal_rows(rank=4, dimension=9).double()
    coefficient = torch.randn(7, 4, dtype=torch.float64)
    operator, rotation = historical_energy_coordinates(coefficient, shared_a)
    shared_a, coefficient, _ = rotate_operator_coordinates(
        shared_a, coefficient, torch.zeros_like(coefficient), rotation
    )
    second_moment = torch.rand(9, dtype=torch.float64) + 0.1

    factors, functional = diagonal_functional_energies(
        shared_a, operator, second_moment
    )
    effective = coefficient @ shared_a / torch.linalg.vector_norm(shared_a)
    direct = torch.trace(
        effective @ torch.diag(second_moment) @ effective.t()
    )

    assert torch.all(factors >= 0)
    assert functional.sum().item() == pytest.approx(
        direct.item(), rel=1e-9, abs=1e-10
    )


def test_unit_second_moment_recovers_operator_order_and_partition():
    shared_a = _orthonormal_rows(rank=3, dimension=7).double()
    operator = torch.tensor([0.05, 0.2, 2.0], dtype=torch.float64)
    factors, functional = diagonal_functional_energies(
        shared_a, operator, torch.ones(7, dtype=torch.float64)
    )
    operator_partition = select_global_low_energy_partition([operator], 0.1)
    functional_partition = select_global_low_energy_partition(
        [functional], 0.1, tie_spectra=[operator]
    )

    assert torch.allclose(factors, torch.ones_like(factors), atol=1e-10)
    assert torch.allclose(functional, operator, atol=1e-10)
    assert torch.equal(
        operator_partition.plastic_masks[0],
        functional_partition.plastic_masks[0],
    )


def test_activation_anisotropy_reverses_direction_order():
    shared_a = torch.eye(2, dtype=torch.float64)
    operator = torch.tensor([10.0, 1.0], dtype=torch.float64)
    factors, functional = diagonal_functional_energies(
        shared_a,
        operator,
        torch.tensor([0.001, 100.0], dtype=torch.float64),
    )

    assert factors.tolist() == pytest.approx([0.001, 100.0])
    assert functional.tolist() == pytest.approx([0.01, 100.0])
    assert int(torch.argmin(operator)) == 1
    assert int(torch.argmin(functional)) == 0


def test_functional_partition_keeps_operator_ties_indivisible():
    operator = torch.tensor([0.2, 0.2, 4.0], dtype=torch.float64)
    first = torch.tensor([0.01, 0.09, 4.0], dtype=torch.float64)
    second = torch.tensor([0.05, 0.05, 4.0], dtype=torch.float64)
    partition_a = select_global_low_energy_partition(
        [first], 0.03, tie_spectra=[operator]
    )
    partition_b = select_global_low_energy_partition(
        [second], 0.03, tie_spectra=[operator]
    )

    assert partition_a.plastic_masks[0].tolist() == [True, True, False]
    assert torch.equal(
        partition_a.plastic_masks[0], partition_b.plastic_masks[0]
    )
    assert partition_a.selected_energy == pytest.approx(0.1)
    assert partition_b.selected_energy == pytest.approx(0.1)


def test_functional_partition_helpers_are_finite_and_deterministic():
    first = [torch.tensor([True, False]), torch.tensor([False, True])]
    second = [torch.tensor([True, True]), torch.tensor([False, False])]
    assert mask_jaccard(first, second) == pytest.approx(1.0 / 3.0)
    assert spearman_rank_correlation(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([3.0, 2.0, 1.0]),
    ) == pytest.approx(-1.0)

    shared_a = torch.eye(2)
    factors, functional = diagonal_functional_energies(
        shared_a, torch.zeros(2), torch.zeros(2)
    )
    assert torch.isfinite(factors).all()
    assert torch.isfinite(functional).all()


def test_partition_gradient_freezes_stable_rows_and_is_tangent():
    torch.manual_seed(7)
    shared_a = _orthonormal_rows()
    gradient = torch.randn_like(shared_a)
    plastic_mask = torch.tensor([False, True, False, True])

    projected = project_partition_gradient(
        gradient, shared_a, plastic_mask
    )

    assert torch.count_nonzero(projected[~plastic_mask]) == 0
    stable = shared_a[~plastic_mask]
    plastic = shared_a[plastic_mask]
    plastic_gradient = projected[plastic_mask]
    assert torch.allclose(
        plastic_gradient @ stable.t(),
        torch.zeros(2, 2),
        atol=2e-6,
        rtol=0,
    )
    tangent_constraint = (
        plastic_gradient @ plastic.t()
        + plastic @ plastic_gradient.t()
    )
    assert torch.allclose(
        tangent_constraint,
        torch.zeros_like(tangent_constraint),
        atol=2e-6,
        rtol=0,
    )


def test_selected_energy_bounds_alignment_residual():
    torch.manual_seed(9)
    old_a = _orthonormal_rows(rank=4, dimension=12).double()
    coefficient = torch.randn(7, 4, dtype=torch.float64)
    eigenvalues, eigenvectors = historical_energy_coordinates(
        coefficient * 2.0, old_a
    )
    old_a, rotated_g, _ = rotate_operator_coordinates(
        old_a,
        coefficient * 2.0,
        torch.zeros(7, 4, dtype=torch.float64),
        eigenvectors,
    )
    partition = select_global_low_energy_partition(
        [eigenvalues], energy_budget=0.25
    )
    plastic = partition.plastic_masks[0]
    stable = old_a[~plastic]
    random_rows = torch.randn(int(plastic.sum()), old_a.shape[1]).double()
    if stable.numel():
        random_rows -= (random_rows @ stable.t()) @ stable
    if random_rows.numel():
        random_rows = torch.linalg.qr(random_rows.t(), mode="reduced")[0].t()
    new_a = old_a.clone()
    new_a[plastic] = random_rows
    historical = rotated_g @ old_a / torch.linalg.vector_norm(old_a)
    aligned = historical @ new_a.t() @ new_a
    residual = torch.linalg.matrix_norm(historical - aligned).square()

    assert residual.item() <= partition.selected_energy + 1e-9


def test_degenerate_energy_cluster_selection_is_rotation_invariant():
    torch.manual_seed(10)
    old_a = _orthonormal_rows(rank=3, dimension=7).double()
    coefficient = torch.diag(torch.tensor([0.2, 0.2, 1.0])).double()
    angle = torch.tensor(0.73, dtype=torch.float64)
    rotation = torch.eye(3, dtype=torch.float64)
    rotation[:2, :2] = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)],
        ]
    )
    spectra = []
    for current_g in (coefficient, coefficient @ rotation):
        values, _ = historical_energy_coordinates(current_g, old_a)
        spectra.append(values)
    first = select_global_low_energy_partition([spectra[0]], 0.08, 1e-8)
    second = select_global_low_energy_partition([spectra[1]], 0.08, 1e-8)

    assert torch.equal(first.plastic_masks[0], second.plastic_masks[0])
    assert first.plastic_masks[0].tolist() == [True, True, False]
    assert first.selected_energy == pytest.approx(second.selected_energy)


def test_retraction_preserves_operators_and_stable_rows():
    torch.manual_seed(11)
    anchor = _orthonormal_rows(rank=4, dimension=10)
    plastic_mask = torch.tensor([False, True, False, True])
    candidate = anchor.clone()
    candidate[~plastic_mask] *= torch.tensor([[0.999], [1.001]])
    candidate[plastic_mask] += 1e-3 * torch.randn_like(candidate[plastic_mask])
    aggregate = torch.randn(8, 4)
    current = torch.randn(8, 4)
    historical_before = (
        aggregate @ candidate / torch.linalg.vector_norm(candidate)
    )
    current_before = current @ candidate

    new_a, new_g, new_b, _, diagnostics = retract_partitioned_coordinates(
        candidate,
        anchor,
        aggregate,
        current,
        plastic_mask,
    )

    assert torch.equal(new_a[~plastic_mask], anchor[~plastic_mask])
    assert torch.allclose(
        new_a @ new_a.t(), torch.eye(4), atol=2e-6, rtol=0
    )
    assert torch.allclose(
        new_g @ new_a / torch.linalg.vector_norm(new_a),
        historical_before,
        atol=3e-6,
        rtol=3e-6,
    )
    assert torch.allclose(
        new_b @ new_a, current_before, atol=3e-6, rtol=3e-6
    )
    assert diagnostics["historical_operator_error"] < 1e-6
    assert diagnostics["current_operator_error"] < 1e-6


def _valid_config(**overrides):
    config = {
        "sa_adaptive_a_enabled": True,
        "sa_adaptive_a_strategy": "operator_energy_partition",
        "sa_train_a_all_tasks": True,
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
        "sa_live_a_absorb_mode": "operator_preserving_absorb",
        "optimizer": "sgd",
    }
    config.update(overrides)
    return config


def test_hoep_config_defaults_and_constraints():
    settings = validate_adaptive_a_config(_valid_config())
    assert settings["hoep_energy_budget"] == pytest.approx(0.05)
    assert settings["hoep_eigenvalue_rtol"] == pytest.approx(1e-6)
    assert settings["hoep_energy_metric"] == "operator"
    assert settings["hoep_functional_diagnostics"] is False

    with pytest.raises(ValueError, match="operator_preserving_absorb"):
        validate_adaptive_a_config(
            _valid_config(sa_live_a_absorb_mode="normalized_absorb")
        )
    with pytest.raises(ValueError, match="sa_hoep_energy_budget"):
        validate_adaptive_a_config(_valid_config(sa_hoep_energy_budget=1.1))
    with pytest.raises(ValueError, match="sa_hoep_eigenvalue_rtol"):
        validate_adaptive_a_config(
            _valid_config(sa_hoep_eigenvalue_rtol=float("nan"))
        )
    with pytest.raises(ValueError, match="sa_hoep_energy_metric"):
        validate_adaptive_a_config(
            _valid_config(sa_hoep_energy_metric="dense")
        )


def test_hoep_eval_calibration_persists_exact_second_moment(tmp_path):
    settings = {
        "train_a_all_tasks": True,
        "cumulative_state": True,
        "cumulative_merge": "live_a_aggregate_b",
        "live_a_coordinate_align": True,
        "live_a_absorb_mode": "operator_preserving_absorb",
        "adaptive_a_enabled": True,
        "adaptive_a_strategy": "operator_energy_partition",
        "hoep_functional_diagnostics": True,
    }
    task_zero = SharedALoRA_ViT_timm(
        _TinyViT(), r=2, filepath=str(tmp_path), cur_task_index=0, **settings
    )
    inputs = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            [[3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]],
        ]
    )
    expected_second_moment = inputs.square().reshape(-1, 4).mean(dim=0)
    before_parameters = {
        name: value.detach().clone()
        for name, value in task_zero.named_parameters()
    }

    task_zero.eval()
    task_zero.prepare_hoep_activation_calibration()
    with torch.no_grad():
        task_zero(inputs)
    summaries = task_zero.finalize_hoep_activation_calibration()

    assert summaries[0]["token_count"] == pytest.approx(4.0)
    wrapper = task_zero._recoverability_wrappers()[0]
    rms, count = wrapper.merged_input_sketch()
    assert count == pytest.approx(4.0)
    assert torch.allclose(rms.square(), expected_second_moment)
    for name, value in task_zero.named_parameters():
        assert torch.equal(value, before_parameters[name])

    with torch.no_grad():
        for module in task_zero.w_Bs:
            module.weight.normal_()
    task_zero.save_lora_parameters(str(tmp_path), 0)
    state = torch.load(
        tmp_path / SA_STATE_FILENAME, map_location="cpu", weights_only=True
    )
    assert len(state["adaptive_a_input_rms"]) == 1
    assert state["adaptive_a_input_counts"].tolist() == pytest.approx([4.0])
    assert torch.allclose(
        state["adaptive_a_input_rms"][0].square(), expected_second_moment
    )

    task_one = SharedALoRA_ViT_timm(
        _TinyViT(), r=2, filepath=str(tmp_path), cur_task_index=1, **settings
    )
    record = task_one.hoep_functional_diagnostic_record()
    assert record["task_id"] == 1
    assert len(record["branches"]) == 2
    assert record["global"]["finite"] is True
    assert 0.0 <= record["global"]["jaccard"] <= 1.0


def test_functional_hoep_refuses_missing_activation_statistics(tmp_path):
    base_settings = {
        "train_a_all_tasks": True,
        "cumulative_state": True,
        "cumulative_merge": "live_a_aggregate_b",
        "live_a_coordinate_align": True,
        "live_a_absorb_mode": "operator_preserving_absorb",
        "adaptive_a_enabled": True,
        "adaptive_a_strategy": "operator_energy_partition",
    }
    task_zero = SharedALoRA_ViT_timm(
        _TinyViT(), r=2, filepath=str(tmp_path), cur_task_index=0,
        **base_settings,
    )
    task_zero.save_lora_parameters(str(tmp_path), 0)

    with pytest.raises(RuntimeError, match="activation statistics"):
        SharedALoRA_ViT_timm(
            _TinyViT(),
            r=2,
            filepath=str(tmp_path),
            cur_task_index=1,
            hoep_energy_metric="functional_diag",
            **base_settings,
        )


def test_hoep_task_boundary_and_optimizer_step_keep_fixed_state(tmp_path):
    settings = {
        "train_a_all_tasks": True,
        "cumulative_state": True,
        "cumulative_merge": "live_a_aggregate_b",
        "live_a_coordinate_align": True,
        "live_a_absorb_mode": "operator_preserving_absorb",
        "adaptive_a_enabled": True,
        "adaptive_a_strategy": "operator_energy_partition",
        "hoep_energy_budget": 0.5,
    }
    task_zero = SharedALoRA_ViT_timm(
        _TinyViT(),
        r=2,
        filepath=str(tmp_path),
        cur_task_index=0,
        **settings,
    )
    with torch.no_grad():
        for module in task_zero.w_As:
            module.weight.add_(0.03 * torch.randn_like(module.weight))
        for module in task_zero.w_Bs:
            module.weight.normal_()
    task_zero.save_lora_parameters(str(tmp_path), 0)
    task_zero_state = torch.load(
        tmp_path / SA_STATE_FILENAME, map_location="cpu", weights_only=True
    )
    task_zero_numel = sum(
        tensor.numel()
        for value in task_zero_state.values()
        for tensor in (value if isinstance(value, list) else [])
        if isinstance(tensor, torch.Tensor)
    )

    task_one = SharedALoRA_ViT_timm(
        _TinyViT(),
        r=2,
        filepath=str(tmp_path),
        cur_task_index=1,
        **settings,
    )
    for module in task_one.w_As:
        assert torch.allclose(
            module.weight @ module.weight.t(),
            torch.eye(2),
            atol=2e-6,
            rtol=0,
        )
    optimizer = torch.optim.SGD(
        [module.weight for module in task_one.w_As]
        + [module.weight for module in task_one.w_Bs],
        lr=0.01,
        momentum=0.9,
        weight_decay=0.01,
    )
    loss = task_one(torch.randn(3, 2, 4)).square().mean()
    loss.backward()
    task_one.apply_operator_energy_partition_gradients(optimizer)
    optimizer.step()
    step = task_one.apply_operator_energy_partition_step(optimizer)

    assert step["max_historical_operator_error"] < 1e-6
    assert step["max_current_operator_error"] < 1e-6
    task_one.save_lora_parameters(str(tmp_path), 1)
    boundary = task_one._last_live_a_coordinate_diagnostics
    assert boundary["predicted_energy_budget"] == pytest.approx(0.5)
    assert boundary["global_squared_residual_ratio"] <= 0.5 + 1e-6
    state = torch.load(
        tmp_path / SA_STATE_FILENAME, map_location="cpu", weights_only=True
    )
    assert "hoep" not in " ".join(state.keys()).lower()
    assert set(state) == {
        "version",
        "task_id",
        "rank",
        "merge_mode",
        "shared_a",
        "aggregate_up",
        "coordinate_aligned",
        "absorb_mode",
        "history_groups",
    }
    task_one_numel = sum(
        tensor.numel()
        for value in state.values()
        for tensor in (value if isinstance(value, list) else [])
        if isinstance(tensor, torch.Tensor)
    )
    assert task_one_numel == task_zero_numel

    torch.manual_seed(123)
    base = _TinyViT()
    base_qkv = base.blocks[0].attn.qkv
    rebuilt = SharedALoRA_ViT_timm(
        base,
        r=2,
        filepath=str(tmp_path),
        cur_task_index=2,
        **settings,
    )
    inputs = torch.randn(3, 2, 4)
    with torch.no_grad():
        base_output = base_qkv(inputs)
        q_update = F.linear(
            F.linear(inputs, state["shared_a"][0]),
            state["aggregate_up"][0]
            / torch.linalg.vector_norm(state["shared_a"][0]),
        )
        v_update = F.linear(
            F.linear(inputs, state["shared_a"][1]),
            state["aggregate_up"][1]
            / torch.linalg.vector_norm(state["shared_a"][1]),
        )
        expected = base_output.clone()
        expected[..., :4] += q_update
        expected[..., -4:] += v_update
        rebuilt_output = rebuilt(inputs)
    assert torch.allclose(rebuilt_output, expected, atol=2e-6, rtol=2e-6)


def test_hoep_cancels_sgd_weight_decay_on_stable_a(tmp_path):
    settings = {
        "train_a_all_tasks": True,
        "cumulative_state": True,
        "cumulative_merge": "live_a_aggregate_b",
        "live_a_coordinate_align": True,
        "live_a_absorb_mode": "operator_preserving_absorb",
        "adaptive_a_enabled": True,
        "adaptive_a_strategy": "operator_energy_partition",
        "hoep_energy_budget": 0.0,
    }
    task_zero = SharedALoRA_ViT_timm(
        _TinyViT(), r=2, filepath=str(tmp_path), cur_task_index=0, **settings
    )
    with torch.no_grad():
        for module in task_zero.w_Bs:
            module.weight.normal_()
    task_zero.save_lora_parameters(str(tmp_path), 0)
    task_one = SharedALoRA_ViT_timm(
        _TinyViT(), r=2, filepath=str(tmp_path), cur_task_index=1, **settings
    )
    assert all(not bool(mask.any()) for mask in task_one._hoep_plastic_masks)
    before_a = [module.weight.detach().clone() for module in task_one.w_As]
    optimizer = torch.optim.SGD(
        [module.weight for module in task_one.w_As],
        lr=0.1,
        momentum=0.0,
        weight_decay=0.2,
    )
    for module in task_one.w_As:
        module.weight.grad = torch.randn_like(module.weight)

    task_one.apply_operator_energy_partition_gradients(optimizer)
    optimizer.step()
    task_one.apply_operator_energy_partition_step(optimizer)

    for before, module in zip(before_a, task_one.w_As):
        assert torch.equal(before, module.weight)
