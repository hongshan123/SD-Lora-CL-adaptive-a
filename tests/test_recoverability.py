import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.recoverability import (
    accessibility_candidates,
    accessibility_energy,
    align_to_anchor,
    choose_global_recoverability_candidates,
    choose_local_recoverability_candidates,
    effective_gradient_cross,
    effective_gradient_energy,
    grassmann_accessibility_direction,
    operator_weighted_recoverability,
    recoverability_energies,
    row_polar_retraction,
)


def _dense_operator_risk(anchor_up, anchor_a, candidate_a, eps=1e-12):
    operator = anchor_up.double() @ anchor_a.double()
    projector = torch.linalg.pinv(candidate_a.double()) @ candidate_a.double()
    residual = operator @ (torch.eye(candidate_a.shape[1], dtype=torch.float64) - projector)
    total = operator.square().sum()
    return residual.square().sum() / (total + eps)


def test_low_rank_recoverability_matches_dense_projection_and_alignment():
    torch.manual_seed(4)
    anchor_up = torch.randn(7, 3, dtype=torch.float64)
    anchor_a = torch.randn(3, 9, dtype=torch.float64)
    candidate_a = torch.randn(3, 9, dtype=torch.float64)

    risk = operator_weighted_recoverability(
        anchor_up, anchor_a, candidate_a
    )
    dense_risk = _dense_operator_risk(anchor_up, anchor_a, candidate_a)
    aligned = align_to_anchor(anchor_up, anchor_a, candidate_a)
    explicit = anchor_up @ anchor_a
    residual = aligned @ candidate_a - explicit

    assert risk.item() == pytest.approx(dense_risk.item(), rel=1e-9, abs=1e-11)
    assert (
        residual.square().sum() / (explicit.square().sum() + 1e-12)
    ).item() == pytest.approx(risk.item(), rel=1e-9, abs=1e-11)


def test_recoverability_is_invariant_to_invertible_factor_coordinates():
    torch.manual_seed(7)
    anchor_up = torch.randn(6, 3, dtype=torch.float64)
    anchor_a = torch.randn(3, 8, dtype=torch.float64)
    candidate_a = torch.randn(3, 8, dtype=torch.float64)
    q_anchor = torch.randn(3, 3, dtype=torch.float64) + 2.0 * torch.eye(3)
    q_candidate = torch.randn(3, 3, dtype=torch.float64) + 2.0 * torch.eye(3)

    reference = operator_weighted_recoverability(
        anchor_up, anchor_a, candidate_a
    )
    transformed = operator_weighted_recoverability(
        anchor_up @ torch.linalg.inv(q_anchor),
        q_anchor @ anchor_a,
        q_candidate @ candidate_a,
    )

    assert transformed.item() == pytest.approx(reference.item(), rel=1e-8)


def test_operator_weighting_separates_equal_grassmann_rotations():
    theta = torch.tensor(math.pi / 6, dtype=torch.float64)
    c, s = torch.cos(theta), torch.sin(theta)
    anchor_a = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    rotate_high_energy = torch.stack(
        [torch.tensor([c, 0.0, s, 0.0]), anchor_a[1]]
    )
    rotate_low_energy = torch.stack(
        [anchor_a[0], torch.tensor([0.0, c, 0.0, s])]
    )
    anchor_up = torch.diag(torch.tensor([10.0, 1.0], dtype=torch.float64))

    projector = anchor_a.t() @ anchor_a
    distance_high = torch.linalg.vector_norm(
        rotate_high_energy.t() @ rotate_high_energy - projector
    )
    distance_low = torch.linalg.vector_norm(
        rotate_low_energy.t() @ rotate_low_energy - projector
    )
    high = operator_weighted_recoverability(
        anchor_up, anchor_a, rotate_high_energy
    )
    low = operator_weighted_recoverability(
        anchor_up, anchor_a, rotate_low_energy
    )

    assert distance_high.item() == pytest.approx(distance_low.item(), rel=1e-12)
    assert high > 50.0 * low


def test_recoverability_handles_zero_history_and_rank_deficient_candidate():
    anchor_a = torch.eye(2, 4, dtype=torch.float64)
    zero_up = torch.zeros(5, 2, dtype=torch.float64)
    rank_deficient = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    risk = operator_weighted_recoverability(
        zero_up, anchor_a, rank_deficient
    )
    total, captured, residual = recoverability_energies(
        zero_up, anchor_a, rank_deficient
    )
    aligned = align_to_anchor(zero_up, anchor_a, rank_deficient)

    assert risk.item() == 0.0
    assert total.item() == captured.item() == residual.item() == 0.0
    assert torch.isfinite(aligned).all()


def test_polar_retraction_produces_row_orthonormal_basis():
    torch.manual_seed(13)
    candidate = torch.randn(4, 11, dtype=torch.float64)
    basis = row_polar_retraction(candidate)

    assert torch.allclose(
        basis @ basis.t(), torch.eye(4, dtype=torch.float64), atol=1e-10
    )


def test_factorized_effective_gradient_statistics_match_dense_gradient():
    torch.manual_seed(17)
    inputs = torch.randn(12, 7, dtype=torch.float64)
    output_grad = torch.randn(12, 5, dtype=torch.float64)
    basis = row_polar_retraction(torch.randn(3, 7, dtype=torch.float64))
    dense_h = output_grad.t() @ inputs

    cross = effective_gradient_cross(inputs, output_grad, basis)
    energy = effective_gradient_energy(inputs, output_grad)
    accessible = accessibility_energy(cross, basis)

    assert torch.allclose(cross, dense_h @ basis.t(), atol=1e-11)
    assert energy.item() == pytest.approx(dense_h.square().sum().item(), rel=1e-10)
    assert accessible.item() == pytest.approx(
        (dense_h @ (basis.t() @ basis)).square().sum().item(), rel=1e-10
    )


def test_grassmann_accessibility_direction_is_horizontal_and_ascending():
    torch.manual_seed(23)
    inputs = torch.randn(16, 8, dtype=torch.float64)
    output_grad = torch.randn(16, 6, dtype=torch.float64)
    basis = row_polar_retraction(torch.randn(3, 8, dtype=torch.float64))
    direction = grassmann_accessibility_direction(inputs, output_grad, basis)
    cross = effective_gradient_cross(inputs, output_grad, basis)
    before = accessibility_energy(cross, basis)
    step = 1e-7
    candidate = row_polar_retraction(basis + step * direction)
    after_cross = effective_gradient_cross(inputs, output_grad, candidate)
    after = accessibility_energy(after_cross, candidate)

    assert torch.allclose(
        direction @ basis.t(), torch.zeros(3, 3, dtype=torch.float64), atol=1e-10
    )
    assert after > before
    finite_difference = (after - before) / step
    expected = 2.0 * direction.square().sum()
    assert finite_difference.item() == pytest.approx(
        expected.item(), rel=2e-4, abs=1e-7
    )


def test_accessibility_candidates_include_unchanged_and_full_steps():
    torch.manual_seed(29)
    inputs = torch.randn(10, 6, dtype=torch.float64)
    output_grad = torch.randn(10, 5, dtype=torch.float64)
    basis = row_polar_retraction(torch.randn(2, 6, dtype=torch.float64))
    direction = grassmann_accessibility_direction(inputs, output_grad, basis)
    candidates = accessibility_candidates(
        basis,
        direction,
        gradient_energy=effective_gradient_energy(inputs, output_grad),
        gammas=(0.0, 0.5, 1.0),
        step_size=0.2,
    )

    assert [item[0] for item in candidates] == [0.0, 0.5, 1.0]
    assert torch.equal(candidates[0][1], basis)
    for _, candidate in candidates:
        assert torch.allclose(
            candidate @ candidate.t(), torch.eye(2, dtype=torch.float64), atol=1e-10
        )


def test_local_and_global_selectors_obey_energy_budget():
    layers = [
        [
            {"gamma": 0.0, "utility": 0.0, "residual": 0.0, "history": 90.0},
            {"gamma": 1.0, "utility": 3.0, "residual": 9.0, "history": 90.0},
        ],
        [
            {"gamma": 0.0, "utility": 0.0, "residual": 0.0, "history": 10.0},
            {"gamma": 1.0, "utility": 2.0, "residual": 1.0, "history": 10.0},
        ],
    ]

    local = choose_local_recoverability_candidates(layers, risk_budget=0.05)
    global_choice = choose_global_recoverability_candidates(
        layers, risk_budget=0.05
    )

    assert local["indices"] == [0, 0]
    assert global_choice["indices"] == [0, 1]
    assert global_choice["risk"] == pytest.approx(0.01)


def test_global_selector_with_zero_history_chooses_maximum_utility():
    layers = [
        [
            {"gamma": 0.0, "utility": 0.0, "residual": 0.0, "history": 0.0},
            {"gamma": 1.0, "utility": 1.0, "residual": 0.0, "history": 0.0},
        ],
        [
            {"gamma": 0.0, "utility": 0.0, "residual": 0.0, "history": 0.0},
            {"gamma": 1.0, "utility": 2.0, "residual": 0.0, "history": 0.0},
        ],
    ]

    selected = choose_global_recoverability_candidates(layers, risk_budget=0.0)

    assert selected["indices"] == [1, 1]
    assert selected["risk"] == 0.0
