"""Tests for the pure global functional-halfspace projection kernel."""

from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import (  # noqa: E402
    decompose_adaptive_a_gradient,
    project_functional_halfspace_directions,
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


def test_functional_halfspace_preserves_proposed_dtype_and_device():
    """Mixed precision stability tensors must not promote the output direction."""
    proposed = [torch.tensor([[-2.0, -2.0, 0.0]], dtype=torch.float16)]
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
        projected, torch.tensor([[-2.0, 2.0, 0.0]], dtype=torch.float16)
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
