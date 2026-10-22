"""Operator-energy partitioning for fixed-state continual LoRA.

The helpers in this module are deliberately independent of the ViT wrapper.
They operate on one LoRA branch with down projection ``A``, historical up
projection ``G``, and current up projection ``B``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class EnergyPartition:
    """Global low-energy allocation over a list of Q/V branch spectra."""

    plastic_masks: tuple[Tensor, ...]
    total_energy: float
    selected_energy: float
    budget_energy: float
    selected_directions: int
    total_directions: int
    tie_groups: int


def canonicalize_operator_coordinates(
    shared_a: Tensor,
    aggregate_up: Tensor,
    current_up: Tensor,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return an exact row-orthonormal SVD gauge for one LoRA branch."""

    if shared_a.ndim != 2 or aggregate_up.ndim != 2 or current_up.ndim != 2:
        raise ValueError("A, G, and B must be matrices")
    if aggregate_up.shape[1] != shared_a.shape[0]:
        raise ValueError("aggregate_up/shared_a rank mismatch")
    if current_up.shape[1] != shared_a.shape[0]:
        raise ValueError("current_up/shared_a rank mismatch")
    if eps <= 0:
        raise ValueError("eps must be positive")
    work = shared_a.detach().to(dtype=torch.float64)
    u, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
    if singular_values.numel() != shared_a.shape[0]:
        raise RuntimeError("thin SVD did not return a full row basis")
    threshold = eps * max(1.0, float(singular_values.max()))
    if bool((singular_values <= threshold).any()):
        raise RuntimeError("shared A is rank deficient during canonicalization")
    factor = u * singular_values.unsqueeze(0)
    old_norm = torch.linalg.vector_norm(work).clamp_min(eps)
    new_norm = torch.linalg.vector_norm(vh).clamp_min(eps)
    canonical_g = (
        aggregate_up.detach().to(dtype=torch.float64)
        @ factor
        * (new_norm / old_norm)
    )
    canonical_b = current_up.detach().to(dtype=torch.float64) @ factor
    return (
        vh.to(shared_a),
        canonical_g.to(aggregate_up),
        canonical_b.to(current_up),
    )


def historical_energy_coordinates(
    aggregate_up: Tensor,
    shared_a: Tensor,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return ascending historical energies and their coordinate rotation.

    For the deployed historical operator ``M = G A / ||A||_F``, the energy
    carried by a coefficient-space direction is governed by
    ``C.T @ C``, where ``C = G / ||A||_F``.  ``eigenvectors`` contains the
    corresponding orthonormal directions as columns.
    """

    if aggregate_up.ndim != 2 or shared_a.ndim != 2:
        raise ValueError("aggregate_up and shared_a must be matrices")
    if aggregate_up.shape[1] != shared_a.shape[0]:
        raise ValueError("aggregate_up/shared_a rank mismatch")
    if eps <= 0:
        raise ValueError("eps must be positive")
    work_dtype = torch.float64
    aggregate = aggregate_up.detach().to(dtype=work_dtype)
    norm_a = torch.linalg.vector_norm(shared_a.detach().to(dtype=work_dtype))
    coefficient = aggregate / norm_a.clamp_min(eps)
    gram = coefficient.t() @ coefficient
    gram = 0.5 * (gram + gram.t())
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    return eigenvalues, eigenvectors


def rotate_operator_coordinates(
    shared_a: Tensor,
    aggregate_up: Tensor,
    current_up: Tensor,
    eigenvectors: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Rotate rank coordinates without changing historical/current operators."""

    rotation = eigenvectors.to(device=shared_a.device, dtype=shared_a.dtype)
    if rotation.shape != (shared_a.shape[0], shared_a.shape[0]):
        raise ValueError("eigenvector rotation has the wrong shape")
    return (
        rotation.t() @ shared_a,
        aggregate_up @ rotation.to(aggregate_up),
        current_up @ rotation.to(current_up),
    )


def _tie_groups(values: Tensor, relative_tolerance: float) -> list[tuple[int, ...]]:
    if values.ndim != 1:
        raise ValueError("energy spectrum must be one-dimensional")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    groups: list[list[int]] = []
    for index, value in enumerate(values.detach().cpu().double().tolist()):
        if not groups:
            groups.append([index])
            continue
        reference = float(values[groups[-1][0]].detach().cpu())
        scale = max(abs(reference), abs(value), torch.finfo(torch.float64).eps)
        if abs(value - reference) <= relative_tolerance * scale:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [tuple(group) for group in groups]


def spectral_tie_group_ids(values: Tensor, relative_tolerance: float) -> Tensor:
    """Return a stable integer group id for each ordered spectral value."""

    groups = _tie_groups(values, relative_tolerance)
    group_ids = torch.empty(values.numel(), dtype=torch.int64)
    for group_id, group in enumerate(groups):
        group_ids[list(group)] = group_id
    return group_ids


def diagonal_functional_energies(
    shared_a: Tensor,
    operator_energies: Tensor,
    input_second_moment: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return activation factors and diagonal functional-energy surrogates.

    ``shared_a`` is expected to be in the historical-energy eigenbasis.  The
    returned values implement ``q_i = a_i diag(v) a_i.T`` and
    ``E_i = lambda_i q_i`` without constructing a dense covariance matrix.
    """

    if shared_a.ndim != 2:
        raise ValueError("shared_a must be a matrix")
    if operator_energies.shape != (shared_a.shape[0],):
        raise ValueError("operator_energies must contain one value per A row")
    if input_second_moment.shape != (shared_a.shape[1],):
        raise ValueError("input_second_moment has the wrong shape")
    if not bool(torch.isfinite(shared_a).all()):
        raise ValueError("shared_a contains non-finite values")
    if not bool(torch.isfinite(operator_energies).all()):
        raise ValueError("operator_energies contains non-finite values")
    if not bool(torch.isfinite(input_second_moment).all()):
        raise ValueError("input_second_moment contains non-finite values")
    if bool((operator_energies < 0).any()):
        raise ValueError("operator_energies must be non-negative")
    if bool((input_second_moment < 0).any()):
        raise ValueError("input_second_moment must be non-negative")

    work_a = shared_a.detach().to(dtype=torch.float64)
    work_v = input_second_moment.detach().to(
        device=work_a.device, dtype=work_a.dtype
    )
    activation_factors = work_a.square() @ work_v
    functional = operator_energies.detach().to(
        device=work_a.device, dtype=work_a.dtype
    ) * activation_factors
    return activation_factors, functional


def select_global_low_energy_partition(
    spectra: Sequence[Tensor],
    energy_budget: float,
    tie_relative_tolerance: float = 1e-6,
    eps: float = 1e-12,
    tie_spectra: Sequence[Tensor] | None = None,
) -> EnergyPartition:
    """Select complete low-energy eigenspace groups under one global budget.

    Equal (within tolerance) eigenvalues from the same branch are indivisible,
    avoiding arbitrary choices inside a degenerate eigenspace.
    """

    if not 0.0 <= energy_budget <= 1.0:
        raise ValueError("energy_budget must be in [0, 1]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if tie_spectra is None:
        tie_spectra = spectra
    if len(tie_spectra) != len(spectra):
        raise ValueError("tie_spectra must match the number of spectra")
    masks = [torch.zeros_like(values, dtype=torch.bool) for values in spectra]
    records = []
    total_energy = 0.0
    total_directions = 0
    for branch_index, (values, tie_values) in enumerate(
        zip(spectra, tie_spectra)
    ):
        if values.ndim != 1:
            raise ValueError("each energy spectrum must be one-dimensional")
        if tie_values.shape != values.shape:
            raise ValueError("tie spectrum shape must match energy spectrum")
        clean = values.detach().cpu().double().clamp_min(0.0)
        clean_ties = tie_values.detach().cpu().double().clamp_min(0.0)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("energy spectrum contains non-finite values")
        if not bool(torch.isfinite(clean_ties).all()):
            raise ValueError("tie spectrum contains non-finite values")
        total_energy += float(clean.sum())
        total_directions += clean.numel()
        for group in _tie_groups(clean_ties, tie_relative_tolerance):
            group_energy = float(clean[list(group)].sum())
            group_level = float(clean[list(group)].mean())
            records.append((group_level, group_energy, branch_index, group))

    budget_energy = energy_budget * total_energy
    selected_energy = 0.0
    selected_directions = 0
    if total_energy <= eps:
        for mask in masks:
            mask.fill_(True)
        selected_directions = total_directions
    else:
        records.sort(key=lambda item: (item[0], item[2], item[3][0]))
        allowance = max(eps, eps * total_energy)
        for _, group_energy, branch_index, group in records:
            if selected_energy + group_energy > budget_energy + allowance:
                break
            masks[branch_index][list(group)] = True
            selected_energy += group_energy
            selected_directions += len(group)

    return EnergyPartition(
        plastic_masks=tuple(masks),
        total_energy=total_energy,
        selected_energy=selected_energy,
        budget_energy=budget_energy,
        selected_directions=selected_directions,
        total_directions=total_directions,
        tie_groups=len(records),
    )


def mask_jaccard(first: Sequence[Tensor], second: Sequence[Tensor]) -> float:
    """Return Jaccard overlap for two branch-mask collections."""

    if len(first) != len(second):
        raise ValueError("mask collections must have the same length")
    intersection = 0
    union = 0
    for left, right in zip(first, second):
        if left.shape != right.shape:
            raise ValueError("corresponding masks must have equal shapes")
        left = left.detach().bool().cpu()
        right = right.detach().bool().cpu()
        intersection += int((left & right).sum())
        union += int((left | right).sum())
    return 1.0 if union == 0 else intersection / union


def spearman_rank_correlation(first: Tensor, second: Tensor) -> float:
    """Return deterministic average-rank Spearman correlation."""

    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Spearman inputs must be equal one-dimensional tensors")
    if first.numel() < 2:
        return 1.0

    def average_ranks(values: Tensor) -> Tensor:
        values = values.detach().cpu().double()
        order = torch.argsort(values, stable=True)
        ranks = torch.empty_like(values)
        start = 0
        while start < values.numel():
            end = start + 1
            while end < values.numel() and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return ranks

    left = average_ranks(first)
    right = average_ranks(second)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(left, right) else 0.0
    return float((left @ right) / denominator)


def project_partition_gradient(
    gradient: Tensor,
    shared_a: Tensor,
    plastic_mask: Tensor,
) -> Tensor:
    """Keep stable rows fixed and make plastic updates Grassmann-tangent."""

    if gradient.shape != shared_a.shape:
        raise ValueError("gradient/shared_a shape mismatch")
    if plastic_mask.shape != (shared_a.shape[0],):
        raise ValueError("plastic_mask has the wrong shape")
    mask = plastic_mask.to(device=shared_a.device, dtype=torch.bool)
    projected = torch.zeros_like(gradient)
    if not bool(mask.any()):
        return projected
    plastic_a = shared_a[mask]
    plastic_gradient = gradient[mask]
    stable_a = shared_a[~mask]
    if stable_a.numel():
        plastic_gradient = plastic_gradient - (
            plastic_gradient @ stable_a.t()
        ) @ stable_a
    symmetric = 0.5 * (
        plastic_gradient @ plastic_a.t()
        + plastic_a @ plastic_gradient.t()
    )
    plastic_gradient = plastic_gradient - symmetric @ plastic_a
    projected[mask] = plastic_gradient
    return projected


def retract_partitioned_coordinates(
    candidate_a: Tensor,
    stable_anchor_a: Tensor,
    aggregate_up: Tensor,
    current_up: Tensor,
    plastic_mask: Tensor,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, float]]:
    """Retract A and preserve both post-step effective operators exactly.

    Stable rows are restored from ``stable_anchor_a``.  Plastic rows are
    orthonormalized in the stable complement.  The returned coordinate map
    reparameterizes G and B so that the historical normalized operator and
    current raw operator match their values at ``candidate_a``.
    """

    if candidate_a.shape != stable_anchor_a.shape:
        raise ValueError("candidate_a/stable_anchor_a shape mismatch")
    rank, _ = candidate_a.shape
    if aggregate_up.shape[1] != rank or current_up.shape[1] != rank:
        raise ValueError("up projections do not match A rank")
    mask = plastic_mask.to(device=candidate_a.device, dtype=torch.bool)
    if mask.shape != (rank,):
        raise ValueError("plastic_mask has the wrong shape")

    work = candidate_a.detach().to(dtype=torch.float64)
    anchor = stable_anchor_a.detach().to(device=work.device, dtype=work.dtype)
    stable = anchor[~mask]
    plastic = work[mask]
    if stable.numel():
        plastic_residual = plastic - (plastic @ stable.t()) @ stable
    else:
        plastic_residual = plastic

    retracted = anchor.clone()
    if plastic_residual.numel():
        q, r = torch.linalg.qr(plastic_residual.t(), mode="reduced")
        diagonal = torch.diagonal(r).abs()
        threshold = eps * max(1.0, float(diagonal.max()))
        if bool((diagonal <= threshold).any()):
            raise RuntimeError("plastic A rows became rank deficient during retraction")
        signs = torch.where(
            torch.diagonal(r) < 0,
            -torch.ones_like(diagonal),
            torch.ones_like(diagonal),
        )
        retracted[mask] = (q * signs.unsqueeze(0)).t()

    identity = torch.eye(rank, dtype=retracted.dtype, device=retracted.device)
    orthogonality_error = torch.linalg.matrix_norm(
        retracted @ retracted.t() - identity
    ) / max(1.0, rank ** 0.5)
    coordinate_map = work @ retracted.t()
    reconstruction_error = torch.linalg.matrix_norm(
        work - coordinate_map @ retracted
    ) / torch.linalg.matrix_norm(work).clamp_min(eps)
    if float(reconstruction_error) > max(1e-6, 100.0 * eps):
        raise RuntimeError(
            "partition retraction changed the candidate row space "
            f"(relative error={float(reconstruction_error):.3e})"
        )

    norm_before = torch.linalg.vector_norm(work).clamp_min(eps)
    norm_after = torch.linalg.vector_norm(retracted).clamp_min(eps)
    aggregate = aggregate_up.detach().to(dtype=torch.float64)
    current = current_up.detach().to(dtype=torch.float64)
    new_aggregate = aggregate @ coordinate_map * (norm_after / norm_before)
    new_current = current @ coordinate_map

    historical_before = aggregate @ work / norm_before
    historical_after = new_aggregate @ retracted / norm_after
    current_before = current @ work
    current_after = new_current @ retracted
    historical_error = torch.linalg.matrix_norm(
        historical_after - historical_before
    ) / torch.linalg.matrix_norm(historical_before).clamp_min(eps)
    current_error = torch.linalg.matrix_norm(
        current_after - current_before
    ) / torch.linalg.matrix_norm(current_before).clamp_min(eps)
    diagnostics = {
        "orthogonality_error": float(orthogonality_error),
        "rowspace_reconstruction_error": float(reconstruction_error),
        "historical_operator_error": float(historical_error),
        "current_operator_error": float(current_error),
    }
    return (
        retracted.to(candidate_a),
        new_aggregate.to(aggregate_up),
        new_current.to(current_up),
        coordinate_map.to(candidate_a),
        diagnostics,
    )
