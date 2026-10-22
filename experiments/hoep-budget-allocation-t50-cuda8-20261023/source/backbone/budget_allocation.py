"""Reserve branch coverage inside the existing global historical-energy cap."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor

from backbone.operator_energy_partition import (
    EnergyPartition, _tie_groups, select_global_low_energy_partition,
)


@dataclass(frozen=True)
class CoveragePartition(EnergyPartition):
    lowest_group_coverage_feasible: bool
    reserved_energy: float
    fallback_to_global: bool
    changed_from_global: bool


def select_with_branch_coverage(
    spectra: Sequence[Tensor],
    energy_budget: float,
    tie_relative_tolerance: float = 1e-6,
    eps: float = 1e-12,
) -> CoveragePartition:
    """Reserve each branch's lowest complete tie group, then fill the budget.

    If the reservation does not fit, return the original global masks. If the
    original masks already cover all branches, preserve them. This is a
    deterministic greedy allocation, not an optimal knapsack solver.
    """
    if not spectra or not math.isfinite(eps) or eps <= 0:
        raise ValueError('nonempty spectra and finite positive eps required')
    clean = [values.detach().cpu().double() for values in spectra]
    for values in clean:
        if (values.ndim != 1 or not values.numel()
                or not bool(torch.isfinite(values).all()) or bool((values < 0).any())
                or bool((values[1:] < values[:-1]).any())):
            raise ValueError('ascending finite nonnegative nonempty spectra required')
    original = select_global_low_energy_partition(
        spectra, energy_budget, tie_relative_tolerance, eps)
    groups = [_tie_groups(values, tie_relative_tolerance) for values in clean]
    reserved = sum(float(values[list(branch_groups[0])].sum())
                   for values, branch_groups in zip(clean, groups))
    allowance = max(eps, eps * original.total_energy)
    feasible = reserved <= original.budget_energy + allowance

    def unchanged(fallback):
        # The original zero-history path reports zero selected energy. Record
        # the actual selected energy here; the same absolute tolerance applies.
        selected = sum(float(values[mask.detach().cpu()].sum())
                       for values, mask in zip(clean, original.plastic_masks))
        return CoveragePartition(
            original.plastic_masks, original.total_energy, selected,
            original.budget_energy, original.selected_directions,
            original.total_directions, original.tie_groups,
            feasible, reserved, fallback, False)

    if not feasible:
        return unchanged(True)
    if original.total_energy <= eps or all(bool(m.any()) for m in original.plastic_masks):
        return unchanged(False)
    masks = [torch.zeros_like(values, dtype=torch.bool) for values in spectra]
    for mask, branch_groups in zip(masks, groups):
        mask[list(branch_groups[0])] = True
    remaining = []
    for index, (values, branch_groups) in enumerate(zip(clean, groups)):
        for group in branch_groups[1:]:
            remaining.append((float(values[list(group)].mean()),
                              float(values[list(group)].sum()), index, group))
    remaining.sort(key=lambda item: (item[0], item[2], item[3][0]))
    selected = reserved
    for _, cost, index, group in remaining:
        if selected + cost <= original.budget_energy + allowance:
            masks[index][list(group)] = True
            selected += cost
    return CoveragePartition(
        tuple(masks), original.total_energy, selected, original.budget_energy,
        sum(int(mask.sum()) for mask in masks), original.total_directions,
        original.tie_groups, True, reserved, False,
        any(not torch.equal(mask, old) for mask, old in zip(masks, original.plastic_masks)))
