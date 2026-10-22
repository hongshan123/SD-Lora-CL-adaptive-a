"""Independent scalar checks of recorded allocation; no model imports."""
import math


def tie_groups(values, relative_tolerance):
    groups = []
    for index, value in enumerate(values):
        if not groups:
            groups.append([index])
        else:
            reference = values[groups[-1][0]]
            scale = max(abs(reference), abs(value), 2.220446049250313e-16)
            if abs(value-reference) <= relative_tolerance*scale:
                groups[-1].append(index)
            else:
                groups.append([index])
    return groups


def original_global_masks(spectra, budget, relative_tolerance, eps):
    total = sum(sum(values) for values in spectra)
    masks = [[False]*len(values) for values in spectra]
    if total <= eps:
        return [[True]*len(values) for values in spectra]
    candidates = []
    for branch, values in enumerate(spectra):
        for group in tie_groups(values, relative_tolerance):
            energy = sum(values[i] for i in group)
            candidates.append((energy/len(group), energy, branch, group))
    candidates.sort(key=lambda item: (item[0], item[2], item[3][0]))
    selected = 0.
    for _, cost, branch, group in candidates:
        if selected+cost > budget*total + max(eps, eps*total):
            break
        for index in group:
            masks[branch][index] = True
        selected += cost
    return masks


def validate_allocation_diagnostic(diagnostic, config):
    spectra = diagnostic['branch_spectra']
    masks = diagnostic['branch_masks']
    counts = diagnostic['branch_counts']
    rank = config['lora_rank']
    assert len(spectra) == len(masks) == len(counts) == 24
    assert diagnostic['partition_policy'] == 'low'
    budget = config['sa_hoep_energy_budget']
    eps = config.get('sa_adaptive_a_eps', 1e-8)
    rtol = config.get('sa_hoep_eigenvalue_rtol', 1e-6)
    assert diagnostic['energy_budget'] == budget
    for values, mask, count in zip(spectra, masks, counts):
        assert len(values) == len(mask) == rank
        assert all(math.isfinite(v) and v >= 0 for v in values)
        assert all(a <= b for a, b in zip(values, values[1:]))
        assert all(type(value) is bool for value in mask)
        assert type(count) is int and count == sum(mask)
        for group in tie_groups(values, rtol):
            assert all(mask[i] == mask[group[0]] for i in group), 'split eigenspace'
    total = sum(sum(values) for values in spectra)
    selected = sum(sum(v for v, active in zip(values, mask) if active)
                   for values, mask in zip(spectra, masks))
    allowance = max(eps, eps*total)
    assert math.isfinite(diagnostic['total_energy'])
    assert math.isfinite(diagnostic['selected_energy'])
    assert abs(total-diagnostic['total_energy']) < allowance+1e-12
    assert abs(selected-diagnostic['selected_energy']) < allowance+1e-12
    assert selected <= budget*total + allowance+1e-12, 'global energy cap exceeded'
    assert diagnostic['selected_directions'] == sum(counts)
    assert diagnostic['total_directions'] == 24*rank
    allocation = config.get('sa_hoep_budget_allocation', 'global')
    assert diagnostic.get('budget_allocation', 'global') == allocation
    reference = original_global_masks(spectra, budget, rtol, eps)
    if allocation == 'global':
        assert masks == reference, 'global control differs'
        return dict(allocation='global', changed_from_global=False,
                    selected_energy_ratio=selected/max(total, 1e-30),
                    selected_directions=sum(counts))
    assert allocation == 'coverage'
    reserved = sum(sum(values[i] for i in tie_groups(values, rtol)[0]) for values in spectra)
    assert math.isfinite(diagnostic['reserved_energy'])
    assert abs(reserved-diagnostic['reserved_energy']) < allowance+1e-12
    feasible = reserved <= budget*total+allowance
    assert type(diagnostic['lowest_group_coverage_feasible']) is bool
    assert diagnostic['lowest_group_coverage_feasible'] == feasible
    assert type(diagnostic['coverage_fallback_to_global']) is bool
    assert diagnostic['coverage_fallback_to_global'] == (not feasible)
    if feasible:
        assert all(count > 0 for count in counts)
        for values, mask in zip(spectra, masks):
            assert all(mask[i] for i in tie_groups(values, rtol)[0])
    else:
        assert masks == reference, 'infeasible coverage must use original allocation'
    changed = masks != reference
    assert type(diagnostic['coverage_changed_from_global']) is bool
    assert diagnostic['coverage_changed_from_global'] == changed
    return dict(allocation='coverage', changed_from_global=changed,
                lowest_group_coverage_feasible=feasible,
                selected_energy_ratio=selected/max(total, 1e-30),
                selected_directions=sum(counts))
