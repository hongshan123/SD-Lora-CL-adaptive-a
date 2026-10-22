import dataclasses
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from backbone.budget_allocation import select_with_branch_coverage
from backbone.operator_energy_partition import select_global_low_energy_partition, _tie_groups


class BudgetAllocation(unittest.TestCase):
    def same_masks(self, left, right):
        self.assertEqual(len(left.plastic_masks), len(right.plastic_masks))
        for a, b in zip(left.plastic_masks, right.plastic_masks):
            self.assertTrue(torch.equal(a, b))

    def check_cost(self, spectra, result, budget, eps=1e-12):
        selected = sum(float(s[m].sum()) for s, m in zip(spectra, result.plastic_masks))
        total = sum(float(s.sum()) for s in spectra)
        self.assertAlmostEqual(selected, result.selected_energy, places=10)
        self.assertLessEqual(selected, budget*total + max(eps, eps*total) + 1e-12)
        self.assertEqual(sum(int(m.sum()) for m in result.plastic_masks), result.selected_directions)

    def test_restores_starved_branch_within_unchanged_budget(self):
        spectra = [torch.tensor([1., 2., 3., 4.], dtype=torch.float64),
                   torch.tensor([10., 20., 100., 100.], dtype=torch.float64)]
        original = select_global_low_energy_partition(spectra, .05)
        self.assertFalse(bool(original.plastic_masks[1].any()))
        changed = select_with_branch_coverage(spectra, .05)
        self.assertTrue(changed.lowest_group_coverage_feasible)
        self.assertFalse(changed.fallback_to_global)
        self.assertTrue(all(bool(m.any()) for m in changed.plastic_masks))
        self.check_cost(spectra, changed, .05)

    def test_already_covered_keeps_original_allocation(self):
        spectra = [torch.tensor([1., 2., 100., 200.], dtype=torch.float64)] * 2
        original = select_global_low_energy_partition(spectra, .05)
        changed = select_with_branch_coverage(spectra, .05)
        self.same_masks(original, changed)
        self.check_cost(spectra, changed, .05)

    def test_infeasible_reservation_returns_original_masks(self):
        spectra = [torch.ones(10, dtype=torch.float64)] * 2
        changed = select_with_branch_coverage(spectra, .05)
        self.assertFalse(changed.lowest_group_coverage_feasible)
        self.assertTrue(changed.fallback_to_global)
        self.same_masks(changed, select_global_low_energy_partition(spectra, .05))
        self.check_cost(spectra, changed, .05)

    def test_complete_degenerate_eigenspaces_are_never_split(self):
        spectra = [torch.tensor([.01, .01, 1., 100.], dtype=torch.float64),
                   torch.tensor([.02, .02, 20., 200.], dtype=torch.float64)]
        changed = select_with_branch_coverage(spectra, .05)
        for s, mask in zip(spectra, changed.plastic_masks):
            for group in _tie_groups(s, 1e-6):
                self.assertTrue(bool((mask[list(group)] == mask[group[0]]).all()))
        self.check_cost(spectra, changed, .05)

    def test_zero_and_negligible_history_report_actual_selected_energy(self):
        for value in [0., 1e-12]:
            with self.subTest(value=value):
                spectra = [torch.full((10,), value, dtype=torch.float64)] * 24
                changed = select_with_branch_coverage(spectra, .05, eps=1e-8)
                self.assertTrue(all(bool(m.all()) for m in changed.plastic_masks))
                self.assertEqual(changed.selected_directions, 240)
                self.check_cost(spectra, changed, .05, eps=1e-8)

    def test_random_spectra_obey_budget_and_feasibility_contract(self):
        generator = torch.Generator().manual_seed(19)
        spectra = [torch.sort(torch.rand(10, generator=generator, dtype=torch.float64)**3 * (i+1)).values
                   for i in range(24)]
        for budget in [0., .01, .05, .2, 1.]:
            with self.subTest(budget=budget):
                changed = select_with_branch_coverage(spectra, budget)
                self.check_cost(spectra, changed, budget)
                if changed.lowest_group_coverage_feasible:
                    self.assertTrue(all(bool(m.any()) for m in changed.plastic_masks))
                else:
                    self.same_masks(changed, select_global_low_energy_partition(spectra, budget))

    def test_positive_rescaling_preserves_masks_away_from_tolerance(self):
        spectra = [torch.tensor([1., 2., 3., 4.], dtype=torch.float64),
                   torch.tensor([10., 20., 100., 100.], dtype=torch.float64)]
        self.same_masks(select_with_branch_coverage(spectra, .05),
                        select_with_branch_coverage([s*1e4 for s in spectra], .05))

    def test_does_not_change_rng(self):
        state = torch.random.get_rng_state().clone()
        select_with_branch_coverage([torch.arange(10, dtype=torch.float64)] * 24, .05)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_invalid_spectra_fail(self):
        for spectra in [[], [torch.tensor([2., 1.])], [torch.tensor([1., float('nan')])],
                        [torch.tensor([-1., 2.])], [torch.empty(0)]]:
            with self.subTest(spectra=spectra), self.assertRaises(ValueError):
                select_with_branch_coverage(spectra, .05)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
