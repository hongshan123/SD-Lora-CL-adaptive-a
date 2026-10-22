import copy
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'source'))
from allocation_checks import original_global_masks, validate_allocation_diagnostic
from backbone.budget_allocation import select_with_branch_coverage
from backbone.operator_energy_partition import select_global_low_energy_partition


class AllocationChecks(unittest.TestCase):
    def setUp(self):
        self.config = dict(lora_rank=10, sa_hoep_energy_budget=.05,
                           sa_adaptive_a_eps=1e-8, sa_hoep_eigenvalue_rtol=1e-6,
                           sa_hoep_budget_allocation='coverage')
        generator = torch.Generator().manual_seed(43)
        self.spectra = [torch.sort(torch.rand(10, generator=generator, dtype=torch.float64)**3*(i+1)).values
                        for i in range(24)]

    def diagnostic(self, spectra=None):
        spectra = self.spectra if spectra is None else spectra
        p = select_with_branch_coverage(spectra, .05, eps=1e-8)
        return dict(branch_spectra=[s.tolist() for s in spectra],
                    branch_masks=[m.tolist() for m in p.plastic_masks],
                    branch_counts=[int(m.sum()) for m in p.plastic_masks],
                    partition_policy='low', energy_budget=.05,
                    total_energy=p.total_energy, selected_energy=p.selected_energy,
                    selected_directions=p.selected_directions, total_directions=p.total_directions,
                    budget_allocation='coverage', reserved_energy=p.reserved_energy,
                    lowest_group_coverage_feasible=p.lowest_group_coverage_feasible,
                    coverage_fallback_to_global=p.fallback_to_global,
                    coverage_changed_from_global=p.changed_from_global)

    def test_scalar_reference_matches_original_torch_allocator(self):
        for spectra in [self.spectra, [torch.ones(10, dtype=torch.float64)]*24,
                        [torch.zeros(10, dtype=torch.float64)]*24]:
            for budget in [.01, .05, .3]:
                expected = select_global_low_energy_partition(spectra, budget, eps=1e-8)
                actual = original_global_masks([s.tolist() for s in spectra], budget, 1e-6, 1e-8)
                self.assertEqual(actual, [m.tolist() for m in expected.plastic_masks])

    def test_valid_enabled_and_fallback_diagnostics(self):
        for spectra in [self.spectra, [torch.ones(10, dtype=torch.float64)]*24,
                        [torch.full((10,), 1e-12, dtype=torch.float64)]*24]:
            diagnostic = self.diagnostic(spectra)
            result = validate_allocation_diagnostic(diagnostic, self.config)
            self.assertEqual(result['changed_from_global'], diagnostic['coverage_changed_from_global'])

    def test_cannot_pass_with_all_rows_exceeding_budget(self):
        d = self.diagnostic([torch.ones(10, dtype=torch.float64)]*24)
        d['branch_masks'] = [[True]*10 for _ in range(24)]
        d['branch_counts'] = [10]*24
        d['selected_directions'] = 240
        d['selected_energy'] = d['total_energy']
        with self.assertRaisesRegex(AssertionError, 'global energy cap exceeded'):
            validate_allocation_diagnostic(d, self.config)

    def test_changed_flag_is_recomputed(self):
        d = self.diagnostic()
        d['coverage_changed_from_global'] = not d['coverage_changed_from_global']
        with self.assertRaises(AssertionError):
            validate_allocation_diagnostic(d, self.config)

    def test_mask_counts_and_lowest_group_cannot_be_faked(self):
        d = self.diagnostic()
        d['branch_counts'][0] += 1
        with self.assertRaises(AssertionError):
            validate_allocation_diagnostic(d, self.config)
        d = self.diagnostic()
        d['lowest_group_coverage_feasible'] = not d['lowest_group_coverage_feasible']
        with self.assertRaises(AssertionError):
            validate_allocation_diagnostic(d, self.config)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
