import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from goal_audit import rounding_evidence


class GoalThresholdTest(unittest.TestCase):
    def test_missing_result_never_passes(self):
        self.assertFalse(rounding_evidence(None,.5)['robustly_above_threshold'])

    def test_rounded_gap_at_target_needs_exact_counts(self):
        evidence=rounding_evidence(.5,.5)
        self.assertFalse(evidence['robustly_above_threshold'])
        self.assertTrue(evidence['requires_exact_evaluation'])

    def test_rounding_can_move_an_apparent_pass_below_target(self):
        evidence=rounding_evidence(.505,.5)
        self.assertFalse(evidence['robustly_above_threshold'])
        self.assertLess(evidence['lower_bound_pp'],.5)

    def test_float_subtraction_does_not_fake_a_strict_margin(self):
        self.assertFalse(rounding_evidence(80.51-80.,.5)['robustly_above_threshold'])

    def test_clear_margin_passes_rounding_bound(self):
        self.assertTrue(rounding_evidence(.515,.5)['robustly_above_threshold'])

    def test_clearly_insufficient_gap_does_not_request_exact_evaluation(self):
        evidence=rounding_evidence(.48,.5)
        self.assertFalse(evidence['robustly_above_threshold'])
        self.assertFalse(evidence['requires_exact_evaluation'])

    def test_exact_counts_can_resolve_a_small_true_gain(self):
        self.assertTrue(rounding_evidence(.5005,.5,score_half_width=0.)['robustly_above_threshold'])
        self.assertFalse(rounding_evidence(.5,.5,score_half_width=0.)['robustly_above_threshold'])

    def test_nonfinite_mean_rejected(self):
        with self.assertRaises(ValueError):rounding_evidence(math.nan,.5)


if __name__=='__main__':unittest.main()
