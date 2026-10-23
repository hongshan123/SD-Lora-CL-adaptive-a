import importlib.util
from pathlib import Path
import unittest
import torch

PATH = Path(__file__).resolve().parents[1] / 'source/backbone/branch_operator_average.py'
SPEC = importlib.util.spec_from_file_location('average_math', PATH)
math = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math)


class OperatorAverageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.generator = torch.Generator().manual_seed(727)
        self.a = torch.stack([torch.linalg.qr(torch.randn(7, 3, generator=self.generator,
                              dtype=torch.float64)).Q.T for _ in range(2)])
        self.b = torch.randn(2, 5, 3, generator=self.generator, dtype=torch.float64)
        self.s = torch.tensor(.8, dtype=torch.float64)

    def test_window_is_last_quarter_with_nontrivial_short_checks(self):
        self.assertEqual(math.averaging_window(20, .25), (16, 5))
        self.assertEqual(math.averaging_window(80, .25), (61, 20))
        self.assertEqual(math.averaging_window(2, .25), (1, 2))
        self.assertEqual(math.averaging_window(1, .25), (1, 1))
        for n, fraction in [(0, .25), (True, .25), (20, 0), (20, 1.1), (20, float('nan'))]:
            with self.assertRaises(ValueError):
                math.averaging_window(n, fraction)

    def test_same_functions_in_rotating_factor_coordinates_have_same_mean(self):
        rotation = torch.stack([torch.linalg.qr(torch.randn(3, 3, generator=self.generator,
                                dtype=torch.float64)).Q for _ in range(2)])
        rotation = rotation @ torch.diag(torch.tensor([1., .5, 2.], dtype=torch.float64))
        a2 = rotation @ self.a
        b2 = self.b @ torch.linalg.inv(rotation)
        mean = math.OperatorMean()
        mean.add(math.current_operators(self.a, self.b, self.s))
        mean.add(math.current_operators(a2, b2, self.s))
        coefficient, projected, record = math.project_to_final_a(mean.mean, a2)
        torch.testing.assert_close(projected, math.current_operators(self.a, self.b, self.s),
                                   atol=1e-12, rtol=1e-12)
        self.assertLess(record['projection_residual_squared_relative'], 1e-25)
        torch.testing.assert_close(coefficient @ a2, projected)

    def test_fixed_a_mean_includes_the_learned_scale(self):
        mean = math.OperatorMean()
        mean.add(math.current_operators(self.a, self.b, self.s))
        mean.add(math.current_operators(self.a, self.b * 3, self.s / 2))
        expected = (self.s * self.b + self.s / 2 * self.b * 3) / 2
        coefficient, projected, _ = math.project_to_final_a(mean.mean, self.a)
        torch.testing.assert_close(coefficient, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(projected, expected @ self.a)

    def test_projection_is_the_least_squares_fit_with_recorded_discarded_energy(self):
        target = torch.randn(2, 5, 7, generator=self.generator, dtype=torch.float64)
        coefficient, projected, record = math.project_to_final_a(target, self.a)
        expected = torch.linalg.lstsq(self.a.transpose(1, 2), target.transpose(1, 2)).solution.transpose(1, 2)
        torch.testing.assert_close(coefficient, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close((target - projected) @ self.a.transpose(1, 2),
                                   torch.zeros(2, 5, 3, dtype=torch.float64), atol=1e-12, rtol=0)
        residual = float((target - projected).square().sum() / target.square().sum())
        self.assertAlmostEqual(residual, record['projection_residual_squared_relative'])

    def test_average_can_exceed_final_norm_and_is_not_normcap(self):
        mean = math.OperatorMean()
        last = math.current_operators(self.a, self.b, self.s)
        mean.add(3 * last)
        mean.add(last)
        _, projected, _ = math.project_to_final_a(mean.mean, self.a)
        self.assertGreater(float(projected.norm()), float(last.norm()))
        torch.testing.assert_close(projected, 2 * last)

    def test_rank_shape_and_nonfinite_inputs_fail_explicitly(self):
        target = math.current_operators(self.a, self.b, self.s)
        with self.assertRaises(ValueError):
            math.project_to_final_a(target, torch.zeros_like(self.a))
        with self.assertRaises(FloatingPointError):
            math.OperatorMean().add(target * float('nan'))
        with self.assertRaises(ValueError):
            math.current_operators(self.a, self.b[:, :, :2], self.s)
        mean = math.OperatorMean();mean.add(target)
        with self.assertRaises(ValueError):
            mean.add(target[:1])

    def test_math_does_not_consume_rng(self):
        before = torch.get_rng_state().clone()
        mean = math.OperatorMean()
        mean.add(math.current_operators(self.a, self.b, self.s))
        mean.add(math.current_operators(self.a, self.b * .9, self.s))
        math.project_to_final_a(mean.mean, self.a)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_accumulator_memory_does_not_grow_with_updates(self):
        target = math.current_operators(self.a, self.b, self.s)
        mean = math.OperatorMean()
        for _ in range(100):mean.add(target)
        self.assertEqual(mean.count, 100)
        self.assertEqual(mean.mean.numel(), target.numel())
        torch.testing.assert_close(mean.mean, target)


if __name__ == '__main__':
    unittest.main()
