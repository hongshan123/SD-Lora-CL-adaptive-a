import math
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from backbone.a_factor_gain import amplify_projected_gradient


class FactorGainTest(unittest.TestCase):
    def test_unit_multiplier_is_bitwise_identity(self):
        p = torch.tensor([[.3, .7]], dtype=torch.float32)
        g = torch.tensor([[.1234567, -.8765432]], dtype=torch.float32)
        out, _ = amplify_projected_gradient(g, p, 1., .1)
        self.assertIs(out, g)
        self.assertTrue(torch.equal(out, g))

    def test_radial_decay_is_not_amplified(self):
        p = torch.tensor([[1., 2.]], dtype=torch.float64)
        tangent = torch.tensor([[.2, -.1]], dtype=torch.float64)
        out, record = amplify_projected_gradient(tangent - .3*p, p, 100., .3)
        torch.testing.assert_close(out + .3*p, 100.*tangent, rtol=1e-13, atol=1e-13)
        self.assertLess(record['relative_error'], 1e-13)

    def test_frozen_rows_have_zero_effective_step(self):
        p = torch.tensor([[1., 2.], [2., -1.]], dtype=torch.float64)
        tangent = torch.tensor([[0., 0.], [.1, .2]], dtype=torch.float64)
        out, _ = amplify_projected_gradient(tangent - .2*p, p, 10., .2)
        torch.testing.assert_close((out + .2*p)[0], torch.zeros(2, dtype=p.dtype), atol=1e-14, rtol=0.)

    def test_sgd_momentum_matches_a_learning_rate_multiplier(self):
        for multiplier in [10., 100.]:
            p = torch.nn.Parameter(torch.tensor([[.2, -.4]], dtype=torch.float64))
            q = torch.nn.Parameter(p.detach().clone())
            first = torch.optim.SGD([p], lr=.01, momentum=.9, weight_decay=.2)
            second = torch.optim.SGD([q], lr=.01*multiplier, momentum=.9, weight_decay=.2)
            for _ in range(4):
                p.grad = p.detach().sin() - .2*p.detach()
                q.grad = q.detach().sin() - .2*q.detach()
                p.grad, _ = amplify_projected_gradient(p.grad, p, multiplier, .2)
                first.step(); second.step()
                torch.testing.assert_close(p, q, atol=1e-12, rtol=1e-12)

    def test_fp32_norm_and_direction(self):
        g = torch.linspace(-1., 1., 7680).reshape(10, 768)
        out, record = amplify_projected_gradient(g, torch.ones_like(g), 100.)
        torch.testing.assert_close(out, 100.*g)
        self.assertLess(record['relative_error'], 1e-6)

    def test_zero_gradient(self):
        g = torch.zeros(3, 4)
        out, record = amplify_projected_gradient(g, torch.zeros_like(g), 10.)
        self.assertTrue(torch.equal(out, g))
        self.assertFalse(record['nonzero'])

    def test_invalid_or_nonfinite_inputs(self):
        for multiplier in [0., -1., math.inf, math.nan]:
            with self.assertRaises(ValueError):
                amplify_projected_gradient(torch.ones(2), torch.ones(2), multiplier)
        with self.assertRaises(FloatingPointError):
            amplify_projected_gradient(torch.tensor([math.nan]), torch.ones(1), 10.)


if __name__ == '__main__':
    unittest.main()
