import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from models.hoep_adaptation import update_energy, response_energy, lagged_prototype_loss


class AdaptationTests(unittest.TestCase):
    def test_weighted_energy_and_gradients_match_dense_operator(self):
        g = torch.Generator().manual_seed(7)
        u = torch.randn(7, 3, generator=g, dtype=torch.double, requires_grad=True)
        a = torch.randn(3, 7, generator=g, dtype=torch.double, requires_grad=True)
        u0 = torch.randn(7, 3, generator=g, dtype=torch.double)
        a0 = torch.randn(3, 7, generator=g, dtype=torch.double)
        x = torch.randn(100, 7, generator=g, dtype=torch.double)
        c = x.T @ x / len(x)
        actual = update_energy(u, a, u0, a0, c)
        expected = ((u @ a - u0 @ a0) @ x.T).square().sum() / len(x)
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual, (u, a), retain_graph=True)
        ge = torch.autograd.grad(expected, (u, a))
        for first, second in zip(ga, ge):
            torch.testing.assert_close(first, second)

    def test_zero_at_anchor_and_under_exact_factor_compensation(self):
        g = torch.Generator().manual_seed(9)
        u = torch.randn(7, 3, generator=g, dtype=torch.double)
        a = torch.randn(3, 7, generator=g, dtype=torch.double)
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.double))
        c = torch.eye(7, dtype=torch.double)
        self.assertLess(float(update_energy(u, a, u, a, c)), 1e-10)
        self.assertLess(float(update_energy(u @ q, q.T @ a, u, a, c)), 1e-10)

    def test_history_input_weighting_and_current_b_are_visible(self):
        a = torch.tensor([[1., 1.]], dtype=torch.double)
        old_up = torch.zeros(2, 1, dtype=torch.double)
        b = torch.tensor([[1.], [0.]], dtype=torch.double, requires_grad=True)
        c = torch.diag(torch.tensor([9., 1.], dtype=torch.double))
        loss = update_energy(b, a, old_up, a, c)
        torch.testing.assert_close(loss, torch.tensor(10., dtype=torch.double))
        self.assertGreater(float(torch.autograd.grad(loss, b)[0].abs().sum()), 0)

    def test_prototype_loss_uses_old_negatives_and_detached_lagged_bank(self):
        x = torch.tensor([[1., .2]], requires_grad=True)
        p = torch.tensor([[1., 0.], [0., 1.]])
        loss = lagged_prototype_loss(x, torch.tensor([1]), p, .1)
        with torch.no_grad():
            p[1] = x.detach()[0]
        loss.backward()  # Must survive EMA mutation and retain feature gradient.
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertIsNone(p.grad)


if __name__ == '__main__':
    unittest.main()
