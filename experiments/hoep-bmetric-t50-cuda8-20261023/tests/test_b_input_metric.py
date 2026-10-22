from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from backbone.b_input_metric import condition_b_gradient


class BInputMetricTests(unittest.TestCase):
    def test_matches_direct_solve_and_preserves_gradient_norm(self):
        generator=torch.Generator().manual_seed(41)
        gradient=torch.randn(7,3,generator=generator,dtype=torch.double)
        x=torch.randn(20,3,generator=generator,dtype=torch.double)
        covariance=x.T@x/len(x)
        metric=covariance/covariance.trace()+torch.eye(3,dtype=torch.double)*(.1/3)
        expected=torch.linalg.solve(metric,gradient.T).T
        expected*=torch.linalg.vector_norm(gradient)/torch.linalg.vector_norm(expected)
        actual,record=condition_b_gradient(gradient,covariance)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.linalg.vector_norm(actual),torch.linalg.vector_norm(gradient))
        self.assertFalse(record['skipped'])

    def test_isotropic_input_is_identity(self):
        gradient=torch.tensor([[1.,2.,3.],[2.,-1.,4.]])
        actual,_=condition_b_gradient(gradient,torch.eye(3))
        torch.testing.assert_close(actual,gradient)

    def test_orthogonal_factor_coordinates_are_equivariant(self):
        generator=torch.Generator().manual_seed(8)
        gradient=torch.randn(9,3,generator=generator,dtype=torch.double)
        x=torch.randn(10,3,generator=generator,dtype=torch.double)
        cov=x.T@x
        old=torch.diag(torch.tensor([1.,5.,9.],dtype=torch.double))
        q,_=torch.linalg.qr(torch.randn(3,3,generator=generator,dtype=torch.double))
        first,_=condition_b_gradient(gradient,cov,old)
        second,_=condition_b_gradient(gradient@q,q.T@cov@q,q.T@old@q)
        torch.testing.assert_close(first@q,second)

    def test_history_metric_reduces_sensitive_direction_in_toy(self):
        gradient=torch.ones(1,2,dtype=torch.double)
        old=torch.diag(torch.tensor([100.,1.],dtype=torch.double))
        actual,_=condition_b_gradient(gradient,torch.eye(2,dtype=torch.double),old)
        self.assertLess(float((actual@old*actual).sum()),float((gradient@old*gradient).sum()))
        torch.testing.assert_close(actual.norm(),gradient.norm())

    def test_zero_gradient_is_finite_noop(self):
        gradient=torch.zeros(7,3)
        actual,record=condition_b_gradient(gradient,torch.eye(3))
        torch.testing.assert_close(actual,gradient);self.assertTrue(record['skipped'])

    def test_nonfinite_input_is_rejected(self):
        with self.assertRaises(FloatingPointError):
            condition_b_gradient(torch.ones(2,2),torch.full((2,2),float('nan')))


if __name__=='__main__':unittest.main()
