"""Average current LoRA functions in their input/output coordinates."""
import math
import torch


def averaging_window(total_steps, fraction):
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError('positive integer optimizer-step count required')
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('fraction must lie in (0,1]')
    count = min(total_steps, max(2, math.ceil(total_steps * fraction)))
    return total_steps - count + 1, count


@torch.no_grad()
def current_operators(down, up, scale):
    if down.ndim != 3 or up.ndim != 3 or scale.numel() != 1:
        raise ValueError('batched down/up matrices and one scalar required')
    if down.shape[0] != up.shape[0] or down.shape[1] != up.shape[2]:
        raise ValueError('incompatible branch/rank dimensions')
    if not all(bool(torch.isfinite(x).all()) for x in (down, up, scale)):
        raise FloatingPointError('nonfinite current operator input')
    return torch.bmm(up.double() * scale.double().reshape(()), down.double())


class OperatorMean:
    def __init__(self):
        self.mean = None
        self.count = 0

    @torch.no_grad()
    def add(self, operator):
        if operator.ndim != 3 or not bool(torch.isfinite(operator).all()):
            raise FloatingPointError('finite batched operators required')
        value = operator.detach().double()
        if self.mean is None:
            self.mean = value.clone()
        else:
            if self.mean.shape != value.shape or self.mean.device != value.device:
                raise ValueError('operator shape/device changed inside one task')
            self.mean.add_(value - self.mean, alpha=1. / (self.count + 1))
        self.count += 1


@torch.no_grad()
def project_to_final_a(operator, down):
    """Least-squares projection; discarded energy is diagnostic, not an Acc bound."""
    if operator.ndim != 3 or down.ndim != 3:
        raise ValueError('batched matrices required')
    if operator.shape[0] != down.shape[0] or operator.shape[2] != down.shape[2]:
        raise ValueError('incompatible branch/input dimensions')
    if not all(bool(torch.isfinite(x).all()) for x in (operator, down)):
        raise FloatingPointError('nonfinite averaging projection')
    a, target = down.double(), operator.double()
    gram = torch.bmm(a, a.transpose(1, 2))
    eigenvalues = torch.linalg.eigvalsh(gram)
    if not bool((eigenvalues[:, 0] > eigenvalues[:, -1] * 1e-12).all()):
        raise ValueError('full row-rank final A required')
    rhs = torch.bmm(target, a.transpose(1, 2))
    coefficient = torch.linalg.solve(gram, rhs.transpose(1, 2)).transpose(1, 2)
    projected = torch.bmm(coefficient, a)
    residual = (target - projected).square().sum(dim=(1, 2))
    energy = target.square().sum(dim=(1, 2))
    return coefficient, projected, dict(
        projection_residual_squared_relative=float(residual.sum() / energy.sum().clamp_min(1e-30)),
        branch_projection_residual_squared_relative=(residual / energy.clamp_min(1e-30)).tolist(),
        mean_operator_frobenius=float(energy.sum().sqrt()),
        projected_operator_frobenius=float(projected.square().sum().sqrt()))
