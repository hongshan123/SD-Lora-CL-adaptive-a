"""Scale a projected SGD direction while retaining its decay cancellation."""
import math
import torch


@torch.no_grad()
def amplify_projected_gradient(gradient, parameter, multiplier, weight_decay=0.):
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('positive finite A multiplier required')
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('nonnegative finite weight decay required')
    if gradient.shape != parameter.shape:
        raise ValueError('gradient and parameter shapes differ')
    if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(parameter).all()):
        raise FloatingPointError('nonfinite A gradient/parameter')
    if multiplier == 1.:
        return gradient, dict(nonzero=False, relative_error=0., before_norm=0., after_norm=0.)
    before = gradient.double() + weight_decay * parameter.double()
    adjusted = gradient.clone().mul_(multiplier)
    if weight_decay:
        adjusted.add_(parameter, alpha=(multiplier - 1.) * weight_decay)
    after = adjusted.double() + weight_decay * parameter.double()
    norm = float(torch.linalg.vector_norm(before))
    error = 0. if norm < 1e-12 else float(
        torch.linalg.vector_norm(after - multiplier * before) / (multiplier * norm))
    if not bool(torch.isfinite(adjusted).all()) or error > 1e-4:
        raise FloatingPointError('A gain failed finite/direction check')
    return adjusted, dict(nonzero=norm >= 1e-12, relative_error=error,
                          before_norm=norm, after_norm=float(torch.linalg.vector_norm(after)))
