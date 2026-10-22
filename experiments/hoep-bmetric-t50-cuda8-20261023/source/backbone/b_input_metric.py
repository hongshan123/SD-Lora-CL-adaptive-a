"""Small input-side metrics for B gradients, with orthogonal gauge equivariance."""
import math
import torch


@torch.no_grad()
def condition_b_gradient(gradient, current_cov, historical_cov=None, *, damping=0.1, history_mix=1.0):
    if damping <= 0 or not math.isfinite(damping):raise ValueError('positive finite damping required')
    if history_mix < 0 or not math.isfinite(history_mix):raise ValueError('invalid history mix')
    if gradient.ndim != 2:raise ValueError('B gradient must be a matrix')
    rank=gradient.shape[1]
    if current_cov.shape != (rank,rank):raise ValueError('covariance shape mismatch')
    if not bool(torch.isfinite(gradient).all()):raise FloatingPointError('nonfinite B gradient')
    cov=current_cov.detach().double()
    if not bool(torch.isfinite(cov).all()):raise FloatingPointError('nonfinite current covariance')
    norm=torch.linalg.vector_norm(gradient.double())
    if float(norm) < 1e-12:return gradient.clone(),dict(skipped=True,reason='zero gradient')
    cov=(cov+cov.T)*0.5
    trace=cov.trace()
    if float(trace) < 1e-12:return gradient.clone(),dict(skipped=True,reason='zero current input energy')
    metric=cov/trace
    if historical_cov is not None and history_mix:
        old=historical_cov.detach().double()
        if old.shape != (rank,rank):raise ValueError('history covariance shape mismatch')
        if not bool(torch.isfinite(old).all()):raise FloatingPointError('nonfinite historical covariance')
        old=(old+old.T)*0.5
        if float(old.trace()) > 1e-12:metric=metric+history_mix*old/old.trace()
    metric=metric+torch.eye(rank,dtype=metric.dtype,device=metric.device)*(damping*metric.trace()/rank)
    chol,info=torch.linalg.cholesky_ex(metric)
    if int(info):raise FloatingPointError('input metric is not positive definite')
    adjusted=torch.cholesky_solve(gradient.double().T,chol).T
    adjusted=adjusted*(norm/torch.linalg.vector_norm(adjusted).clamp_min(1e-30))
    adjusted=adjusted.to(gradient.dtype)
    after=torch.linalg.vector_norm(adjusted.double())
    error=float((after-norm).abs()/norm)
    cosine=float((gradient.double()*adjusted.double()).sum()/(norm*after).clamp_min(1e-30))
    if not bool(torch.isfinite(adjusted).all()) or error>1e-5:
        raise FloatingPointError('B metric failed finite/norm-preservation check')
    return adjusted,dict(skipped=False,relative_norm_error=error,cosine=cosine)
