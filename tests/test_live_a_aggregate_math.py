"""Core math tests for Live-A Aggregate-B (bank == aggregate, incl. gradients).

These tests use a local standalone aggregate implementation matching the
planned ``live_a_aggregate_b`` semantics (historical branch ``G A x / ||A||``
with a *live* shared A).  They do not depend on the new backbone mode yet.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def bank_forward(a, b_list, scales, x):
    out = torch.zeros_like(x)
    for b, s in zip(b_list, scales):
        denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-8
        out = out + s * F.linear(F.linear(x, a), b) / denom
    return out


def aggregate_forward(a, g, x):
    return F.linear(F.linear(x, a), g / torch.linalg.vector_norm(a))


def make_g(b_list, scales):
    g = torch.zeros_like(b_list[0])
    for b, s in zip(b_list, scales):
        g = g + s * b / (torch.linalg.vector_norm(b) + 1e-8)
    return g


@pytest.mark.parametrize("n_tasks", [1, 3, 5])
@pytest.mark.parametrize("b_scale", [0.3, 1.0, 3.0])
def test_aggregate_matches_bank_forward(n_tasks, b_scale):
    torch.manual_seed(0)
    dim, rank, n = 16, 4, 8
    a = torch.randn(rank, dim, dtype=torch.float64)
    x = torch.randn(n, dim, dtype=torch.float64)
    b_list = [
        b_scale * torch.randn(dim, rank, dtype=torch.float64)
        for _ in range(n_tasks)
    ]
    scales = [torch.tensor(0.5 + 0.3 * i, dtype=torch.float64) for i in range(n_tasks)]
    g = make_g(b_list, scales)
    bank = bank_forward(a, b_list, scales, x)
    agg = aggregate_forward(a, g, x)
    assert torch.allclose(agg, bank, atol=1e-7, rtol=1e-7)


def test_gradient_with_respect_to_a_matches():
    torch.manual_seed(1)
    dim, rank, n, tasks = 16, 4, 8, 4
    b_list = [torch.randn(dim, rank) for _ in range(tasks)]
    scales = [torch.tensor(0.4 + 0.2 * i) for i in range(tasks)]
    g = make_g(b_list, scales)
    x = torch.randn(n, dim)
    y = torch.randn(n, dim)
    a_val = torch.randn(rank, dim)

    def grads(forward_fn):
        a = nn.Parameter(a_val.clone())
        opt = torch.optim.SGD([a], lr=0.1)
        loss = ((forward_fn(a, x) - y) ** 2).mean()
        loss.backward()
        return a.grad.clone()

    grad_bank = grads(lambda a, z: bank_forward(a, b_list, scales, z))
    grad_agg = grads(lambda a, z: aggregate_forward(a, g, z))
    assert torch.allclose(grad_agg, grad_bank, atol=1e-6, rtol=1e-6)


def test_gradient_with_respect_to_input_matches():
    torch.manual_seed(2)
    dim, rank, n, tasks = 16, 4, 8, 4
    a = torch.randn(rank, dim)
    b_list = [torch.randn(dim, rank) for _ in range(tasks)]
    scales = [torch.tensor(0.4 + 0.2 * i) for i in range(tasks)]
    g = make_g(b_list, scales)
    y = torch.randn(n, dim)
    x = torch.randn(n, dim, requires_grad=True)

    def input_grads(forward_fn):
        x.grad = None
        loss = ((forward_fn(a, x) - y) ** 2).mean()
        loss.backward()
        return x.grad.clone()

    grad_bank = input_grads(lambda a_, z: bank_forward(a_, b_list, scales, z))
    grad_agg = input_grads(lambda a_, z: aggregate_forward(a_, g, z))
    assert torch.allclose(grad_agg, grad_bank, atol=1e-6, rtol=1e-6)


def test_frozen_scales_equivalent_to_folded_g():
    """Folding frozen historical scales into G is exact for forward and dL/dA."""
    torch.manual_seed(3)
    dim, rank, n, tasks = 16, 4, 8, 4
    b_list = [torch.randn(dim, rank) for _ in range(tasks)]
    scales = [torch.tensor(0.4 + 0.2 * i) for i in range(tasks)]
    g = make_g(b_list, scales)
    a = nn.Parameter(torch.randn(rank, dim))
    x = torch.randn(n, dim)
    y = torch.randn(n, dim)

    # Bank with frozen scales.
    loss_bank = ((bank_forward(a, b_list, scales, x) - y) ** 2).mean()
    loss_bank.backward()
    grad_bank = a.grad.clone()

    # Aggregate with folded G and live A.
    a.grad = None
    loss_agg = ((aggregate_forward(a, g, x) - y) ** 2).mean()
    loss_agg.backward()
    grad_agg = a.grad.clone()

    assert torch.allclose(grad_agg, grad_bank, atol=1e-6, rtol=1e-6)
