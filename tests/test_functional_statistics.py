import torch
from torch import nn

from backbone.functional_statistics import QKVFunctionStatistics


class TinyQKV(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class TwoLayerQKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = TinyQKV(2)
        self.second = TinyQKV(2)

    def forward(self, x):
        first = self.first(x)
        second = self.second(first[..., :2] + first[..., -2:])
        return second[..., :2] + second[..., -2:]


def test_qkv_statistics_match_reference_gradients_and_preserve_parameter_grads():
    torch.manual_seed(7)
    model = TwoLayerQKV()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(3, 4, 2)
    reference_outputs = []
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, output: reference_outputs.append(output)
        )
        for module in (model.first, model.second)
    ]
    reference_input = x.detach().clone().requires_grad_(True)
    reference_loss = model(reference_input).square().mean()
    reference_grads = torch.autograd.grad(reference_loss, reference_outputs)
    for handle in handles:
        handle.remove()

    with QKVFunctionStatistics([model.first, model.second]) as collector:
        collector.begin_batch()
        loss = model(x).square().mean()
        collector.accumulate(loss, batch_size=3)
        statistics = collector.summary()

    assert statistics["input_counts"] == [12, 12]
    assert statistics["output_counts"] == [12, 12, 12, 12]
    assert torch.allclose(statistics["input_moments"][0], x.square().mean((0, 1)))
    first = reference_outputs[0]
    second_input = first[..., :2] + first[..., -2:]
    assert torch.allclose(
        statistics["input_moments"][1], second_input.square().mean((0, 1))
    )
    for layer, gradient in enumerate(reference_grads):
        q = (gradient[..., :2] * 3).square().mean((0, 1))
        v = (gradient[..., -2:] * 3).square().mean((0, 1))
        assert torch.allclose(statistics["output_sensitivities"][2 * layer], q)
        assert torch.allclose(statistics["output_sensitivities"][2 * layer + 1], v)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_qkv_statistics_hooks_are_removed_after_context_exit():
    model = TwoLayerQKV()
    with QKVFunctionStatistics([model.first, model.second]):
        assert len(model.first._forward_hooks) == 1
    assert len(model.first._forward_hooks) == 0
    assert len(model.first._forward_pre_hooks) == 0
