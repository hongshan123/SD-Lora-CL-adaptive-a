import copy
import pytest
import torch
from torch import nn

from backbone.structure_ablation import make_structure_backbone


def test_disabled_adaptive_a_never_runs_retraction():
    from types import SimpleNamespace
    from models.sa_sdlora import Learner
    disabled = SimpleNamespace(_sa_adaptive_a_enabled=False,
                               _sa_adaptive_a_strategy="operator_energy_partition")
    # No network is needed: the disabled hook must return before looking it up.
    assert Learner._after_optimizer_step(disabled) is None


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.qkv = nn.Linear(8, 24)
        self.blocks = nn.ModuleList([block])
        self.head = nn.Identity()

    def forward(self, x):
        return self.blocks[0].attn.qkv(x)


def build(base, path, mode, task, training=True):
    return make_structure_backbone(
        structure_mode=mode, vit_model=copy.deepcopy(base), r=3,
        filepath=str(path), cur_task_index=task, index=task == 0,
        cumulative_state=True, cumulative_merge="live_a_aggregate_b",
        train_a_all_tasks=mode == "independent_bank",
        live_a_coordinate_align=False,
        live_a_absorb_mode="operator_preserving_absorb",
        initialize_training_partition=training)


@pytest.mark.parametrize("mode", ["independent_bank", "frozen_bank"])
def test_bank_exact_deployment_and_history_immutability(tmp_path, mode):
    torch.manual_seed(17)
    base = TinyViT()
    x = torch.randn(4, 5, 8)
    m = build(base, tmp_path, mode, 0)
    with torch.no_grad():
        for a in m.w_As:
            a.weight.add_(torch.randn_like(a.weight)*.03)
        for b in m.w_Bs:
            b.weight.normal_(0, .02)
    expected = m(x).detach()
    a0 = [a.weight.detach().clone() for a in m.w_As]
    m.save_lora_parameters(str(tmp_path), 0)
    rebuilt = build(base, tmp_path, mode, 1, False)
    torch.testing.assert_close(rebuilt(x), expected, rtol=1e-5, atol=1e-6)
    training = build(base, tmp_path, mode, 1)
    torch.testing.assert_close(training(x), expected, rtol=1e-5, atol=1e-6)
    if mode == "frozen_bank":
        for a, old in zip(training.w_As, a0):
            assert not a.weight.requires_grad
            torch.testing.assert_close(a.weight, old, rtol=0, atol=0)
    else:
        assert not torch.equal(training.w_As[0].weight, a0[0])
    with torch.no_grad():
        for b in training.w_Bs:
            b.weight.normal_(0, .02)
    second = training(x).detach()
    training.save_lora_parameters(str(tmp_path), 1)
    again = build(base, tmp_path, mode, 2, False)
    torch.testing.assert_close(again(x), second, rtol=1e-5, atol=1e-6)
    saved = torch.load(tmp_path/"sa_state.pt", weights_only=True)
    assert len(saved["entries"]) == 2


def test_frozen_bank_matches_exact_fold_forward_and_input_gradient(tmp_path):
    torch.manual_seed(19)
    base = TinyViT()
    bank_dir, fold_dir = tmp_path/"bank", tmp_path/"fold"
    for task in range(3):
        bank = build(base, bank_dir, "frozen_bank", task)
        fold = build(base, fold_dir, "frozen_fold", task)
        with torch.no_grad():
            for b, f in zip(bank.w_Bs, fold.w_Bs):
                b.weight.normal_(0, .02)
                f.weight.copy_(b.weight)
        x = torch.randn(2, 4, 8, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        bank(x).square().sum().backward()
        fold(y).square().sum().backward()
        torch.testing.assert_close(bank(x), fold(y), rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=2e-6)
        for b, f in zip(bank.w_Bs, fold.w_Bs):
            torch.testing.assert_close(b.weight.grad, f.weight.grad, rtol=1e-5, atol=2e-6)
        bank.save_lora_parameters(str(bank_dir), task)
        fold.save_lora_parameters(str(fold_dir), task)
