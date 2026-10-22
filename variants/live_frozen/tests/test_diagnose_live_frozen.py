import torch
from torch import nn

from scripts.diagnose_live_frozen import (
    build_stage_backbone,
    fixed_head,
    mean_prototypes,
    set_stage,
    summarize_stage,
    verify_formal_prototypes,
)
from scripts.evaluate_sa_sdlora import build_merged_backbone


def test_prototypes_are_class_ordered_and_use_normalized_image_features():
    features = torch.tensor([[2., 0.], [0., 1.], [2., 0.]])
    labels = torch.tensor([2, 3, 2])
    new = mean_prototypes(features, labels, [2, 3])
    old = {0: torch.tensor([1., 0.]), 1: torch.tensor([0., 1.])}
    head = fixed_head(old, new, 4)
    assert torch.allclose(head, torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]))


def test_stage_switch_changes_qv_operator_without_changing_base_qkv():
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Module()])
            self.blocks[0].attn = nn.Module()
            self.blocks[0].attn.qkv = nn.Linear(2, 6, bias=False)
            self.head = nn.Identity()

    base = TinyBackbone()
    old = {"shared_a": [torch.tensor([[1., 0.]]), torch.tensor([[1., 0.]])],
           "merged_b": [torch.tensor([[1.], [0.]]), torch.tensor([[0.], [1.]])]}
    new = {"shared_a": [torch.tensor([[0., 1.]]), torch.tensor([[0., 1.]])],
           "merged_b": [torch.tensor([[2.], [0.]]), torch.tensor([[0.], [2.]])]}
    model = build_stage_backbone(base, old)
    weight_before = model.blocks[0].attn.qkv.qkv.weight.detach().clone()
    set_stage(model, new)
    wrapper = model.blocks[0].attn.qkv
    assert torch.allclose(wrapper.b_q @ wrapper.a_q, torch.tensor([[0., 2.], [0., 0.]]))
    assert torch.equal(wrapper.qkv.weight, weight_before)
    assert torch.equal(old["shared_a"][0], torch.tensor([[1., 0.]]))
    assert torch.equal(old["merged_b"][0], torch.tensor([[1.], [0.]]))


def test_stage_summary_counts_global_old_new_and_flips():
    labels = torch.tensor([0, 1, 2])
    before = torch.tensor([[.9, .2, .1], [.2, .8, .7], [.1, .2, .9]])
    after = torch.tensor([[.2, .1, .9], [.2, .8, .7], [.1, .8, .3]])
    result = summarize_stage(after, labels, 2, before)
    assert result["global_top1"] == 100 / 3
    assert result["old"]["top1"] == 50.0
    assert result["old"]["predicted_new_rate"] == 50.0
    assert result["flips_vs_previous"]["old"]["correct_to_wrong"] == 1


def test_formal_new_prototype_recompute_must_match_training_artifact():
    expected = {2: torch.tensor([1., 0.]), 3: torch.tensor([0., 1.])}
    close = {2: torch.tensor([1., 0.]), 3: torch.tensor([0., 1.000001])}
    assert verify_formal_prototypes(close, expected) < 1e-4
    wrong = {2: torch.tensor([0., 1.]), 3: torch.tensor([0., 1.])}
    import pytest
    with pytest.raises(RuntimeError, match="formal new prototype"):
        verify_formal_prototypes(wrong, expected)
