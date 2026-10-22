import torch

from scripts.counterfactual_g_head_swap import evaluate_grid, graft_old_rows, metrics


def test_graft_preserves_target_new_rows():
    anchor = torch.tensor([[1., 0.], [0., 1.]])
    target = torch.tensor([[2., 0.], [0., 2.], [3., 3.]])
    grafted = graft_old_rows(anchor, target, 2)
    assert torch.equal(grafted[:2], anchor)
    assert torch.equal(grafted[2:], target[2:])
    assert torch.equal(target[0], torch.tensor([2., 0.]))


def test_full_grid_uses_identical_labels_and_features():
    features = {
        "anchor": torch.tensor([[1., 0.], [0., 1.]]),
        "target": torch.tensor([[0., 1.], [1., 0.]]),
        "labels": torch.tensor([0, 1]),
    }
    old_head = torch.tensor([[1., 0.], [0., 1.]])
    target_head = torch.tensor([[0., 1.], [1., 0.], [-1., -1.]])
    result, predictions = evaluate_grid(features, old_head, target_head, 2)
    assert len(result) == 4
    assert result["anchor_G__anchor_rows"]["top1"] == 100.0
    assert result["target_G__target_rows"]["top1"] == 100.0
    assert result["anchor_G__target_rows"]["top1"] == 0.0
    assert result["target_G__anchor_rows"]["top1"] == 0.0
    assert all(value.shape == (2,) for value in predictions.values())


def test_restricted_accuracy_separates_later_class_competition():
    logits = torch.tensor([[0.9, 0.1, 1.0], [0.2, 0.8, 0.1]])
    result, _ = metrics(logits, torch.tensor([0, 1]), 2)
    assert result["top1"] == 50.0
    assert result["old_restricted_top1"] == 100.0
    assert result["predicted_later_rate"] == 50.0


def test_fc_shadow_uses_linear_bias_and_grafts_old_bias():
    features = {
        "anchor": torch.tensor([[1., 0.]]),
        "target": torch.tensor([[1., 0.]]),
        "labels": torch.tensor([0]),
    }
    old_weight = torch.tensor([[0., 0.]])
    new_weight = torch.tensor([[0., 0.], [0., 0.]])
    result, _ = evaluate_grid(
        features, old_weight, new_weight, 1,
        torch.tensor([2.]), torch.tensor([0., 1.]),
    )
    assert result["target_G__anchor_rows"]["top1"] == 100.0
    assert result["target_G__target_rows"]["top1"] == 0.0
