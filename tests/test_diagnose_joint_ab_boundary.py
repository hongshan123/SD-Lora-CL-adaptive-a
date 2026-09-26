import torch

from scripts.diagnose_joint_ab_boundary import (
    aligned_counterfactual,
    capped_current_up,
    paired_prediction_changes,
    summarize_group_logits,
)


def test_aligned_counterfactual_keeps_current_operator_and_projects_history():
    old_a = torch.tensor([[1.0, 0.0]])
    old_g = torch.tensor([[2.0], [0.0]])
    new_a = torch.tensor([[1.0, 1.0]])
    current_b = torch.tensor([[0.0], [1.0]])
    previous = {"shared_a": [old_a], "aggregate_up": [old_g]}
    pre = {
        "task_id": 1,
        "current_scale": torch.tensor(0.5),
        "branches": [{"down": new_a, "current_up": current_b}],
    }
    state = aligned_counterfactual(previous, pre, "operator_preserving_absorb")
    expected_history = old_g @ old_a / old_a.norm()
    expected_history = expected_history @ torch.linalg.pinv(new_a) @ new_a
    actual = state["merged_b"][0] @ state["shared_a"][0]
    expected = expected_history + 0.5 * current_b @ new_a
    assert torch.allclose(actual, expected, atol=1e-6)


def test_capped_current_up_matches_absorbed_operator():
    a = torch.tensor([[2.0, 0.0]])
    b = torch.tensor([[3.0], [4.0]])
    scale = torch.tensor(0.5)
    capped, gain = capped_current_up(a, b, scale, "bounded_norm_calibrated_absorb")
    assert abs(gain - 0.1) < 1e-6
    assert torch.allclose(scale * capped @ a, 0.1 * scale * b @ a)


def test_group_logits_reports_true_margin_and_old_to_new_errors():
    logits = torch.tensor([
        [0.9, 0.1, 0.8],
        [0.4, 0.5, 0.7],
        [0.1, 0.2, 0.6],
    ])
    labels = torch.tensor([0, 1, 2])
    groups = summarize_group_logits(logits, labels, old_count=2)
    assert groups["old"]["count"] == 2
    assert groups["old"]["top1"] == 50.0
    assert groups["old"]["predicted_new_rate"] == 50.0
    assert abs(groups["old"]["mean_true_margin"] - (-0.05)) < 1e-6
    assert groups["new"]["top1"] == 100.0


def test_paired_prediction_changes_counts_opposite_correctness_flips():
    labels = torch.tensor([0, 1, 2])
    before = torch.tensor([[0.8, 0.1, 0.2], [0.1, 0.7, 0.3], [0.1, 0.8, 0.4]])
    after = torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.2, 0.8], [0.1, 0.2, 0.9]])
    result = paired_prediction_changes(before, after, labels, old_count=2)
    assert result["old"] == {
        "count": 2, "prediction_changed": 1,
        "correct_to_wrong": 1, "wrong_to_correct": 0,
    }
    assert result["new"] == {
        "count": 1, "prediction_changed": 1,
        "correct_to_wrong": 0, "wrong_to_correct": 1,
    }
