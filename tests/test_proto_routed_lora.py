import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.proto_routed_lora import (
    mask_logits_to_tasks,
    select_task_by_cosine,
    targets_to_task_ids,
)


def test_cosine_router_selects_nearest_task():
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([[0.9, 0.1], [0.2, 0.8]])

    selected, scores = select_task_by_cosine(query, prototypes)

    assert selected.tolist() == [0, 1]
    assert scores.shape == (2, 2)


def test_targets_map_to_saved_task_ranges():
    targets = torch.tensor([0, 9, 10, 19, 20, 29])

    task_ids = targets_to_task_ids(targets, [[0, 10], [10, 20], [20, 30]])

    assert task_ids.tolist() == [0, 0, 1, 1, 2, 2]


def test_task_mask_removes_classes_outside_routed_task():
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    routed_tasks = torch.tensor([0, 1])

    masked = mask_logits_to_tasks(logits, routed_tasks, [[0, 6], [6, 12]])

    assert torch.equal(masked[0, :6], logits[0, :6])
    assert torch.equal(masked[1, 6:], logits[1, 6:])
    assert torch.all(masked[0, 6:] < -1e20)
    assert torch.all(masked[1, :6] < -1e20)
