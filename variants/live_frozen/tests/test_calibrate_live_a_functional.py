import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from scripts.calibrate_live_a_functional import calibrate_task, merge_task_statistics
from utils.rng_utils import rng_state_hash


def task(value, count):
    return {
        "input_moments": [torch.tensor([value, value + 1.0])],
        "output_sensitivities": [
            torch.tensor([1.0, 3.0]), torch.tensor([3.0, 1.0])
        ],
        "input_counts": [count],
        "output_counts": [count, count],
    }


def test_streaming_statistics_are_count_weighted_with_constant_shapes():
    first = task(2.0, 10)
    second = task(8.0, 30)
    history = merge_task_statistics(None, first)
    merged = merge_task_statistics(history, second)
    assert len(merged["input_moments"]) == 1
    assert len(merged["output_sensitivities"]) == 2
    assert merged["input_counts"] == [40]
    assert merged["output_counts"] == [40, 40]
    assert torch.allclose(merged["input_moments"][0], torch.tensor([6.5, 7.5]))
    for value in merged["output_sensitivities"]:
        assert torch.allclose(value.mean(), torch.tensor(1.0), atol=1e-6)


def test_streaming_statistics_reject_missing_branch():
    history = merge_task_statistics(None, task(2.0, 10))
    incomplete = task(3.0, 10)
    incomplete["output_sensitivities"].pop()
    try:
        merge_task_statistics(history, incomplete)
    except ValueError as error:
        assert "branch" in str(error)
    else:
        raise AssertionError("missing Q/V branch must fail")


def test_streaming_statistics_reject_zero_calibration_count():
    empty = task(2.0, 0)
    with pytest.raises(ValueError, match="count"):
        merge_task_statistics(None, empty)


def test_real_calibration_entry_preserves_rng_even_when_model_creation_uses_it(
    tmp_path, monkeypatch
):
    class TinyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.ModuleDict({"attn": nn.ModuleDict({"qkv": nn.Linear(2, 6)})})
            ])
            self.head = nn.Identity()

        def forward(self, x):
            out = self.blocks[0].attn.qkv(x)
            return (out[..., :2] + out[..., -2:]).mean(dim=1)

    class Manager:
        def get_task_size(self, _task):
            return 2

        def get_dataset(self, *_args, **_kwargs):
            return TensorDataset(
                torch.arange(2), torch.tensor([
                    [[1.0, 0.0], [0.5, 0.0]],
                    [[0.0, 1.0], [0.0, 0.5]],
                ]), torch.tensor([0, 1]),
            )

    snapshot = tmp_path / "task_000"
    snapshot.mkdir()
    torch.save({
        "task_id": 0,
        "shared_a": [torch.tensor([[1.0, 0.0]])] * 2,
        "merged_b": [torch.zeros(2, 1)] * 2,
    }, snapshot / "sa_merged_lora.pt")
    torch.save({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])},
               snapshot / "sa_prototypes.pt")
    monkeypatch.setattr("scripts.calibrate_live_a_functional.timm.create_model",
                        lambda *_args, **_kwargs: TinyViT())
    before = rng_state_hash()
    statistics, samples = calibrate_task(snapshot, Manager(), 2, torch.device("cpu"), 0)
    assert samples == 2
    assert rng_state_hash() == before
    assert statistics["model_tensor_hash_preserved"] is True
