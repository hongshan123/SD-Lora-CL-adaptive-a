from pathlib import Path

import torch
from torch import nn
import pytest

from backbone.lora import ParameterWrapper
from backbone.sa_lora import _LiveAAggregateQKV, _SensitivityBudgetedGQKV
from utils.sa_task_snapshots import audit_snapshot, capture_pre_merge, save_post_merge
from scripts.diagnose_main_branch_intrusion import summarize_intrusion


class FakeBackbone:
    def __init__(self, wrapper, down, up, scale):
        self.lora_vit = nn.Module()
        self.lora_vit.blocks = nn.ModuleList([nn.Module()])
        self.lora_vit.blocks[0].attn = nn.Module()
        self.lora_vit.blocks[0].attn.qkv = wrapper
        self.w_As = [down, down]
        self.w_Bs = [up, up]
        self.wrapped_param = [nn.Module()]
        self.wrapped_param[0].param = nn.Parameter(scale)
        self.cumulative_merge = "live_a_aggregate_b"


def test_pre_merge_snapshot_reconstructs_both_branch_outputs():
    torch.manual_seed(4)
    down = nn.Linear(5, 2, bias=False)
    up = nn.Linear(2, 5, bias=False)
    scale = torch.tensor(0.7)
    aggregate = torch.randn(5, 2)
    wrapper = _LiveAAggregateQKV(
        nn.Linear(5, 15), down, down, up, up, aggregate, aggregate,
        [nn.Identity()], 0,
    )
    backbone = FakeBackbone(wrapper, down, up, scale)
    fc = nn.Linear(5, 3)
    snapshot = capture_pre_merge(backbone, fc, {0: torch.randn(5)}, 1, 2, 3)
    inputs = torch.randn(2, 4, 5)
    with torch.no_grad():
        expected_history = wrapper._norm_live_a(inputs, down.weight, aggregate)
        expected_current = scale * up(down(inputs))
        branch = snapshot["branches"][0]
        actual_history = torch.nn.functional.linear(
            torch.nn.functional.linear(inputs, branch["down"]),
            branch["historical_up"] / (branch["down"].norm() + 1e-8),
        )
        actual_current = snapshot["current_scale"] * torch.nn.functional.linear(
            torch.nn.functional.linear(inputs, branch["down"]), branch["current_up"]
        )
    assert torch.allclose(actual_history, expected_history, atol=1e-6)
    assert torch.allclose(actual_current, expected_current, atol=1e-6)
    assert snapshot["task_id"] == 1
    assert len(snapshot["old_prototypes"]) == 1


def test_sbgc_pre_merge_snapshot_reconstructs_fixed_p_and_current_branch():
    torch.manual_seed(17)
    down = nn.Linear(5, 2, bias=False)
    up = nn.Linear(2, 5, bias=False)
    scale = torch.tensor(0.7)
    projection = torch.randn(2, 5)
    aggregate = torch.randn(5, 2)
    wrapper = _SensitivityBudgetedGQKV(
        nn.Linear(5, 15), down, down, up, up,
        projection, aggregate, projection, aggregate,
        [nn.Identity()], 0,
    )
    backbone = FakeBackbone(wrapper, down, up, scale)
    backbone.cumulative_merge = "sensitivity_budgeted_g"
    snapshot = capture_pre_merge(backbone, nn.Linear(5, 3), {}, 1, 2, 3)
    branch = snapshot["branches"][0]
    inputs = torch.randn(2, 4, 5)
    historical = torch.nn.functional.linear(
        torch.nn.functional.linear(inputs, branch["historical_down"]),
        branch["historical_up"],
    )
    current = snapshot["current_scale"] * torch.nn.functional.linear(
        torch.nn.functional.linear(inputs, branch["down"]),
        branch["current_up"],
    )
    expected_historical = torch.nn.functional.linear(
        torch.nn.functional.linear(inputs, wrapper.projection_q),
        wrapper.unified_up_q,
    )
    expected_current = scale * up(down(inputs))
    assert snapshot["version"] == 2
    assert torch.allclose(historical, expected_historical)
    assert torch.allclose(current, expected_current)


def test_anchored_snapshot_records_independent_history_basis():
    torch.manual_seed(31)
    down = nn.Linear(5, 2, bias=False)
    up = nn.Linear(2, 5, bias=False)
    scale = torch.tensor(0.7)
    wrapper = _LiveAAggregateQKV(
        nn.Linear(5, 15), down, down, up, up,
        torch.randn(5, 2), torch.randn(5, 2),
        [ParameterWrapper(nn.Parameter(scale.clone()))], 0,
        history_forward="anchored",
    )
    with torch.no_grad():
        down.weight.add_(torch.randn_like(down.weight))
    backbone = FakeBackbone(wrapper, down, up, scale)
    snapshot = capture_pre_merge(backbone, nn.Linear(5, 3), {}, 1, 2, 3)
    assert snapshot["version"] == 3
    assert snapshot["history_forward"] == "anchored"
    inputs = torch.randn(2, 4, 5)
    for branch, expected in zip(snapshot["branches"], wrapper.historical_output(inputs)):
        actual = torch.nn.functional.linear(
            torch.nn.functional.linear(inputs, branch["historical_down"]),
            branch["historical_up"],
        )
        assert torch.equal(actual, expected)
        assert not torch.equal(branch["down"], branch["historical_down"])


def test_anchored_snapshot_replay_keeps_current_and_historical_bases(monkeypatch):
    from scripts.causal_subspace_intervention import TaskQKV
    from scripts.diagnose_main_branch_intrusion import build_model

    class TinyTaskModel(nn.Module):
        def __init__(self, state, old_weight, old_bias, new_classes, live):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.8))
            self.fc = nn.Linear(5, len(old_weight) + new_classes)
            self.wrappers = nn.ModuleList([
                TaskQKV(
                    nn.Linear(5, 15), state["projection_down"][index],
                    state["unified_up"][index], state["projection_down"][index + 1],
                    state["unified_up"][index + 1], lambda: self.scale, live,
                ) for index in range(0, 24, 2)
            ])

    monkeypatch.setattr("scripts.diagnose_main_branch_intrusion.TaskModel", TinyTaskModel)
    torch.manual_seed(32)
    historical_down = torch.eye(5)[:2]
    current_down = torch.eye(5)[2:4]
    historical_up = torch.randn(5, 2)
    current_up = torch.randn(5, 2)
    snapshot = {
        "version": 3, "history_forward": "anchored",
        "normalize_current_branch": False, "known_classes": 2, "total_classes": 3,
        "current_scale": torch.tensor(0.7), "fc_weight": torch.randn(3, 5),
        "fc_bias": torch.randn(3), "branches": [{
            "down": current_down, "current_up": current_up,
            "historical_down": historical_down, "historical_up": historical_up,
        } for _ in range(24)],
    }
    model = build_model(snapshot, torch.device("cpu"))
    x = torch.randn(2, 4, 5)
    wrapper = model.wrappers[0]
    raw = wrapper.qkv(x)
    change = x @ (historical_up @ historical_down + 0.7 * current_up @ current_down).T
    expected = torch.cat((raw[..., :5] + change, raw[..., 5:10], raw[..., -5:] + change), -1)
    assert torch.allclose(wrapper(x), expected, atol=1e-6, rtol=1e-6)
    invalid = {**snapshot, "version": 1}
    with pytest.raises(ValueError, match="explicit historical factors"):
        build_model(invalid, torch.device("cpu"))


def test_post_merge_copies_task_local_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("sa_state.pt", "sa_prototypes.pt", "sa_merged_lora.pt", "CLs_weight1.pt", "CLs_bias1.pt"):
        (run_dir / name).write_bytes(name.encode())
    (run_dir / "task_snapshots" / "task_001").mkdir(parents=True)
    (run_dir / "task_snapshots" / "task_001" / "pre_merge.pt").write_bytes(b"pre")
    destination = save_post_merge(run_dir, 1, {"dataset": "cub", "seed": 1})
    assert (destination / "sa_state.pt").read_bytes() == b"sa_state.pt"
    assert (destination / "sa_prototypes.pt").read_bytes() == b"sa_prototypes.pt"
    assert (destination / "CLs_weight1.pt").is_file()
    assert (destination / "config.json").is_file()
    assert (destination / "complete.json").is_file()
    assert audit_snapshot(destination)["verified"]
    assert not (run_dir / "task_snapshots" / "task_000" / "sa_state.pt").exists()


def test_audit_detects_corrupted_snapshot(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("sa_state.pt", "sa_prototypes.pt", "sa_merged_lora.pt", "CLs_weight0.pt", "CLs_bias0.pt"):
        (run_dir / name).write_bytes(name.encode())
    directory = run_dir / "task_snapshots" / "task_000"
    directory.mkdir(parents=True)
    (directory / "pre_merge.pt").write_bytes(b"pre")
    save_post_merge(run_dir, 0, {"seed": 1})
    (directory / "sa_state.pt").write_bytes(b"changed")
    try:
        audit_snapshot(directory)
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("corrupted snapshot passed checksum audit")


def test_intrusion_metric_counts_old_to_new_errors_with_fixed_head():
    labels = torch.tensor([0, 1, 2])
    historical = torch.tensor([[4., 1., 0.], [0., 4., 1.], [0., 1., 4.]])
    full = torch.tensor([[1., 0., 4.], [0., 3., 2.], [0., 4., 1.]])
    result = summarize_intrusion(labels, historical, full, old_count=2)
    assert result["old"]["history_top1"] == 100.0
    assert result["old"]["full_top1"] == 50.0
    assert result["old"]["history_correct_to_full_new_error_rate"] == 50.0
    assert result["old"]["full_restricted_top1"] == 100.0
    assert result["new"]["full_top1"] == 0.0
