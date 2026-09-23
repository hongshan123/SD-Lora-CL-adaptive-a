from pathlib import Path

import torch
from torch import nn

from backbone.sa_lora import _LiveAAggregateQKV
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
