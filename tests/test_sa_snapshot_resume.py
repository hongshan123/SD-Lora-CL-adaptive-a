import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.sa_snapshot_resume import restore_sbgc_snapshot
from utils import sa_snapshot_resume
from utils.sa_task_snapshots import audit_snapshot


class _Manager:
    nb_tasks = 10

    def get_task_size(self, task):
        return 20


class _Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.fc = None
        self.prototypes = None

    def update_fc(self, count):
        self.fc = nn.Linear(6, count)

    def set_prototypes(self, values):
        self.prototypes = values


class _Learner:
    def __init__(self):
        self.network = _Network()
        self._device = torch.device("cpu")
        self.feature_dim = 6
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0

    def _raw_network(self):
        return self.network

    def update_network(self, index, task_index):
        assert index is False
        backbone = nn.Identity()
        backbone.task_id = task_index
        return backbone


def _fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source" / "task_snapshots" / "task_001"
    source.mkdir(parents=True)
    config = {
        "model_name": "sa_sdlora", "dataset": "cub", "init_cls": 20,
        "increment": 20, "prefix": "source", "seed": 1,
        "backbone_type": "vit_base_patch16_224", "sa_cumulative_merge":
        "sensitivity_budgeted_g", "sa_g_budget_scope": "global",
        "lora_rank": 10, "batch_size": 64, "max_tasks": 10,
    }
    (source / "config.json").write_text(json.dumps(config))
    torch.save({}, source / "pre_merge.pt")
    torch.save({"version": 8, "task_id": 2, "budget_scope": "global"},
               source / "sa_state.pt")
    torch.save({}, source / "sa_merged_lora.pt")
    prototypes = {index: torch.ones(6) for index in range(40)}
    torch.save(prototypes, source / "sa_prototypes.pt")
    weight = torch.randn(40, 6)
    bias = torch.randn(40)
    torch.save(weight, source / "CLs_weight1.pt")
    torch.save(bias, source / "CLs_bias1.pt")
    audit_snapshot(source, finalize=True)
    log = tmp_path / "logs" / "sa_sdlora" / "cub" / "0" / "20"
    log.mkdir(parents=True)
    (log / "source_1_vit_base_patch16_224.log").write_text(
        "2026 [trainer.py] => CNN: {'00-19': np.float64(90.0)}\n"
        "CNN top1 curve: [np.float64(90.0)]\n"
        "CNN top5 curve: [np.float64(99.0)]\n"
        "2026 [trainer.py] => CNN: {'00-19': np.float64(85.0), '20-39': np.float64(87.0)}\n"
        "CNN top1 curve: [np.float64(90.0), np.float64(86.0)]\n"
        "CNN top5 curve: [np.float64(99.0), np.float64(98.0)]\n"
    )
    destination = tmp_path / "destination"
    destination.mkdir()
    args = dict(config, sa_resume_snapshot=str(source), filepath=str(destination), rank=0)
    return source, destination, args, weight, bias


def test_snapshot_resume_restores_state_head_prototypes_and_full_history(
    tmp_path, monkeypatch
):
    source, destination, args, weight, bias = _fixture(tmp_path, monkeypatch)
    learner = _Learner()
    start, metrics = restore_sbgc_snapshot(args, learner, _Manager())
    assert start == 2
    assert learner._cur_task == 1
    assert learner._known_classes == learner._total_classes == 40
    assert learner.network.backbone.task_id == 2
    assert torch.equal(learner.network.fc.weight, weight)
    assert torch.equal(learner.network.fc.bias, bias)
    assert len(learner.network.prototypes) == 40
    assert metrics == {
        "top1": [90.0, 86.0],
        "top5": [99.0, 98.0],
        "matrix": [[90.0], [85.0, 87.0]],
    }
    assert (destination / "sa_state.pt").read_bytes() == (source / "sa_state.pt").read_bytes()


def test_snapshot_resume_rejects_protocol_mismatch_without_copying(tmp_path, monkeypatch):
    _, destination, args, _, _ = _fixture(tmp_path, monkeypatch)
    args["batch_size"] = 32
    with pytest.raises(ValueError, match="batch_size"):
        restore_sbgc_snapshot(args, _Learner(), _Manager())
    assert not (destination / "sa_state.pt").exists()


def _live_fixture(tmp_path, monkeypatch):
    source, destination, args, weight, bias = _fixture(tmp_path, monkeypatch)
    config = json.loads((source / "config.json").read_text())
    config.pop("sa_g_budget_scope")
    config.update(
        sa_cumulative_merge="live_a_aggregate_b", sa_cumulative_state=True,
        sa_train_a_all_tasks=True, sa_deterministic_training=True,
        sa_live_a_coordinate_align=True, max_tasks=2,
    )
    (source / "config.json").write_text(json.dumps(config))
    torch.save({"version": 4, "task_id": 2, "merge_mode": "live_a_aggregate_b"},
               source / "sa_state.pt")
    (source / "complete.json").unlink()
    audit_snapshot(source, finalize=True)
    args = dict(config, max_tasks=10, sa_freeze_a_after_tasks=2,
                sa_resume_snapshot=str(source), filepath=str(destination), rank=0)
    return source, destination, args, weight, bias


def test_live_snapshot_forks_a_completed_prefix_with_full_metric_history(tmp_path, monkeypatch):
    assert hasattr(sa_snapshot_resume, "restore_task_snapshot"), "Live-A boundary resume is missing"
    source, destination, args, weight, bias = _live_fixture(tmp_path, monkeypatch)
    learner = _Learner()
    start, metrics = sa_snapshot_resume.restore_task_snapshot(args, learner, _Manager())
    assert start == 2
    assert learner._known_classes == 40 and learner._cur_task == 1
    assert torch.equal(learner.network.fc.weight, weight)
    assert torch.equal(learner.network.fc.bias, bias)
    assert metrics["top1"] == [90.0, 86.0]
    assert (destination / "sa_state.pt").read_bytes() == (source / "sa_state.pt").read_bytes()
    lineage = json.loads((destination / "snapshot_resume.json").read_text())
    assert lineage["source_snapshot"] == str(source.resolve())
    assert lineage["sha256"] == json.loads((source / "complete.json").read_text())["sha256"]


def test_live_snapshot_rejects_a_different_prefix_policy(tmp_path, monkeypatch):
    assert hasattr(sa_snapshot_resume, "restore_task_snapshot"), "Live-A boundary resume is missing"
    _, destination, args, _, _ = _live_fixture(tmp_path, monkeypatch)
    args["sa_freeze_a_after_tasks"] = 1
    with pytest.raises(ValueError, match="prefix A policy"):
        sa_snapshot_resume.restore_task_snapshot(args, _Learner(), _Manager())
    assert not (destination / "sa_state.pt").exists()


def test_live_snapshot_rejects_an_added_training_setting(tmp_path, monkeypatch):
    _, destination, args, _, _ = _live_fixture(tmp_path, monkeypatch)
    args["sa_hbd_lambda"] = 0.1
    with pytest.raises(ValueError, match="sa_hbd_lambda"):
        sa_snapshot_resume.restore_task_snapshot(args, _Learner(), _Manager())
    assert not (destination / "sa_state.pt").exists()


@pytest.mark.parametrize("fixture", [_fixture, _live_fixture])
def test_ddp_restore_synchronizes_before_and_after_rank_zero_copy(tmp_path, monkeypatch, fixture):
    _, destination, args, _, _ = fixture(tmp_path, monkeypatch)
    observed = []
    monkeypatch.setattr(sa_snapshot_resume.dist, "is_available", lambda: True)
    monkeypatch.setattr(sa_snapshot_resume.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sa_snapshot_resume.dist, "barrier", lambda: observed.append(
        (destination / "sa_state.pt").exists()
    ))
    sa_snapshot_resume.restore_task_snapshot(args, _Learner(), _Manager())
    assert observed == [False, True]
