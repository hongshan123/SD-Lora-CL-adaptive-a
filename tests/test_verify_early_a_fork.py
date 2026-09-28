"""A fork audit must prove the requested policy and source identity."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import verify_early_a_fork as verifier
from utils.sa_task_snapshots import audit_snapshot


def _fixture(tmp_path, monkeypatch, policy="freeze"):
    source = tmp_path / "prefix" / "task_snapshots" / "task_002"
    run = tmp_path / "suffix"
    target = run / "task_snapshots" / "task_003"
    config = {
        "sa_resume_snapshot": str(source), "max_tasks": 4,
        "sa_train_a_all_tasks": True,
    }
    if policy == "freeze":
        config["sa_freeze_a_after_tasks"] = 3
    state = {
        "shared_a": [torch.ones(10, 768) for _ in range(24)],
        "aggregate_up": [torch.zeros(768, 10) for _ in range(24)],
    }
    for directory, task in ((source, 2), (target, 3)):
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps(config))
        torch.save(dict(state, task_id=task + 1), directory / "sa_state.pt")
        for filename in ("pre_merge.pt", "sa_merged_lora.pt", "sa_prototypes.pt",
                         f"CLs_weight{task}.pt", f"CLs_bias{task}.pt"):
            torch.save({}, directory / filename)
        audit_snapshot(directory, finalize=True)
    lineage = {
        "source_snapshot": str(source.resolve()),
        "sha256": json.loads((source / "complete.json").read_text())["sha256"],
    }
    (run / "snapshot_resume.json").write_text(json.dumps(lineage))
    monkeypatch.setattr(verifier, "_previous_metrics", lambda *args: {
        "top1": [90.0, 89.0, 88.0, 87.0], "top5": [99.0] * 4,
    })
    return source, run


def test_fork_audit_requires_the_expected_policy(tmp_path, monkeypatch):
    source, run = _fixture(tmp_path, monkeypatch, policy="freeze")
    with pytest.raises(ValueError, match="policy"):
        verifier.verify_fork(run, expected_source=source, expected_policy="live")


def test_fork_audit_rejects_a_different_source(tmp_path, monkeypatch):
    source, run = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="source"):
        verifier.verify_fork(run, expected_source=tmp_path / "other", expected_policy="freeze")


def test_fork_audit_rejects_mismatched_import_fingerprint(tmp_path, monkeypatch):
    source, run = _fixture(tmp_path, monkeypatch)
    lineage = json.loads((run / "snapshot_resume.json").read_text())
    lineage["sha256"]["sa_state.pt"] = "wrong"
    (run / "snapshot_resume.json").write_text(json.dumps(lineage))
    with pytest.raises(ValueError, match="fingerprint"):
        verifier.verify_fork(run, expected_source=source, expected_policy="freeze")


def test_fork_audit_accepts_exact_frozen_source(tmp_path, monkeypatch):
    source, run = _fixture(tmp_path, monkeypatch)
    result = verifier.verify_fork(run, expected_source=source, expected_policy="freeze")
    assert result["verified"] and result["frozen_a_bit_identical"]


def test_pair_audit_rejects_mismatched_prefix_metrics(monkeypatch):
    def fake_verify(run, source, policy):
        return {
            "source_sha256": {"sa_state.pt": "same"}, "suffix_tasks": [3, 9],
            "top1": [90.0, 89.0, 88.0 if policy == "freeze" else 87.0, 80.0],
            "top5": [99.0] * 4,
        }
    monkeypatch.setattr(verifier, "verify_fork", fake_verify)
    with pytest.raises(ValueError, match="prefix metrics"):
        verifier.verify_pair("freeze", "live", "source")


def test_pair_audit_accepts_different_suffix_metrics(monkeypatch):
    def fake_verify(run, source, policy):
        return {
            "source_sha256": {"sa_state.pt": "same"}, "suffix_tasks": [3, 9],
            "top1": [90.0, 89.0, 88.0, 80.0 if policy == "freeze" else 81.0],
            "top5": [99.0] * 4,
        }
    monkeypatch.setattr(verifier, "verify_fork", fake_verify)
    assert verifier.verify_pair("freeze", "live", "source")["paired_verified"]
